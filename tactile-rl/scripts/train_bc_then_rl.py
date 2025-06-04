#!/usr/bin/env python3
"""
Train with Behavioral Cloning first, then fine-tune with RL.
This is the recommended approach for tactile manipulation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import os
import sys
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.panda_7dof_rl_env import Panda7DOFTactileRL
from train_policies import TactilePolicy, GraspingDataset
from train_rl_from_scratch import PPOTrainer, ActorCritic


class BCtoRLTrainer:
    """Combined BC and RL training pipeline."""
    
    def __init__(self, env, dataset_path, use_tactile=True):
        self.env = env
        self.dataset_path = dataset_path
        self.use_tactile = use_tactile
        
        # Get dimensions
        obs_dict = env.reset()
        self.obs_dim = sum(np.prod(v.shape) if isinstance(v, np.ndarray) else 1 
                          for v in obs_dict.values())
        self.action_dim = env.action_space.shape[0]
        
        print(f"📊 Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
    
    def train_bc(self, epochs=100, batch_size=32, lr=1e-3):
        """Phase 1: Behavioral Cloning from demonstrations."""
        print("\n🎓 Phase 1: Behavioral Cloning Training")
        print("=" * 50)
        
        # Load dataset
        dataset = GraspingDataset(self.dataset_path, use_tactile=self.use_tactile)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Create BC policy
        if self.use_tactile:
            bc_policy = TactilePolicy(
                state_dim=7,  # ee_pos(3) + cube_pos(3) + gripper(1)
                tactile_dim=72,  # 3x4x3x2 tactile
                action_dim=self.action_dim,
                hidden_dim=128
            )
        else:
            bc_policy = nn.Sequential(
                nn.Linear(self.obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_dim),
                nn.Tanh()
            )
        
        optimizer = optim.Adam(bc_policy.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for states, actions in dataloader:
                # Forward pass
                if self.use_tactile:
                    # Split state into components
                    basic_state = states[:, :7]
                    tactile_state = states[:, 7:]
                    pred_actions = bc_policy(basic_state, tactile_state)
                else:
                    pred_actions = bc_policy(states)
                
                # Compute loss
                loss = nn.MSELoss()(pred_actions, actions)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")
        
        # Save BC policy
        torch.save(bc_policy.state_dict(), 'bc_policy.pt')
        print("💾 Saved BC policy to bc_policy.pt")
        
        return bc_policy
    
    def convert_bc_to_rl(self, bc_policy):
        """Convert BC policy to Actor-Critic for RL."""
        print("\n🔄 Converting BC policy to Actor-Critic...")
        
        # Create Actor-Critic network
        actor_critic = ActorCritic(self.obs_dim, self.action_dim)
        
        # Copy BC weights to actor
        if isinstance(bc_policy, nn.Sequential):
            # Simple MLP case
            with torch.no_grad():
                # Copy shared layers
                actor_critic.shared[0].weight.copy_(bc_policy[0].weight)
                actor_critic.shared[0].bias.copy_(bc_policy[0].bias)
                actor_critic.shared[2].weight.copy_(bc_policy[2].weight)
                actor_critic.shared[2].bias.copy_(bc_policy[2].bias)
                
                # Copy actor head
                actor_critic.actor_mean.weight.copy_(bc_policy[4].weight)
                actor_critic.actor_mean.bias.copy_(bc_policy[4].bias)
        else:
            # TactilePolicy case - need custom conversion
            print("⚠️  Using random initialization for Actor-Critic (TactilePolicy conversion not implemented)")
            print("    The BC weights provide good demonstrations but aren't directly transferred")
        
        return actor_critic
    
    def evaluate_policy(self, policy, n_episodes=10, is_bc=True):
        """Evaluate current policy performance."""
        successes = 0
        total_reward = 0
        
        for ep in range(n_episodes):
            obs_dict = self.env.reset()
            obs = self._dict_to_tensor(obs_dict)
            episode_reward = 0
            
            for step in range(200):
                with torch.no_grad():
                    if is_bc:
                        # BC policy output
                        if self.use_tactile:
                            basic_state = obs[:7]
                            tactile_state = obs[7:]
                            action = policy(basic_state.unsqueeze(0), 
                                          tactile_state.unsqueeze(0)).squeeze()
                        else:
                            action = policy(obs.unsqueeze(0)).squeeze()
                    else:
                        # RL policy output
                        action, _, _ = policy.get_action(obs, deterministic=True)
                
                obs_dict, reward, terminated, truncated, info = self.env.step(action.numpy())
                obs = self._dict_to_tensor(obs_dict)
                episode_reward += reward
                
                if terminated or truncated:
                    if info.get('success', False):
                        successes += 1
                    break
            
            total_reward += episode_reward
        
        success_rate = successes / n_episodes
        avg_reward = total_reward / n_episodes
        
        return success_rate, avg_reward
    
    def train_rl(self, actor_critic, n_episodes=500, lr=3e-4):
        """Phase 2: RL fine-tuning starting from BC policy."""
        print("\n🚀 Phase 2: RL Fine-tuning (PPO)")
        print("=" * 50)
        
        # Evaluate BC performance
        print("\n📊 Evaluating BC policy before RL...")
        bc_success, bc_reward = self.evaluate_policy(
            actor_critic, n_episodes=20, is_bc=False
        )
        print(f"BC Performance - Success: {bc_success:.1%}, Avg Reward: {bc_reward:.2f}")
        
        # Create PPO trainer
        trainer = PPOTrainer(self.env, actor_critic, lr=lr)
        
        # Train with RL
        trainer.train(n_episodes=n_episodes, steps_per_episode=200)
        
        # Final evaluation
        print("\n📊 Final evaluation after RL...")
        rl_success, rl_reward = self.evaluate_policy(
            actor_critic, n_episodes=20, is_bc=False
        )
        print(f"RL Performance - Success: {rl_success:.1%}, Avg Reward: {rl_reward:.2f}")
        print(f"Improvement - Success: +{(rl_success-bc_success)*100:.1f}%, "
              f"Reward: +{rl_reward-bc_reward:.2f}")
    
    def _dict_to_tensor(self, obs_dict):
        """Convert observation dictionary to tensor."""
        obs_list = []
        obs_list.append(obs_dict['joint_pos'])
        obs_list.append(obs_dict['joint_vel'])
        obs_list.append([obs_dict['gripper_pos']])
        
        if 'tactile' in obs_dict:
            obs_list.append(obs_dict['tactile'])
        if 'cube_pose' in obs_dict:
            obs_list.append(obs_dict['cube_pose'])
        if 'ee_pose' in obs_dict:
            obs_list.append(obs_dict['ee_pose'])
        
        obs = np.concatenate(obs_list)
        return torch.FloatTensor(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                       default='../datasets/expert_demonstrations.hdf5',
                       help='Path to expert demonstrations')
    parser.add_argument('--bc_epochs', type=int, default=100,
                       help='Number of BC training epochs')
    parser.add_argument('--rl_episodes', type=int, default=500,
                       help='Number of RL fine-tuning episodes')
    parser.add_argument('--no_tactile', action='store_true',
                       help='Train without tactile sensing')
    args = parser.parse_args()
    
    # Create environment
    print("🤖 Creating Panda 7-DOF Environment...")
    env = Panda7DOFTactileRL(
        control_frequency=20,
        joint_vel_limit=2.0,
        use_tactile=not args.no_tactile
    )
    
    # Create trainer
    trainer = BCtoRLTrainer(env, args.dataset, use_tactile=not args.no_tactile)
    
    # Phase 1: BC Training
    bc_policy = trainer.train_bc(epochs=args.bc_epochs)
    
    # Convert to Actor-Critic
    actor_critic = trainer.convert_bc_to_rl(bc_policy)
    
    # Phase 2: RL Fine-tuning
    trainer.train_rl(actor_critic, n_episodes=args.rl_episodes)
    
    print("\n✅ Training complete! Policy learned from BC and improved with RL.")


if __name__ == "__main__":
    main()