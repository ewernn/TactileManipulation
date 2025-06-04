#!/usr/bin/env python3
"""
Train a Panda 7-DOF robot policy from scratch using RL (PPO).
Fixed version that uses the corrected environment with blocks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import sys
import argparse
from collections import deque
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.panda_7dof_rl_env_fixed import Panda7DOFTactileRL


class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs):
        features = self.shared(obs)
        
        # Actor
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        # Critic
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            return action_mean, value
        
        # Sample from normal distribution
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


class PPOTrainer:
    """PPO algorithm implementation."""
    
    def __init__(self, env, actor_critic, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01):
        self.env = env
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, obs_batch, action_batch, log_prob_batch, value_batch,
               advantage_batch, return_batch, epochs=10):
        """Update policy using PPO."""
        
        for _ in range(epochs):
            # Get current policy outputs
            action_mean, action_std, values = self.actor_critic(obs_batch)
            dist = Normal(action_mean, action_std)
            log_probs = dist.log_prob(action_batch).sum(dim=-1)
            
            # Compute ratios
            ratios = torch.exp(log_probs - log_prob_batch)
            
            # Clipped objective
            surr1 = ratios * advantage_batch
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantage_batch
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = ((values.squeeze() - return_batch) ** 2).mean()
            
            # Entropy bonus
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
    
    def train(self, n_episodes=1000, steps_per_episode=200, update_every=2048):
        """Main training loop."""
        
        # Storage
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        done_buffer = []
        
        episode_rewards = deque(maxlen=100)
        episode_successes = deque(maxlen=100)
        total_steps = 0
        
        print("\n🚀 Starting PPO Training from Scratch!")
        print("=" * 50)
        
        for episode in range(n_episodes):
            obs_dict = self.env.reset()
            obs = self._dict_to_tensor(obs_dict)
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Get action from policy
                with torch.no_grad():
                    action, log_prob, value = self.actor_critic.get_action(obs)
                
                # Step environment
                next_obs_dict, reward, terminated, truncated, info = self.env.step(
                    action.numpy()
                )
                done = terminated or truncated
                
                # Store transition
                obs_buffer.append(obs)
                action_buffer.append(action)
                log_prob_buffer.append(log_prob)
                value_buffer.append(value.squeeze())
                reward_buffer.append(reward)
                done_buffer.append(done)
                
                episode_reward += reward
                total_steps += 1
                
                # Update observation
                obs = self._dict_to_tensor(next_obs_dict)
                
                # Update policy
                if total_steps % update_every == 0:
                    self._ppo_update(obs_buffer, action_buffer, log_prob_buffer,
                                   value_buffer, reward_buffer, done_buffer, obs)
                    
                    # Clear buffers
                    obs_buffer.clear()
                    action_buffer.clear()
                    log_prob_buffer.clear()
                    value_buffer.clear()
                    reward_buffer.clear()
                    done_buffer.clear()
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_successes.append(float(info.get('success', 0)))
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                avg_success = np.mean(episode_successes) if episode_successes else 0
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Success Rate: {avg_success:.1%} | "
                      f"Steps: {total_steps:6d}")
                
                # Save checkpoint
                if episode % 100 == 0 and episode > 0:
                    self._save_checkpoint(episode)
    
    def _dict_to_tensor(self, obs_dict):
        """Convert observation dictionary to tensor."""
        # Concatenate all observations in a consistent order
        obs_list = []
        
        # Robot state
        obs_list.append(obs_dict['joint_pos'])
        obs_list.append(obs_dict['joint_vel'])
        obs_list.append(obs_dict['gripper_pos'])
        
        # Block state
        obs_list.append(obs_dict['target_block_pose'])
        
        # End-effector
        obs_list.append(obs_dict['ee_pose'])
        
        # Tactile (if available)
        if 'tactile' in obs_dict:
            obs_list.append(obs_dict['tactile'])
        
        obs = np.concatenate(obs_list)
        return torch.FloatTensor(obs)
    
    def _ppo_update(self, obs_buffer, action_buffer, log_prob_buffer,
                    value_buffer, reward_buffer, done_buffer, last_obs):
        """Perform PPO update."""
        
        # Convert to tensors
        obs_batch = torch.stack(obs_buffer)
        action_batch = torch.stack(action_buffer)
        log_prob_batch = torch.stack(log_prob_buffer)
        value_batch = torch.stack(value_buffer)
        reward_batch = torch.FloatTensor(reward_buffer)
        done_batch = torch.FloatTensor(done_buffer)
        
        # Get last value
        with torch.no_grad():
            _, _, last_value = self.actor_critic(last_obs)
            last_value = last_value.squeeze()
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            reward_batch, value_batch, done_batch, last_value
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update
        self.update(obs_batch, action_batch, log_prob_batch, value_batch,
                   advantages, returns)
    
    def _save_checkpoint(self, episode):
        """Save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        path = f'checkpoints/ppo_tactile_ep{episode}.pt'
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--no_tactile', action='store_true',
                       help='Train without tactile sensing')
    parser.add_argument('--test', action='store_true',
                       help='Just test the environment')
    args = parser.parse_args()
    
    # Create environment
    print("🤖 Creating Panda 7-DOF RL Environment...")
    env = Panda7DOFTactileRL(
        control_frequency=20,
        joint_vel_limit=2.0,
        use_tactile=not args.no_tactile
    )
    
    # Get observation dimension
    obs_dict = env.reset()
    obs_dim = sum(np.prod(v.shape) if isinstance(v, np.ndarray) else 1 
                  for v in obs_dict.values())
    action_dim = env.action_space.shape[0]
    
    print(f"📊 Observation dimension: {obs_dim}")
    print(f"🎮 Action dimension: {action_dim}")
    print(f"🔧 Tactile sensing: {'Enabled' if not args.no_tactile else 'Disabled'}")
    
    if args.test:
        print("\n🧪 Running environment test...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: reward={reward:.3f}, block_height={info['block_height']:.3f}")
            if terminated or truncated:
                break
        print("✅ Test complete!")
        return
    
    # Create networks
    actor_critic = ActorCritic(obs_dim, action_dim, args.hidden_dim)
    
    # Create trainer
    trainer = PPOTrainer(env, actor_critic, lr=args.lr)
    
    # Train
    trainer.train(n_episodes=args.episodes)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()