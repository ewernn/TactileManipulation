"""
Train baseline and tactile-enhanced policies using behavioral cloning.
Compare performance to demonstrate tactile sensing advantage.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GraspingDataset(Dataset):
    """Dataset for grasping demonstrations."""
    
    def __init__(self, hdf5_path, use_tactile=True):
        self.use_tactile = use_tactile
        self.data = []
        
        with h5py.File(hdf5_path, 'r') as f:
            for demo_key in f.keys():
                if not demo_key.startswith('demo_'):
                    continue
                    
                demo = f[demo_key]
                
                # Get data
                ee_pos = demo['obs/ee_pos'][:]
                cube_pos = demo['obs/cube_pos'][:]
                gripper_width = demo['obs/gripper_width'][:]
                actions = demo['actions'][:]
                
                # Create state vectors
                for i in range(len(actions)):
                    # Basic state: ee_pos, cube_pos, gripper_width
                    state = np.concatenate([
                        ee_pos[i],
                        cube_pos[i],
                        [gripper_width[i]]
                    ])
                    
                    # Add tactile if available (placeholder for now)
                    if self.use_tactile:
                        # Simulate tactile based on gripper width and cube distance
                        distance = np.linalg.norm(ee_pos[i] - cube_pos[i])
                        contact = distance < 0.05 and gripper_width[i] < 0.05
                        tactile = np.random.randn(24) * 0.1  # Placeholder
                        if contact:
                            tactile += np.random.rand(24) * 2  # Add contact signal
                        state = np.concatenate([state, tactile])
                    
                    self.data.append((state, actions[i]))
                    
        print(f"Loaded {len(self.data)} samples from {hdf5_path}")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        state, action = self.data[idx]
        return torch.FloatTensor(state), torch.FloatTensor(action)


class BaselinePolicy(nn.Module):
    """MLP policy without tactile input."""
    
    def __init__(self, state_dim=7, action_dim=4, hidden_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are normalized to [-1, 1]
        )
        
    def forward(self, state):
        return self.net(state)


class TactilePolicy(nn.Module):
    """Policy with tactile input processing."""
    
    def __init__(self, state_dim=7, tactile_dim=24, action_dim=4, hidden_dim=128):
        super().__init__()
        
        # Tactile encoder
        self.tactile_encoder = nn.Sequential(
            nn.Linear(tactile_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Main policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        # Split state and tactile
        basic_state = state[:, :7]
        tactile = state[:, 7:]
        
        # Encode tactile
        tactile_features = self.tactile_encoder(tactile)
        
        # Combine and predict
        combined = torch.cat([basic_state, tactile_features], dim=1)
        return self.policy_net(combined)


def train_policy(model, dataloader, num_epochs=50, lr=1e-3, device='cpu'):
    """Train policy using behavioral cloning."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for states, actions in progress:
            states = states.to(device)
            actions = actions.to(device)
            
            # Forward pass
            pred_actions = model(states)
            loss = criterion(pred_actions, actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            progress.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")
        
    return losses


def evaluate_policy(model, env_class, num_episodes=20, use_tactile=False, device='cpu'):
    """Evaluate policy performance."""
    
    from environments.simple_tactile_env import SimpleTactileGraspingEnv
    
    model.eval()
    env = SimpleTactileGraspingEnv(use_tactile=use_tactile)
    
    successes = 0
    lift_heights = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # Convert observation to state vector
            state = np.concatenate([
                obs['ee_pos'],
                obs['cube_pose'][:3],
                [obs['joint_pos'][3] + obs['joint_pos'][4]]  # gripper width
            ])
            
            if use_tactile:
                # Add placeholder tactile (would use real tactile in full implementation)
                tactile = np.zeros(24)
                state = np.concatenate([state, tactile])
                
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(state_tensor).cpu().numpy()[0]
                
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
        if info['success']:
            successes += 1
        lift_heights.append(info['cube_lifted'])
        
    success_rate = successes / num_episodes
    avg_lift = np.mean(lift_heights)
    
    return {
        'success_rate': success_rate,
        'avg_lift_height': avg_lift,
        'lift_heights': lift_heights
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, 
                       default='../../datasets/tactile_grasping/direct_demos.hdf5')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    # Create datasets
    baseline_dataset = GraspingDataset(args.data_path, use_tactile=False)
    tactile_dataset = GraspingDataset(args.data_path, use_tactile=True)
    
    baseline_loader = DataLoader(baseline_dataset, batch_size=args.batch_size, shuffle=True)
    tactile_loader = DataLoader(tactile_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create models
    baseline_model = BaselinePolicy()
    tactile_model = TactilePolicy()
    
    print("\nTraining baseline policy (without tactile)...")
    baseline_losses = train_policy(baseline_model, baseline_loader, 
                                 num_epochs=args.num_epochs, lr=args.lr, device=args.device)
    
    print("\nTraining tactile-enhanced policy...")
    tactile_losses = train_policy(tactile_model, tactile_loader,
                                num_epochs=args.num_epochs, lr=args.lr, device=args.device)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_losses, label='Baseline (no tactile)', linewidth=2)
    plt.plot(tactile_losses, label='Tactile-enhanced', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('../../datasets/tactile_grasping/training_curves.png')
    print("\nSaved training curves to datasets/tactile_grasping/training_curves.png")
    
    # Evaluate policies
    print("\nEvaluating baseline policy...")
    baseline_results = evaluate_policy(baseline_model, None, num_episodes=20, 
                                     use_tactile=False, device=args.device)
    
    print("\nEvaluating tactile policy...")
    tactile_results = evaluate_policy(tactile_model, None, num_episodes=20,
                                    use_tactile=True, device=args.device)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nBaseline Policy (no tactile):")
    print(f"  Success rate: {baseline_results['success_rate']:.1%}")
    print(f"  Average lift height: {baseline_results['avg_lift_height']:.3f}m")
    
    print(f"\nTactile-Enhanced Policy:")
    print(f"  Success rate: {tactile_results['success_rate']:.1%}")
    print(f"  Average lift height: {tactile_results['avg_lift_height']:.3f}m")
    
    improvement = (tactile_results['success_rate'] - baseline_results['success_rate']) / baseline_results['success_rate']
    print(f"\nImprovement with tactile: {improvement:.1%}")
    
    # Save models
    torch.save(baseline_model.state_dict(), '../../datasets/tactile_grasping/baseline_policy.pth')
    torch.save(tactile_model.state_dict(), '../../datasets/tactile_grasping/tactile_policy.pth')
    print("\nSaved trained models to datasets/tactile_grasping/")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rates
    policies = ['Baseline', 'Tactile']
    success_rates = [baseline_results['success_rate'], tactile_results['success_rate']]
    ax1.bar(policies, success_rates, color=['blue', 'green'])
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Grasping Success Rate')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(success_rates):
        ax1.text(i, v + 0.05, f'{v:.1%}', ha='center')
        
    # Lift heights distribution
    ax2.boxplot([baseline_results['lift_heights'], tactile_results['lift_heights']], 
                labels=policies)
    ax2.set_ylabel('Lift Height (m)')
    ax2.set_title('Lift Height Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../datasets/tactile_grasping/policy_comparison.png')
    print("Saved comparison plot to datasets/tactile_grasping/policy_comparison.png")


if __name__ == "__main__":
    main()