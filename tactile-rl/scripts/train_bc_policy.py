#!/usr/bin/env python3
"""
Train Behavior Cloning (BC) policy from expert demonstrations.
Designed to work both locally and on Google Colab with T4/A100 GPUs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import argparse
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple, Optional


class BCDataset(Dataset):
    """Dataset for behavior cloning from expert demonstrations."""
    
    def __init__(self, hdf5_path: str, normalize: bool = True):
        """Load expert demonstrations from HDF5 file."""
        self.normalize = normalize
        
        # Load all demonstrations into memory (small dataset)
        with h5py.File(hdf5_path, 'r') as f:
            print(f"Loading {f.attrs['num_demos']} demonstrations...")
            
            self.observations = []
            self.actions = []
            
            for i in range(f.attrs['num_demos']):
                demo = f[f'demo_{i}']
                
                # Get observations - combine proprio, tactile, and object state
                obs = demo['observations']
                proprio = obs['proprio'][:]  # 14D
                tactile = obs['tactile'][:]  # 24D
                object_state = obs['object_state'][:]  # 14D
                
                # Concatenate to form 52D observation
                combined_obs = np.concatenate([proprio, tactile, object_state], axis=1)
                
                self.observations.append(combined_obs)
                self.actions.append(demo['actions'][:])
            
            # Concatenate all demos
            self.observations = np.concatenate(self.observations, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
            
            print(f"Total samples: {len(self.observations)}")
            print(f"Observation shape: {self.observations.shape}")
            print(f"Action shape: {self.actions.shape}")
        
        # Compute normalization statistics
        if self.normalize:
            self.obs_mean = self.observations.mean(axis=0)
            self.obs_std = self.observations.std(axis=0) + 1e-6
            self.action_mean = self.actions.mean(axis=0)
            self.action_std = self.actions.std(axis=0) + 1e-6
            
            # Save normalization stats
            self.norm_stats = {
                'obs_mean': self.obs_mean.tolist(),
                'obs_std': self.obs_std.tolist(),
                'action_mean': self.action_mean.tolist(),
                'action_std': self.action_std.tolist()
            }
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]
        
        if self.normalize:
            obs = (obs - self.obs_mean) / self.obs_std
            action = (action - self.action_mean) / self.action_std
        
        return {
            'observation': torch.FloatTensor(obs),
            'action': torch.FloatTensor(action)
        }


class BCPolicy(nn.Module):
    """Behavior Cloning policy network."""
    
    def __init__(self, obs_dim: int = 52, action_dim: int = 8, 
                 hidden_dims: List[int] = [256, 256], dropout: float = 0.1):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build MLP
        layers = []
        dims = [obs_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(dims[i+1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        
        # Output head
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs):
        features = self.encoder(obs)
        actions = self.action_head(features)
        
        # Tanh to ensure actions are in [-1, 1]
        actions = torch.tanh(actions)
        
        return actions


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    losses = []
    
    for batch in dataloader:
        obs = batch['observation'].to(device)
        target_actions = batch['action'].to(device)
        
        # Forward pass
        pred_actions = model(obs)
        
        # MSE loss
        loss = F.mse_loss(pred_actions, target_actions)
        
        # Add L2 penalty on velocities (first 7 dims)
        velocity_penalty = 0.01 * (pred_actions[:, :7] ** 2).mean()
        loss = loss + velocity_penalty
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)


def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            obs = batch['observation'].to(device)
            target_actions = batch['action'].to(device)
            
            pred_actions = model(obs)
            loss = F.mse_loss(pred_actions, target_actions)
            
            losses.append(loss.item())
    
    return np.mean(losses)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, 
                   norm_stats, save_path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'norm_stats': norm_stats,
        'model_config': {
            'obs_dim': model.obs_dim,
            'action_dim': model.action_dim
        }
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train BC policy")
    parser.add_argument('--demos', type=str, required=True, help='Path to expert demos HDF5')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--save_dir', type=str, default='./bc_checkpoints')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu', 'mps'])
    parser.add_argument('--log_interval', type=int, default=10)
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading demonstrations from {args.demos}")
    dataset = BCDataset(args.demos, normalize=True)
    
    # Split into train/val
    n_samples = len(dataset)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda')
    )
    
    # Create model
    model = BCPolicy(
        obs_dim=52,
        action_dim=8,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    training_history = []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                dataset.norm_stats,
                os.path.join(args.save_dir, 'best_model.pt')
            )
        
        # Regular checkpoints
        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                dataset.norm_stats,
                os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )
        
        # Logging
        if epoch % args.log_interval == 0:
            elapsed = time.time() - start_time
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Total: {elapsed/60:.1f}min")
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs-1, train_loss, val_loss,
        dataset.norm_stats,
        os.path.join(args.save_dir, 'final_model.pt')
    )
    
    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()