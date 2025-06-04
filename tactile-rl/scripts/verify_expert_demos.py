#!/usr/bin/env python3
"""
Verify the expert demonstrations HDF5 file format and contents.
"""

import h5py
import numpy as np
import os

def verify_demonstrations(hdf5_path: str):
    """Verify the demonstration file structure and contents."""
    print(f"\nVerifying: {hdf5_path}")
    print("="*60)
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check attributes
        print("\nFile Attributes:")
        for key in f.attrs.keys():
            print(f"  {key}: {f.attrs[key]}")
        
        num_demos = f.attrs['num_demos']
        print(f"\nNumber of demonstrations: {num_demos}")
        
        # Check each demo
        for demo_idx in range(min(2, num_demos)):  # Check first 2 demos
            print(f"\n{'='*40}")
            print(f"Demo {demo_idx}:")
            
            demo_grp = f[f'demo_{demo_idx}']
            
            # Check attributes
            print("\n  Attributes:")
            for key in demo_grp.attrs.keys():
                val = demo_grp.attrs[key]
                if isinstance(val, float):
                    print(f"    {key}: {val:.3f}")
                else:
                    print(f"    {key}: {val}")
            
            # Check datasets
            print("\n  Datasets:")
            print(f"    actions: shape={demo_grp['actions'].shape}, dtype={demo_grp['actions'].dtype}")
            print(f"    rewards: shape={demo_grp['rewards'].shape}, dtype={demo_grp['rewards'].dtype}")
            print(f"    tactile_readings: shape={demo_grp['tactile_readings'].shape}, dtype={demo_grp['tactile_readings'].dtype}")
            
            # Check observations
            print("\n  Observations:")
            obs_grp = demo_grp['observations']
            for key in sorted(obs_grp.keys()):
                data = obs_grp[key]
                print(f"    {key}: shape={data.shape}, dtype={data.dtype}")
            
            # Sample some data
            print("\n  Sample Data:")
            actions = demo_grp['actions'][:]
            print(f"    First action: {actions[0]}")
            print(f"    Last action: {actions[-1]}")
            print(f"    Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
            
            tactile = demo_grp['tactile_readings'][:]
            print(f"    Tactile range: [{np.min(tactile):.3f}, {np.max(tactile):.3f}]")
            
            # Check trajectory
            ee_pos = obs_grp['ee_pos'][:]
            print(f"    EE trajectory: start={ee_pos[0]}, end={ee_pos[-1]}")
            
            target_pos = obs_grp['target_block_pos'][:]
            print(f"    Red block: start={target_pos[0]}, end={target_pos[-1]}")
    
    print(f"\n{'='*60}")
    print("✅ Verification complete!")

if __name__ == "__main__":
    # Find the most recent demo file
    demo_dir = "../datasets/expert"
    if os.path.exists(demo_dir):
        files = [f for f in os.listdir(demo_dir) if f.endswith('.hdf5')]
        if files:
            # Sort by timestamp and get most recent
            files.sort(reverse=True)
            latest_file = os.path.join(demo_dir, files[0])
            verify_demonstrations(latest_file)
        else:
            print("No HDF5 files found in", demo_dir)
    else:
        print(f"Directory {demo_dir} does not exist")