#!/usr/bin/env python3
"""
Debug script to print end effector and block positions during demonstration replay.
"""

import numpy as np
import h5py
import os
import sys
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments'))
from environments.panda_demo_env import PandaDemoEnv


def debug_demo_positions(dataset_path, demo_idx=0):
    """
    Replay demonstration and print positions every 5 frames.
    """
    print(f"Loading demonstration {demo_idx} from {dataset_path}")
    
    # Initialize environment
    env = PandaDemoEnv(camera_name="demo_cam")
    
    with h5py.File(dataset_path, 'r') as f:
        # Check if demo exists
        demo_key = f'demo_{demo_idx}'
        if demo_key not in f:
            print(f"Error: {demo_key} not found in dataset")
            return
        
        demo_group = f[demo_key]
        
        # Get data
        actions = demo_group['actions'][:]
        joint_positions = demo_group['observations/joint_pos'][:]
        gripper_positions = demo_group['observations/gripper_pos'][:]
        target_block_positions = demo_group['observations/target_block_pos'][:]
        rewards = demo_group['rewards'][:]
        
        n_steps = len(actions)
        print(f"\nReplaying {n_steps} steps...")
        print(f"Initial block position from dataset: {target_block_positions[0]}")
        print(f"Final block position from dataset: {target_block_positions[-1]}")
        print(f"Block movement: {np.linalg.norm(target_block_positions[-1] - target_block_positions[0]):.4f} m")
        
        # Reset environment
        obs = env.reset()
        
        # Set initial joint positions
        for i in range(7):
            joint_addr = env.model.jnt_qposadr[env.arm_joint_ids[i]]
            env.data.qpos[joint_addr] = joint_positions[0, i]
        
        # Forward to update
        mujoco.mj_forward(env.model, env.data)
        
        # Get end effector site ID
        ee_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        
        # Print header
        print("\n" + "="*80)
        print(f"{'Step':>5} | {'EE Position (x,y,z)':>25} | {'Block Position (x,y,z)':>25} | {'Gripper':>8} | {'Reward':>8}")
        print("="*80)
        
        # Replay the demonstration
        for step in range(n_steps):
            # Apply action
            obs = env.step(actions[step])
            
            # Get positions every 5 steps or at critical moments
            if step % 5 == 0 or step in [0, 49, 50, 51, 70, 71, n_steps-1]:
                # Get end effector position
                if ee_site_id >= 0:
                    ee_pos = env.data.site_xpos[ee_site_id].copy()
                else:
                    # Fallback to hand body
                    hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
                    ee_pos = env.data.xpos[hand_id].copy()
                
                # Get block position from environment
                actual_block_pos = env.data.xpos[env.target_block_id].copy()
                
                # Gripper state
                gripper_open = gripper_positions[step] > 0.5
                gripper_str = "OPEN" if gripper_open else "CLOSED"
                
                # Print info
                print(f"{step:5d} | {ee_pos[0]:8.4f}, {ee_pos[1]:8.4f}, {ee_pos[2]:8.4f} | "
                      f"{actual_block_pos[0]:8.4f}, {actual_block_pos[1]:8.4f}, {actual_block_pos[2]:8.4f} | "
                      f"{gripper_str:>8} | {rewards[step]:8.2f}")
                
                # Check for discrepancies
                dataset_block_pos = target_block_positions[step]
                pos_diff = np.linalg.norm(actual_block_pos - dataset_block_pos)
                if pos_diff > 0.001:
                    print(f"      ⚠️  Block position mismatch! Dataset: {dataset_block_pos}, "
                          f"Actual: {actual_block_pos}, Diff: {pos_diff:.4f}")
        
        print("="*80)
        
        # Final analysis
        print("\n📊 Summary:")
        print(f"Total reward: {np.sum(rewards):.2f}")
        print(f"Final reward: {rewards[-1]:.2f}")
        
        # Check if block was lifted
        initial_block_z = target_block_positions[0, 2]
        final_block_z = target_block_positions[-1, 2]
        lift_height = final_block_z - initial_block_z
        print(f"\nBlock lift analysis:")
        print(f"  Initial height: {initial_block_z:.4f} m")
        print(f"  Final height: {final_block_z:.4f} m")
        print(f"  Lift height: {lift_height:.4f} m ({lift_height*1000:.1f} mm)")
        
        # Analyze phases
        print("\n🔍 Phase Analysis:")
        
        # Approach phase (0-50)
        ee_start = env.data.site_xpos[ee_site_id].copy() if ee_site_id >= 0 else env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")].copy()
        print(f"\n1. Approach (steps 0-50):")
        print(f"   EE movement: {np.linalg.norm(ee_start - ee_pos):.4f} m")
        
        # Grasp phase (50-70)
        print(f"\n2. Grasp (steps 50-70):")
        print(f"   Gripper closes at step ~50")
        
        # Lift phase (70-150)
        print(f"\n3. Lift (steps 70-150):")
        print(f"   Block lifted: {lift_height*1000:.1f} mm")
        
        # Check for success
        if lift_height > 0.005:  # 5mm threshold
            print(f"\n✅ SUCCESS: Block lifted {lift_height*1000:.1f} mm")
        else:
            print(f"\n❌ FAILURE: Block only lifted {lift_height*1000:.1f} mm (need > 5mm)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug demonstration positions")
    parser.add_argument("--dataset", type=str, 
                       default="../../datasets/expert_demonstrations.hdf5",
                       help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=int, default=0,
                       help="Demo index to debug")
    
    args = parser.parse_args()
    
    debug_demo_positions(args.dataset, args.demo)