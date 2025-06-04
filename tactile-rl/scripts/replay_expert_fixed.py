#!/usr/bin/env python3
"""
Fixed replay script that properly handles gripper control.
"""

import numpy as np
import h5py
import os
import sys
import mujoco
import imageio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments'))
from environments.panda_demo_env import PandaDemoEnv


def replay_demo_fixed(dataset_path, demo_idx=0, output_path=None):
    """
    Replay a demonstration with fixed gripper control.
    """
    print(f"Loading demonstration {demo_idx} from {dataset_path}")
    
    # Initialize environment
    env = PandaDemoEnv(camera_name="demo_cam")
    
    frames = []
    
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
        tactile_readings = demo_group['tactile_readings'][:]
        rewards = demo_group['rewards'][:]
        target_positions = demo_group['observations/target_block_pos'][:]
        
        n_steps = len(actions)
        print(f"Replaying {n_steps} steps...")
        print(f"Action space: {actions.shape}")
        print(f"First action: {actions[0]}")
        print(f"Gripper actions: min={actions[:, 7].min()}, max={actions[:, 7].max()}")
        
        # Reset environment
        obs = env.reset()
        
        # Print initial info
        print(f"\nInitial state:")
        print(f"  Gripper joint ID: {env.gripper_joint1_id}")
        print(f"  Control dimension: {env.model.nu}")
        print(f"  Initial gripper pos: {obs['gripper_pos']}")
        
        # Render initial frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Replay the demonstration
        for step in tqdm(range(n_steps), desc="Replaying"):
            # Apply the full action from the dataset
            obs = env.step(actions[step])
            
            # Check gripper state periodically
            if step % 30 == 0:
                print(f"\nStep {step}:")
                print(f"  Action: {actions[step]}")
                print(f"  Gripper action: {actions[step, 7]}")
                print(f"  Gripper pos from obs: {obs['gripper_pos']}")
                print(f"  Tactile sum: {np.sum(obs['tactile']):.1f}")
            
            # Render frame (every other step)
            if step % 2 == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        
        # Add extra frames at the end
        for _ in range(15):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # Print summary
        print(f"\nDemo Summary:")
        print(f"Total steps: {n_steps}")
        print(f"Total reward: {np.sum(rewards):.2f}")
        print(f"Final tactile: {np.sum(tactile_readings[-1]):.2f}")
        print(f"Block movement: {np.linalg.norm(target_positions[-1] - target_positions[0]):.4f}m")
        
        # Analyze gripper behavior
        gripper_close_steps = np.where(actions[:, 7] > 0.5)[0]
        print(f"\nGripper analysis:")
        print(f"  Close commands at steps: {gripper_close_steps[:10]}..." if len(gripper_close_steps) > 10 else f"  Close commands at steps: {gripper_close_steps}")
        print(f"  Total close commands: {len(gripper_close_steps)}")
    
    # Save video
    if frames:
        if output_path is None:
            output_path = f"../../datasets/expert_demo_{demo_idx}_fixed.mp4"
        
        print(f"\nSaving video with {len(frames)} frames...")
        imageio.mimsave(output_path, frames, fps=30, quality=8, codec='libx264')
        print(f"Video saved to: {output_path}")
        
        # GIF version
        gif_path = output_path.replace('.mp4', '.gif')
        gif_frames = frames[::3]
        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
        print(f"GIF saved to: {gif_path}")
        
        return len(frames)
    else:
        print("No frames generated")
        return 0


def diagnose_gripper_issue(dataset_path):
    """Diagnose why the gripper isn't working."""
    print("\n🔍 Diagnosing gripper control issue...")
    
    # Check the environment
    env = PandaDemoEnv(camera_name="demo_cam")
    
    print(f"\nEnvironment info:")
    print(f"  Model nu (controls): {env.model.nu}")
    print(f"  Model nq (positions): {env.model.nq}")
    
    # List all actuators
    print(f"\nActuators:")
    for i in range(env.model.nu):
        actuator_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {actuator_name}")
    
    # Check gripper joints
    print(f"\nGripper joints:")
    for name in ["finger_joint1", "finger_joint2", "left_finger_joint", "right_finger_joint"]:
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            print(f"  {name}: id={joint_id}")
    
    # Test gripper control
    print(f"\nTesting gripper control...")
    obs = env.reset()
    print(f"Initial gripper state: {obs['gripper_pos']}")
    
    # Try to close gripper
    action = np.zeros(8)
    action[7] = 1.0  # Close command
    obs = env.step(action)
    print(f"After close command: {obs['gripper_pos']}")
    
    # Try to open gripper
    action[7] = -1.0  # Open command
    obs = env.step(action)
    print(f"After open command: {obs['gripper_pos']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed replay of expert demonstrations")
    parser.add_argument("--dataset", type=str, 
                       default="../../datasets/expert_demonstrations.hdf5",
                       help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=int, default=0, 
                       help="Demonstration index to replay")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for video")
    parser.add_argument("--diagnose", action="store_true",
                       help="Diagnose gripper issues")
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_gripper_issue(args.dataset)
    else:
        replay_demo_fixed(args.dataset, args.demo, args.output)