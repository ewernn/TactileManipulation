#!/usr/bin/env python3
"""
Replay expert demonstrations from HDF5 file and create a video.
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

def replay_and_record_demo(dataset_path, demo_idx=0, output_path=None):
    """
    Replay a demonstration from HDF5 file and record video.
    
    Args:
        dataset_path: Path to HDF5 dataset
        demo_idx: Index of demonstration to replay
        output_path: Output path for video
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
        
        n_steps = len(actions)
        print(f"Replaying {n_steps} steps...")
        
        # Reset environment
        obs = env.reset()
        
        # Set initial joint positions
        for i in range(7):
            joint_addr = env.model.jnt_qposadr[env.arm_joint_ids[i]]
            env.data.qpos[joint_addr] = joint_positions[0, i]
        
        # Set initial gripper position
        # The last control is for gripper (index 7 for 8D control)
        gripper_ctrl = 0.04 if gripper_positions[0] > 0.5 else 0.0
        env.data.ctrl[7] = gripper_ctrl
        
        # Forward to update
        mujoco.mj_forward(env.model, env.data)
        
        # Render initial frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Replay the demonstration
        for step in tqdm(range(n_steps), desc="Replaying"):
            # Apply action
            obs = env.step(actions[step])
            
            # Render frame (every other step to keep video manageable)
            if step % 2 == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
        
        # Add extra frames at the end to show final state
        for _ in range(15):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # Print summary
        print(f"\nDemo Summary:")
        print(f"Total steps: {n_steps}")
        print(f"Total reward: {np.sum(rewards):.2f}")
        print(f"Max tactile reading: {np.max(tactile_readings):.2f}")
        print(f"Final reward: {rewards[-1]:.2f}")
    
    # Save video
    if frames:
        if output_path is None:
            output_path = f"../../datasets/expert_demo_{demo_idx}_replay.mp4"
        
        print(f"\nSaving video with {len(frames)} frames...")
        imageio.mimsave(output_path, frames, fps=30, quality=8, codec='libx264')
        print(f"Video saved to: {output_path}")
        
        # Also save GIF
        gif_path = output_path.replace('.mp4', '.gif')
        gif_frames = frames[::3]  # Every 3rd frame
        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
        print(f"GIF saved to: {gif_path}")
        
        return len(frames)
    else:
        print("No frames generated")
        return 0


def create_demo_montage(dataset_path, n_demos=4):
    """
    Create a montage video showing multiple demonstrations.
    """
    import matplotlib.pyplot as plt
    
    print(f"Creating montage of {n_demos} demonstrations...")
    
    env = PandaDemoEnv(camera_name="demo_cam")
    
    # Collect key frames from each demo
    demo_frames = []
    
    with h5py.File(dataset_path, 'r') as f:
        for demo_idx in range(min(n_demos, 30)):  # Max 30 demos in dataset
            demo_key = f'demo_{demo_idx}'
            if demo_key not in f:
                continue
                
            print(f"Processing {demo_key}...")
            demo_group = f[demo_key]
            
            actions = demo_group['actions'][:]
            joint_positions = demo_group['observations/joint_pos'][:]
            
            # Reset and set initial state
            env.reset()
            for i in range(7):
                joint_addr = env.model.jnt_qposadr[env.arm_joint_ids[i]]
                env.data.qpos[joint_addr] = joint_positions[0, i]
            mujoco.mj_forward(env.model, env.data)
            
            # Collect frames at key points
            key_steps = [0, 30, 60, 90, 120, 149]  # Start, approach, descend, grasp, lift, end
            frames = []
            
            for step in range(150):
                env.step(actions[step])
                if step in key_steps:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
            
            demo_frames.append(frames)
    
    # Create montage figure
    n_demos_actual = len(demo_frames)
    if n_demos_actual > 0:
        fig, axes = plt.subplots(n_demos_actual, 6, figsize=(18, 3*n_demos_actual))
        if n_demos_actual == 1:
            axes = axes.reshape(1, -1)
        
        phase_names = ['Start', 'Approach', 'Descend', 'Grasp', 'Lift', 'End']
        
        for demo_idx, frames in enumerate(demo_frames):
            for frame_idx, frame in enumerate(frames[:6]):
                axes[demo_idx, frame_idx].imshow(frame)
                axes[demo_idx, frame_idx].axis('off')
                if demo_idx == 0:
                    axes[demo_idx, frame_idx].set_title(phase_names[frame_idx])
                if frame_idx == 0:
                    axes[demo_idx, frame_idx].text(-50, frame.shape[0]//2, f'Demo {demo_idx}', 
                                                   rotation=90, va='center', fontsize=12)
        
        plt.tight_layout()
        montage_path = "../../datasets/expert_demos_montage.png"
        plt.savefig(montage_path, dpi=150, bbox_inches='tight')
        print(f"Montage saved to: {montage_path}")
        plt.close()
        
        return n_demos_actual
    else:
        print("No demonstrations found")
        return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Replay expert demonstrations")
    parser.add_argument("--dataset", type=str, 
                       default="../../datasets/expert_demonstrations.hdf5",
                       help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=int, default=0, 
                       help="Demonstration index to replay")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for video")
    parser.add_argument("--montage", action="store_true",
                       help="Create montage of multiple demos")
    parser.add_argument("--n_demos", type=int, default=4,
                       help="Number of demos for montage")
    
    args = parser.parse_args()
    
    if args.montage:
        create_demo_montage(args.dataset, args.n_demos)
    else:
        replay_and_record_demo(args.dataset, args.demo, args.output)