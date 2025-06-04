#!/usr/bin/env python3
"""
Simple demonstration showing the robot moving forward.
"""

import numpy as np
import sys
import os
import mujoco
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments'))
from environments.panda_demo_env import PandaDemoEnv


def simple_forward_demo():
    """Create a simple demo showing forward motion."""
    
    # Use original environment
    env = PandaDemoEnv()
    obs = env.reset()
    
    frames = []
    
    print("\n" + "="*60)
    print("SIMPLE FORWARD MOTION DEMONSTRATION")
    print("="*60)
    
    # Get initial position
    hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    initial_pos = env.data.xpos[hand_id].copy()
    
    print(f"\nInitial hand position: {initial_pos}")
    print(f"Target block position: {obs['target_block_pos']}")
    
    # Add initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    # Show the problem: robot facing backward
    print("\n🔍 Demonstrating the orientation problem:")
    
    # Try to move "forward" with positive joint velocities
    print("\n1. Attempting to reach forward (positive velocities)...")
    for i in range(30):
        action = np.array([0.3, 0.3, 0.3, -0.3, 0.1, 0, 0, -1])
        obs = env.step(action)
        
        if i % 5 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    
    hand_pos = env.data.xpos[hand_id].copy()
    print(f"   After 30 steps: {hand_pos}")
    print(f"   Movement: {hand_pos - initial_pos}")
    
    # Reset and try rotating base
    print("\n2. Trying to rotate base to face blocks...")
    obs = env.reset()
    
    for i in range(50):
        # Try to rotate joint 1
        action = np.array([0.5, 0, 0, 0, 0, 0, 0, -1])
        obs = env.step(action)
        
        if i % 10 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            joint1_pos = obs['joint_pos'][0]
            print(f"   Step {i}: Joint 1 = {joint1_pos:.3f} rad ({np.degrees(joint1_pos):.1f}°)")
    
    # Show final orientation
    hand_quat = env.data.xquat[hand_id]
    rotmat = np.zeros(9)
    mujoco.mju_quat2Mat(rotmat, hand_quat)
    rotmat = rotmat.reshape(3, 3)
    gripper_direction = rotmat[:, 2]
    
    print(f"\n3. Final gripper direction: {gripper_direction}")
    print(f"   X-component: {gripper_direction[0]:.3f} (negative = backward)")
    
    # Add text overlay showing the problem
    for _ in range(20):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    
    # Save video
    if frames:
        output_path = "../../datasets/orientation_problem_demo.mp4"
        print(f"\n💾 Saving video with {len(frames)} frames...")
        imageio.mimsave(output_path, frames, fps=15, quality=8, codec='libx264')
        print(f"Video saved to: {output_path}")
        
        # GIF
        gif_path = output_path.replace('.mp4', '.gif')
        gif_frames = frames[::2]
        imageio.mimsave(gif_path, gif_frames, fps=8, loop=0)
        print(f"GIF saved to: {gif_path}")
    
    print("\n" + "="*60)
    print("SUMMARY: Robot cannot reach blocks because:")
    print("1. Gripper faces backward (negative X direction)")
    print("2. Blocks are in positive X direction")
    print("3. Even with Joint 1 rotation, hand quaternion keeps it backward")
    print("4. Solution: Need to fix XML file or move blocks behind robot")
    print("="*60)


if __name__ == "__main__":
    simple_forward_demo()