#!/usr/bin/env python3
"""
Run a single expert demonstration and save as video.
"""

import numpy as np
import mujoco
import cv2
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Import the expert policy from create_expert_demos.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ExpertPolicy:
    """Expert policy optimized for 110 degree wrist configuration."""
    
    def __init__(self):
        self.phase = "approach"
        self.step_count = 0
        self.phase_steps = {
            "approach": 50,
            "pre_grasp": 40, 
            "descend": 35,
            "grasp": 30,
            "lift": 45
        }
        
    def get_action(self, obs):
        """Get expert action for current observation."""
        actions = np.zeros(8)
        
        if self.phase == "approach":
            # Strong forward motion to overcome initial distance
            actions[0] = 2.0   # x forward
            actions[2] = -0.3  # slight down for stability
            
        elif self.phase == "pre_grasp":
            # Continue forward with descent preparation
            actions[0] = 1.5   # maintain forward
            actions[2] = -0.5  # prepare descent
            
        elif self.phase == "descend":
            # Aggressive final descent
            actions[0] = 0.5   # slow forward
            actions[2] = -2.0  # strong down
            
        elif self.phase == "grasp":
            # Close gripper with micro-adjustments
            actions[7] = 1.0   # close gripper
            actions[0] = 0.1   # tiny forward
            actions[2] = -0.2  # gentle down pressure
            
        elif self.phase == "lift":
            # Strong lift with backward motion for stability
            actions[2] = 3.0   # strong up
            actions[0] = -0.5  # slight back for balance
            actions[7] = 1.0   # maintain grip
        
        # Update phase
        self.step_count += 1
        if self.step_count >= self.phase_steps[self.phase]:
            phases = list(self.phase_steps.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.step_count = 0
                
        return actions

def run_expert_demo():
    """Run a single expert demonstration and save video."""
    
    # Load the environment
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environments.tactile_grasping_env import TactileGraspingEnv
    
    env = TactileGraspingEnv(
        xml_path="../franka_emika_panda/panda_demo_scene.xml",
        use_tactile=True,
        max_episode_steps=200
    )
    
    # Initialize expert policy
    expert = ExpertPolicy()
    
    # Video setup
    frames = []
    tactile_data = defaultdict(list)
    
    # Reset environment
    obs = env.reset()
    
    print("Running expert demonstration...")
    print("-" * 50)
    
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 200:
        # Get expert action
        action = expert.get_action(obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render frame
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
        
        # Collect tactile data
        if 'tactile' in obs:
            # Split tactile data (first half is left, second half is right)
            tactile_size = len(obs['tactile']) // 2
            tactile_data['left'].append(obs['tactile'][:tactile_size])
            tactile_data['right'].append(obs['tactile'][tactile_size:])
        
        # Print progress
        if step % 20 == 0:
            print(f"Step {step}: Phase={expert.phase}, Reward={reward:.3f}")
            if 'tactile' in obs:
                tactile_size = len(obs['tactile']) // 2
                left_mean = np.mean(obs['tactile'][:tactile_size])
                right_mean = np.mean(obs['tactile'][tactile_size:])
                print(f"  Tactile - Left: {left_mean:.3f}, Right: {right_mean:.3f}")
        
        step += 1
    
    print("-" * 50)
    print(f"Demo complete! Total reward: {total_reward:.2f}")
    print(f"Success: {info.get('success', False)}")
    
    # Save video
    save_video(frames, "expert_demo.mp4")
    
    # Create tactile visualization if available
    if tactile_data['left']:
        create_tactile_plot(tactile_data, "expert_demo_tactile.png")
    
    env.close()
    return total_reward, info.get('success', False)

def save_video(frames, filename, fps=30):
    """Save frames as video."""
    if not frames:
        print("No frames to save!")
        return
        
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to {filename}")

def create_tactile_plot(tactile_data, filename):
    """Create a plot of tactile sensor readings over time."""
    fig = plt.figure(figsize=(12, 8))
    
    # Convert to numpy arrays
    left_data = np.array(tactile_data['left'])
    right_data = np.array(tactile_data['right'])
    
    # Main plot - average tactile readings
    ax1 = plt.subplot(2, 1, 1)
    steps = range(len(left_data))
    ax1.plot(steps, np.mean(left_data, axis=1), label='Left Finger', color='blue')
    ax1.plot(steps, np.mean(right_data, axis=1), label='Right Finger', color='red')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Average Tactile Reading')
    ax1.set_title('Tactile Sensor Readings During Expert Demo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Spatial distribution at key moments
    key_steps = [0, len(steps)//3, 2*len(steps)//3, len(steps)-1]
    
    for idx, step in enumerate(key_steps):
        if step < len(left_data) and idx < 4:
            # Reshape to grid (assuming 6x6 grid)
            grid_size = 6  # 36 sensors = 6x6 grid
            try:
                left_grid = left_data[step].reshape(grid_size, grid_size)
                right_grid = right_data[step].reshape(grid_size, grid_size)
            except:
                # If reshape fails, skip spatial plot
                continue
            
            # Create subplots for spatial view
            ax_left = plt.subplot(4, 4, 9 + idx)
            ax_right = plt.subplot(4, 4, 13 + idx)
            
            # Plot heatmaps
            max_val = max(left_data.max(), right_data.max()) if left_data.max() > 0 else 1
            im1 = ax_left.imshow(left_grid, cmap='viridis', vmin=0, vmax=max_val)
            im2 = ax_right.imshow(right_grid, cmap='viridis', vmin=0, vmax=max_val)
            
            ax_left.set_title(f'L@{step}', fontsize=8)
            ax_right.set_title(f'R@{step}', fontsize=8)
            ax_left.axis('off')
            ax_right.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Tactile plot saved to {filename}")
    plt.close()

if __name__ == "__main__":
    # Run the demonstration
    reward, success = run_expert_demo()
    
    print(f"\nDemo complete!")
    print(f"Files created:")
    print(f"  - expert_demo.mp4 (video)")
    print(f"  - expert_demo_tactile.png (tactile visualization)")