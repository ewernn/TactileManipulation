#!/usr/bin/env python3
"""
Create a polished expert demonstration video with multiple camera angles.
"""

import numpy as np
import mujoco
import cv2
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.simple_tactile_env import SimpleTactileGraspingEnv

class SimpleExpertPolicy:
    """Simple expert policy for block grasping."""
    
    def __init__(self):
        self.phase = "approach"
        self.step_count = 0
        self.phase_steps = {
            "approach": 40,
            "descend": 30,
            "grasp": 20,
            "lift": 30
        }
        
    def get_action(self, obs):
        """Get expert action based on current phase."""
        # Get current gripper position
        ee_pos = obs['ee_pos']
        block_pos = obs['cube_pose'][:3]
        
        # Simple proportional control
        action = np.zeros(4)  # [x, y, z, gripper]
        
        if self.phase == "approach":
            # Move towards block horizontally
            action[0] = 2.0 * (block_pos[0] - ee_pos[0])
            action[1] = 2.0 * (block_pos[1] - ee_pos[1])
            
        elif self.phase == "descend":
            # Move down to grasp height
            action[0] = 1.0 * (block_pos[0] - ee_pos[0])
            action[1] = 1.0 * (block_pos[1] - ee_pos[1])
            action[2] = -1.5  # Move down
            
        elif self.phase == "grasp":
            # Close gripper
            action[3] = 1.0
            action[2] = -0.5  # Maintain downward pressure
            
        elif self.phase == "lift":
            # Lift up
            action[2] = 2.0
            action[3] = 1.0  # Keep gripper closed
        
        # Update phase
        self.step_count += 1
        if self.step_count >= self.phase_steps[self.phase]:
            phases = list(self.phase_steps.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.step_count = 0
                print(f"  -> Switching to phase: {self.phase}")
                
        return action

def create_multi_view_frame(env, views=['agentview', 'sideview']):
    """Create a frame with multiple camera views."""
    frames = []
    
    # Define camera positions if not in model
    camera_configs = {
        'agentview': {
            'pos': [0.5, 0.0, 0.8],
            'xyaxes': [0.0, -1.0, 0.0, -0.4, 0.0, 0.92]
        },
        'sideview': {
            'pos': [0.0, -0.8, 0.6],
            'xyaxes': [1.0, 0.0, 0.0, 0.0, 0.3, 0.95]
        }
    }
    
    for view in views:
        # Create renderer for this view
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        
        # Update camera if needed
        if view in camera_configs:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.fixedcamid = -1
            cam.trackbodyid = -1
            
            # Set position and orientation
            config = camera_configs[view]
            cam.lookat[:] = [0.4, 0.0, 0.5]  # Look at workspace center
            cam.distance = 0.8
            cam.azimuth = 90 if view == 'sideview' else 0
            cam.elevation = -20
            
            renderer.update_scene(env.data, camera=cam)
        else:
            renderer.update_scene(env.data)
            
        frame = renderer.render()
        frames.append(frame)
    
    # Combine frames side by side
    if len(frames) > 1:
        combined = np.hstack(frames)
    else:
        combined = frames[0]
        
    return combined

def main():
    """Run expert demo and create video."""
    
    # Create environment
    env = SimpleTactileGraspingEnv()
    expert = SimpleExpertPolicy()
    
    # Video setup
    frames = []
    fps = 30
    
    # Reset environment
    obs = env.reset()
    
    print("Creating expert demonstration video...")
    print("-" * 50)
    
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 120:
        # Get expert action
        action = expert.get_action(obs)
        
        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Create multi-view frame
        frame = create_multi_view_frame(env, views=['agentview'])
        
        # Add text overlay
        text_color = (255, 255, 255)
        cv2.putText(frame, f"Phase: {expert.phase}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(frame, f"Step: {step}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(frame, f"Reward: {total_reward:.1f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Add phase indicator bar
        phase_colors = {
            'approach': (0, 255, 0),    # Green
            'descend': (255, 255, 0),   # Yellow  
            'grasp': (255, 0, 0),       # Red
            'lift': (0, 0, 255)         # Blue
        }
        color = phase_colors.get(expert.phase, (255, 255, 255))
        progress = expert.step_count / expert.phase_steps[expert.phase]
        bar_width = int(200 * progress)
        cv2.rectangle(frame, (20, 110), (20 + bar_width, 130), color, -1)
        cv2.rectangle(frame, (20, 110), (220, 130), text_color, 2)
        
        frames.append(frame)
        
        if step % 30 == 0:
            print(f"Step {step}: Phase={expert.phase}, Reward={reward:.3f}")
        
        step += 1
    
    print("-" * 50)
    print(f"Demo complete! Total reward: {total_reward:.2f}")
    
    # Save video
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('expert_demo_polished.mp4', fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"\nVideo saved to expert_demo_polished.mp4")
        print(f"Duration: {len(frames)/fps:.1f} seconds")
        print(f"Resolution: {width}x{height}")
    
    env.close()

if __name__ == "__main__":
    main()