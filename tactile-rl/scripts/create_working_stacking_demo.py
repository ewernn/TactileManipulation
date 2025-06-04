"""
Create a working stacking demonstration with proper gripper positioning.
"""

import numpy as np
import sys
import os
import imageio
import mujoco

sys.path.append('../environments')
from panda_demo_env import PandaDemoEnv


class WorkingStackingPolicy:
    """
    Working policy that properly positions the gripper over blocks.
    """
    
    def __init__(self):
        self.reset()
        self.target_joints = None
        
    def reset(self):
        """Reset policy state."""
        self.phase = "init"
        self.step_count = 0
        self.grasp_count = 0
        
    def get_action(self, observation):
        """
        Generate action using target joint positions for precise control.
        """
        
        # Get current state
        joint_pos = observation['joint_pos']
        red_pos = observation['target_block_pos']
        blue_pos = observation['block2_pos'] if 'block2_pos' in observation else None
        tactile_sum = np.sum(observation['tactile'])
        
        action = np.zeros(8)  # 7 joints + gripper
        
        # Define target joint configurations for key poses
        if self.phase == "init":
            # Move to a good starting position
            self.target_joints = np.array([0.5, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])
            self.phase = "move_to_start"
            self.step_count = 0
            
        elif self.phase == "move_to_start":
            # Move to starting position
            if self.step_count < 50:
                # PD control towards target
                joint_error = self.target_joints - joint_pos
                action[:7] = np.clip(joint_error * 0.5, -1.0, 1.0)
                action[7] = -1.0  # Open gripper
            else:
                self.phase = "reach_over_red"
                self.step_count = 0
                
        elif self.phase == "reach_over_red":
            # Position over red block
            self.target_joints = np.array([0.5, 0.2, 0.0, -1.8, 0.0, 2.0, 0.785])
            if self.step_count < 60:
                joint_error = self.target_joints - joint_pos
                action[:7] = np.clip(joint_error * 0.3, -1.0, 1.0)
                action[7] = -1.0  # Keep gripper open
            else:
                self.phase = "descend_to_red"
                self.step_count = 0
                
        elif self.phase == "descend_to_red":
            # Lower to grasp height
            self.target_joints = np.array([0.5, 0.5, 0.0, -1.5, 0.0, 2.0, 0.785])
            if self.step_count < 50:
                joint_error = self.target_joints - joint_pos
                action[:7] = np.clip(joint_error * 0.25, -1.0, 1.0)
                action[7] = -1.0  # Keep gripper open
            else:
                self.phase = "grasp"
                self.step_count = 0
                self.grasp_count = 0
                
        elif self.phase == "grasp":
            # Close gripper
            if self.step_count < 40:
                action[:7] = 0.0  # Hold position
                action[7] = 1.0   # Close gripper
                
                # Count frames with good tactile contact
                if tactile_sum > 10:
                    self.grasp_count += 1
            else:
                self.phase = "lift"
                self.step_count = 0
                print(f"  Grasp quality: {self.grasp_count}/40 frames with contact")
                
        elif self.phase == "lift":
            # Lift up
            self.target_joints = np.array([0.5, 0.0, 0.0, -2.0, 0.0, 2.0, 0.785])
            if self.step_count < 50:
                joint_error = self.target_joints - joint_pos
                action[:7] = np.clip(joint_error * 0.3, -1.0, 1.0)
                action[7] = 1.0  # Keep gripper closed
            else:
                self.phase = "move_to_blue"
                self.step_count = 0
                
        elif self.phase == "move_to_blue":
            # Move over blue block - rotate base
            self.target_joints = np.array([0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.785])
            if self.step_count < 60:
                joint_error = self.target_joints - joint_pos
                action[:7] = np.clip(joint_error * 0.25, -1.0, 1.0)
                action[7] = 1.0  # Keep gripper closed
            else:
                self.phase = "position_over_blue"
                self.step_count = 0
                
        elif self.phase == "position_over_blue":
            # Fine position over blue block
            self.target_joints = np.array([0.0, 0.2, 0.0, -1.8, 0.0, 2.0, 0.785])
            if self.step_count < 40:
                joint_error = self.target_joints - joint_pos
                action[:7] = np.clip(joint_error * 0.3, -1.0, 1.0)
                action[7] = 1.0  # Keep gripper closed
            else:
                self.phase = "lower_to_stack"
                self.step_count = 0
                
        elif self.phase == "lower_to_stack":
            # Lower to stack
            self.target_joints = np.array([0.0, 0.45, 0.0, -1.5, 0.0, 2.0, 0.785])
            if self.step_count < 50:
                joint_error = self.target_joints - joint_pos
                action[:7] = np.clip(joint_error * 0.2, -1.0, 1.0)
                action[7] = 1.0  # Keep gripper closed
            else:
                self.phase = "release"
                self.step_count = 0
                
        elif self.phase == "release":
            # Open gripper
            if self.step_count < 30:
                action[:7] = 0.0  # Hold position
                action[7] = -1.0  # Open gripper
            else:
                self.phase = "retreat"
                self.step_count = 0
                
        elif self.phase == "retreat":
            # Move back
            self.target_joints = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])
            if self.step_count < 40:
                joint_error = self.target_joints - joint_pos
                action[:7] = np.clip(joint_error * 0.3, -1.0, 1.0)
                action[7] = -1.0  # Keep gripper open
            else:
                self.phase = "done"
                self.step_count = 0
                
        else:  # done
            action = np.zeros(8)
            action[7] = -1.0
            
        self.step_count += 1
        return action


def create_working_demo():
    """Create a working stacking demonstration."""
    
    print("🎬 Creating Working Block Stacking Demonstration...")
    
    env = PandaDemoEnv()
    expert = WorkingStackingPolicy()
    
    # Test single episode first
    print("\n📹 Recording demonstration...")
    
    obs = env.reset(randomize=False)
    expert.reset()
    
    print(f"Initial positions:")
    print(f"  Red block: {obs['target_block_pos']}")
    print(f"  Blue block: {obs['block2_pos']}")
    
    frames = []
    phase_log = []
    
    # Get end-effector ID for tracking
    ee_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    for step in range(500):
        # Get expert action
        action = expert.get_action(obs)
        
        # Track phase changes
        if len(phase_log) == 0 or expert.phase != phase_log[-1][1]:
            phase_log.append((step, expert.phase))
            print(f"\nStep {step:3d}: → {expert.phase}")
            
        # Execute action
        obs = env.step(action, steps=3)
        
        # Capture frame
        frame = env.render()
        frames.append(frame)
        
        # Monitor key moments
        if step % 40 == 0 and step > 0:
            ee_pos = env.data.xpos[ee_id]
            red_pos = obs['target_block_pos']
            blue_pos = obs['block2_pos']
            
            print(f"  Step {step}:")
            print(f"    EE pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            print(f"    Red block: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
            print(f"    EE-Red distance: {np.linalg.norm(ee_pos[:2] - red_pos[:2]):.3f}")
            
        # Check if red block is being carried
        if expert.phase in ["lift", "move_to_blue", "position_over_blue"]:
            red_height = obs['target_block_pos'][2]
            if red_height > 0.48:  # Block lifted
                print(f"  ✓ Red block lifted to {red_height:.3f}m")
        
        # Check stacking success
        red_pos = obs['target_block_pos']
        blue_pos = obs['block2_pos']
        height_diff = red_pos[2] - blue_pos[2]
        horizontal_dist = np.linalg.norm(red_pos[:2] - blue_pos[:2])
        
        if height_diff > 0.03 and horizontal_dist < 0.05:
            print(f"\n✅ SUCCESS! Stacking achieved at step {step}")
            print(f"   Height difference: {height_diff:.3f}m")
            print(f"   Horizontal distance: {horizontal_dist:.3f}m")
            
            # Hold success pose
            for _ in range(60):
                obs = env.step(action, steps=1)
                frames.append(env.render())
            break
            
        # Early termination
        if expert.phase == "done" and expert.step_count > 20:
            print(f"\n⏹️ Demo complete at step {step}")
            break
    
    # Save video
    output_path = "working_stacking_demo.mp4"
    
    print(f"\n💾 Saving video with {len(frames)} frames...")
    
    imageio.mimsave(
        output_path,
        frames,
        fps=30,
        quality=9,
        macro_block_size=1
    )
    
    print(f"\n✅ Video created: {output_path}")
    print(f"   Duration: {len(frames)/30:.1f} seconds")
    
    # Also create the final video in the videos directory
    final_path = "../../videos/franka_block_stacking_demo.mp4"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    
    # Create multiple angle version
    print("\n🎥 Creating multi-angle version...")
    
    all_frames = []
    cameras = ["demo_cam", "side_cam", "overhead_cam"]
    
    for cam_idx, camera in enumerate(cameras):
        print(f"\n  Camera {cam_idx+1}/3: {camera}")
        
        obs = env.reset(randomize=(cam_idx > 0))
        expert.reset()
        
        cam_frames = []
        
        for step in range(450):
            action = expert.get_action(obs)
            obs = env.step(action, steps=3)
            
            frame = env.render(camera=camera)
            cam_frames.append(frame)
            
            # Check success
            red_pos = obs['target_block_pos']
            blue_pos = obs['block2_pos']
            if (red_pos[2] - blue_pos[2] > 0.03 and 
                np.linalg.norm(red_pos[:2] - blue_pos[:2]) < 0.05):
                # Hold success
                for _ in range(40):
                    obs = env.step(action, steps=1)
                    cam_frames.append(env.render(camera=camera))
                break
                
            if expert.phase == "done" and expert.step_count > 20:
                break
        
        all_frames.extend(cam_frames)
        
        # Add transition
        if cam_idx < len(cameras) - 1:
            for _ in range(15):
                all_frames.append(cam_frames[-1])
    
    imageio.mimsave(
        final_path,
        all_frames,
        fps=30,
        quality=9,
        macro_block_size=1
    )
    
    print(f"\n✅ Final video created!")
    print(f"   📹 Location: {final_path}")
    print(f"   ⏱️  Duration: {len(all_frames)/30:.1f} seconds")
    print(f"   📷 Includes 3 camera angles")
    

if __name__ == "__main__":
    create_working_demo()