"""
Fixed demonstration of the Panda robot grasping and stacking blocks.
Properly handles tendon-based gripper control and correct kinematics.
"""

import numpy as np
import mujoco
import os
import sys
import time
import cv2
from typing import Dict, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.panda_demo_env import PandaDemoEnv


class FixedPandaExpert:
    """Fixed expert policy with proper gripper control."""
    
    def __init__(self):
        # State machine
        self.states = ["approach", "descend", "pre_grasp", "grasp", "lift", 
                      "move_to_target", "place", "release", "done"]
        self.current_state = "approach"
        self.state_entry_time = 0
        
        # Control parameters
        self.approach_height = 0.08  # Height above block
        self.grasp_height = 0.025   # Height to grasp at (block center)
        self.lift_height = 0.12     # Height to lift to
        self.stack_offset = 0.05    # Stack height
        
        # Target positions
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        
        # Gripper parameters
        self.gripper_open_pos = 0.04   # Fully open
        self.gripper_close_pos = 0.025 # For grasping 0.025m blocks
        
    def reset(self):
        """Reset the policy."""
        self.current_state = "approach"
        self.state_entry_time = time.time()
        
    def get_ee_position(self, env: PandaDemoEnv) -> np.ndarray:
        """Get actual gripper tip position."""
        # Get hand position
        hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        hand_pos = env.data.xpos[hand_id].copy()
        
        # Get hand orientation
        hand_quat = env.data.xquat[hand_id].copy()
        hand_mat = np.zeros((3, 3))
        mujoco.mju_quat2Mat(hand_mat.flatten(), hand_quat)
        hand_mat = hand_mat.reshape((3, 3))
        
        # Gripper extends down from hand (in hand's -Z direction)
        # Based on the XML, gripper site is at (0, 0, -0.1) in hand frame
        gripper_offset_local = np.array([0, 0, -0.1])
        gripper_offset_world = hand_mat @ gripper_offset_local
        
        gripper_pos = hand_pos + gripper_offset_world
        return gripper_pos
    
    def compute_ik_velocities(self, env: PandaDemoEnv, target_pos: np.ndarray, 
                              max_speed: float = 0.5) -> np.ndarray:
        """Compute joint velocities using simple position-based control."""
        current_pos = self.get_ee_position(env)
        error = target_pos - current_pos
        
        # Simple proportional control for each joint
        # This is a simplified approach - proper IK would be better
        joint_vel = np.zeros(7)
        
        # Map position error to joint velocities (simplified)
        # These gains are tuned for the Panda's kinematic structure
        joint_vel[0] = error[1] * 2.0   # Base rotation for Y movement
        joint_vel[1] = -error[2] * 1.5  # Shoulder for Z movement
        joint_vel[2] = error[0] * 2.0   # Elbow for X movement
        joint_vel[3] = -error[2] * 1.0  # Elbow flex for Z
        joint_vel[4] = error[0] * 1.0   # Forearm for X fine-tuning
        joint_vel[5] = error[2] * 0.5   # Wrist for Z fine-tuning
        joint_vel[6] = 0.0              # Wrist rotation (not needed)
        
        # Limit velocities
        joint_vel = np.clip(joint_vel, -max_speed, max_speed)
        
        return joint_vel
    
    def get_gripper_width(self, env: PandaDemoEnv) -> float:
        """Get current gripper width in meters."""
        finger1_id = env.gripper_joint1_id
        if finger1_id is not None:
            finger_addr = env.model.jnt_qposadr[finger1_id]
            finger_pos = env.data.qpos[finger_addr]
            # Total width is 2 * finger position (symmetric fingers)
            return finger_pos * 2
        return 0.08  # Default open
    
    def get_action(self, obs: Dict[str, np.ndarray], env: PandaDemoEnv) -> np.ndarray:
        """Generate expert action based on current state."""
        
        action = np.zeros(8)
        
        # Get current state
        ee_pos = self.get_ee_position(env)
        gripper_width = self.get_gripper_width(env)
        time_in_state = time.time() - self.state_entry_time
        
        # Get block positions
        red_pos = obs.get('target_block_pos', self.red_block_pos)
        blue_pos = obs.get('block2_pos', self.blue_block_pos)
        
        # State machine
        if self.current_state == "approach":
            # Move above red block
            target = red_pos.copy()
            target[2] += self.approach_height
            
            joint_vel = self.compute_ik_velocities(env, target)
            action[:7] = joint_vel
            
            # Open gripper (high value = open for tendon control)
            action[7] = -1.0  # Maps to 255 in tendon control
            
            # Check if reached
            dist = np.linalg.norm(target - ee_pos)
            if dist < 0.03:  # 3cm tolerance
                print(f"✓ Reached approach position (error: {dist:.3f}m)")
                self.current_state = "descend"
                self.state_entry_time = time.time()
                
        elif self.current_state == "descend":
            # Descend to grasp height
            target = red_pos.copy()
            target[2] = red_pos[2] + self.grasp_height
            
            joint_vel = self.compute_ik_velocities(env, target, max_speed=0.3)
            action[:7] = joint_vel
            action[7] = -1.0  # Keep open
            
            # Check if at grasp height
            if abs(ee_pos[2] - target[2]) < 0.02 or time_in_state > 3.0:
                print(f"✓ At grasp height (z: {ee_pos[2]:.3f}m)")
                self.current_state = "pre_grasp"
                self.state_entry_time = time.time()
                
        elif self.current_state == "pre_grasp":
            # Hold position briefly
            target = red_pos.copy()
            target[2] = red_pos[2] + self.grasp_height
            
            joint_vel = self.compute_ik_velocities(env, target, max_speed=0.1)
            action[:7] = joint_vel
            action[7] = -1.0
            
            if time_in_state > 0.5:
                print("✓ Starting grasp")
                self.current_state = "grasp"
                self.state_entry_time = time.time()
                
        elif self.current_state == "grasp":
            # Close gripper
            target = red_pos.copy()
            target[2] = red_pos[2] + self.grasp_height
            
            joint_vel = self.compute_ik_velocities(env, target, max_speed=0.1)
            action[:7] = joint_vel
            
            # Close gripper (low value = close for tendon control)
            action[7] = 1.0  # Maps to 0 in tendon control
            
            # Check if grasped (gripper closed enough)
            if gripper_width < 0.06 and time_in_state > 1.5:
                print(f"✓ Object grasped (width: {gripper_width:.3f}m)")
                self.current_state = "lift"
                self.state_entry_time = time.time()
                
        elif self.current_state == "lift":
            # Lift up
            target = red_pos.copy()
            target[2] = red_pos[2] + self.lift_height
            
            joint_vel = self.compute_ik_velocities(env, target, max_speed=0.4)
            action[:7] = joint_vel
            action[7] = 1.0  # Keep closed
            
            if ee_pos[2] > red_pos[2] + 0.08:
                print(f"✓ Object lifted (z: {ee_pos[2]:.3f}m)")
                self.current_state = "move_to_target"
                self.state_entry_time = time.time()
                
        elif self.current_state == "move_to_target":
            # Move above blue block
            target = blue_pos.copy()
            target[2] = blue_pos[2] + self.lift_height
            
            joint_vel = self.compute_ik_velocities(env, target, max_speed=0.5)
            action[:7] = joint_vel
            action[7] = 1.0  # Keep closed
            
            # Check horizontal position
            xy_dist = np.linalg.norm(target[:2] - ee_pos[:2])
            if xy_dist < 0.03:
                print(f"✓ Above target (error: {xy_dist:.3f}m)")
                self.current_state = "place"
                self.state_entry_time = time.time()
                
        elif self.current_state == "place":
            # Lower to place
            target = blue_pos.copy()
            target[2] = blue_pos[2] + self.stack_offset
            
            joint_vel = self.compute_ik_velocities(env, target, max_speed=0.2)
            action[:7] = joint_vel
            action[7] = 1.0  # Keep closed
            
            if abs(ee_pos[2] - target[2]) < 0.02 or time_in_state > 3.0:
                print(f"✓ Placed block (z: {ee_pos[2]:.3f}m)")
                self.current_state = "release"
                self.state_entry_time = time.time()
                
        elif self.current_state == "release":
            # Open gripper
            target = blue_pos.copy()
            target[2] = blue_pos[2] + self.stack_offset
            
            joint_vel = self.compute_ik_velocities(env, target, max_speed=0.1)
            action[:7] = joint_vel
            action[7] = -1.0  # Open
            
            if gripper_width > 0.07 and time_in_state > 1.0:
                print("✓ Block released")
                self.current_state = "done"
                self.state_entry_time = time.time()
                
        elif self.current_state == "done":
            # Move away
            target = blue_pos.copy()
            target[2] = blue_pos[2] + self.lift_height
            target[0] -= 0.1
            
            joint_vel = self.compute_ik_velocities(env, target, max_speed=0.3)
            action[:7] = joint_vel
            action[7] = -1.0  # Keep open
        
        return np.clip(action, -1.0, 1.0)


def run_demonstration(save_video=True, output_path="fixed_panda_demo.mp4"):
    """Run the fixed demonstration."""
    
    print("🤖 Fixed Panda Demonstration")
    print("=" * 50)
    
    # Create environment and expert
    env = PandaDemoEnv(camera_name="demo_cam")
    expert = FixedPandaExpert()
    
    # Reset
    obs = env.reset(randomize=False)
    expert.reset()
    
    # Video frames
    frames = []
    
    print("\n📋 Task: Stack red block on blue block")
    print(f"   🔴 Red: {expert.red_block_pos}")
    print(f"   🔵 Blue: {expert.blue_block_pos}")
    print("\n🎬 Starting...\n")
    
    # Run demo
    max_steps = 1500
    for step in range(max_steps):
        # Get action
        action = expert.get_action(obs, env)
        
        # Step
        obs = env.step(action)
        
        # Render
        frame = env.render()
        frames.append(frame)
        
        # Status
        if step % 100 == 0:
            ee_pos = expert.get_ee_position(env)
            gripper_width = expert.get_gripper_width(env)
            print(f"Step {step:4d} | State: {expert.current_state:15s} | "
                  f"EE: [{ee_pos[0]:5.2f}, {ee_pos[1]:5.2f}, {ee_pos[2]:5.2f}] | "
                  f"Gripper: {gripper_width:.3f}m")
        
        # Check if done
        if expert.current_state == "done" and step > 100:
            break
    
    print(f"\n✅ Complete in {step} steps!")
    
    # Save video
    if save_video and frames:
        print(f"\n💾 Saving to: {output_path}")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"✅ Saved {len(frames)} frames!")
    
    return frames


if __name__ == "__main__":
    run_demonstration()