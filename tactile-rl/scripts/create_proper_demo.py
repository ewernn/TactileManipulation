"""
Create a proper demonstration of the Panda robot grasping and stacking blocks.
This script creates an expert policy that can reach, grasp, and stack blocks.
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


class PandaExpertPolicy:
    """Expert policy for Panda grasping and stacking demonstration."""
    
    def __init__(self):
        # State machine states
        self.states = ["approach", "descend", "pre_grasp", "grasp", "lift", "move_to_target", "place", "release", "done"]
        self.current_state = "approach"
        self.state_entry_time = 0
        
        # Control gains
        self.kp_pos = 10.0  # Position control gain
        self.kp_ori = 5.0   # Orientation control gain
        
        # Grasp parameters
        self.grasp_height_offset = 0.06  # Height above block to start approach
        self.grasp_width = 0.045  # Slightly wider than block (0.025 * 2)
        
        # Target positions
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        self.stack_height_offset = 0.05  # Height to place red on blue
        
    def reset(self):
        """Reset the expert policy."""
        self.current_state = "approach"
        self.state_entry_time = time.time()
        self.gripper_close_start_time = None
        
    def get_ee_pose(self, env: PandaDemoEnv) -> Tuple[np.ndarray, np.ndarray]:
        """Get end-effector position and orientation."""
        # Get hand position
        hand_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ee_pos = env.data.xpos[hand_body_id].copy()
        ee_quat = env.data.xquat[hand_body_id].copy()
        
        # Adjust for gripper offset (gripper extends down from hand)
        # In the hand frame, the gripper extends in the -Z direction
        gripper_offset = np.array([0, 0, -0.1])  # 10cm down from hand
        
        # Rotate offset to world frame
        ee_mat = np.zeros((3, 3))
        mujoco.mju_quat2Mat(ee_mat.flatten(), ee_quat)
        ee_mat = ee_mat.reshape((3, 3))
        world_offset = ee_mat @ gripper_offset
        
        gripper_pos = ee_pos + world_offset
        
        return gripper_pos, ee_quat
    
    def compute_jacobian(self, env: PandaDemoEnv) -> np.ndarray:
        """Compute the Jacobian for the arm joints."""
        # Get jacobian for end-effector
        hand_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        
        # Position and rotation jacobians
        jac_pos = np.zeros((3, env.model.nv))
        jac_rot = np.zeros((3, env.model.nv))
        
        # Compute jacobian
        mujoco.mj_jacBody(env.model, env.data, jac_pos, jac_rot, hand_body_id)
        
        # Extract only the arm joint columns (first 7 joints)
        jac_arm = np.vstack([jac_pos[:, :7], jac_rot[:, :7]])
        
        return jac_arm
    
    def position_control_ik(self, env: PandaDemoEnv, target_pos: np.ndarray, 
                           target_quat: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute joint velocities to reach target position using inverse kinematics.
        """
        # Get current end-effector pose
        ee_pos, ee_quat = self.get_ee_pose(env)
        
        # Position error
        pos_error = target_pos - ee_pos
        
        # Orientation error (if target orientation specified)
        if target_quat is not None:
            # Compute quaternion error
            quat_error = np.zeros(4)
            mujoco.mju_mulQuat(quat_error, target_quat, mujoco.mju_negQuat(ee_quat))
            
            # Convert to axis-angle
            angle = 2 * np.arccos(np.clip(quat_error[0], -1, 1))
            if angle > 0:
                axis = quat_error[1:] / np.sin(angle / 2)
                ori_error = angle * axis
            else:
                ori_error = np.zeros(3)
        else:
            ori_error = np.zeros(3)
        
        # Stack position and orientation errors
        error = np.hstack([pos_error * self.kp_pos, ori_error * self.kp_ori])
        
        # Get Jacobian
        J = self.compute_jacobian(env)
        
        # Compute pseudo-inverse
        J_pinv = np.linalg.pinv(J)
        
        # Compute joint velocities
        joint_vel = J_pinv @ error
        
        # Scale down for safety
        max_vel = 2.0
        joint_vel = np.clip(joint_vel, -max_vel, max_vel)
        
        return joint_vel
    
    def get_action(self, obs: Dict[str, np.ndarray], env: PandaDemoEnv) -> np.ndarray:
        """Get expert action based on current state."""
        
        # Initialize action array
        action = np.zeros(8)
        
        # Get current gripper position and state
        ee_pos, ee_quat = self.get_ee_pose(env)
        gripper_opening = obs['gripper_pos']  # Normalized 0-1
        
        # Get block positions
        red_block_pos = obs.get('target_block_pos', self.red_block_pos)
        blue_block_pos = obs.get('block2_pos', self.blue_block_pos)
        
        # State machine logic
        time_in_state = time.time() - self.state_entry_time
        
        if self.current_state == "approach":
            # Move above red block
            target_pos = red_block_pos.copy()
            target_pos[2] += self.grasp_height_offset
            
            # Use inverse kinematics to get joint velocities
            joint_vel = self.position_control_ik(env, target_pos)
            action[:7] = joint_vel
            
            # Open gripper
            action[7] = -1.0  # Open
            
            # Check if close enough
            pos_error = np.linalg.norm(target_pos - ee_pos)
            if pos_error < 0.02:  # 2cm tolerance
                print(f"✓ Reached approach position (error: {pos_error:.3f}m)")
                self.current_state = "descend"
                self.state_entry_time = time.time()
                
        elif self.current_state == "descend":
            # Descend to grasp height
            target_pos = red_block_pos.copy()
            target_pos[2] += 0.01  # Just above block
            
            joint_vel = self.position_control_ik(env, target_pos)
            action[:7] = joint_vel
            
            # Keep gripper open
            action[7] = -1.0
            
            # Check if at grasp height
            if ee_pos[2] < red_block_pos[2] + 0.03 or time_in_state > 2.0:
                print(f"✓ Reached grasp height (z: {ee_pos[2]:.3f}m)")
                self.current_state = "pre_grasp"
                self.state_entry_time = time.time()
                
        elif self.current_state == "pre_grasp":
            # Brief pause before grasping
            # Hold position
            target_pos = red_block_pos.copy()
            target_pos[2] += 0.01
            
            joint_vel = self.position_control_ik(env, target_pos)
            action[:7] = joint_vel * 0.1  # Slower movement
            
            # Keep gripper open
            action[7] = -1.0
            
            if time_in_state > 0.5:
                print("✓ Starting grasp")
                self.current_state = "grasp"
                self.state_entry_time = time.time()
                self.gripper_close_start_time = time.time()
                
        elif self.current_state == "grasp":
            # Close gripper
            # Hold position steady
            target_pos = red_block_pos.copy()
            target_pos[2] += 0.01
            
            joint_vel = self.position_control_ik(env, target_pos)
            action[:7] = joint_vel * 0.1  # Very slow movement
            
            # Close gripper
            action[7] = 1.0  # Close
            
            # Check if gripper is closed enough
            if gripper_opening < 0.4 and time_in_state > 1.0:  # Given enough time to close
                print(f"✓ Grasp complete (gripper: {gripper_opening:.3f})")
                self.current_state = "lift"
                self.state_entry_time = time.time()
                
        elif self.current_state == "lift":
            # Lift the block
            target_pos = red_block_pos.copy()
            target_pos[2] += 0.1  # Lift 10cm
            
            joint_vel = self.position_control_ik(env, target_pos)
            action[:7] = joint_vel
            
            # Keep gripper closed
            action[7] = 1.0
            
            # Check if lifted
            if ee_pos[2] > red_block_pos[2] + 0.08:
                print(f"✓ Block lifted (z: {ee_pos[2]:.3f}m)")
                self.current_state = "move_to_target"
                self.state_entry_time = time.time()
                
        elif self.current_state == "move_to_target":
            # Move to above blue block
            target_pos = blue_block_pos.copy()
            target_pos[2] += self.stack_height_offset + 0.05  # Extra clearance
            
            joint_vel = self.position_control_ik(env, target_pos)
            action[:7] = joint_vel
            
            # Keep gripper closed
            action[7] = 1.0
            
            # Check if above target
            pos_error = np.linalg.norm(target_pos[:2] - ee_pos[:2])
            if pos_error < 0.02:
                print(f"✓ Reached target position (error: {pos_error:.3f}m)")
                self.current_state = "place"
                self.state_entry_time = time.time()
                
        elif self.current_state == "place":
            # Lower to place on blue block
            target_pos = blue_block_pos.copy()
            target_pos[2] += self.stack_height_offset
            
            joint_vel = self.position_control_ik(env, target_pos)
            action[:7] = joint_vel * 0.5  # Slower for precision
            
            # Keep gripper closed
            action[7] = 1.0
            
            # Check if at place height
            if ee_pos[2] < blue_block_pos[2] + self.stack_height_offset + 0.02 or time_in_state > 2.0:
                print(f"✓ Block placed (z: {ee_pos[2]:.3f}m)")
                self.current_state = "release"
                self.state_entry_time = time.time()
                
        elif self.current_state == "release":
            # Open gripper to release
            # Hold position
            target_pos = blue_block_pos.copy()
            target_pos[2] += self.stack_height_offset
            
            joint_vel = self.position_control_ik(env, target_pos)
            action[:7] = joint_vel * 0.1
            
            # Open gripper
            action[7] = -1.0
            
            if gripper_opening > 0.8 and time_in_state > 1.0:
                print("✓ Block released")
                self.current_state = "done"
                self.state_entry_time = time.time()
                
        elif self.current_state == "done":
            # Move up and away
            target_pos = blue_block_pos.copy()
            target_pos[2] += 0.15
            target_pos[0] -= 0.05
            
            joint_vel = self.position_control_ik(env, target_pos)
            action[:7] = joint_vel
            
            # Keep gripper open
            action[7] = -1.0
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action


def create_demonstration(save_video=True, video_path="panda_demo.mp4"):
    """Create a demonstration of the Panda robot grasping and stacking blocks."""
    
    print("🤖 Creating Panda Grasping Demonstration")
    print("=" * 50)
    
    # Create environment and expert
    env = PandaDemoEnv(camera_name="demo_cam")
    expert = PandaExpertPolicy()
    
    # Reset environment
    obs = env.reset(randomize=False)
    expert.reset()
    
    # Video setup
    frames = []
    
    # Run demonstration
    max_steps = 1000
    step = 0
    
    print("\n📋 Task: Pick up red block and stack on blue block")
    print(f"   🔴 Red block at: {expert.red_block_pos}")
    print(f"   🔵 Blue block at: {expert.blue_block_pos}")
    print("\n🎬 Starting demonstration...\n")
    
    while step < max_steps and expert.current_state != "done":
        # Get expert action
        action = expert.get_action(obs, env)
        
        # Step environment
        obs = env.step(action)
        
        # Render and save frame
        frame = env.render()
        frames.append(frame)
        
        # Status update every 50 steps
        if step % 50 == 0:
            ee_pos, _ = expert.get_ee_pose(env)
            print(f"Step {step:4d} | State: {expert.current_state:15s} | "
                  f"EE pos: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}] | "
                  f"Gripper: {obs['gripper_pos']:.3f}")
        
        step += 1
    
    print(f"\n✅ Demonstration complete in {step} steps!")
    
    # Save video if requested
    if save_video and frames:
        print(f"\n💾 Saving video to: {video_path}")
        
        # Convert frames to video
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"✅ Video saved! ({len(frames)} frames)")
    
    return frames


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Panda grasping demonstration")
    parser.add_argument('--no-video', action='store_true', help='Disable video saving')
    parser.add_argument('--output', type=str, default='panda_demo.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    # Create demonstration
    frames = create_demonstration(
        save_video=not args.no_video,
        video_path=args.output
    )
    
    print("\n🎉 Demonstration script complete!")