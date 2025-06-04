"""
Working demonstration with proper joint configurations for the Panda robot.
Uses pre-computed joint positions for key poses.
"""

import numpy as np
import mujoco
import os
import sys
import time
import cv2
from typing import Dict, Tuple, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.panda_demo_env import PandaDemoEnv


class WorkingPandaExpert:
    """Expert with pre-computed joint configurations."""
    
    def __init__(self):
        # State machine
        self.states = ["init", "approach", "descend", "grasp", "lift", 
                      "move", "place", "release", "done"]
        self.current_state = "init"
        self.state_start_time = 0
        self.state_start_joints = None
        
        # Key joint configurations (7 joints)
        # These are tuned for the specific scene layout
        self.joint_configs = {
            # Home position
            "home": np.array([0, -0.5, 0, -2.3, 0, 1.8, 0.785]),
            
            # Above red block (0.05, 0.0, 0.445)
            "above_red": np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785]),
            
            # At red block (ready to grasp)
            "at_red": np.array([0.0, -0.1, 0.0, -1.8, 0.0, 1.7, 0.785]),
            
            # Lifted position
            "lifted": np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.785]),
            
            # Above blue block (0.05, -0.15, 0.445)
            "above_blue": np.array([-0.3, -0.4, 0.0, -2.0, 0.0, 1.6, 0.785]),
            
            # At blue block (ready to place)
            "at_blue": np.array([-0.3, -0.2, 0.0, -1.8, 0.0, 1.6, 0.785]),
            
            # Final position
            "final": np.array([-0.3, -0.5, 0.0, -2.2, 0.0, 1.7, 0.785])
        }
        
        # Gripper states
        self.gripper_states = {
            "open": -1.0,   # Maps to 255 (tendon released)
            "close": 1.0    # Maps to 0 (tendon pulled)
        }
        
        # Timing for each state (seconds)
        self.state_durations = {
            "init": 1.0,
            "approach": 2.0,
            "descend": 2.0,
            "grasp": 2.0,
            "lift": 2.0,
            "move": 3.0,
            "place": 2.0,
            "release": 1.5,
            "done": 1.0
        }
        
    def reset(self):
        """Reset the expert."""
        self.current_state = "init"
        self.state_start_time = time.time()
        self.state_start_joints = None
        
    def interpolate_joints(self, start: np.ndarray, end: np.ndarray, 
                          t: float, duration: float) -> np.ndarray:
        """Smoothly interpolate between joint configurations."""
        # Use cosine interpolation for smooth acceleration/deceleration
        alpha = t / duration
        alpha = np.clip(alpha, 0, 1)
        # Cosine interpolation
        alpha_smooth = 0.5 * (1 - np.cos(alpha * np.pi))
        
        return start + (end - start) * alpha_smooth
    
    def get_current_joints(self, env: PandaDemoEnv) -> np.ndarray:
        """Get current joint positions."""
        joints = np.zeros(7)
        for i, joint_id in enumerate(env.arm_joint_ids):
            if joint_id is not None:
                addr = env.model.jnt_qposadr[joint_id]
                joints[i] = env.data.qpos[addr]
        return joints
    
    def joints_to_velocities(self, current: np.ndarray, target: np.ndarray, 
                            dt: float = 0.05) -> np.ndarray:
        """Convert target joint positions to velocities."""
        error = target - current
        # Simple P controller with saturation
        kp = 5.0
        velocities = kp * error
        # Limit velocities
        max_vel = 2.0
        velocities = np.clip(velocities, -max_vel, max_vel)
        return velocities
    
    def get_action(self, obs: Dict[str, np.ndarray], env: PandaDemoEnv) -> np.ndarray:
        """Generate action based on current state."""
        
        action = np.zeros(8)
        
        # Get timing
        time_in_state = time.time() - self.state_start_time
        current_joints = self.get_current_joints(env)
        
        # Initialize start joints if needed
        if self.state_start_joints is None:
            self.state_start_joints = current_joints.copy()
        
        # State machine
        if self.current_state == "init":
            # Move to home position
            target_joints = self.joint_configs["home"]
            interp_joints = self.interpolate_joints(
                self.state_start_joints, target_joints, 
                time_in_state, self.state_durations["init"]
            )
            
            velocities = self.joints_to_velocities(current_joints, interp_joints)
            action[:7] = velocities
            action[7] = self.gripper_states["open"]
            
            if time_in_state > self.state_durations["init"]:
                print("✓ Initialized at home position")
                self.transition_to("approach")
                
        elif self.current_state == "approach":
            # Move above red block
            target_joints = self.joint_configs["above_red"]
            interp_joints = self.interpolate_joints(
                self.state_start_joints, target_joints,
                time_in_state, self.state_durations["approach"]
            )
            
            velocities = self.joints_to_velocities(current_joints, interp_joints)
            action[:7] = velocities
            action[7] = self.gripper_states["open"]
            
            if time_in_state > self.state_durations["approach"]:
                print("✓ Reached position above red block")
                self.transition_to("descend")
                
        elif self.current_state == "descend":
            # Descend to grasp position
            target_joints = self.joint_configs["at_red"]
            interp_joints = self.interpolate_joints(
                self.state_start_joints, target_joints,
                time_in_state, self.state_durations["descend"]
            )
            
            velocities = self.joints_to_velocities(current_joints, interp_joints)
            action[:7] = velocities
            action[7] = self.gripper_states["open"]
            
            if time_in_state > self.state_durations["descend"]:
                print("✓ At grasp position")
                self.transition_to("grasp")
                
        elif self.current_state == "grasp":
            # Close gripper
            target_joints = self.joint_configs["at_red"]
            velocities = self.joints_to_velocities(current_joints, target_joints)
            action[:7] = velocities * 0.1  # Hold position
            
            # Gradually close gripper
            if time_in_state < 1.0:
                action[7] = self.gripper_states["open"]
            else:
                action[7] = self.gripper_states["close"]
            
            if time_in_state > self.state_durations["grasp"]:
                print("✓ Object grasped")
                self.transition_to("lift")
                
        elif self.current_state == "lift":
            # Lift up
            target_joints = self.joint_configs["lifted"]
            interp_joints = self.interpolate_joints(
                self.state_start_joints, target_joints,
                time_in_state, self.state_durations["lift"]
            )
            
            velocities = self.joints_to_velocities(current_joints, interp_joints)
            action[:7] = velocities
            action[7] = self.gripper_states["close"]
            
            if time_in_state > self.state_durations["lift"]:
                print("✓ Object lifted")
                self.transition_to("move")
                
        elif self.current_state == "move":
            # Move to above blue block
            target_joints = self.joint_configs["above_blue"]
            interp_joints = self.interpolate_joints(
                self.state_start_joints, target_joints,
                time_in_state, self.state_durations["move"]
            )
            
            velocities = self.joints_to_velocities(current_joints, interp_joints)
            action[:7] = velocities
            action[7] = self.gripper_states["close"]
            
            if time_in_state > self.state_durations["move"]:
                print("✓ Reached position above blue block")
                self.transition_to("place")
                
        elif self.current_state == "place":
            # Lower to place
            target_joints = self.joint_configs["at_blue"]
            interp_joints = self.interpolate_joints(
                self.state_start_joints, target_joints,
                time_in_state, self.state_durations["place"]
            )
            
            velocities = self.joints_to_velocities(current_joints, interp_joints)
            action[:7] = velocities
            action[7] = self.gripper_states["close"]
            
            if time_in_state > self.state_durations["place"]:
                print("✓ Block placed")
                self.transition_to("release")
                
        elif self.current_state == "release":
            # Open gripper
            target_joints = self.joint_configs["at_blue"]
            velocities = self.joints_to_velocities(current_joints, target_joints)
            action[:7] = velocities * 0.1  # Hold position
            action[7] = self.gripper_states["open"]
            
            if time_in_state > self.state_durations["release"]:
                print("✓ Block released")
                self.transition_to("done")
                
        elif self.current_state == "done":
            # Move to final position
            target_joints = self.joint_configs["final"]
            interp_joints = self.interpolate_joints(
                self.state_start_joints, target_joints,
                time_in_state, self.state_durations["done"]
            )
            
            velocities = self.joints_to_velocities(current_joints, interp_joints)
            action[:7] = velocities
            action[7] = self.gripper_states["open"]
        
        return np.clip(action, -1.0, 1.0)
    
    def transition_to(self, new_state: str):
        """Transition to a new state."""
        self.current_state = new_state
        self.state_start_time = time.time()
        self.state_start_joints = None  # Will be set on next get_action


def create_working_demo(save_video=True, output_path="working_panda_demo.mp4"):
    """Create a working demonstration."""
    
    print("🤖 Working Panda Demonstration")
    print("=" * 50)
    
    # Create environment and expert
    env = PandaDemoEnv(camera_name="demo_cam")
    expert = WorkingPandaExpert()
    
    # Reset
    obs = env.reset(randomize=False)
    expert.reset()
    
    # Video frames
    frames = []
    
    print("\n📋 Task: Pick up red block and stack on blue block")
    print("🎬 Starting demonstration...\n")
    
    # Calculate total expected time
    total_time = sum(expert.state_durations.values())
    max_steps = int(total_time * 30) + 100  # 30 FPS + buffer
    
    # Run demonstration
    for step in range(max_steps):
        # Get action
        action = expert.get_action(obs, env)
        
        # Step environment
        obs = env.step(action)
        
        # Render
        frame = env.render()
        frames.append(frame)
        
        # Status updates
        if step % 60 == 0:  # Every 2 seconds
            joints = expert.get_current_joints(env)
            print(f"Step {step:4d} | State: {expert.current_state:10s} | "
                  f"J1: {joints[0]:5.2f} | J2: {joints[1]:5.2f}")
        
        # Check if done
        if expert.current_state == "done" and step > 300:
            time_in_done = time.time() - expert.state_start_time
            if time_in_done > expert.state_durations["done"]:
                break
    
    print(f"\n✅ Demonstration complete in {step} steps!")
    
    # Save video
    if save_video and frames:
        print(f"\n💾 Saving video to: {output_path}")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"✅ Video saved! ({len(frames)} frames)")
    
    return frames


if __name__ == "__main__":
    create_working_demo()