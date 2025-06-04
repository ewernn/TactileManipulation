"""
Final working demonstration with correct joint configurations.
The robot base is at (-0.5, 0, 0) and needs to reach blocks at (0.05, y, 0.445).
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


class FinalPandaExpert:
    """Expert with correct joint configurations for the actual scene."""
    
    def __init__(self):
        # State machine
        self.states = ["init", "approach_red", "descend_red", "grasp", "lift", 
                      "approach_blue", "descend_blue", "release", "retreat", "done"]
        self.current_state = "init"
        self.state_start_time = 0
        self.state_start_joints = None
        
        # Joint configurations - carefully tuned for reaching from (-0.5, 0, 0) to (0.05, y, 0.445)
        # The robot needs significant forward reach (0.55m in X direction)
        self.joint_configs = {
            # Starting position
            "home": np.array([0, -0.5, 0, -2.3, 0, 1.8, 0.785]),
            
            # Reaching toward red block at (0.05, 0.0, 0.445)
            # Need to lean forward significantly
            "above_red": np.array([0.0, 0.3, 0.0, -2.2, 0.0, 2.5, 1.57]),
            
            # Lower to grasp red block
            "at_red": np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.3, 1.57]),
            
            # Lift configuration
            "lifted": np.array([0.0, 0.2, 0.0, -2.3, 0.0, 2.5, 1.57]),
            
            # Reaching toward blue block at (0.05, -0.15, 0.445)
            # Need to rotate base and lean forward
            "above_blue": np.array([-0.5, 0.3, 0.0, -2.2, 0.0, 2.5, 1.57]),
            
            # Lower to place on blue block
            "at_blue": np.array([-0.5, 0.5, 0.0, -2.0, 0.0, 2.3, 1.57]),
            
            # Retreat position
            "retreat": np.array([-0.3, -0.3, 0.0, -2.3, 0.0, 2.0, 0.785])
        }
        
        # Gripper commands for tendon control
        self.gripper_open = -1.0   # Maps to 255
        self.gripper_close = 1.0   # Maps to 0
        
        # State durations
        self.state_durations = {
            "init": 1.5,
            "approach_red": 3.0,
            "descend_red": 2.0,
            "grasp": 2.0,
            "lift": 2.0,
            "approach_blue": 3.0,
            "descend_blue": 2.0,
            "release": 1.5,
            "retreat": 2.0,
            "done": 1.0
        }
        
    def reset(self):
        """Reset the expert."""
        self.current_state = "init"
        self.state_start_time = time.time()
        self.state_start_joints = None
        
    def get_current_joints(self, env: PandaDemoEnv) -> np.ndarray:
        """Get current joint positions."""
        joints = np.zeros(7)
        for i, joint_id in enumerate(env.arm_joint_ids):
            if joint_id is not None:
                addr = env.model.jnt_qposadr[joint_id]
                joints[i] = env.data.qpos[addr]
        return joints
    
    def interpolate_joints(self, start: np.ndarray, target: np.ndarray, 
                          progress: float) -> np.ndarray:
        """Smooth interpolation with cosine easing."""
        progress = np.clip(progress, 0, 1)
        # Cosine interpolation for smooth motion
        smooth_progress = 0.5 * (1 - np.cos(progress * np.pi))
        return start + (target - start) * smooth_progress
    
    def joints_to_action(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Convert target joints to velocity action."""
        error = target - current
        # P control with appropriate gains
        kp = 3.0
        velocities = kp * error
        # Limit velocities
        return np.clip(velocities, -1.0, 1.0)
    
    def get_hand_position(self, env: PandaDemoEnv) -> np.ndarray:
        """Get current hand position."""
        hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        return env.data.xpos[hand_id].copy()
    
    def get_action(self, obs: Dict[str, np.ndarray], env: PandaDemoEnv) -> np.ndarray:
        """Generate action based on current state."""
        
        action = np.zeros(8)
        
        # Get current state
        time_in_state = time.time() - self.state_start_time
        current_joints = self.get_current_joints(env)
        
        # Initialize start joints
        if self.state_start_joints is None:
            self.state_start_joints = current_joints.copy()
        
        # Calculate progress in current state
        duration = self.state_durations.get(self.current_state, 2.0)
        progress = time_in_state / duration
        
        # State machine
        if self.current_state == "init":
            target = self.joint_configs["home"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = self.gripper_open
            
            if progress >= 1.0:
                print("✓ Initialized")
                self.transition_to("approach_red")
                
        elif self.current_state == "approach_red":
            target = self.joint_configs["above_red"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = self.gripper_open
            
            if progress >= 1.0:
                hand_pos = self.get_hand_position(env)
                print(f"✓ Above red block (hand at: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}])")
                self.transition_to("descend_red")
                
        elif self.current_state == "descend_red":
            target = self.joint_configs["at_red"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = self.gripper_open
            
            if progress >= 1.0:
                print("✓ At grasp position")
                self.transition_to("grasp")
                
        elif self.current_state == "grasp":
            # Hold position
            target = self.joint_configs["at_red"]
            action[:7] = self.joints_to_action(current_joints, target) * 0.3
            
            # Close gripper gradually
            if progress < 0.3:
                action[7] = self.gripper_open
            else:
                action[7] = self.gripper_close
            
            if progress >= 1.0:
                print("✓ Object grasped")
                self.transition_to("lift")
                
        elif self.current_state == "lift":
            target = self.joint_configs["lifted"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = self.gripper_close
            
            if progress >= 1.0:
                print("✓ Object lifted")
                self.transition_to("approach_blue")
                
        elif self.current_state == "approach_blue":
            target = self.joint_configs["above_blue"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = self.gripper_close
            
            if progress >= 1.0:
                hand_pos = self.get_hand_position(env)
                print(f"✓ Above blue block (hand at: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}])")
                self.transition_to("descend_blue")
                
        elif self.current_state == "descend_blue":
            target = self.joint_configs["at_blue"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = self.gripper_close
            
            if progress >= 1.0:
                print("✓ Block placed")
                self.transition_to("release")
                
        elif self.current_state == "release":
            # Hold position
            target = self.joint_configs["at_blue"]
            action[:7] = self.joints_to_action(current_joints, target) * 0.3
            action[7] = self.gripper_open
            
            if progress >= 1.0:
                print("✓ Block released")
                self.transition_to("retreat")
                
        elif self.current_state == "retreat":
            target = self.joint_configs["retreat"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = self.gripper_open
            
            if progress >= 1.0:
                print("✓ Task complete!")
                self.transition_to("done")
                
        elif self.current_state == "done":
            target = self.joint_configs["retreat"]
            action[:7] = self.joints_to_action(current_joints, target) * 0.1
            action[7] = self.gripper_open
        
        return action
    
    def transition_to(self, new_state: str):
        """Transition to new state."""
        self.current_state = new_state
        self.state_start_time = time.time()
        self.state_start_joints = None


def create_final_demo(save_video=True, output_path="final_panda_demo.mp4", 
                     use_side_camera=False):
    """Create the final working demonstration."""
    
    print("🤖 Final Panda Demonstration")
    print("=" * 50)
    
    # Create environment
    camera_name = "side_cam" if use_side_camera else "demo_cam"
    env = PandaDemoEnv(camera_name=camera_name)
    expert = FinalPandaExpert()
    
    # Reset
    obs = env.reset(randomize=False)
    expert.reset()
    
    # Frames for video
    frames = []
    
    print("\n📋 Task: Pick up red block and stack on blue block")
    print(f"📹 Using camera: {camera_name}")
    print("🎬 Starting...\n")
    
    # Calculate total time
    total_time = sum(expert.state_durations.values())
    max_steps = int(total_time * 30) + 200  # 30 FPS + buffer
    
    # Run demonstration
    for step in range(max_steps):
        # Get action
        action = expert.get_action(obs, env)
        
        # Step environment
        obs = env.step(action)
        
        # Render
        frame = env.render()
        frames.append(frame)
        
        # Periodic status
        if step % 90 == 0:  # Every 3 seconds
            hand_pos = expert.get_hand_position(env)
            joints = expert.get_current_joints(env)
            print(f"Step {step:4d} | State: {expert.current_state:12s} | "
                  f"Hand: [{hand_pos[0]:5.2f}, {hand_pos[1]:5.2f}, {hand_pos[2]:5.2f}]")
        
        # Check completion
        if expert.current_state == "done":
            time_in_done = time.time() - expert.state_start_time
            if time_in_done > 1.0:
                break
    
    print(f"\n🎉 Complete in {step} steps!")
    
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
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--side-camera', action='store_true', help='Use side camera view')
    parser.add_argument('--output', type=str, default='final_panda_demo.mp4')
    args = parser.parse_args()
    
    create_final_demo(
        save_video=True,
        output_path=args.output,
        use_side_camera=args.side_camera
    )