"""
Complete working demonstration with proper timing and gripper control.
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


class CompletePandaExpert:
    """Complete expert policy that performs the full stacking task."""
    
    def __init__(self):
        # State machine
        self.states = ["init", "approach_red", "descend_red", "grasp", "lift", 
                      "approach_blue", "descend_blue", "release", "retreat", "done"]
        self.current_state = "init"
        self.state_start_time = 0
        self.state_start_joints = None
        
        # Joint configurations
        self.joint_configs = {
            "home": np.array([0, -0.5, 0, -2.3, 0, 1.8, 0.785]),
            "above_red": np.array([0.0, 0.3, 0.0, -2.2, 0.0, 2.5, 1.57]),
            "at_red": np.array([0.0, 0.5, 0.0, -2.0, 0.0, 2.3, 1.57]),
            "lifted": np.array([0.0, 0.2, 0.0, -2.3, 0.0, 2.5, 1.57]),
            "above_blue": np.array([-0.5, 0.3, 0.0, -2.2, 0.0, 2.5, 1.57]),
            "at_blue": np.array([-0.5, 0.5, 0.0, -2.0, 0.0, 2.3, 1.57]),
            "retreat": np.array([-0.3, -0.3, 0.0, -2.3, 0.0, 2.0, 0.785])
        }
        
        # State durations - adjusted for proper gripper timing
        self.state_durations = {
            "init": 2.0,
            "approach_red": 3.0,
            "descend_red": 2.5,
            "grasp": 3.0,      # More time to close gripper
            "lift": 2.5,
            "approach_blue": 3.5,
            "descend_blue": 2.5,
            "release": 2.0,    # More time to open gripper
            "retreat": 2.0,
            "done": 1.0
        }
        
        # Track block positions
        self.red_block_grasped = False
        
    def reset(self):
        """Reset the expert."""
        self.current_state = "init"
        self.state_start_time = time.time()
        self.state_start_joints = None
        self.red_block_grasped = False
        
    def get_current_joints(self, env: PandaDemoEnv) -> np.ndarray:
        """Get current joint positions."""
        joints = np.zeros(7)
        for i, joint_id in enumerate(env.arm_joint_ids):
            if joint_id is not None:
                addr = env.model.jnt_qposadr[joint_id]
                joints[i] = env.data.qpos[addr]
        return joints
    
    def get_gripper_width(self, env: PandaDemoEnv) -> float:
        """Get actual gripper width."""
        if env.gripper_joint1_id is not None:
            addr1 = env.model.jnt_qposadr[env.gripper_joint1_id]
            addr2 = env.model.jnt_qposadr[env.gripper_joint2_id]
            return env.data.qpos[addr1] + env.data.qpos[addr2]
        return 0.08
    
    def interpolate_joints(self, start: np.ndarray, target: np.ndarray, 
                          progress: float) -> np.ndarray:
        """Smooth interpolation."""
        progress = np.clip(progress, 0, 1)
        smooth_progress = 0.5 * (1 - np.cos(progress * np.pi))
        return start + (target - start) * smooth_progress
    
    def joints_to_action(self, current: np.ndarray, target: np.ndarray, 
                        gain: float = 3.0) -> np.ndarray:
        """Convert target joints to velocity action."""
        error = target - current
        velocities = gain * error
        return np.clip(velocities, -1.0, 1.0)
    
    def get_hand_position(self, env: PandaDemoEnv) -> np.ndarray:
        """Get hand position."""
        hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        return env.data.xpos[hand_id].copy()
    
    def get_block_positions(self, env: PandaDemoEnv) -> Tuple[np.ndarray, np.ndarray]:
        """Get red and blue block positions."""
        red_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        blue_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        return env.data.xpos[red_id].copy(), env.data.xpos[blue_id].copy()
    
    def get_action(self, obs: Dict[str, np.ndarray], env: PandaDemoEnv) -> np.ndarray:
        """Generate action based on current state."""
        
        action = np.zeros(8)
        
        # Get current state
        time_in_state = time.time() - self.state_start_time
        current_joints = self.get_current_joints(env)
        gripper_width = self.get_gripper_width(env)
        
        # Initialize start joints
        if self.state_start_joints is None:
            self.state_start_joints = current_joints.copy()
        
        # Calculate progress
        duration = self.state_durations[self.current_state]
        progress = time_in_state / duration
        
        # Get block positions
        red_pos, blue_pos = self.get_block_positions(env)
        
        # State machine
        if self.current_state == "init":
            target = self.joint_configs["home"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = -1.0  # Open gripper
            
            if progress >= 1.0:
                print("✓ Initialized at home")
                self.transition_to("approach_red")
                
        elif self.current_state == "approach_red":
            target = self.joint_configs["above_red"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = -1.0  # Keep open
            
            if progress >= 0.8:  # Check position
                hand_pos = self.get_hand_position(env)
                dist_to_red = np.linalg.norm(hand_pos[:2] - red_pos[:2])
                if dist_to_red < 0.15 or progress >= 1.0:
                    print(f"✓ Above red block (distance: {dist_to_red:.3f}m)")
                    self.transition_to("descend_red")
                
        elif self.current_state == "descend_red":
            target = self.joint_configs["at_red"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp, gain=2.0)  # Slower
            action[7] = -1.0  # Keep open
            
            if progress >= 1.0:
                print(f"✓ At grasp position (gripper width: {gripper_width:.3f}m)")
                self.transition_to("grasp")
                
        elif self.current_state == "grasp":
            # Hold position steady
            target = self.joint_configs["at_red"]
            action[:7] = self.joints_to_action(current_joints, target, gain=1.0)
            
            # Close gripper gradually
            if progress < 0.2:
                action[7] = -1.0  # Still open
            else:
                action[7] = 1.0   # Close
            
            # Check if grasped
            if progress > 0.7 and gripper_width < 0.06:  # Gripper closed around block
                self.red_block_grasped = True
                print(f"✓ Object grasped (width: {gripper_width:.3f}m)")
                self.transition_to("lift")
            elif progress >= 1.0:
                print(f"✓ Grasp complete (width: {gripper_width:.3f}m)")
                self.transition_to("lift")
                
        elif self.current_state == "lift":
            target = self.joint_configs["lifted"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp, gain=2.5)
            action[7] = 1.0  # Keep closed
            
            if progress >= 0.8:
                # Check if block is lifted
                if red_pos[2] > 0.50:  # Block lifted above threshold
                    print(f"✓ Object lifted (height: {red_pos[2]:.3f}m)")
                    self.transition_to("approach_blue")
                elif progress >= 1.0:
                    self.transition_to("approach_blue")
                
        elif self.current_state == "approach_blue":
            target = self.joint_configs["above_blue"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = 1.0  # Keep closed
            
            if progress >= 0.8:
                hand_pos = self.get_hand_position(env)
                dist_to_blue = np.linalg.norm(hand_pos[:2] - blue_pos[:2])
                if dist_to_blue < 0.15 or progress >= 1.0:
                    print(f"✓ Above blue block (distance: {dist_to_blue:.3f}m)")
                    self.transition_to("descend_blue")
                
        elif self.current_state == "descend_blue":
            target = self.joint_configs["at_blue"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp, gain=1.5)  # Careful
            action[7] = 1.0  # Keep closed
            
            if progress >= 1.0:
                print(f"✓ Block placed (red at: {red_pos[2]:.3f}m)")
                self.transition_to("release")
                
        elif self.current_state == "release":
            # Hold position
            target = self.joint_configs["at_blue"]
            action[:7] = self.joints_to_action(current_joints, target, gain=0.5)
            action[7] = -1.0  # Open gripper
            
            if progress > 0.5 and gripper_width > 0.08:
                print(f"✓ Block released (width: {gripper_width:.3f}m)")
                self.transition_to("retreat")
            elif progress >= 1.0:
                self.transition_to("retreat")
                
        elif self.current_state == "retreat":
            target = self.joint_configs["retreat"]
            interp = self.interpolate_joints(self.state_start_joints, target, progress)
            action[:7] = self.joints_to_action(current_joints, interp)
            action[7] = -1.0  # Keep open
            
            if progress >= 1.0:
                # Check final positions
                print(f"\n📊 Final positions:")
                print(f"   Red block: {red_pos}")
                print(f"   Blue block: {blue_pos}")
                print(f"   Stack height: {red_pos[2] - blue_pos[2]:.3f}m")
                print("\n✅ Task complete!")
                self.transition_to("done")
                
        elif self.current_state == "done":
            target = self.joint_configs["retreat"]
            action[:7] = self.joints_to_action(current_joints, target, gain=0.2)
            action[7] = -1.0
        
        return action
    
    def transition_to(self, new_state: str):
        """Transition to new state."""
        self.current_state = new_state
        self.state_start_time = time.time()
        self.state_start_joints = None


def create_complete_demo(save_video=True, output_path="complete_panda_demo.mp4", 
                        camera="demo_cam", max_duration=30.0):
    """Create the complete demonstration."""
    
    print("🤖 Complete Panda Stacking Demonstration")
    print("=" * 50)
    
    # Create environment
    env = PandaDemoEnv(camera_name=camera)
    expert = CompletePandaExpert()
    
    # Reset
    obs = env.reset(randomize=False)
    expert.reset()
    
    # Video frames
    frames = []
    
    print(f"\n📋 Task: Stack red block on blue block")
    print(f"📹 Camera: {camera}")
    print("🎬 Starting...\n")
    
    # Run demonstration
    start_time = time.time()
    step = 0
    
    while True:
        # Get action
        action = expert.get_action(obs, env)
        
        # Step environment
        obs = env.step(action)
        
        # Render
        frame = env.render()
        frames.append(frame)
        
        # Status updates
        if step % 60 == 0:  # Every 2 seconds
            elapsed = time.time() - start_time
            print(f"[{elapsed:5.1f}s] Step {step:4d} | State: {expert.current_state:12s}")
        
        step += 1
        
        # Check completion
        if expert.current_state == "done":
            done_time = time.time() - expert.state_start_time
            if done_time > 1.0:
                break
        
        # Safety timeout
        if time.time() - start_time > max_duration:
            print("\n⚠️  Timeout reached")
            break
    
    print(f"\n🎉 Demonstration complete!")
    print(f"   Total steps: {step}")
    print(f"   Duration: {time.time() - start_time:.1f}s")
    
    # Save video
    if save_video and frames:
        print(f"\n💾 Saving video to: {output_path}")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"✅ Video saved! ({len(frames)} frames @ 30fps = {len(frames)/30:.1f}s)")
    
    return frames


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default='demo_cam',
                       choices=['demo_cam', 'side_cam', 'overhead_cam', 'wrist_cam'],
                       help='Camera view to use')
    parser.add_argument('--output', type=str, default='complete_panda_demo.mp4')
    parser.add_argument('--no-video', action='store_true', help='Skip video saving')
    args = parser.parse_args()
    
    create_complete_demo(
        save_video=not args.no_video,
        output_path=args.output,
        camera=args.camera
    )