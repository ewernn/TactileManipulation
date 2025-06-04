#!/usr/bin/env python3
"""
Test expert demonstration using the original approach.
"""

import numpy as np
import mujoco
import cv2
import sys
import os

# Add path to environments
sys.path.append('../environments')

# Import the environment
try:
    from panda_demo_env import PandaDemoEnv
except ImportError:
    print("Error: Could not import PandaDemoEnv")
    print("Trying direct load...")
    
    # Fallback: implement minimal version
    class PandaDemoEnv:
        def __init__(self):
            xml_path = "../franka_emika_panda/panda_demo_scene.xml"
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
            
            # Find joint IDs
            self.arm_joint_ids = []
            for i in range(1, 8):
                joint_name = f"joint{i}"
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.arm_joint_ids.append(joint_id if joint_id != -1 else None)
            
            # Camera
            self.camera_name = "demo_cam"
            
        def reset(self):
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            
            # Set initial joint configuration
            joint_vals = [0, -0.1, 0, -2.0, 0, 1.5, 0]
            for i, val in enumerate(joint_vals):
                if i < len(self.arm_joint_ids) and self.arm_joint_ids[i] is not None:
                    joint_addr = self.model.jnt_qposadr[self.arm_joint_ids[i]]
                    self.data.qpos[joint_addr] = val
                    self.data.ctrl[i] = val
            
            # Open gripper
            self.data.ctrl[7] = 255 if self.model.nu > 7 else 0
            
            mujoco.mj_forward(self.model, self.data)
            
            return self._get_observation()
        
        def _get_observation(self):
            # Get IDs
            ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            red_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
            blue_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
            
            # Build observation
            obs = {
                'joint_pos': np.array([self.data.qpos[self.model.jnt_qposadr[jid]] 
                                      for jid in self.arm_joint_ids if jid is not None]),
                'target_block_pos': self.data.xpos[red_id].copy() if red_id != -1 else np.zeros(3),
                'block2_pos': self.data.xpos[blue_id].copy() if blue_id != -1 else np.zeros(3),
                'ee_pos': self.data.xpos[ee_id].copy() if ee_id != -1 else np.zeros(3),
                'gripper_pos': np.array([self.data.ctrl[7] / 255.0]) if self.model.nu > 7 else np.array([1.0]),
                'tactile': np.zeros(72)  # Placeholder
            }
            
            return obs
        
        def step(self, action, steps=1):
            # Apply velocity control
            action = np.clip(action, -1.0, 1.0)
            joint_vels = action[:7] * 2.0  # Scale to rad/s
            gripper_cmd = action[7] if len(action) > 7 else 0.0
            
            # Integrate velocities
            for i, vel in enumerate(joint_vels):
                if i < len(self.arm_joint_ids) and self.arm_joint_ids[i] is not None:
                    joint_addr = self.model.jnt_qposadr[self.arm_joint_ids[i]]
                    current_pos = self.data.qpos[joint_addr]
                    target_pos = current_pos + vel * 0.05  # 20Hz control
                    self.data.ctrl[i] = target_pos
            
            # Gripper
            if self.model.nu > 7:
                # Map [-1, 1] to [255, 0]
                self.data.ctrl[7] = 127.5 * (1 - gripper_cmd)
            
            # Step physics
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
            
            obs = self._get_observation()
            return obs, 0, False, False, {}
        
        def render(self, camera=None):
            if camera is None:
                camera = self.camera_name
            
            try:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                self.renderer.update_scene(self.data, camera=cam_id)
            except:
                self.renderer.update_scene(self.data)
            
            return self.renderer.render()


class TestExpertPolicy:
    """Simple expert policy for testing."""
    
    def __init__(self):
        self.phase = "approach"
        self.step_count = 0
        
    def reset(self):
        self.phase = "approach"
        self.step_count = 0
    
    def get_action(self, observation):
        """Generate action based on phase."""
        action = np.zeros(8)
        
        if self.phase == "approach":
            if self.step_count < 40:
                # Move forward and slightly down
                action = np.array([0.0, 0.2, 0.15, -0.25, 0.1, -0.05, 0.0, -1.0])
            else:
                self.phase = "descend"
                self.step_count = 0
                
        elif self.phase == "descend":
            if self.step_count < 30:
                # Move down to grasp
                action = np.array([0.0, 0.15, 0.2, -0.15, -0.1, -0.05, 0.0, -1.0])
            else:
                self.phase = "grasp"
                self.step_count = 0
                
        elif self.phase == "grasp":
            if self.step_count < 20:
                # Close gripper
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            else:
                self.phase = "lift"
                self.step_count = 0
                
        elif self.phase == "lift":
            if self.step_count < 30:
                # Lift up
                action = np.array([0.0, -0.15, -0.15, 0.15, 0.05, 0.0, 0.0, 1.0])
            else:
                self.phase = "done"
                self.step_count = 0
                
        elif self.phase == "done":
            # Stay still
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        self.step_count += 1
        return action


def main():
    """Test expert demonstration."""
    print("Testing expert demonstration...")
    
    # Create environment
    env = PandaDemoEnv()
    expert = TestExpertPolicy()
    
    # Reset
    obs = env.reset()
    expert.reset()
    
    # Storage for video
    frames = []
    
    # Run demonstration
    max_steps = 200
    
    for step in range(max_steps):
        # Get expert action
        action = expert.get_action(obs)
        
        # Step environment
        obs, _, _, _, _ = env.step(action)
        
        # Render
        frame = env.render()
        
        # Add text overlays
        cv2.putText(frame, f"EXPERT TEST", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 2)
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"Step: {step}", (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # EE position
        ee_pos = obs['ee_pos']
        cv2.putText(frame, f"EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]", 
                    (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Block position
        block_pos = obs['target_block_pos']
        cv2.putText(frame, f"Block: [{block_pos[0]:.2f}, {block_pos[1]:.2f}, {block_pos[2]:.2f}]", 
                    (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        frames.append(frame)
        
        # Print status
        if step % 20 == 0:
            print(f"\nStep {step} | Phase: {expert.phase}")
            print(f"  EE pos: {ee_pos}")
            print(f"  Block pos: {block_pos}")
            print(f"  Gripper: {'CLOSED' if action[7] > 0 else 'OPEN'}")
    
    # Save video
    os.makedirs("../../videos/121pm", exist_ok=True)
    output_path = "../../videos/121pm/test_expert_demo.mp4"
    
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20, (width, height))  # 20 fps
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"\n✓ Saved video to: {output_path}")

if __name__ == "__main__":
    main()