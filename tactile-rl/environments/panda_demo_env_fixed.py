"""
Fixed demo environment with correct robot orientation.
"""

import numpy as np
import mujoco
import os
from typing import Dict, Tuple, Optional

from tactile_sensor import TactileSensor


class PandaDemoEnvFixed:
    """
    Fixed demo environment with robot properly oriented to face workspace.
    """
    
    def __init__(self, use_fixed_scene=True):
        # Load scene
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if use_fixed_scene:
            xml_path = os.path.join(base_dir, "franka_emika_panda", "panda_demo_scene_fixed.xml")
        else:
            xml_path = os.path.join(base_dir, "franka_emika_panda", "panda_demo_scene.xml")
        
        print(f"Loading scene from: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Setup renderer
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        
        # Initialize tactile sensor
        self.tactile_sensor = TactileSensor(
            model=self.model,
            data=self.data,
            n_taxels_x=3, 
            n_taxels_y=4,
            left_finger_name="left_finger",
            right_finger_name="right_finger"
        )
        
        # Get model IDs
        self._setup_model_ids()
        
        print("🎬 Fixed Panda Demo Environment Ready!")
        print(f"   🎯 Control dim: {self.model.nu}")
    
    def _setup_model_ids(self):
        """Get IDs for bodies, joints, and sites."""
        # Arm joints
        joint_names = [f"joint{i}" for i in range(1, 8)]
        self.arm_joint_ids = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.arm_joint_ids.append(joint_id if joint_id >= 0 else None)
        
        # Gripper joints
        self.gripper_joint1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.gripper_joint2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        
        # End effector site
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        
        # Blocks
        self.target_block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        self.block2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        self.block3_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block3")
        
        if self.target_block_id >= 0:
            print(f"   🧱 Found blocks at IDs: {self.target_block_id}, {self.block2_id}, {self.block3_id}")
    
    def reset(self):
        """Reset environment to initial state."""
        
        # Reset physics
        mujoco.mj_resetData(self.model, self.data)
        
        # Set arm to home position - NO rotation, face forward naturally
        home_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.920, 0.785])
        
        for i, (joint_id, qpos) in enumerate(zip(self.arm_joint_ids, home_qpos)):
            if joint_id is not None:
                joint_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[joint_addr] = qpos
        
        # Open gripper
        if self.gripper_joint1_id is not None:
            gripper_addr1 = self.model.jnt_qposadr[self.gripper_joint1_id] 
            gripper_addr2 = self.model.jnt_qposadr[self.gripper_joint2_id]
            self.data.qpos[gripper_addr1] = 0.04  # Open
            self.data.qpos[gripper_addr2] = 0.04  # Open
        
        # Forward simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Check orientation
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        hand_quat = self.data.xquat[hand_id]
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, hand_quat)
        rotmat = rotmat.reshape(3, 3)
        gripper_direction = rotmat[:, 2]
        
        if self.ee_site_id >= 0:
            ee_pos = self.data.site_xpos[self.ee_site_id]
        else:
            ee_pos = self.data.xpos[hand_id]
        
        print(f"\n   🤖 Reset complete:")
        print(f"      EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        print(f"      Gripper direction: [{gripper_direction[0]:.3f}, {gripper_direction[1]:.3f}, {gripper_direction[2]:.3f}]")
        
        return self._get_observation()
    
    def step(self, action, steps=1):
        """Apply action and step simulation."""
        
        # Apply joint velocities
        for i in range(7):
            if self.arm_joint_ids[i] is not None:
                joint_addr = self.model.jnt_qposadr[self.arm_joint_ids[i]]
                current_pos = self.data.qpos[joint_addr]
                
                # Velocity control - integrate to position
                target_pos = current_pos + action[i] * 0.05  # Scale velocity
                
                # Apply joint limits
                joint_id = self.arm_joint_ids[i]
                joint_range = self.model.jnt_range[joint_id]
                if joint_range[0] < joint_range[1]:  # Has limits
                    target_pos = np.clip(target_pos, joint_range[0], joint_range[1])
                
                # Set control target
                self.data.ctrl[i] = target_pos
        
        # Apply gripper control
        if self.model.nu > 7:
            # Convert gripper command to tendon control
            # -1 = open (255), +1 = close (0)
            gripper_cmd = action[7]
            gripper_ctrl = 127.5 * (1 - gripper_cmd)
            self.data.ctrl[7] = gripper_ctrl
        
        # Step physics
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation state."""
        obs = {}
        
        # Joint positions and velocities
        joint_pos = np.zeros(7)
        joint_vel = np.zeros(7)
        
        for i, joint_id in enumerate(self.arm_joint_ids):
            if joint_id is not None:
                joint_addr = self.model.jnt_qposadr[joint_id]
                vel_addr = self.model.jnt_dofadr[joint_id]
                joint_pos[i] = self.data.qpos[joint_addr]
                joint_vel[i] = self.data.qvel[vel_addr]
        
        obs['joint_pos'] = joint_pos
        obs['joint_vel'] = joint_vel
        
        # Gripper position
        if self.gripper_joint1_id is not None:
            gripper_addr = self.model.jnt_qposadr[self.gripper_joint1_id]
            gripper_opening = self.data.qpos[gripper_addr] / 0.04
            obs['gripper_pos'] = gripper_opening
        else:
            obs['gripper_pos'] = 0.5
        
        # Tactile readings
        left_tactile, right_tactile = self.tactile_sensor.get_readings(self.model, self.data)
        obs['tactile'] = np.concatenate([left_tactile.flatten(), right_tactile.flatten()])
        
        # Block positions
        if self.target_block_id >= 0:
            obs['target_block_pos'] = self.data.xpos[self.target_block_id].copy()
        
        return obs
    
    def render(self, camera="demo_cam"):
        """Render current state."""
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()