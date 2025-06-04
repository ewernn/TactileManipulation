"""
Enhanced demo environment with better camera angles and scene setup.
"""

import numpy as np
import mujoco
import os
from typing import Dict, Tuple, Optional

from .tactile_sensor import TactileSensor


class PandaDemoEnv:
    """
    Enhanced demo environment for creating videos with:
    - Better camera angles
    - Table and multiple blocks
    - Professional lighting
    """
    
    def __init__(self, camera_name="demo_cam"):
        # Load enhanced demo scene
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = os.path.join(base_dir, "franka_emika_panda", "panda_demo_scene.xml")
        
        print(f"Loading demo scene from: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Setup renderer with better camera
        self.camera_name = camera_name
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
        
        print("🎬 Panda Demo Environment Ready!")
        print(f"   📹 Camera: {camera_name}")
        print(f"   🎥 Resolution: 1280x720")
        print(f"   🎯 Control dim: {self.model.nu}")
        
    def _setup_model_ids(self):
        """Get MuJoCo IDs for joints, bodies, etc."""
        
        # Find 7 arm joints
        self.arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.arm_joint_ids = []
        
        for name in self.arm_joint_names:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.arm_joint_ids.append(jid)
            except:
                print(f"   ⚠️  Joint not found: {name}")
        
        # Find gripper joints  
        try:
            self.gripper_joint1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1')
            self.gripper_joint2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint2')
        except:
            self.gripper_joint1_id = None
            self.gripper_joint2_id = None
        
        # Find blocks
        try:
            self.target_block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
            self.block2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
            self.block3_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block3")
            print(f"   🧱 Found blocks: target_block, block2, block3")
        except:
            print("   ⚠️  Some blocks not found")
    
    def reset(self, randomize=True):
        """Reset environment to initial state."""
        
        # Reset physics
        mujoco.mj_resetData(self.model, self.data)
        
        # Set arm to home position - adjusted for table-level blocks
        home_qpos = np.array([0, -0.5, 0, -2.3, 0, 1.8, 0.785])  # Better starting position
        
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
        
        # Randomize block positions slightly if requested
        if randomize:
            if hasattr(self, 'target_block_id'):
                # Red block - randomize around the reachable position
                block_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.target_block_id]]
                self.data.qpos[block_addr:block_addr+3] = [
                    -0.15 + np.random.uniform(-0.02, 0.02),
                    0.0 + np.random.uniform(-0.02, 0.02),
                    0.445  # On table
                ]
            
            if hasattr(self, 'block2_id'):
                # Blue block - randomize position
                block_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.block2_id]]
                self.data.qpos[block_addr:block_addr+3] = [
                    -0.15 + np.random.uniform(-0.02, 0.02),
                    -0.15 + np.random.uniform(-0.02, 0.02),
                    0.445  # On table
                ]
        
        # Forward dynamics to settle
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_observation()
    
    def step(self, action, steps=1):
        """Execute action for specified number of physics steps."""
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Extract joint velocities and gripper command
        joint_vels = action[:7] * 2.0  # Scale to rad/s
        gripper_cmd = action[7] if len(action) > 7 else 0.0
        
        # Apply joint velocity control
        for i, (joint_id, vel) in enumerate(zip(self.arm_joint_ids, joint_vels)):
            if joint_id is not None and i < self.model.nu:
                # Get current position
                joint_addr = self.model.jnt_qposadr[joint_id]
                current_pos = self.data.qpos[joint_addr]
                
                # Integrate velocity to get target position  
                target_pos = current_pos + vel * 0.05  # 20 Hz control
                
                # Apply joint limits
                joint_range = self.model.jnt_range[joint_id]
                if joint_range[0] < joint_range[1]:  # Has limits
                    target_pos = np.clip(target_pos, joint_range[0], joint_range[1])
                
                # Set control target
                self.data.ctrl[i] = target_pos
        
        # Apply gripper control
        if self.gripper_joint1_id is not None and self.model.nu > 7:
            # Gripper command: -1 = open, +1 = close
            # Tendon control: 0 = close (pull tendons), 255 = open (release tendons)
            # Map: -1 → 255 (open), +1 → 0 (close)
            gripper_ctrl = 127.5 * (1 - gripper_cmd)  # -1 → 255, +1 → 0
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
        if hasattr(self, 'target_block_id'):
            obs['target_block_pos'] = self.data.xpos[self.target_block_id].copy()
        
        if hasattr(self, 'block2_id'):
            obs['block2_pos'] = self.data.xpos[self.block2_id].copy()
        
        return obs
    
    def render(self, camera=None):
        """Render current state with specified camera."""
        
        if camera is None:
            camera = self.camera_name
            
        # Set camera
        try:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            self.renderer.update_scene(self.data, camera=cam_id)
        except:
            # Fallback to default camera
            self.renderer.update_scene(self.data)
        
        return self.renderer.render()
    
    def get_available_cameras(self):
        """Get list of available camera names."""
        cameras = []
        for i in range(self.model.ncam):
            cam_name = self.model.camera(i).name
            if cam_name:
                cameras.append(cam_name)
        return cameras


# Test the enhanced environment
if __name__ == "__main__":
    print("🧪 Testing Enhanced Demo Environment...")
    
    try:
        env = PandaDemoEnv()
        
        print(f"Available cameras: {env.get_available_cameras()}")
        
        # Test reset
        obs = env.reset()
        print(f"✅ Reset successful")
        
        # Test different camera views
        cameras = ["demo_cam", "side_cam", "overhead_cam"]
        
        for cam in cameras:
            try:
                frame = env.render(camera=cam)
                print(f"✅ {cam}: {frame.shape}")
            except:
                print(f"❌ {cam}: not available")
        
        print(f"Environment ready for enhanced video creation!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()