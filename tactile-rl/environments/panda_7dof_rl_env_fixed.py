"""
7-DOF Panda Tactile Environment for RL - Fixed version that loads scene with blocks
Clean implementation without external dependencies.
"""

import numpy as np
import mujoco
import os
from typing import Dict, Tuple, Optional

from .tactile_sensor import TactileSensor


class ActionSpace:
    """Simple action space definition."""
    def __init__(self, low, high, shape):
        self.low = np.array(low)
        self.high = np.array(high) 
        self.shape = shape
    
    def sample(self):
        return np.random.uniform(self.low, self.high, self.shape)


class ObservationSpace:
    """Simple observation space definition.""" 
    def __init__(self, spaces_dict):
        self.spaces = spaces_dict


class Panda7DOFTactileRL:
    """
    7-DOF Panda with tactile sensing for RL training - Fixed version with blocks.
    
    Key Features:
    - ✅ 7-DOF arm control via velocity commands  
    - ✅ Tactile sensing integration
    - ✅ Includes blocks in the scene for manipulation
    - ✅ No external Gym dependencies
    
    Action Space (8D):
        [0-6]: Joint velocity commands [-1, +1] rad/s
        [7]:   Gripper command [-1=open, +1=close]
    
    Observation Space:
        - joint_pos: 7 joint angles
        - joint_vel: 7 joint velocities
        - gripper_pos: Gripper opening [0-1]
        - tactile: 72D tactile readings
        - target_block_pose: 7D block pose (pos + quat)
        - ee_pose: 7D end-effector pose
    """
    
    def __init__(
        self,
        max_episode_steps: int = 500,
        success_height: float = 0.15,
        control_frequency: int = 20,  # Hz - realistic for real robots
        joint_vel_limit: float = 2.0,  # rad/s - realistic joint speeds
        use_tactile: bool = True,
        **kwargs
    ):
        self.max_episode_steps = max_episode_steps
        self.success_height = success_height
        self.control_frequency = control_frequency
        self.joint_vel_limit = joint_vel_limit
        self.use_tactile = use_tactile
        self.dt = 1.0 / control_frequency
        
        # Load 7-DOF Panda model with scene
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = os.path.join(base_dir, "franka_emika_panda", "panda_rl_scene.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Panda scene model not found at {xml_path}")
        
        print(f"Loading Panda scene model from: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize tactile sensor
        if self.use_tactile:
            self.tactile_sensor = TactileSensor(
                model=self.model,
                data=self.data,
                n_taxels_x=3, 
                n_taxels_y=4,
                left_finger_name="left_finger",
                right_finger_name="right_finger"
            )
        
        # Setup model IDs and spaces
        self._setup_model_ids()
        self._setup_action_observation_spaces()
        
        # Episode state
        self.current_step = 0
        self.initial_block_pose = None
        
        print("🤖 Panda 7-DOF Tactile RL Environment Ready!")
        print(f"   🎮 Action dim: {self.action_space.shape}")
        print(f"   👁️  Observation keys: {list(self.observation_space.spaces.keys())}")
        print(f"   🕐 Control frequency: {control_frequency} Hz")
        print(f"   🚀 Max joint velocity: {joint_vel_limit} rad/s")
        print(f"   🔧 Tactile sensing: {'Enabled' if use_tactile else 'Disabled'}")
    
    def _setup_model_ids(self):
        """Get MuJoCo IDs for joints, bodies, etc."""
        
        # Find 7 arm joints
        self.arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.arm_joint_ids = []
        
        for name in self.arm_joint_names:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.arm_joint_ids.append(jid)
                print(f"   Found joint: {name} (ID: {jid})")
            except:
                print(f"   ⚠️  Joint not found: {name}")
        
        # Find gripper joints  
        try:
            self.gripper_joint1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1')
            self.gripper_joint2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint2')
            print(f"   Found gripper joints: finger_joint1, finger_joint2")
        except:
            print("   ⚠️  Gripper joints not found")
            self.gripper_joint1_id = None
            self.gripper_joint2_id = None
        
        # Find target block (red block)
        try:
            self.target_block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
            print(f"   Found target block (red)")
        except:
            print("   ⚠️  Target block not found")
            self.target_block_body_id = None
        
        # Find other blocks
        try:
            self.block2_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
            self.block3_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block3")
            print(f"   Found additional blocks (green, blue)")
        except:
            self.block2_body_id = None
            self.block3_body_id = None
        
        # Find end-effector
        try:
            self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            print(f"   Found end-effector (hand)")
        except:
            print("   ⚠️  End-effector not found")
            self.ee_body_id = None
        
        print(f"   🔧 Model has {self.model.njnt} joints, {self.model.nbody} bodies")
        print(f"   🎯 Control dim: {self.model.nu}")
    
    def _setup_action_observation_spaces(self):
        """Define action and observation spaces."""
        
        # Action: 7 joint velocities + 1 gripper
        self.action_space = ActionSpace(
            low=[-1.0] * 8,
            high=[1.0] * 8, 
            shape=(8,)
        )
        
        # Observations
        obs_spaces = {
            'joint_pos': {'shape': (7,), 'low': -3.14, 'high': 3.14},
            'joint_vel': {'shape': (7,), 'low': -10.0, 'high': 10.0},
            'gripper_pos': {'shape': (1,), 'low': 0.0, 'high': 1.0},
            'target_block_pose': {'shape': (7,), 'low': -2.0, 'high': 2.0}, 
            'ee_pose': {'shape': (7,), 'low': -2.0, 'high': 2.0}
        }
        
        if self.use_tactile:
            obs_spaces['tactile'] = {'shape': (72,), 'low': 0.0, 'high': 10.0}
        
        self.observation_space = ObservationSpace(obs_spaces)
    
    def reset(self, randomize=True):
        """Reset environment to initial state."""
        
        # Reset physics
        mujoco.mj_resetData(self.model, self.data)
        
        # Use the rl_home keyframe if available
        try:
            key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "rl_home")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            pass  # Successfully reset to keyframe
        except:
            # Manual reset if no keyframe
            print("   Manual reset (no keyframe)")
            
            # Set arm to home position
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
        
        # Randomize block positions if requested
        if randomize and self.target_block_body_id is not None:
            # Find the joint for the target block
            try:
                block_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube:joint")
                joint_addr = self.model.jnt_qposadr[block_joint_id]
                
                # Randomize position on table
                self.data.qpos[joint_addr] = 0.4 + np.random.uniform(-0.05, 0.05)  # x
                self.data.qpos[joint_addr + 1] = np.random.uniform(-0.1, 0.1)    # y  
                self.data.qpos[joint_addr + 2] = 0.44  # z (on table)
                
                # Keep orientation upright
                self.data.qpos[joint_addr + 3:joint_addr + 7] = [1, 0, 0, 0]  # quaternion
            except:
                pass
        
        # Forward dynamics to settle
        mujoco.mj_forward(self.model, self.data)
        
        # Store initial state
        if self.target_block_body_id is not None:
            self.initial_block_pose = self.data.xpos[self.target_block_body_id].copy()
        else:
            self.initial_block_pose = np.array([0.4, 0.0, 0.44])
            
        self.current_step = 0
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one environment step with velocity control."""
        
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Extract joint velocities and gripper command
        joint_vels = action[:7] * self.joint_vel_limit  # Scale to rad/s
        gripper_cmd = action[7]  # -1=open, +1=close
        
        # Apply joint velocity control (using position integration)
        for i, (joint_id, vel) in enumerate(zip(self.arm_joint_ids, joint_vels)):
            if joint_id is not None and i < self.model.nu:
                # Get current position
                joint_addr = self.model.jnt_qposadr[joint_id]
                current_pos = self.data.qpos[joint_addr]
                
                # Integrate velocity to get target position
                target_pos = current_pos + vel * self.dt
                
                # Apply joint limits
                joint_range = self.model.jnt_range[joint_id]
                if joint_range[0] < joint_range[1]:  # Has limits
                    target_pos = np.clip(target_pos, joint_range[0], joint_range[1])
                
                # Set control target (position control)
                self.data.ctrl[i] = target_pos
        
        # Apply gripper control
        if self.gripper_joint1_id is not None and self.model.nu > 7:
            # Map [-1, 1] to [255, 0] for the tendon-based gripper control
            gripper_ctrl = 127.5 * (1 - gripper_cmd)  # -1 → 255 (open), +1 → 0 (closed)
            self.data.ctrl[7] = np.clip(gripper_ctrl, 0, 255)
        
        # Step physics for desired control frequency
        physics_steps = max(1, int(self.dt / self.model.opt.timestep))
        
        for _ in range(physics_steps):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(obs, action)
        
        # Check termination
        success = self._check_success(obs)
        terminated = success
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            'success': success,
            'block_height': obs['target_block_pose'][2],
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
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
        
        obs['joint_pos'] = joint_pos.astype(np.float32)
        obs['joint_vel'] = joint_vel.astype(np.float32)
        
        # Gripper position
        if self.gripper_joint1_id is not None:
            gripper_addr = self.model.jnt_qposadr[self.gripper_joint1_id]
            gripper_opening = self.data.qpos[gripper_addr] / 0.04  # Normalize
            obs['gripper_pos'] = np.array([gripper_opening], dtype=np.float32)
        else:
            obs['gripper_pos'] = np.array([0.5], dtype=np.float32)  # Default
        
        # Tactile readings
        if self.use_tactile:
            left_tactile, right_tactile = self.tactile_sensor.get_readings(self.model, self.data)
            tactile = np.concatenate([left_tactile.flatten(), right_tactile.flatten()])
            obs['tactile'] = tactile.astype(np.float32)
        
        # Target block pose
        if self.target_block_body_id is not None:
            block_pos = self.data.xpos[self.target_block_body_id].copy()
            block_quat = self.data.xquat[self.target_block_body_id].copy()
        else:
            block_pos = np.array([0.4, 0.0, 0.44])
            block_quat = np.array([1, 0, 0, 0])  # Identity quaternion
        
        obs['target_block_pose'] = np.concatenate([block_pos, block_quat]).astype(np.float32)
        
        # End-effector pose
        if self.ee_body_id is not None:
            ee_pos = self.data.xpos[self.ee_body_id].copy()
            ee_quat = self.data.xquat[self.ee_body_id].copy()
        else:
            # Approximate from last joint
            ee_pos = np.array([0.5, 0.0, 0.8])  # Placeholder
            ee_quat = np.array([1, 0, 0, 0])    # Placeholder
            
        obs['ee_pose'] = np.concatenate([ee_pos, ee_quat]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, obs, action):
        """Compute reward for RL training."""
        reward = 0.0
        
        # Height reward - main objective
        block_height = obs['target_block_pose'][2]
        initial_height = self.initial_block_pose[2]
        height_diff = block_height - initial_height
        height_reward = np.clip(height_diff / self.success_height, 0, 2) * 10
        reward += height_reward
        
        # Distance to block reward - encourage approach
        ee_pos = obs['ee_pose'][:3]
        block_pos = obs['target_block_pose'][:3]
        distance = np.linalg.norm(ee_pos - block_pos)
        distance_reward = np.exp(-distance * 5) * 2  # Exponential decay
        reward += distance_reward
        
        # Tactile contact reward - encourage grasping
        if self.use_tactile:
            tactile_sum = np.sum(obs['tactile'])
            contact_reward = np.clip(tactile_sum / 5.0, 0, 1) * 3
            reward += contact_reward
        
        # Gripper reward - close when near block
        if distance < 0.1:  # Near block
            gripper_closed = 1 - obs['gripper_pos'][0]  # 1 when closed
            reward += gripper_closed * 2
        
        # Action smoothness penalty - encourage smooth control
        action_penalty = -np.sum(np.abs(action)) * 0.01
        reward += action_penalty
        
        # Joint velocity penalty - discourage jerky motion
        vel_penalty = -np.sum(np.abs(obs['joint_vel'])) * 0.005
        reward += vel_penalty
        
        return float(reward)
    
    def _check_success(self, obs):
        """Check if grasping task succeeded."""
        block_height = obs['target_block_pose'][2]
        initial_height = self.initial_block_pose[2]
        lifted = block_height - initial_height
        
        return lifted > self.success_height
    
    def render(self):
        """Render current state."""
        if not hasattr(self, 'renderer'):
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        self.renderer.update_scene(self.data)
        return self.renderer.render()


# Test the environment
if __name__ == "__main__":
    print("🧪 Testing Panda 7-DOF RL Environment (Fixed)...")
    
    try:
        env = Panda7DOFTactileRL()
        
        # Test reset
        obs = env.reset()
        print(f"✅ Reset successful")
        print(f"   Observation keys: {list(obs.keys())}")
        print(f"   Joint positions: {obs['joint_pos']}")
        print(f"   Target block position: {obs['target_block_pose'][:3]}")
        if 'tactile' in obs:
            print(f"   Tactile shape: {obs['tactile'].shape}")
        
        # Test a few steps
        print(f"\n🎮 Testing environment steps...")
        
        for step in range(5):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   Step {step+1}: reward={reward:.3f}, success={info['success']}, block_height={info['block_height']:.3f}")
            
            if terminated or truncated:
                print(f"   Episode ended after {step+1} steps")
                break
        
        print(f"\n✅ Environment test successful!")
        print(f"   This is a proper 7-DOF RL environment with blocks")
        print(f"   Ready for velocity-based control and RL training")
        
    except Exception as e:
        print(f"❌ Error testing environment: {e}")
        import traceback
        traceback.print_exc()