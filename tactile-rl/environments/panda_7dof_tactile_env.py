"""
7-DOF Panda Tactile Grasping Environment
Designed for RL with sim-to-real transfer in mind.
"""

import numpy as np
import gym
from gym import spaces
import mujoco
import os
from typing import Dict, Tuple, Optional
from scipy.spatial.transform import Rotation

from .tactile_sensor import TactileSensor


class Panda7DOFTactileEnv(gym.Env):
    """
    7-DOF Panda robot with tactile sensing for RL training.
    
    Action Space: 8-dimensional continuous
        - 7 joint velocity commands [-1, 1] 
        - 1 gripper command [-1=open, 1=close]
    
    Observation Space: Dictionary with:
        - joint_pos: 7 joint positions
        - joint_vel: 7 joint velocities  
        - gripper_pos: gripper opening [0=closed, 1=open]
        - tactile: 72-dim tactile readings (3x4x3x2 fingers)
        - cube_pose: 7-dim cube pose (pos + quat)
        - ee_pose: 7-dim end-effector pose (pos + quat)
    """
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 500,
        success_height: float = 0.15,
        control_frequency: int = 20,  # Hz
        **kwargs
    ):
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.success_height = success_height
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency  # 0.05 seconds per step
        
        # Load 7-DOF Panda model
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = os.path.join(base_dir, "franka_emika_panda", "panda.xml")
        
        # Add a scene with objects
        self._create_scene_xml()
        scene_path = os.path.join(base_dir, "franka_emika_panda", "panda_7dof_scene.xml")
        
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize tactile sensor
        self.tactile_sensor = TactileSensor(
            n_taxels_x=3, n_taxels_y=4,
            left_finger_name="left_finger",
            right_finger_name="right_finger"
        )
        
        # Get important IDs
        self._setup_ids()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.initial_cube_pose = None
        
        print("🤖 Panda7DOFTactileEnv initialized!")
        print(f"   Control frequency: {control_frequency} Hz")
        print(f"   Action space: {self.action_space.shape}")
        print(f"   Observation space keys: {list(self.observation_space.spaces.keys())}")
    
    def _create_scene_xml(self):
        """Create a complete scene XML with 7-DOF Panda and objects."""
        scene_xml = '''
<mujoco model="panda_7dof_tactile_scene">
  <include file="panda.xml"/>
  
  <worldbody>
    <!-- Table -->
    <body name="table" pos="0.5 0 0.4">
      <geom type="box" size="0.4 0.4 0.02" rgba="0.8 0.7 0.6 1"/>
    </body>
    
    <!-- Cube to grasp -->
    <body name="cube" pos="0.5 0 0.44">
      <freejoint/>
      <geom name="cube" type="box" size="0.025 0.025 0.025" 
            rgba="0 1 0 1" contype="1" conaffinity="1" friction="1 0.5 0.5"/>
    </body>
  </worldbody>
</mujoco>'''
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scene_path = os.path.join(base_dir, "franka_emika_panda", "panda_7dof_scene.xml")
        
        with open(scene_path, 'w') as f:
            f.write(scene_xml)
    
    def _setup_ids(self):
        """Get MuJoCo IDs for important bodies/joints."""
        # Joint IDs (7-DOF arm + 2 gripper)
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4', 
            'joint5', 'joint6', 'joint7'
        ]
        
        self.joint_ids = []
        for name in self.joint_names:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.joint_ids.append(jid)
            except:
                print(f"Warning: Joint {name} not found")
        
        # Gripper joints
        try:
            self.gripper_joint1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1')
            self.gripper_joint2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint2')
        except:
            print("Warning: Gripper joints not found")
        
        # Body IDs
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        # End-effector site
        try:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        except:
            print("Warning: End-effector site not found")
    
    def _setup_spaces(self):
        """Define action and observation spaces."""
        
        # Action space: 7 joint velocities + 1 gripper command
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Observation space
        obs_spaces = {
            'joint_pos': spaces.Box(low=-3.14, high=3.14, shape=(7,), dtype=np.float32),
            'joint_vel': spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32), 
            'gripper_pos': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'tactile': spaces.Box(low=0.0, high=10.0, shape=(72,), dtype=np.float32),
            'cube_pose': spaces.Box(low=-2.0, high=2.0, shape=(7,), dtype=np.float32),
            'ee_pose': spaces.Box(low=-2.0, high=2.0, shape=(7,), dtype=np.float32)
        }
        
        self.observation_space = spaces.Dict(obs_spaces)
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        
        # Set robot to home position
        home_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        for i, qpos in enumerate(home_qpos):
            if i < len(self.joint_ids):
                joint_addr = self.model.jnt_qposadr[self.joint_ids[i]]
                self.data.qpos[joint_addr] = qpos
        
        # Open gripper
        if hasattr(self, 'gripper_joint1'):
            gripper_addr1 = self.model.jnt_qposadr[self.gripper_joint1]
            gripper_addr2 = self.model.jnt_qposadr[self.gripper_joint2]
            self.data.qpos[gripper_addr1] = 0.04
            self.data.qpos[gripper_addr2] = 0.04
        
        # Randomize cube position slightly
        cube_x = 0.5 + np.random.uniform(-0.05, 0.05)
        cube_y = 0.0 + np.random.uniform(-0.05, 0.05)
        cube_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.cube_body_id]]
        self.data.qpos[cube_addr:cube_addr+3] = [cube_x, cube_y, 0.44]
        
        # Forward dynamics to settle
        mujoco.mj_forward(self.model, self.data)
        
        # Store initial state
        self.initial_cube_pose = self.data.xpos[self.cube_body_id].copy()
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one environment step."""
        
        # Apply velocity control to joints
        joint_vels = action[:7] * 2.0  # Scale to reasonable velocities
        
        # Apply joint velocities (using position control with small deltas)
        for i, vel in enumerate(joint_vels):
            if i < len(self.joint_ids):
                joint_addr = self.model.jnt_qposadr[self.joint_ids[i]]
                current_pos = self.data.qpos[joint_addr]
                target_pos = current_pos + vel * self.dt
                
                # Apply joint limits
                joint_range = self.model.jnt_range[self.joint_ids[i]]
                if joint_range[0] < joint_range[1]:  # Has limits
                    target_pos = np.clip(target_pos, joint_range[0], joint_range[1])
                
                # Set control target
                if i < self.model.nu:
                    self.data.ctrl[i] = target_pos
        
        # Apply gripper control
        gripper_cmd = action[7]  # -1=open, +1=close
        gripper_pos = 0.02 * (1 - gripper_cmd)  # Convert to position
        
        if hasattr(self, 'gripper_joint1'):
            # Set gripper control (assuming last actuators are gripper)
            if self.model.nu > 7:
                self.data.ctrl[7] = gripper_pos * 255  # Scale for tendon actuator
        
        # Step physics multiple times for control frequency
        steps_per_control = int(self.model.opt.timestep * self.control_frequency)
        steps_per_control = max(1, steps_per_control)
        
        for _ in range(steps_per_control):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._compute_reward(obs)
        
        # Check termination
        terminated = self._check_success(obs)
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            'success': terminated,
            'cube_height': obs['cube_pose'][2],
            'ee_pos': obs['ee_pose'][:3]
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation."""
        obs = {}
        
        # Joint positions and velocities
        joint_pos = np.zeros(7)
        joint_vel = np.zeros(7)
        
        for i, joint_id in enumerate(self.joint_ids):
            if i < 7:
                joint_addr = self.model.jnt_qposadr[joint_id]
                vel_addr = self.model.jnt_dofadr[joint_id]
                joint_pos[i] = self.data.qpos[joint_addr]
                joint_vel[i] = self.data.qvel[vel_addr]
        
        obs['joint_pos'] = joint_pos.astype(np.float32)
        obs['joint_vel'] = joint_vel.astype(np.float32)
        
        # Gripper position
        if hasattr(self, 'gripper_joint1'):
            gripper_addr = self.model.jnt_qposadr[self.gripper_joint1]
            gripper_pos = self.data.qpos[gripper_addr] / 0.04  # Normalize to [0,1]
            obs['gripper_pos'] = np.array([gripper_pos], dtype=np.float32)
        else:
            obs['gripper_pos'] = np.array([0.0], dtype=np.float32)
        
        # Tactile readings
        left_reading, right_reading = self.tactile_sensor.get_readings(self.model, self.data)
        tactile = np.concatenate([left_reading.flatten(), right_reading.flatten()])
        obs['tactile'] = tactile.astype(np.float32)
        
        # Cube pose
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        cube_quat = self.data.xquat[self.cube_body_id].copy()
        obs['cube_pose'] = np.concatenate([cube_pos, cube_quat]).astype(np.float32)
        
        # End-effector pose
        if hasattr(self, 'ee_site_id'):
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            ee_quat = np.array([1, 0, 0, 0])  # Default orientation
        else:
            # Approximate end-effector from last link
            ee_pos = np.array([0.5, 0, 0.8])  # Default position
            ee_quat = np.array([1, 0, 0, 0])
        
        obs['ee_pose'] = np.concatenate([ee_pos, ee_quat]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, obs):
        """Compute reward for current state."""
        reward = 0.0
        
        # Height reward - encourage lifting cube
        cube_height = obs['cube_pose'][2]
        initial_height = self.initial_cube_pose[2]
        height_reward = np.clip((cube_height - initial_height) / self.success_height, 0, 1)
        reward += height_reward * 10.0
        
        # Approach reward - encourage moving toward cube
        ee_pos = obs['ee_pose'][:3]
        cube_pos = obs['cube_pose'][:3]
        distance = np.linalg.norm(ee_pos - cube_pos)
        approach_reward = np.exp(-distance * 5.0)
        reward += approach_reward
        
        # Tactile contact reward - encourage making contact
        tactile_sum = np.sum(obs['tactile'])
        contact_reward = np.clip(tactile_sum / 10.0, 0, 1)
        reward += contact_reward * 2.0
        
        # Stability penalty - discourage large joint velocities
        vel_penalty = -np.sum(np.abs(obs['joint_vel'])) * 0.01
        reward += vel_penalty
        
        return float(reward)
    
    def _check_success(self, obs):
        """Check if episode is successful."""
        cube_height = obs['cube_pose'][2]
        initial_height = self.initial_cube_pose[2]
        lifted = cube_height - initial_height
        
        return lifted > self.success_height
    
    def render(self, mode='human'):
        """Render the environment."""
        if not hasattr(self, 'renderer'):
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        self.renderer.update_scene(self.data)
        return self.renderer.render()


# Test the environment
if __name__ == "__main__":
    env = Panda7DOFTactileEnv()
    
    print("Testing environment...")
    obs, info = env.reset()
    print(f"Initial observation keys: {obs.keys()}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, success={info['success']}")
        
        if terminated or truncated:
            break
    
    print("Environment test complete!")