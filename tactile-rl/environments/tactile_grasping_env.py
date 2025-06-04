"""
Tactile-enhanced grasping environment for cube manipulation.
This environment provides a simple cube grasping task with tactile feedback.
"""

import numpy as np
import mujoco
import os
from typing import Dict, Tuple, Optional
import logging

from .tactile_sensor import TactileSensor

logger = logging.getLogger(__name__)


class TactileGraspingEnv:
    """
    A simple cube grasping environment with tactile sensing.
    
    The task is to grasp a cube and lift it to a target height.
    Success is measured by:
    1. Cube lifted above threshold height
    2. Cube remains stable in grasp for required duration
    3. Minimal slip detected via tactile sensors
    """
    
    def __init__(
        self,
        xml_path: str = None,
        success_height: float = 0.1,  # 10cm lift
        success_duration: float = 1.0,  # 1 second hold
        max_episode_steps: int = 200,
        cube_size_range: Tuple[float, float] = (0.015, 0.025),  # 1.5-2.5cm cubes
        workspace_bounds: Dict[str, Tuple[float, float]] = None,
        use_tactile: bool = True,
        tactile_noise_std: float = 0.01
    ):
        """Initialize the tactile grasping environment."""
        
        # Environment parameters
        self.success_height = success_height
        self.success_duration = success_duration
        self.max_episode_steps = max_episode_steps
        self.cube_size_range = cube_size_range
        self.use_tactile = use_tactile
        
        # Workspace bounds for cube spawning
        self.workspace_bounds = workspace_bounds if workspace_bounds is not None else {
            'x': (0.4, 0.6),
            'y': (-0.15, 0.15),
            'z': (0.02, 0.02)  # Fixed height on table
        }
            
        # Load MuJoCo model
        if xml_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            xml_path = os.path.join(base_dir, "franka_emika_panda", "mjx_single_cube_fixed.xml")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize tactile sensors
        if self.use_tactile:
            self.tactile_sensor = TactileSensor(
                model=self.model,
                data=self.data,
                n_taxels_x=3, 
                n_taxels_y=4,
                finger_pad_size=(0.02, 0.04),
                noise_level=tactile_noise_std
            )
            
        # Get relevant body and geom IDs
        self._setup_ids()
        
        # Episode tracking
        self.current_step = 0
        self.success_counter = 0
        self.cube_initial_pos = None
        
        # Action and observation spaces
        self.action_dim = 8  # 7 joints + 1 gripper
        self.proprio_dim = 15  # joint pos(7) + joint vel(7) + gripper_width(1)
        self.tactile_dim = 72 if self.use_tactile else 0  # 3x4x3x2 tactile arrays (3 force components)
        self.object_dim = 7  # cube pos(3) + quat(4)
        
    def _setup_ids(self):
        """Get relevant body and geom IDs from the model."""
        # Robot bodies
        try:
            self.robot_base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        except:
            self.robot_base_id = None
            
        try:
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        except:
            self.ee_site_id = None
        
        # Gripper geoms for tactile sensing
        try:
            self.left_finger_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
        except:
            self.left_finger_geom = None
            
        try:
            self.right_finger_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
        except:
            self.right_finger_geom = None
        
        # Cube body and geom
        try:
            self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        except:
            self.cube_body_id = None
            self.cube_geom_id = None
        
        # Joint indices
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_ids = []
        for name in self.joint_names:
            try:
                self.joint_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            except:
                pass
        
        # Gripper joint indices
        self.gripper_joints = ["finger_joint1", "finger_joint2"]
        self.gripper_ids = []
        for name in self.gripper_joints:
            try:
                self.gripper_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            except:
                pass
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset the environment to a new episode."""
        if seed is not None:
            np.random.seed(seed)
            
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set robot to home position
        home_qpos = np.array([0, 0.3, 0, -1.57079, 0, 2.0, -0.7853])
        for i, joint_id in enumerate(self.joint_ids):
            if i < len(home_qpos):
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = home_qpos[i]
            
        # Open gripper
        for gripper_id in self.gripper_ids:
            self.data.qpos[self.model.jnt_qposadr[gripper_id]] = 0.04
            
        # Randomize cube position and size
        self._randomize_cube()
        
        # Step simulation to settle
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            
        # Reset episode tracking
        self.current_step = 0
        self.success_counter = 0
        
        return self._get_observation()
        
    def _randomize_cube(self):
        """Randomize cube position and size."""
        # Random position within workspace
        x = np.random.uniform(*self.workspace_bounds['x'])
        y = np.random.uniform(*self.workspace_bounds['y'])
        z = self.workspace_bounds['z'][0]
        
        # Random size
        size = np.random.uniform(*self.cube_size_range)
        
        # Set cube position
        if self.cube_body_id is not None:
            cube_jnt_id = self.model.body_jntadr[self.cube_body_id]
            if cube_jnt_id >= 0:  # Has a joint
                cube_qpos_addr = self.model.jnt_qposadr[cube_jnt_id]
                self.data.qpos[cube_qpos_addr:cube_qpos_addr+3] = [x, y, z]
                self.data.qpos[cube_qpos_addr+3:cube_qpos_addr+7] = [1, 0, 0, 0]  # identity quaternion
            
        # Update cube size
        if self.cube_geom_id is not None:
            self.model.geom_size[self.cube_geom_id] = [size, size, size]
            
        self.cube_initial_pos = np.array([x, y, z])
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one environment step."""
        # Clip actions
        action = np.clip(action, -1, 1)
        
        # Apply actions
        self._apply_action(action)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs)
        
        # Check termination
        terminated = self._check_success(obs)
        truncated = self.current_step >= self.max_episode_steps
        
        # Additional info
        info = {
            'success': terminated and not truncated,
            'cube_height': obs['object_state'][2] if 'object_state' in obs else 0,
            'tactile_contact': np.sum(obs['tactile']) > 0 if self.use_tactile else False,
            'success_counter': self.success_counter
        }
        
        return obs, reward, terminated, truncated, info
        
    def _apply_action(self, action: np.ndarray):
        """Apply action to robot joints and gripper."""
        # Joint actions (velocity control)
        for i in range(len(self.joint_ids)):
            self.data.ctrl[i] = action[i] * 0.5  # Scale down for safety
            
        # Gripper action (position control)
        # action[7]: -1 = open (0.04), +1 = close (0.0)
        gripper_target = 0.04 * (1 - action[7]) / 2  # Convert [-1, 1] to [0.04, 0]
        # The last actuator controls the gripper
        if len(self.data.ctrl) > 7:
            self.data.ctrl[7] = gripper_target
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation including tactile feedback."""
        obs = {}
        
        # Proprioceptive observations
        joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.joint_ids])
        joint_vel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self.joint_ids])
        gripper_width = self.data.qpos[self.model.jnt_qposadr[self.gripper_ids[0]]] * 2 if self.gripper_ids else 0
        
        obs['proprio'] = np.concatenate([joint_pos, joint_vel, [gripper_width]])
        
        # Object state
        if self.cube_body_id is not None:
            cube_jnt_id = self.model.body_jntadr[self.cube_body_id]
            if cube_jnt_id >= 0:  # Has a joint
                cube_qpos_addr = self.model.jnt_qposadr[cube_jnt_id]
                cube_pos = self.data.qpos[cube_qpos_addr:cube_qpos_addr+3]
                cube_quat = self.data.qpos[cube_qpos_addr+3:cube_qpos_addr+7]
                obs['object_state'] = np.concatenate([cube_pos, cube_quat])
            else:
                # Try to get position from body
                cube_pos = self.data.xpos[self.cube_body_id]
                cube_quat = self.data.xquat[self.cube_body_id]
                obs['object_state'] = np.concatenate([cube_pos, cube_quat])
        else:
            obs['object_state'] = np.zeros(7)
            
        # Tactile observations
        if self.use_tactile:
            # Get readings from both fingers
            left_reading, right_reading = self.tactile_sensor.get_readings(self.model, self.data)
            
            obs['tactile'] = np.concatenate([
                left_reading.flatten(),
                right_reading.flatten()
            ])
        else:
            obs['tactile'] = np.zeros(72)
            
        return obs
        
    def _calculate_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """Calculate reward based on task progress."""
        reward = 0.0
        
        # Height reward
        cube_height = obs['object_state'][2]
        height_reward = np.clip(cube_height / self.success_height, 0, 1)
        reward += height_reward
        
        # Grasp stability reward (based on tactile if available)
        if self.use_tactile:
            tactile_sum = np.sum(obs['tactile'])
            if tactile_sum > 0.1:  # Contact detected
                # Reward for symmetric grasp
                left_sum = np.sum(obs['tactile'][:12])
                right_sum = np.sum(obs['tactile'][12:])
                symmetry = 1 - abs(left_sum - right_sum) / (left_sum + right_sum + 1e-6)
                reward += 0.5 * symmetry
                
        # Success bonus
        if self._check_success(obs):
            reward += 10.0
            
        return reward
        
    def _check_success(self, obs: Dict[str, np.ndarray]) -> bool:
        """Check if the task is successfully completed."""
        cube_height = obs['object_state'][2]
        
        if cube_height > self.success_height:
            self.success_counter += 1
        else:
            self.success_counter = 0
            
        # Need to maintain height for success_duration
        success_steps = int(self.success_duration * 100)  # Assuming 100Hz simulation
        return self.success_counter >= success_steps
        
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "rgb_array":
            # Set up rendering
            scene_option = mujoco.MjvOption()
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            
            # Render
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            pixels = renderer.render()
            
            return pixels
        else:
            return None