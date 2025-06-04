"""
Simplified tactile grasping environment with working physics.
Uses a 3-DOF arm with position control for easier manipulation.
"""

import numpy as np
import mujoco
import os
from typing import Dict, Tuple, Optional
import logging

from .tactile_sensor import TactileSensor

logger = logging.getLogger(__name__)


class SimpleTactileGraspingEnv:
    """
    A simplified grasping environment with tactile sensing.
    Uses position control for more reliable grasping.
    """
    
    def __init__(
        self,
        render_mode: str = "rgb_array",
        max_episode_steps: int = 200,
        use_tactile: bool = True,
        success_height: float = 0.1,
        cube_size_range: Tuple[float, float] = (0.02, 0.03)
    ):
        """Initialize the environment."""
        
        self.max_episode_steps = max_episode_steps
        self.use_tactile = use_tactile
        self.success_height = success_height
        self.cube_size_range = cube_size_range
        
        # Load model
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml_path = os.path.join(base_dir, "franka_emika_panda", "panda_tactile_grasp.xml")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get IDs
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        
        # Finger geoms for tactile
        self.left_finger_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad")
        self.right_finger_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_pad")
        
        # Joint IDs
        self.arm_lift_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_lift")
        self.arm_extend_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_extend")
        self.gripper_rotate_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_rotate")
        self.left_finger_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_joint")
        self.right_finger_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger_joint")
        
        # Initialize tactile sensors if enabled
        if self.use_tactile:
            self.tactile_sensor = TactileSensor(
                model=self.model,
                data=self.data,
                n_taxels_x=3,
                n_taxels_y=4,
                finger_pad_size=(0.016, 0.06),
                noise_level=0.01
            )
            
        # Spaces
        self.action_dim = 4  # lift, extend, rotate, gripper
        self.observation_dim = {
            'joint_pos': 5,
            'ee_pos': 3,
            'cube_pos': 7,
            'tactile': 72 if use_tactile else 0
        }
        
        # Episode tracking
        self.current_step = 0
        self.initial_cube_height = None
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
            
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial arm position (slightly raised)
        self.data.ctrl[0] = 0.1  # Lift
        self.data.ctrl[1] = 0.0  # Extend
        self.data.ctrl[2] = 0.0  # Rotate
        self.data.ctrl[3] = 0.04  # Open gripper
        self.data.ctrl[4] = 0.04
        
        # Let it settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
            
        # Randomize cube size
        cube_size = np.random.uniform(*self.cube_size_range)
        self.model.geom_size[self.cube_geom_id] = [cube_size, cube_size, cube_size]
        
        # Record initial cube height
        self.initial_cube_height = self.data.xpos[self.cube_body_id][2]
        
        # Reset tracking
        self.current_step = 0
        
        return self._get_observation()
        
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one environment step."""
        
        # Apply action (position control)
        # action[0]: arm lift (-1 to 1) -> (-0.3 to 0.3)
        # action[1]: arm extend (-1 to 1) -> (-0.2 to 0.2)
        # action[2]: gripper rotate (-1 to 1) -> (-1.57 to 1.57)
        # action[3]: gripper open/close (-1 to 1) -> (0.04 to 0)
        
        self.data.ctrl[0] = np.clip(action[0] * 0.3, -0.3, 0.3)
        self.data.ctrl[1] = np.clip(action[1] * 0.2, -0.2, 0.2)
        self.data.ctrl[2] = np.clip(action[2] * 1.57, -1.57, 1.57)
        
        # Gripper: -1 = open (0.04), +1 = close (0.0)
        gripper_pos = 0.02 * (1 - action[3])
        self.data.ctrl[3] = gripper_pos
        self.data.ctrl[4] = gripper_pos
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs)
        
        # Check termination
        cube_height = self.data.xpos[self.cube_body_id][2]
        lifted = cube_height - self.initial_cube_height
        
        terminated = lifted > self.success_height
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            'success': terminated,
            'cube_lifted': lifted,
            'cube_height': cube_height
        }
        
        return obs, reward, terminated, truncated, info
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        obs = {}
        
        # Joint positions - fix the indexing
        joint_pos = np.zeros(5)
        joint_pos[0] = self.data.ctrl[0]  # arm lift position
        joint_pos[1] = self.data.ctrl[1]  # arm extend position  
        joint_pos[2] = self.data.ctrl[2]  # gripper rotate position
        joint_pos[3] = self.data.ctrl[3]  # left finger position
        joint_pos[4] = self.data.ctrl[4]  # right finger position
        obs['joint_pos'] = joint_pos
        
        # End effector position
        obs['ee_pos'] = self.data.site_xpos[self.ee_site_id].copy()
        
        # Cube pose
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        cube_quat = self.data.xquat[self.cube_body_id].copy()
        obs['cube_pose'] = np.concatenate([cube_pos, cube_quat])
        
        # Tactile readings
        if self.use_tactile:
            left_reading, right_reading = self.tactile_sensor.get_readings(self.model, self.data)
            obs['tactile'] = np.concatenate([left_reading.flatten(), right_reading.flatten()])
        else:
            obs['tactile'] = np.zeros(72)
            
        return obs
        
    def _calculate_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """Calculate reward."""
        reward = 0.0
        
        # Height reward
        cube_height = obs['cube_pose'][2]
        height_bonus = np.clip((cube_height - self.initial_cube_height) / self.success_height, 0, 1)
        reward += height_bonus
        
        # Distance to cube reward (encourage approaching)
        ee_pos = obs['ee_pos']
        cube_pos = obs['cube_pose'][:3]
        distance = np.linalg.norm(ee_pos - cube_pos)
        distance_penalty = -0.1 * np.clip(distance, 0, 1)
        reward += distance_penalty
        
        # Tactile reward if using tactile
        if self.use_tactile:
            tactile_sum = np.sum(obs['tactile'])
            if tactile_sum > 0.5:  # Contact detected
                # Reward symmetric grasp
                left_sum = np.sum(obs['tactile'][:36])
                right_sum = np.sum(obs['tactile'][36:])
                symmetry = 1 - abs(left_sum - right_sum) / (left_sum + right_sum + 1e-6)
                reward += 0.2 * symmetry
                
        return reward
        
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()
        return None