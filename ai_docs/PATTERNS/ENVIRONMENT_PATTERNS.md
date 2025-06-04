# 🌍 Environment Design Patterns

## 🎯 Standard Environment Structure

```python
import gymnasium as gym
import mujoco
import numpy as np
from .tactile_sensor import TactileSensor

class CustomTactileEnv(gym.Env):
    def __init__(self, xml_path: str, render_mode=None):
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize tactile sensor
        self.tactile_sensor = TactileSensor(n_taxels_x=3, n_taxels_y=4)
        
        # Define spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = self._create_obs_space()
        
    def step(self, action):
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        
        return obs, reward, done, False, {}
```

## 🔧 Core Methods

### **Action Application**
```python
def _apply_action(self, action):
    # Map [-1,1] to joint ranges
    for i, joint_name in enumerate(self.arm_joints):
        qpos_id = self.model.joint(joint_name).qposadr[0]
        joint_range = self.model.jnt_range[self.model.joint(joint_name).id]
        target = np.interp(action[i], [-1, 1], joint_range)
        self.data.qpos[qpos_id] = target
    
    # Gripper control (binary)
    gripper_target = 1.0 if action[-1] > 0 else 0.0
    self.data.qpos[self.gripper_qpos_id] = gripper_target
```

### **Observation Collection**
```python
def _get_obs(self):
    # Robot state
    robot_qpos = self.data.qpos[:len(self.arm_joints)]
    robot_qvel = self.data.qvel[:len(self.arm_joints)]
    
    # Object state
    object_pos = self.data.qpos[self.object_qpos_id:self.object_qpos_id+3]
    object_vel = self.data.qvel[self.object_qvel_id:self.object_qvel_id+3]
    
    # Tactile readings
    left_tactile, right_tactile = self.tactile_sensor.get_readings(self.model, self.data)
    
    return {
        'robot': np.concatenate([robot_qpos, robot_qvel]),
        'object': np.concatenate([object_pos, object_vel]),
        'tactile': np.concatenate([left_tactile.flatten(), right_tactile.flatten()])
    }
```

### **Reward Design**
```python
def _compute_reward(self):
    # Object lift height
    object_height = self.data.qpos[self.object_qpos_id + 2]
    lift_reward = max(0, object_height - self.initial_height)
    
    # Grasp stability (tactile feedback)
    left_tactile, right_tactile = self.tactile_sensor.get_readings(self.model, self.data)
    grasp_force = np.sum(left_tactile[:,:,0]) + np.sum(right_tactile[:,:,0])
    grasp_reward = min(1.0, grasp_force / 10.0)  # Normalize
    
    # Combine rewards
    return lift_reward * 10.0 + grasp_reward * 5.0
```

## 🎯 Key Patterns

### **Object Spawning**
```python
def reset(self, seed=None, options=None):
    # Reset robot to home position
    self.data.qpos[:] = self.init_qpos
    
    # Random object placement
    object_pos = self.np_random.uniform(
        [0.3, -0.2, 0.8],  # workspace bounds
        [0.7, 0.2, 0.8]
    )
    self.data.qpos[self.object_qpos_id:self.object_qpos_id+3] = object_pos
    
    mujoco.mj_forward(self.model, self.data)
    return self._get_obs(), {}
```

### **Collision Detection**
```python
def _check_collisions(self):
    # Check for unwanted collisions
    for i in range(self.data.ncon):
        contact = self.data.contact[i]
        geom1_name = self.model.geom(contact.geom1).name
        geom2_name = self.model.geom(contact.geom2).name
        
        # Table collision = failure
        if 'table' in geom1_name and 'object' in geom2_name:
            return True
    return False
```

### **Success Criteria**
```python
def _is_success(self):
    # Object lifted above threshold
    object_height = self.data.qpos[self.object_qpos_id + 2]
    height_success = object_height > (self.initial_height + 0.1)
    
    # Stable grasp (tactile contact)
    left_tactile, right_tactile = self.tactile_sensor.get_readings(self.model, self.data)
    has_contact = (np.sum(left_tactile) > 0.1) and (np.sum(right_tactile) > 0.1)
    
    return height_success and has_contact
```

## 🚨 Common Pitfalls

### **Space Definition**
```python
# ❌ Wrong: Hard-coded observation size
obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(100,))

# ✅ Correct: Calculate from components
def _create_obs_space(self):
    robot_dim = len(self.arm_joints) * 2  # pos + vel
    object_dim = 6  # pos + vel (3D)
    tactile_dim = 3 * 4 * 3 * 2  # 3x4x3 per finger, 2 fingers
    
    total_dim = robot_dim + object_dim + tactile_dim
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,))
```

### **Contact Processing**
```python
# ❌ Wrong: Missing contact check
tactile_readings = self.tactile_sensor.get_readings(self.model, self.data)

# ✅ Correct: Handle no contacts
if self.data.ncon == 0:
    tactile_readings = np.zeros((2, 3, 4, 3))  # Empty readings
else:
    tactile_readings = self.tactile_sensor.get_readings(self.model, self.data)
```

## 🔗 Example Environments
- Simple: `simple_tactile_env.py` - 3-DOF, single cube
- Advanced: `tactile_grasping_env.py` - 7-DOF, complex objects
- Test: `test_simple_robot.py` - Debug/validation script