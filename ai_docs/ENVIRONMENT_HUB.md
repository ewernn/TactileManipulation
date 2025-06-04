# 🌍 Environment Hub

*Central documentation for MuJoCo environment implementations*

## 🚀 Quick Reference

### **Key Files**
- **Primary RL Environment**: `tactile-rl/environments/panda_7dof_rl_env.py` (110° config)
- **Demo Environment**: `tactile-rl/environments/panda_demo_env.py` (110° config)
- **Simple Environment**: `tactile-rl/environments/simple_tactile_env.py`
- **Tactile Sensor**: `tactile-rl/environments/tactile_sensor.py`

### **Common Operations**
- Create environment → `env = SimpleTactileEnv(render_mode='human')`
- Reset environment → `obs, info = env.reset()`
- Step simulation → `obs, reward, done, truncated, info = env.step(action)`
- Render frame → `frame = env.render()`

## 🏗️ Architecture

### **Environment Structure**
```
SimpleTactileEnv (Gymnasium)
├── MuJoCo Model & Data
├── TactileSensor Integration
├── Observation Space
│   ├── Robot State (joint pos/vel)
│   ├── Object State (pose)
│   └── Tactile Readings (24 taxels × 3 forces)
└── Action Space (joint controls)
```

### **Key Components**

#### **1. Environment Class**
```python
class SimpleTactileEnv(gym.Env):
    def __init__(self, 
                 xml_path="panda_tactile_grasp.xml",
                 render_mode=None,
                 tactile_enabled=True):
        # Load MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize tactile
        if tactile_enabled:
            self.tactile_sensor = TactileSensor()
        
        # Define spaces
        self._setup_spaces()
```

#### **2. Observation Space**
```python
def _create_obs_space(self):
    """Define observation space structure"""
    return gym.spaces.Dict({
        'robot_joint_pos': gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(self.n_joints,)
        ),
        'robot_joint_vel': gym.spaces.Box(
            low=-10, high=10, shape=(self.n_joints,)
        ),
        'gripper_pos': gym.spaces.Box(
            low=0, high=0.08, shape=(2,)
        ),
        'object_pos': gym.spaces.Box(
            low=-1, high=1, shape=(3,)
        ),
        'object_quat': gym.spaces.Box(
            low=-1, high=1, shape=(4,)
        ),
        'tactile_left': gym.spaces.Box(
            low=0, high=10, shape=(3, 4, 3)
        ),
        'tactile_right': gym.spaces.Box(
            low=0, high=10, shape=(3, 4, 3)
        )
    })
```

#### **3. Action Space**
```python
def _create_action_space(self):
    """Define action space"""
    # For 3-DOF + gripper
    return gym.spaces.Box(
        low=-1, high=1, shape=(4,), dtype=np.float32
    )
    
    # For 7-DOF + gripper
    # return gym.spaces.Box(
    #     low=-1, high=1, shape=(8,), dtype=np.float32
    # )
```

## 🔧 Implementation Details

### **Step Function**
```python
def step(self, action):
    # Apply actions
    self._apply_action(action)
    
    # Step physics
    mujoco.mj_step(self.model, self.data)
    
    # Get observations
    obs = self._get_obs()
    
    # Calculate reward
    reward = self._compute_reward()
    
    # Check termination
    done = self._check_termination()
    truncated = self.step_count >= self.max_steps
    
    # Collect info
    info = self._get_info()
    
    self.step_count += 1
    
    return obs, reward, done, truncated, info
```

### **Observation Collection**
```python
def _get_obs(self):
    """Collect all observations"""
    # Robot state
    robot_qpos = self.data.qpos[:self.n_joints].copy()
    robot_qvel = self.data.qvel[:self.n_joints].copy()
    
    # Gripper state
    gripper_qpos = self.data.qpos[self.gripper_joint_ids].copy()
    
    # Object state
    obj_pos = self.data.body('object').xpos.copy()
    obj_quat = self.data.body('object').xquat.copy()
    
    # Tactile readings
    if self.tactile_enabled:
        left_tactile, right_tactile = self.tactile_sensor.get_readings(
            self.model, self.data
        )
    else:
        left_tactile = np.zeros((3, 4, 3))
        right_tactile = np.zeros((3, 4, 3))
    
    return {
        'robot_joint_pos': robot_qpos,
        'robot_joint_vel': robot_qvel,
        'gripper_pos': gripper_qpos,
        'object_pos': obj_pos,
        'object_quat': obj_quat,
        'tactile_left': left_tactile,
        'tactile_right': right_tactile
    }
```

### **Action Processing**
```python
def _apply_action(self, action):
    """Apply normalized actions to robot"""
    # Scale actions to control range
    if self.control_mode == 'position':
        # Map [-1, 1] to joint ranges
        for i in range(self.n_arm_joints):
            joint_range = self.joint_ranges[i]
            scaled_action = (action[i] + 1) / 2  # [0, 1]
            target = joint_range[0] + scaled_action * (joint_range[1] - joint_range[0])
            self.data.ctrl[i] = target
        
        # Gripper: -1 = open, +1 = closed
        gripper_target = 0.04 * (1 - action[-1]) / 2
        self.data.ctrl[self.n_arm_joints] = gripper_target
```

### **Reward Design**
```python
def _compute_reward(self):
    """Task-specific reward function"""
    reward = 0.0
    
    # Object lifted reward
    obj_height = self.data.body('object').xpos[2]
    if obj_height > 0.1:  # 10cm threshold
        reward += 10.0
    
    # Grasp quality from tactile
    if self.tactile_enabled:
        left_force = np.sum(self.last_tactile_left[:, :, 0])
        right_force = np.sum(self.last_tactile_right[:, :, 0])
        
        # Balanced grasp bonus
        balance = 1 - abs(left_force - right_force) / (left_force + right_force + 1e-6)
        reward += balance * 2.0
    
    # Distance penalty (optional)
    ee_pos = self.data.site('ee_site').xpos
    obj_pos = self.data.body('object').xpos
    distance = np.linalg.norm(ee_pos - obj_pos)
    reward -= distance * 0.1
    
    return reward
```

## 🎯 Environment Variants

### **1. Simple Tactile Environment**
- **File**: `simple_tactile_env.py`
- **Robot**: 3-DOF arm + gripper
- **Object**: Single cube
- **Focus**: Reliable grasping demos

### **2. Original Complex Environment**
- **File**: `tactile_grasping_env.py`
- **Robot**: 7-DOF Panda
- **Objects**: Multiple options
- **Status**: Has integration issues

### **3. Custom Variants**
```python
# Create with different objects
env = SimpleTactileEnv(
    xml_path="custom_scene.xml",
    tactile_enabled=True,
    max_steps=500
)

# Disable tactile for baseline
env_baseline = SimpleTactileEnv(
    tactile_enabled=False
)
```

## 🐛 Troubleshooting

### **Reset Issues**
```python
def reset(self, seed=None):
    super().reset(seed=seed)
    
    # Reset simulation
    mujoco.mj_resetData(self.model, self.data)
    
    # Randomize object position
    if self.randomize_object:
        self.data.qpos[self.obj_joint_id:self.obj_joint_id+3] = [
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            0.05  # Starting height
        ]
    
    # Forward kinematics
    mujoco.mj_forward(self.model, self.data)
    
    return self._get_obs(), {}
```

### **Rendering Problems**
```python
# Check renderer initialization
if render_mode == 'human':
    self.renderer = mujoco.Renderer(self.model)
    
# In render()
if self.render_mode == 'human':
    self.renderer.update_scene(self.data)
    return self.renderer.render()
```

### **Physics Instability**
- Reduce timestep in XML: `<option timestep="0.001"/>`
- Add damping to joints
- Check collision parameters
- Verify mass/inertia values

## 📊 Performance Metrics

### **Environment Speed**
- Simulation: ~2000 Hz (0.5ms per step)
- With rendering: ~60 Hz
- With tactile: ~1800 Hz

### **Memory Usage**
- Base environment: ~50 MB
- With rendering: ~200 MB
- Per episode data: ~5 MB

## 🔗 Related Documentation
- Robot models → `/ai_docs/ROBOT_HUB.md`
- Tactile sensing → `/ai_docs/TACTILE_HUB.md`
- Data collection → `/ai_docs/DATA_HUB.md`
- XML configurations → `franka_emika_panda/` directory