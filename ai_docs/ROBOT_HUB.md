# 🤖 Robot Models Hub

## 🚀 Quick Reference

### **Key Files**
- 7-DOF Panda: `tactile-rl/franka_emika_panda/panda.xml`
- 3-DOF Simple: `tactile-rl/franka_emika_panda/panda_tactile_grasp.xml` 
- Test scripts: `panda_7dof_grasp.py`, `test_simple_robot.py`

### **Control Functions**
- `data.qpos[joint_id] = value` → Set joint positions
- `data.ctrl[actuator_id] = value` → Apply control
- `data.qpos[-1]` → Gripper position (0=open, 1=closed)

## 🏗️ Available Configurations

### **1. 7-DOF Franka Panda (Full)**
```python
# Control: 7 arm joints + 1 gripper
# panda.xml, mjx_panda.xml
action_space = spaces.Box(low=-1, high=1, shape=(8,))
# Actions: [joint1, joint2, ..., joint7, gripper]
```

### **2. 3-DOF Simplified (Reliable)**
```python
# Control: 3 arm joints + 1 gripper  
# panda_tactile_grasp.xml
action_space = spaces.Box(low=-1, high=1, shape=(4,))
# Actions: [shoulder, elbow, wrist, gripper]
```

## 🔧 Control Mapping

### **Position Control**
```python
# Map normalized actions [-1, 1] to joint limits
def apply_action(self, action):
    # Arm joints
    for i, joint_name in enumerate(self.arm_joints):
        joint_id = self.model.joint(joint_name).id
        qpos_id = self.model.joint(joint_name).qposadr[0]
        
        # Scale to joint limits
        joint_range = self.model.jnt_range[joint_id]
        target_pos = np.interp(action[i], [-1, 1], joint_range)
        self.data.qpos[qpos_id] = target_pos
    
    # Gripper (binary: >0 = close, <0 = open)
    gripper_target = 1.0 if action[-1] > 0 else 0.0
    self.data.qpos[self.gripper_qpos_id] = gripper_target
```

### **Important Joint Limits**
```python
# 7-DOF Panda joint limits (radians)
joint_limits = {
    'joint1': [-2.8973, 2.8973],   # ±166°
    'joint2': [-1.7628, 1.7628],   # ±101° 
    'joint3': [-2.8973, 2.8973],   # ±166°
    'joint4': [-3.0718, -0.0698],  # 4° to 176°
    'joint5': [-2.8973, 2.8973],   # ±166°
    'joint6': [-0.0175, 3.7525],   # 1° to 215°
    'joint7': [-2.8973, 2.8973]    # ±166°
}
```

## 🎯 Working Setups

### **Current Stable**: 3-DOF
- XML: `panda_tactile_grasp.xml`
- Control: Position-based, 300Hz
- Success: 100% grasp rate
- Use for: New implementations, testing

### **Advanced**: 7-DOF  
- XML: `panda.xml`
- Control: Joint position/velocity
- Status: Working, requires tuning
- Use for: Complex manipulation, research

## 🚨 Common Issues

### **Wrong Control Dimensions**
```python
# Check model DOF
print(f"Model DOF: {env.model.nq}")  # Should match action space
print(f"Control dim: {env.model.nu}")
```

### **Gripper Not Working**
- Check `contype="1" conaffinity="1"` on finger geoms
- Verify gripper joint limits: `[0, 1]` for position control
- Test contact detection: `data.ncon > 0`

### **Joint Limits Exceeded**
- Always clip actions to `[-1, 1]` range
- Check joint limit scaling in `apply_action`
- Monitor `data.qpos` values vs `model.jnt_range`

## 🔗 Related Files
- Assets: `tactile-rl/franka_emika_panda/assets/`
- Environments: `simple_tactile_env.py`, `tactile_grasping_env.py`
- Tests: `test_simple_robot.py`, `debug_grasp_physics.py`