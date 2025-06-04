# 🎮 Robot Control Patterns

*Common patterns for robot control implementation and action processing*

## 🎯 Control Mode Patterns

### **Position Control**
```python
class PositionController:
    """Direct position control for joints"""
    def __init__(self, model, control_limits=None):
        self.model = model
        self.n_joints = model.nu
        
        # Get control ranges from model
        if control_limits is None:
            self.control_limits = model.actuator_ctrlrange.copy()
        else:
            self.control_limits = control_limits
    
    def apply_action(self, data, action):
        """
        Apply normalized action [-1, 1] as position targets
        """
        # Clip action to valid range
        action = np.clip(action, -1, 1)
        
        # Map to control range
        for i in range(self.n_joints):
            low, high = self.control_limits[i]
            # Linear mapping from [-1, 1] to [low, high]
            normalized = (action[i] + 1) / 2  # [0, 1]
            data.ctrl[i] = low + normalized * (high - low)
```

### **Velocity Control**
```python
class VelocityController:
    """Velocity-based control"""
    def __init__(self, model, max_velocities=None):
        self.model = model
        self.n_joints = model.nu
        
        if max_velocities is None:
            # Default max velocities (rad/s)
            self.max_velocities = np.ones(self.n_joints) * 2.0
        else:
            self.max_velocities = max_velocities
    
    def apply_action(self, data, action):
        """
        Apply normalized action as velocity commands
        """
        action = np.clip(action, -1, 1)
        
        # Scale by max velocities
        target_velocities = action * self.max_velocities
        
        # Simple P controller to track velocities
        current_vel = data.qvel[:self.n_joints]
        vel_error = target_velocities - current_vel
        
        # Apply control (assuming position actuators)
        data.ctrl[:self.n_joints] = data.qpos[:self.n_joints] + vel_error * 0.05
```

### **Impedance Control**
```python
class ImpedanceController:
    """Compliant control with virtual spring-damper"""
    def __init__(self, model, stiffness=100, damping=10):
        self.model = model
        self.stiffness = stiffness
        self.damping = damping
        
    def apply_action(self, data, target_pos, target_vel=None):
        """
        Impedance control law: F = K(x_d - x) + D(v_d - v)
        """
        if target_vel is None:
            target_vel = np.zeros_like(target_pos)
        
        # Current state
        current_pos = data.qpos[:len(target_pos)]
        current_vel = data.qvel[:len(target_vel)]
        
        # Compute control
        pos_error = target_pos - current_pos
        vel_error = target_vel - current_vel
        
        # Impedance control law
        control = self.stiffness * pos_error + self.damping * vel_error
        
        # Apply to actuators
        data.ctrl[:len(control)] = control
```

## 🔧 Action Space Patterns

### **Joint Space Control**
```python
def create_joint_space_action(n_arm_joints, n_gripper_joints):
    """Standard joint space action"""
    return {
        'arm': np.zeros(n_arm_joints),      # Joint positions/velocities
        'gripper': np.zeros(n_gripper_joints) # Gripper control
    }

def apply_joint_action(env, action_dict):
    """Apply joint space action"""
    # Arm joints
    arm_action = action_dict['arm']
    env.data.ctrl[:env.n_arm_joints] = arm_action
    
    # Gripper
    gripper_action = action_dict['gripper']
    if env.n_gripper_joints == 1:
        # Simple gripper: single value
        env.data.ctrl[env.n_arm_joints] = gripper_action[0]
    else:
        # Multi-joint gripper
        env.data.ctrl[env.n_arm_joints:] = gripper_action
```

### **End-Effector Space Control**
```python
class EndEffectorController:
    """Control in Cartesian space"""
    def __init__(self, model, ee_site_name='ee_site'):
        self.model = model
        self.ee_site_id = model.site(ee_site_name).id
        
    def get_ee_pose(self, data):
        """Get current end-effector pose"""
        ee_pos = data.site_xpos[self.ee_site_id].copy()
        ee_mat = data.site_xmat[self.ee_site_id].reshape(3, 3)
        return ee_pos, ee_mat
    
    def apply_ee_action(self, data, delta_pos, delta_rot=None):
        """Apply end-effector space action"""
        # Get current pose
        current_pos, current_mat = self.get_ee_pose(data)
        
        # Compute target
        target_pos = current_pos + delta_pos
        
        # Use inverse kinematics (simplified for 3-DOF)
        joint_targets = self.simple_ik(target_pos)
        
        # Apply to joints
        data.ctrl[:3] = joint_targets
    
    def simple_ik(self, target_pos):
        """Simplified IK for 3-DOF arm"""
        # Geometric IK solution
        x, y, z = target_pos
        
        # Base rotation
        theta1 = np.arctan2(y, x)
        
        # Arm extension
        r = np.sqrt(x**2 + y**2)
        
        # Elbow and wrist angles (simplified)
        theta2 = np.arctan2(z, r)
        theta3 = 0  # Keep wrist straight
        
        return np.array([theta1, theta2, theta3])
```

### **Hybrid Control**
```python
class HybridController:
    """Mix position and force control"""
    def __init__(self, model, force_dims=[2], position_dims=[0, 1]):
        self.model = model
        self.force_dims = force_dims      # Dimensions for force control
        self.position_dims = position_dims # Dimensions for position control
        
    def apply_hybrid_action(self, data, position_targets, force_targets):
        """Apply hybrid position/force control"""
        control = np.zeros(self.model.nu)
        
        # Position control for specified dimensions
        for i, dim in enumerate(self.position_dims):
            if dim < len(control):
                control[dim] = position_targets[i]
        
        # Force control for specified dimensions
        for i, dim in enumerate(self.force_dims):
            if dim < len(control):
                # Simple force control (would need force sensor)
                control[dim] = self.force_to_position(force_targets[i], dim)
        
        data.ctrl[:] = control
```

## 🎯 Gripper Control Patterns

### **Binary Gripper**
```python
def control_binary_gripper(data, action, gripper_joint_id):
    """Simple open/close gripper control"""
    # Action: -1 = fully open, +1 = fully closed
    gripper_range = 0.04  # 4cm travel per finger
    
    # Map to position
    target_pos = gripper_range * (1 - action) / 2
    data.ctrl[gripper_joint_id] = target_pos
```

### **Continuous Gripper**
```python
class GripperController:
    """Advanced gripper control with force feedback"""
    def __init__(self, model, 
                 gripper_joint_ids,
                 min_force=0.5,
                 max_force=10.0):
        self.model = model
        self.joint_ids = gripper_joint_ids
        self.min_force = min_force
        self.max_force = max_force
        self.gripping = False
        
    def control(self, data, action, tactile_feedback=None):
        """
        Intelligent gripper control
        action: desired grip force normalized to [-1, 1]
        """
        if action > 0:  # Closing command
            if tactile_feedback is not None:
                # Use tactile to modulate grip
                current_force = np.sum(tactile_feedback)
                
                if current_force < self.min_force:
                    # Keep closing
                    grip_vel = 0.01
                elif current_force > self.max_force:
                    # Too much force, open slightly
                    grip_vel = -0.001
                else:
                    # Maintain position
                    grip_vel = 0
                    self.gripping = True
            else:
                # No tactile, use fixed closing speed
                grip_vel = 0.01 * action
        else:
            # Opening command
            grip_vel = 0.02 * action  # Faster opening
            self.gripping = False
        
        # Apply velocities
        for joint_id in self.joint_ids:
            current_pos = data.qpos[joint_id]
            data.ctrl[joint_id] = current_pos + grip_vel
```

### **Adaptive Gripper**
```python
def adaptive_grip_control(data, tactile_left, tactile_right, 
                         target_force=2.0, kp=0.1):
    """Grip with target force using tactile feedback"""
    # Current grip force
    left_force = np.sum(tactile_left[:, :, 0])
    right_force = np.sum(tactile_right[:, :, 0])
    total_force = left_force + right_force
    
    # Force error
    force_error = target_force - total_force
    
    # Proportional control
    grip_adjustment = kp * force_error
    
    # Apply symmetric adjustment
    current_grip = data.qpos[gripper_joint_id]
    new_grip = np.clip(current_grip + grip_adjustment, 0, 0.04)
    
    data.ctrl[gripper_joint_id] = new_grip
```

## 🐛 Control Troubleshooting

### **Oscillations**
```python
class DampedController:
    """Add damping to reduce oscillations"""
    def __init__(self, damping_factor=0.1):
        self.damping = damping_factor
        self.last_action = None
        
    def apply_action(self, data, action):
        """Apply action with damping"""
        if self.last_action is None:
            self.last_action = action
        
        # Smooth action
        smoothed_action = (
            (1 - self.damping) * action + 
            self.damping * self.last_action
        )
        
        self.last_action = smoothed_action
        return smoothed_action
```

### **Joint Limits**
```python
def safe_joint_control(model, data, target_positions):
    """Ensure joint limits are respected"""
    safe_targets = np.zeros_like(target_positions)
    
    for i in range(len(target_positions)):
        if model.jnt_limited[i]:
            low, high = model.jnt_range[i]
            # Add safety margin
            margin = 0.05 * (high - low)
            safe_targets[i] = np.clip(
                target_positions[i],
                low + margin,
                high - margin
            )
        else:
            safe_targets[i] = target_positions[i]
    
    data.ctrl[:len(safe_targets)] = safe_targets
```

### **Control Saturation**
```python
def apply_control_with_limits(data, desired_control, max_change=0.1):
    """Limit control change rate"""
    current_control = data.ctrl.copy()
    
    # Compute change
    control_change = desired_control - current_control
    
    # Limit change magnitude
    change_magnitude = np.linalg.norm(control_change)
    if change_magnitude > max_change:
        control_change = control_change * (max_change / change_magnitude)
    
    # Apply limited control
    data.ctrl[:] = current_control + control_change
```

## 📊 Control Performance Patterns

### **Control Metrics**
```python
def compute_control_metrics(trajectory):
    """Analyze control performance"""
    metrics = {}
    
    # Tracking error
    if 'desired_pos' in trajectory and 'actual_pos' in trajectory:
        tracking_error = np.abs(
            trajectory['desired_pos'] - trajectory['actual_pos']
        )
        metrics['mean_tracking_error'] = np.mean(tracking_error)
        metrics['max_tracking_error'] = np.max(tracking_error)
    
    # Control effort
    if 'actions' in trajectory:
        actions = np.array(trajectory['actions'])
        metrics['mean_control_effort'] = np.mean(np.abs(actions))
        metrics['control_smoothness'] = np.mean(np.abs(np.diff(actions, axis=0)))
    
    # Stability
    if 'velocities' in trajectory:
        velocities = np.array(trajectory['velocities'])
        metrics['velocity_variance'] = np.var(velocities, axis=0)
    
    return metrics
```

## 🔗 Related Patterns
- Environment integration → `/ai_docs/PATTERNS/ENVIRONMENT_PATTERNS.md`
- Sensor feedback → `/ai_docs/PATTERNS/SENSOR_PATTERNS.md`
- Action processing → `/ai_docs/ENVIRONMENT_HUB.md#action-processing`