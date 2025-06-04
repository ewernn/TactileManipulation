# Control Strategy: Position vs Velocity Control

## Overview

The Panda robot XML uses **position-controlled actuators** with high-gain servos, not velocity control. This fundamentally changes how we approach expert demonstrations vs RL training.

## Key Discovery

### Actuator Configuration
```xml
<actuator>
  <general class="panda" name="actuator1" joint="joint1" 
           gainprm="4500" biasprm="0 -4500 -450"/>
  <!-- High gain (4500) = stiff position control -->
</actuator>
```

- **Gains**: 4500, 3500, 2000 (very high = position servos)
- **Bias type**: Affine (PD control with damping)
- **Control input**: `ctrl[i]` sets **target position**, not velocity

### Why Original Demos Failed
1. Code assumed velocity control: `target_pos = current_pos + velocity * dt`
2. With dt=0.002s and velocity=0.5 rad/s → increment of 0.001 rad
3. Position servo tries to move to position 0.001 rad away (tiny!)
4. Result: Robot barely moves (250x slower than intended)

## Updated Pipeline

### 1. Expert Demonstrations (Position Control)
```python
# Direct position targets for clean trajectories
target_joints = {
    "home": [0, -0.1, 0, -2.0, 0, 1.5, 0],
    "above_red": [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
    "grasp_red": [0.0, -0.1, 0.0, -1.95, 0.0, 1.6, 0.0],
    # ... etc
}

# Apply position control
for i in range(7):
    data.ctrl[i] = target_joints[phase][i]
```

**Benefits:**
- Repeatable, precise trajectories
- Higher success rate (83% as mentioned)
- Clean demonstrations for BC
- Standard practice in robotics

### 2. RL Training (Velocity Control)
```python
# RL environment simulates velocity control
class Panda7DOFTactileRL:
    def step(self, action):
        # Action is velocity command
        for i in range(7):
            current_pos = self.data.qpos[14 + i]
            # Integrate velocity to position
            target_pos = current_pos + action[i] * self.dt
            # Apply to position-controlled actuator
            self.data.ctrl[i] = target_pos
```

**Benefits:**
- Matches real robot interface
- Better sim-to-real transfer
- Handles dynamics/disturbances
- Reactive to tactile feedback

### 3. The Complete Pipeline

```
Position Control          Velocity Control
(Expert Demos)     →     (RL Training)
     ↓                        ↓
Clean trajectories       Robust control
     ↓                        ↓
     └─────→ BC ──────→ RL ←─┘
           Learning    Fine-tuning
```

1. **Expert demos with position control** → Clean, successful trajectories
2. **BC training** → Learn task structure from ideal demos
3. **RL fine-tuning with velocity control** → Add robustness, adapt to real dynamics

## Implementation Notes

### For Expert Demonstrations
- Use direct joint targets
- Interpolate smoothly between waypoints
- Tune positions empirically or with IK
- Slower movements for reliability

### For RL Training
- Wrap position actuators with velocity interface
- Scale velocity commands appropriately
- Account for control frequency (20Hz typical)
- Add action limits for safety

### Critical Coordinate Frame Fix
```python
# CORRECT - Robot faces workspace
home_qpos = [0, -0.785, 0, -2.356, 0, 1.920, 0.785]  # Joint 0 = 0

# WRONG - Robot faces away
home_qpos = [3.14159, -0.785, 0, -2.356, 0, 1.920, 0.785]  # Joint 0 = π
```

## Why This Two-Stage Approach Works

1. **Position control gives "ideal" demonstrations**
   - What the task should look like
   - Free from control noise/dynamics

2. **BC learns high-level task structure**
   - Approach → Grasp → Lift → Move → Place
   - Approximate joint trajectories

3. **RL with velocity control adds robustness**
   - Handles real dynamics
   - Responds to tactile feedback
   - Generalizes to variations

The action space transformation happens naturally - BC provides good initialization that RL refines with the actual control interface.