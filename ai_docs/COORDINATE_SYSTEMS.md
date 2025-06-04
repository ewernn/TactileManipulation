# 🌐 TACTILE MANIPULATION COORDINATE SYSTEMS

## ⚠️ CRITICAL REFERENCE - READ THIS FIRST

### **🎯 Problem This Solves**
Default Franka Panda configurations often have the gripper **facing away from the workspace**, causing failed grasping attempts. This document establishes the correct coordinate system conventions.

---

## 🔧 MuJoCo World Frame

```yaml
Origin: Robot base (link0) at (0, 0, 0)
Axes:
  +X: Forward → Toward workspace/manipulation targets  
  +Y: Left → Robot's left side when facing forward
  +Z: Up → Vertical (away from floor)
```

### **Workspace Layout**
```
    +Y (Left)
     ↑
     |
  🤖 |----→ +X (Forward to blocks)
   Base    
     |    
     ↓ Floor (Z=0)
     
  Blocks: X ∈ [0.25, 0.4], Y ∈ [-0.1, 0.1], Z=0.44
```

---

## 🤖 Robot Joint Configuration

### **❌ WRONG - Default Franka Home Position**
```python
# This makes gripper face BACKWARD (-X direction)
home_qpos = [0, -0.785, 0, -2.356, 0, 1.571, -0.785]
#            ^                        ^      ^
#         Joint 1=0              Joint 6   Joint 7
#        (faces -X)            (90° down) (45° twist)
```

### **✅ CORRECT - 110° Wrist Configuration**  
```python
# Optimized for forward reach with 110° wrist angle
home_qpos = [3.14159, -0.785, 0, -2.356, 0, 1.920, 0.785]
#            ^^^^^^^                      ^  ^^^^^  ^^^^^
#           Joint 1=π              Joint 6=1.920  Joint 7=0.785
#         (faces +X)              (110° wrist)    (45° twist)
```

### **Joint Functions**
| Joint | Function | Correct Value | Why |
|-------|----------|---------------|-----|
| 1 | Base rotation | `π` (180°) | Face workspace (+X) |
| 2 | Shoulder | `-0.785` (-45°) | Lower arm toward table |
| 3 | Elbow | `0` | Neutral |
| 4 | Forearm | `-2.356` (-135°) | Position for horizontal reach |
| 5 | Wrist roll | `0` | Neutral |
| 6 | Wrist pitch | `1.920` (110°) | Optimal forward reach angle |
| 7 | Wrist yaw | `0.785` (45°) | Improves grasp approach |

---

## 📹 Camera Coordinate Systems

### **Wrist Camera (Eye-in-Hand)**
```yaml
Position: Relative to hand frame
Orientation: Looking in +X direction (toward blocks)
Purpose: Manipulation feedback, RL training
Frame: Moves with gripper
```

### **External Cameras**
```yaml
demo_cam: 
  - Position: (1.2, -0.8, 1.0)
  - Purpose: Cinematic demonstrations
  
overhead_cam:
  - Position: (0.5, 0, 2.0) 
  - Purpose: Top-down strategic view
  
side_cam:
  - Position: (0.2, -1.5, 0.8)
  - Purpose: Profile view of manipulation
```

---

## 🔍 Verification Methods

### **1. Gripper Direction Test**
```python
import mujoco
import numpy as np

# Get hand orientation
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
hand_quat = data.xquat[hand_id]

# Convert to rotation matrix
rotmat = np.zeros(9)
mujoco.mju_quat2Mat(rotmat, hand_quat)
rotmat = rotmat.reshape(3, 3)

# Check gripper pointing direction (Z-axis of hand frame)
gripper_direction = rotmat[:, 2]
print(f"Gripper direction: {gripper_direction}")

# Verification
if gripper_direction[0] > 0.9:
    print("✅ CORRECT: Gripper pointing toward workspace (+X)")
elif gripper_direction[0] < -0.9:
    print("❌ WRONG: Gripper pointing away from workspace (-X)")
else:
    print("⚠️  UNCLEAR: Gripper direction ambiguous")
```

### **2. Block Reachability Test**
```python
# Check if blocks are in reachable workspace
def check_workspace():
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    target_pos = data.xpos[target_id]
    
    print(f"Target block position: {target_pos}")
    
    # Blocks should be in +X direction from robot base
    if target_pos[0] > 0.2:
        print("✅ Block in correct workspace (+X)")
    else:
        print("❌ Block behind robot or too close")
        
    return target_pos[0] > 0.2 and abs(target_pos[1]) < 0.15
```

### **3. Visual Verification**
```python
# Render from side camera to see gripper orientation
frame = env.render(camera="side_cam")
# Gripper should be horizontal and pointing toward blocks
```

---

## 📄 XML Configuration

### **Keyframe Setup**
```xml
<keyframe>
  <key name="home" 
       qpos="3.14159 -0.785 0 -2.356 0 0 0 0.04 0.04 0.35 0 0.44 1 0 0 0 0.28 -0.08 0.44 1 0 0 0"
       ctrl="3.14159 -0.785 0 -2.356 0 0 0 255"/>
  <!--    ^^^^^^^ Joint 1 = π (FACE WORKSPACE)         -->
  <!--                              ^^^^^ Joint 6,7 = 0 (HORIZONTAL) -->
</keyframe>
```

### **Camera Definitions**
```xml
<!-- Wrist-mounted camera for manipulation -->
<body name="wrist_camera" pos="0 0 0.08">
  <camera name="wrist_cam" pos="0 0 0" xyaxes="1 0 0 0 0 -1" fovy="70"/>
</body>
```

---

## 🚨 Common Mistakes

### **1. Wrong Base Orientation**
- **Symptom**: Robot reaches but misses blocks completely
- **Cause**: Joint 1 = 0 instead of π
- **Fix**: Set `home_qpos[0] = 3.14159`

### **2. Hanging Gripper**  
- **Symptom**: Gripper points downward instead of forward
- **Cause**: Joint 6 = 1.571 (90°) instead of 0
- **Fix**: Set `home_qpos[5] = 0`

### **3. Environment vs XML Mismatch**
- **Symptom**: XML keyframe is correct but robot still wrong
- **Cause**: Environment code overrides XML in `reset()`
- **Fix**: Update `home_qpos` array in environment class

### **4. Camera Coordinate Confusion**
- **Symptom**: Wrist camera sees floor instead of workspace
- **Cause**: Camera mounted with wrong orientation
- **Fix**: Use `xyaxes="1 0 0 0 0 -1"` for downward-looking

---

## 📚 Usage in Code

### **Environment Reset**
```python
def reset(self):
    # ALWAYS set correct home position
    home_qpos = np.array([3.14159, -0.785, 0, -2.356, 0, 0, 0])
    
    for i, qpos in enumerate(home_qpos):
        joint_addr = self.model.jnt_qposadr[self.arm_joint_ids[i]]
        self.data.qpos[joint_addr] = qpos
    
    mujoco.mj_forward(self.model, self.data)
    return self._get_observation()
```

### **Action Space Mapping**
```python
def step(self, action):
    # Action space assumes correct initial orientation
    # action[0] = Joint 1 velocity (rotation around base)
    # action[5] = Joint 6 velocity (wrist pitch)
    # action[6] = Joint 7 velocity (wrist yaw)
    # action[7] = Gripper command (-1=open, +1=close)
```

---

## 🎯 Summary Checklist

Before deploying any tactile manipulation system:

- [ ] ✅ Joint 1 = π (gripper faces +X workspace)
- [ ] ✅ Joint 6 = 0 (horizontal gripper) 
- [ ] ✅ Joint 7 = 0 (no twist)
- [ ] ✅ Blocks positioned at X > 0.25
- [ ] ✅ Wrist camera looks toward workspace
- [ ] ✅ Gripper direction test passes (X-component > 0.9)
- [ ] ✅ Visual verification from side camera

**Remember**: Coordinate system errors cause 100% failure rates in manipulation tasks. Always verify before training!