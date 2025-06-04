# 🎯 TACTILE MANIPULATION IMPLEMENTATION TEMPLATE

## ⚡ QUICK START (Copy & Paste)

### **🚨 Got Errors? Use This:**
```
@ai_docs/MAIN_TEMPLATE.md

**Error Type:** [Runtime/Build/Test/Import/Physics/RL]
**Priority:** [Critical/High/Medium/Low]

**Error Output:**
```
[Paste full error output/traceback here]
```

**Context:**
- What were you trying to do?
- What file/module?
- Recent changes?

**Request:** [Specific fix needed]
```

### **✨ New Feature? Use This:**
```
@ai_docs/MAIN_TEMPLATE.md

"Create [FEATURE] ([detail_1], [detail_2], [detail_3]) 
in [LOCATION] with [INTEGRATIONS]"

Examples:
- "Create RL policy (PPO, tactile observations, velocity control) for 7-DOF grasping with domain randomization"
- "Create slip detection (tactile time series, threshold-based, real-time) in tactile_sensor.py with grasp controller"
- "Create multi-object environment (3 cubes, different sizes, random placement) in environments/ with tactile sensing"
```

### **🔧 Bug Fix? Use This:**
```
@ai_docs/MAIN_TEMPLATE.md

"Fix [ISSUE] ([symptom_1], [symptom_2], [symptom_3]) 
in [MODULE] affecting [FUNCTIONALITY]"

Examples:
- "Fix action space mapping (velocity scaling, control frequency mismatch, joint limits) in panda_7dof_rl_env.py affecting RL training"
- "Fix gripper collision (fingers pass through objects, no contact forces, contype=0) in panda.xml affecting grasping"
- "Fix forward kinematics (end-effector pose calculation, rotation matrices) in RL environment affecting reward computation"
```

## 🚀 IMPLEMENTATION PROTOCOL

### **Phase 1: Analysis**
1. **Parse Request** - Understand exact requirements
2. **Scan Codebase** - Find existing patterns in tactile-rl/
3. **Check Dependencies** - Verify mujoco, numpy, etc.
4. **Plan Approach** - Design implementation

### **Phase 2: Implementation**
1. **Create/Update Models** - XML files for robots/objects
2. **Environment Logic** - Python environment classes
3. **Sensor Integration** - Tactile sensor updates
4. **Scripts** - Data collection/testing scripts
5. **Documentation** - Update relevant ai_docs/

### **Phase 3: Validation**
1. **Run Tests** - Execute test scripts
2. **Check Physics** - Verify collision/contacts work
3. **Validate Data** - Ensure sensors give readings
4. **Update Docs** - Reflect changes

## 🌐 COORDINATE SYSTEM REFERENCE

### **⚠️ CRITICAL: Franka Panda Coordinate System**
```yaml
World Frame (MuJoCo):
  - Origin: Robot base (link0)
  - +X: Forward toward workspace/blocks
  - +Y: Left (robot's perspective)
  - +Z: Up (vertical)

Robot Configuration:
  - Base at origin: (0, 0, 0)
  - Blocks in workspace: +X direction (X > 0.2)
  - Default home position: Joint 1 = π (180°) to face workspace

Panda Joint Mapping (7-DOF):
  - Joint 0: Base rotation (Z-axis)
  - Joint 1: Shoulder lift (rotation to face workspace)
  - Joint 2: Shoulder rotation (upper arm twist)
  - Joint 3: Elbow flexion (primary reach control)
  - Joint 4: Forearm rotation (wrist roll)
  - Joint 5: Wrist flexion (wrist pitch)
  - Joint 6: Wrist rotation (final orientation)
  - Joint 7: Gripper control (finger opening)

Gripper Orientation:
  - Joint 0 = 0: Base faces +X workspace (CRITICAL - not π!)
  - Joint 5 = 1.920: 110° wrist angle (optimal for forward reach)
  - Joint 6 = 0.785: 45° twist for optimal grasp angle
  - Hand XML: quat="0.707 0 0.707 0" rotates gripper to point forward
  - Result: Gripper points horizontally forward toward workspace

Camera Coordinate System:
  - Wrist camera follows gripper orientation
  - Looking direction: Forward along gripper (+X world)
  - Camera config: xyaxes="0 1 0 0 0 1" (orientation #2)
  - Position: pos="0 0 0.08" (80mm above gripper)
  - Camera looks along +X, up is +Z (world vertical)
  - Eye-in-hand configuration for consistent viewpoint
```

## 📋 TACTILE MANIPULATION PATTERNS

### **Project Structure**
```
tactile-rl/
├── environments/           # RL Environment implementations
│   ├── tactile_sensor.py      # Core tactile sensing (3x4 taxels)
│   ├── panda_7dof_rl_env.py  # 7-DOF RL environment (CURRENT)
│   └── *_env.py              # Other environment classes
├── franka_emika_panda/        # Robot models & XML
│   ├── panda.xml             # Full 7-DOF Panda (MAIN MODEL)
│   ├── hand.xml              # Gripper-only model
│   └── assets/               # Meshes & textures
├── scripts/                  # Experiments & utilities
│   ├── test_*.py            # Testing scripts
│   ├── train_*.py           # RL training scripts
│   └── collect_*.py         # Data collection
└── datasets/                # Output data location
```

### **Current Working System:**
- **✅ panda_7dof_rl_env.py** - Main RL environment (7-DOF + tactile)
- **✅ panda_demo_env.py** - Enhanced demo environment with multiple cameras
- **✅ tactile_sensor.py** - Complete tactile implementation
- **✅ panda_demo_scene.xml** - Full scene with table and blocks in reach
- **✅ create_expert_demos.py** - Expert demonstration generator

### **⚠️ CRITICAL: Control Strategy**
The Panda XML uses **position-controlled actuators** (high-gain servos), NOT velocity control!
- Expert demos: Use position control for clean trajectories
- RL training: Simulate velocity control on top of position actuators
- See `CONTROL_STRATEGY.md` for full details

### **7-DOF RL Environment Pattern**
```python
# environments/panda_7dof_rl_env.py
import numpy as np
import mujoco
from tactile_sensor import TactileSensor

class Panda7DOFTactileRL:
    """7-DOF Panda with velocity control for RL + sim-to-real"""
    
    def __init__(self, control_frequency=20, joint_vel_limit=2.0):
        # Load 7-DOF Panda
        self.model = mujoco.MjModel.from_xml_path("panda.xml")
        self.data = mujoco.MjData(self.model)
        
        # CRITICAL: Set home position to face workspace (+X direction)
        self.home_qpos = np.array([
            0,        # Joint 0: Base rotation (0 = face +X, π = face -X)
            -0.785,   # Joint 1: Shoulder lift
            0,        # Joint 2: Shoulder rotation
            -2.356,   # Joint 3: Elbow flexion
            0,        # Joint 4: Forearm rotation
            1.920,    # Joint 5: Wrist flexion (110° for optimal reach)
            0.785     # Joint 6: Wrist rotation (45° for grasp)
        ])
        
        # Initialize tactile sensor  
        self.tactile_sensor = TactileSensor(
            model=self.model, data=self.data,
            n_taxels_x=3, n_taxels_y=4
        )
        
        # Action: [joint_vels x7, gripper_cmd x1]
        # Observation: {joint_pos, joint_vel, tactile, cube_pose, ee_pose}
    
    def reset(self):
        # CRITICAL: Always start facing workspace
        for i, qpos in enumerate(self.home_qpos):
            joint_addr = self.model.jnt_qposadr[self.arm_joint_ids[i]]
            self.data.qpos[joint_addr] = qpos
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_observation()
    
    def step(self, action):
        # Apply velocity control (sim-to-real friendly)
        # NOTE: Actuators are position servos, so we integrate velocity to position
        joint_vels = action[:7] * self.joint_vel_limit
        gripper_cmd = action[7]  # -1=open, +1=close
        
        # Integrate velocities to positions for position-controlled actuators
        for i, vel in enumerate(joint_vels):
            joint_addr = self.model.jnt_qposadr[self.arm_joint_ids[i]]
            current_pos = self.data.qpos[joint_addr]
            target_pos = current_pos + vel * self.dt
            self.data.ctrl[i] = target_pos
        
        # Step physics at control frequency
        mujoco.mj_step(self.model, self.data)
        
        return obs, reward, terminated, truncated, info
```

### **Tactile Integration Pattern**
```python
# Get tactile readings in RL environment
def _get_observation(self):
    # Get 72-dimensional tactile readings (3x4x3 per finger x2 fingers)
    left_tactile, right_tactile = self.tactile_sensor.get_readings(
        self.model, self.data
    )
    
    # Combine all observations for RL
    obs = {
        'joint_pos': joint_positions,          # 7D robot state
        'joint_vel': joint_velocities,         # 7D robot velocities  
        'gripper_pos': gripper_opening,        # 1D gripper state
        'tactile': np.concatenate([            # 72D tactile readings
            left_tactile.flatten(), 
            right_tactile.flatten()
        ]),
        'cube_pose': cube_position_quaternion, # 7D object state
        'ee_pose': end_effector_pose          # 7D end-effector state
    }
    
    return obs
```

### **RL Training Pattern**
```python
# scripts/train_tactile_policy.py
import numpy as np
from environments.panda_7dof_rl_env import Panda7DOFTactileRL

def train_rl_policy():
    # Create RL environment
    env = Panda7DOFTactileRL(
        control_frequency=20,    # 20 Hz control (realistic)
        joint_vel_limit=2.0      # 2 rad/s max (safe)
    )
    
    # RL training loop
    for episode in range(n_episodes):
        obs = env.reset()
        
        for step in range(max_steps):
            # Policy predicts action from observations
            action = policy.predict(obs)  # 8D: [joint_vels x7, gripper x1]
            
            # Environment step with velocity control
            obs, reward, done, truncated, info = env.step(action)
            
            # Train policy with tactile feedback
            policy.update(obs, action, reward)
            
            if done or truncated:
                break
```

### **Sim-to-Real Transfer Pattern**
```python
# Real robot deployment
class RealPandaController:
    def __init__(self):
        # Same interface as simulation
        self.control_frequency = 20  # Hz
        self.joint_vel_limit = 2.0   # rad/s
    
    def step(self, action):
        # Convert RL action to robot commands
        joint_vels = action[:7] * self.joint_vel_limit
        gripper_cmd = action[7]
        
        # Send to real robot (velocity control)
        self.robot.set_joint_velocities(joint_vels)
        self.robot.set_gripper_position(gripper_cmd)
        
        # Read real tactile sensors
        tactile_readings = self.tactile_hardware.read()
        
        return self._get_real_observation()
```

## ⚠️ MANDATORY DOCUMENTATION UPDATE

**After EVERY implementation:**

1. **Check Impact** - What documentation needs updating?
   - New environments → Update ENVIRONMENT_HUB.md
   - New sensors → Update TACTILE_HUB.md
   - New robots → Update ROBOT_HUB.md
   - New patterns → Update pattern library

2. **Update Hubs** - Keep feature documentation current
3. **Update Navigator** - Add new quick actions if needed
4. **Keep concise** - Focus on essential patterns and functions

### **Update Triggers:**
- ✅ New environment → Update environment documentation
- ✅ New XML model → Update robot/object documentation
- ✅ New data collection → Update data patterns
- ✅ New training approach → Update training documentation

## 🐛 Common Issues & Solutions

### **⚠️ CONTROL MODE CONFUSION (CRITICAL)**
```python
# Problem: Robot barely moves despite velocity commands
# Symptom: Commanded 0.5 rad/s, actual movement 0.002 rad/s
# Root cause: Actuators are position servos, not velocity controlled!

# Diagnostic to check actuator type:
print(f"Actuator gain: {model.actuator_gainprm[0, 0]}")  # 4500 = position servo
print(f"Bias type: {model.actuator_biastype[0]}")  # 1 = affine (PD control)

# Solution for expert demos: Use position control directly
data.ctrl[i] = target_position  # NOT velocity!

# Solution for RL: Integrate velocity to position
target_pos = current_pos + velocity_cmd * dt
data.ctrl[i] = target_pos
```

### **⚠️ COORDINATE SYSTEM ISSUES (CRITICAL)**
```python
# Problem: Gripper pointing away from blocks
# Symptom: Robot reaches but can't grasp, wrong direction
# Solution: Check home position Joint 1 value

# CORRECT - Default Franka position (faces +X)
home_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.920, 0.785])  # Joint 0 = 0

# WRONG - Faces away from workspace (-X direction)  
home_qpos = np.array([3.14159, -0.785, 0, -2.356, 0, 1.920, 0.785])  # Joint 0 = π

# Verify gripper direction
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
hand_quat = data.xquat[hand_id]
rotmat = np.zeros(9)
mujoco.mju_quat2Mat(rotmat, hand_quat) 
gripper_direction = rotmat.reshape(3,3)[:, 2]  # Z-axis = pointing direction
print(f"Gripper X-component: {gripper_direction[0]}")  # Should be > 0.9
```

### **Action Space Issues**
```python
# Check action dimensions match environment
print(f"Action space: {env.action_space.shape}")  # Should be (8,) for 7-DOF
print(f"Control dim: {env.model.nu}")  # Should be 8 (7 joints + 1 gripper)

# Debug velocity scaling
joint_vels = action[:7] * env.joint_vel_limit
print(f"Max joint vel: {np.max(np.abs(joint_vels))} rad/s")  # Should be ≤ 2.0
```

### **Tactile Sensor Issues**
```python
# Verify tactile sensor finds finger bodies
print(f"Finger IDs: {env.tactile_sensor.finger_ids}")  # Should find left/right

# Check tactile readings shape
left, right = env.tactile_sensor.get_readings(env.model, env.data)
print(f"Tactile shapes: {left.shape}, {right.shape}")  # Should be (3,4,3) each
```

### **RL Training Issues**
```python
# Check observation space consistency
obs = env.reset()
print(f"Observation keys: {obs.keys()}")
print(f"Tactile range: [{np.min(obs['tactile'])}, {np.max(obs['tactile'])}]")

# Verify reward computation
reward = env._compute_reward(obs, action)
print(f"Reward components: height={height_reward}, contact={contact_reward}")
```

### **XML Keyframe Issues**
```xml
<!-- CORRECT - Robot facing workspace -->
<key name="home" qpos="0 -0.785 0 -2.356 0 1.920 0.785 0.04 0.04 ..." />
<!--                ^ Joint 0 = 0 to face +X (workspace) direction -->
<!--                           ^^^^^ Joint 3 = -2.356 (elbow bent) -->
<!--                                      ^^^^^ Joint 5 = 1.920 (110°) -->
<!--                                             ^^^^^ Joint 6 = 0.785 (45°) -->
```

### **Collision/Physics Issues**
```python
# Check contact detection
print(f"Number of contacts: {env.data.ncon}")
if env.data.ncon > 0:
    for i in range(min(5, env.data.ncon)):
        contact = env.data.contact[i]
        print(f"Contact {i}: geom1={contact.geom1}, geom2={contact.geom2}")
```

## 📊 Testing Protocol

Always test new implementations:

```bash
# 1. Test 7-DOF RL environment
cd tactile-rl/environments
python panda_7dof_rl_env.py

# 2. Test tactile sensor integration
cd tactile-rl/environments  
python tactile_sensor.py

# 3. Test action space and observations
cd tactile-rl/scripts
python test_rl_env.py

# 4. Test RL training loop
python train_tactile_policy.py --test_mode --n_episodes 5
```

## 🎯 Current Status Summary

### **✅ Working Components:**
- **Control Strategy Clarified** - Position servos for demos, velocity interface for RL
- **Coordinate System Fixed** - Joint 0 = 0 to face workspace (not π!)
- **Expert Demonstration System** - Now using proper position control
- **Tactile Demo Environment** (`panda_demo_env.py`) - Position-controlled
- **7-DOF RL Environment** (`panda_7dof_rl_env.py`) - Velocity interface over position servos
- **Tactile Sensing** - 3x4 taxels per finger (72D total) with accurate contact detection
- **Clean Scripts Directory** - Reduced from 106 to 12 essential scripts

### **🎮 Current Configuration:**
- ✅ **Wrist Angle**: 110° (1.920 rad) - optimal for forward reach
- ✅ **Block Distance**: 0.688m forward from gripper start position
- ✅ **Success Threshold**: 5mm lift (realistic for manipulation)
- ✅ **Control Frequency**: 20Hz (sim-to-real compatible)
- ✅ **Action Space**: 8D continuous (7 joint velocities + 1 gripper)

### **📊 Recent Updates:**
1. **Control Mode Discovery** - Identified position servos, not velocity control
2. **Coordinate Fix** - Corrected Joint 0 = 0 (not π) to face workspace
3. **Two-Stage Pipeline** - Position control for demos → Velocity interface for RL
4. **Documentation** - Added CONTROL_STRATEGY.md explaining the approach
5. **Diagnostic Tools** - Created scripts to analyze control and movement issues

### **🚀 Next Steps:**
1. **Generate Expert Demos** - Create position-controlled demonstrations
2. **Train BC Policy** - Learn from clean position-controlled trajectories
3. **RL Fine-tuning** - Switch to velocity interface for robust control
4. **Evaluate Performance** - Test sim-to-real transfer capabilities

## 📏 Stacking Reward Metric

### **Reward Function for Block Stacking**
```python
def compute_stacking_reward(red_pos, blue_pos, red_quat, blue_quat):
    """Compute reward for how well red block is stacked on blue block."""
    
    # 1. Vertical alignment - red should be directly above blue
    xy_offset = np.sqrt((red_pos[0] - blue_pos[0])**2 + (red_pos[1] - blue_pos[1])**2)
    alignment_reward = np.exp(-10 * xy_offset)  # Exponential decay, max 1.0
    
    # 2. Height reward - red bottom should be at blue top
    red_bottom = red_pos[2] - block_half_size  # 0.025m for standard blocks
    blue_top = blue_pos[2] + block_half_size
    height_error = abs(red_bottom - blue_top)
    height_reward = np.exp(-20 * height_error)  # More sensitive to height
    
    # 3. Orientation alignment - blocks should have similar orientation
    z_alignment = np.dot(red_mat[:, 2], blue_mat[:, 2])
    orientation_reward = (z_alignment + 1) / 2  # Map from [-1, 1] to [0, 1]
    
    # 4. Stability - red block should not be tilted
    upright_score = red_z_axis[2]  # How much Z axis points up
    stability_reward = max(0, upright_score)
    
    # 5. Contact reward - bonus if blocks are touching
    contact_bonus = 1.0 if height_error < 0.005 else 0.0  # 5mm tolerance
    
    # Combine rewards with weights
    total_reward = (
        0.3 * alignment_reward +    # 30% for XY alignment
        0.3 * height_reward +       # 30% for correct height
        0.2 * orientation_reward +  # 20% for orientation match
        0.1 * stability_reward +    # 10% for stability
        0.1 * contact_bonus        # 10% for contact
    )
    
    return total_reward, {
        'alignment': alignment_reward,
        'height': height_reward,
        'orientation': orientation_reward,
        'stability': stability_reward,
        'contact': contact_bonus,
        'xy_offset': xy_offset,
        'height_error': height_error
    }
```

### **Expert Demonstration Achievement**
- **Total Reward**: 0.812 (81.2%)
- **Alignment**: 0.855 (15.6mm XY offset)
- **Height**: 0.852 (8.0mm error)
- **Orientation**: 1.000 (perfect alignment)
- **Stability**: 1.000 (perfectly upright)
- **Contact**: 0.0 (requires < 5mm for bonus)

### **Expert Demonstration Details**
```yaml
Ultimate Demo Configuration:
  - Approach clearance: 61mm (target 65mm)
  - Grasp stability: < 1mm movement during grasp
  - Block rotation: 10° on joint 7 for alignment
  - Smart placement: Stops when stack clearance = 0
  - Two-stage descent: Fast then slow approach
  - Total duration: 4.1 seconds
  - Video FPS: 120 (4x speed for ~4 second playback)

Demo Script: create_ultimate_final_demo.py
  - Linear interpolation for descend phases
  - Hold position during grasp (no interpolation)
  - Smart placement with early stopping
  - Stacking reward visualization
  - Saves to: videos/121pm/panda_ultimate_final_demo.mp4
```