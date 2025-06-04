# 📹 Demonstration & Training Guide

*Complete guide for generating expert demonstrations and training policies*

## 🎯 Quick Start

### **Generate Expert Demonstrations**
```bash
cd tactile-rl/scripts
python create_expert_demos.py
```
- Creates 30 demonstrations with 83% success rate
- Saves to `datasets/expert_demonstrations.hdf5`
- Generates visualization video

### **Train Behavioral Cloning**
```bash
python train_policies.py --mode bc --demos datasets/expert_demonstrations.hdf5
```

### **Fine-tune with RL**
```bash
python train_policies.py --mode rl --pretrained models/bc_policy.pt
```

## 📊 How the System Works

### **1. Expert Demonstration Pipeline**

The system uses **scripted expert policies** to generate demonstrations:

```python
# create_expert_demos.py
class ExpertPolicy:
    def get_action(self, observation):
        # Phases: approach → position → descend → grasp → lift
        if self.phase == "approach":
            action = [0.15, 0.25, 0.15, -0.3, 0.1, 0.1, 0.0, -1.0]
        elif self.phase == "grasp":
            action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # ... etc
```

**Key Components:**
- **Observations**: Joint positions, gripper state, tactile readings (72D), block position
- **Actions**: 8D vector [7 joint velocities + 1 gripper command]
- **Success**: Block lifted >5mm (realistic threshold)

### **2. Environment Architecture**

```
PandaDemoEnv
├── MuJoCo Physics (panda_demo_scene.xml)
│   ├── 7-DOF Panda arm
│   ├── Parallel-jaw gripper (tendon-driven)
│   └── Blocks with freejoints
├── Tactile Sensor (3x4 taxels per finger)
│   ├── Left finger: 36D readings
│   └── Right finger: 36D readings
└── Control Interface
    ├── Joint control: Position targets
    └── Gripper control: 0-255 tendon force
```

### **3. Critical Parameters**

**Gripper Control Mapping:**
```python
# Action space: -1 (open) to +1 (close)
# Tendon control: 0 (close) to 255 (open)
gripper_ctrl = 127.5 * (1 - gripper_cmd)
```

**Block Positioning:**
- Base position: [0.35, 0.0, 0.44]
- Randomization: ±0.02m in X/Y
- Success threshold: 5mm lift

**Gripper Physics:**
- Force range: 800N (increased from 200N)
- Gain: 0.5 (increased from 0.05)
- Control range: 0-255

### **4. Data Format (HDF5)**

```
expert_demonstrations.hdf5
├── Attributes
│   ├── n_demos: 30
│   ├── successful_demos: 25
│   └── success_rate: 0.833
└── Demonstrations
    └── demo_0/
        ├── observations/
        │   ├── joint_pos: (150, 7)
        │   ├── joint_vel: (150, 7)
        │   ├── gripper_pos: (150,)
        │   ├── tactile: (150, 72)
        │   └── target_block_pos: (150, 3)
        ├── actions: (150, 8)
        ├── rewards: (150,)
        └── success: True
```

## 🔧 Common Issues & Solutions

### **Low Success Rate**
```python
# Check block positioning
obs = env.reset(randomize=False)
print(f"Block at: {obs['target_block_pos']}")  # Should be ~[0.35, 0, 0.44]

# Verify gripper strength
print(f"Gripper force: {env.model.actuator_forcerange[7]}")  # Should be [-800, 800]
```

### **Gripper Not Closing**
```python
# Test gripper control directly
env.data.ctrl[7] = 0    # Should close gripper
env.data.ctrl[7] = 255  # Should open gripper
```

### **Block Not Lifting**
- Check success threshold (should be 0.005m not 0.05m)
- Verify physics steps per action (use 3-5 for stability)
- Ensure block randomization keeps it reachable

## 🚀 Training Pipeline

### **Stage 1: Behavioral Cloning**
```python
# Learn from demonstrations
policy = BCPolicy(obs_dim=..., act_dim=8)
for demo in dataset:
    loss = MSE(policy(demo.obs), demo.actions)
    optimizer.step()
```

### **Stage 2: RL Fine-tuning**
```python
# Improve with reinforcement learning
env = Panda7DOFTactileRL()
for episode in range(n_episodes):
    obs = env.reset()
    for step in range(max_steps):
        action = policy(obs) + exploration_noise
        obs, reward, done = env.step(action)
        # Update policy with PPO/SAC
```

### **Stage 3: Domain Randomization**
- Vary block mass: 0.1-0.2kg
- Vary friction: 0.5-1.5
- Vary positions: ±2cm
- Add visual noise

## 📈 Performance Metrics

**Current System:**
- Demo success rate: 83.3%
- Average lift height: 7-9mm
- Tactile contact rate: 95%+
- Gripper stability: High

**Target Performance:**
- BC policy success: >70%
- RL policy success: >90%
- Sim-to-real transfer: >60%