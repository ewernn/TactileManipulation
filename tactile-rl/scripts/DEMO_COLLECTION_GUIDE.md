# 🎯 Demonstration Collection Step-by-Step Guide

## Overview: How Expert Demonstrations Work

The demonstration system creates expert trajectories for training imitation learning policies. Here's exactly what happens:

## 📦 Environment Setup (What's Already Configured)

### Scene Components:
1. **Franka Panda Robot** (7-DOF arm + gripper)
   - Base at origin (0, 0, 0)
   - Home position configured to face workspace
   - Gripper with tactile sensors (3x4 taxels per finger)

2. **Table** 
   - Height: 0.4m
   - Position: In front of robot (+X direction)

3. **Blocks** (Pre-positioned in XML)
   - **Red Block** (target): Position (0.15, 0.0, 0.445) - 15cm forward from robot base
   - **Blue Block**: Position (0.15, -0.15, 0.445) - 15cm to the right of red block
   - Block size: 5cm x 5cm x 5cm (red), 4cm x 4cm x 4cm (blue)

4. **Cameras**
   - `demo_cam`: Main camera (1280x720) - angled view
   - `side_cam`: Side view
   - `overhead_cam`: Top-down view
   - `wrist_cam`: Eye-in-hand camera on gripper

## 🤖 Expert Policy Execution Timeline

### Phase 1: "approach" (40 steps = 2 seconds at 20Hz)
- **Goal**: Move end-effector above red block
- **Action**: Joint velocities to reach target position
- **Duration**: 40 timesteps
- **Gripper**: Open (-1.0)

### Phase 2: "pre_grasp" (30 steps = 1.5 seconds)
- **Goal**: Fine positioning directly above block
- **Action**: Small adjustments for alignment
- **Duration**: 30 timesteps
- **Gripper**: Open (-1.0)

### Phase 3: "descend" (25 steps = 1.25 seconds)
- **Goal**: Lower gripper to grasp height
- **Action**: Controlled downward movement
- **Duration**: 25 timesteps
- **Gripper**: Open (-1.0)

### Phase 4: "grasp" (20 steps = 1 second)
- **Goal**: Close gripper on block
- **Action**: No joint movement, close gripper
- **Duration**: 20 timesteps
- **Gripper**: Close (+1.0)

### Phase 5: "lift" (30 steps = 1.5 seconds)
- **Goal**: Lift block 5mm off table
- **Action**: Upward movement
- **Duration**: 30 timesteps
- **Gripper**: Closed (+1.0)

### Phase 6: "move_to_blue" (40 steps = 2 seconds)
- **Goal**: Move red block over blue block
- **Action**: Lateral movement
- **Duration**: 40 timesteps
- **Gripper**: Closed (+1.0)

### Phase 7: "position_above_blue" (20 steps = 1 second)
- **Goal**: Fine positioning above blue block
- **Action**: Small adjustments
- **Duration**: 20 timesteps
- **Gripper**: Closed (+1.0)

### Phase 8: "place" (30 steps = 1.5 seconds)
- **Goal**: Lower red block onto blue block
- **Action**: Controlled descent
- **Duration**: 30 timesteps
- **Gripper**: Closed (+1.0)

### Phase 9: "release" (15 steps = 0.75 seconds)
- **Goal**: Release block
- **Action**: Open gripper
- **Duration**: 15 timesteps
- **Gripper**: Open (-1.0)

### Phase 10: "retreat" (25 steps = 1.25 seconds)
- **Goal**: Move away from stacked blocks
- **Action**: Backward and upward movement
- **Duration**: 25 timesteps
- **Gripper**: Open (-1.0)

**Total Duration**: 275 steps = 13.75 seconds per demonstration

## 📊 Data Collection Process

### For Each Demonstration:
1. **Environment Reset**
   - Robot moves to home position
   - Blocks placed at fixed positions (with optional small randomization)
   - Gripper opens fully

2. **Expert Policy Execution**
   - Policy runs through all phases sequentially
   - Actions are 8D: [7 joint velocities, 1 gripper command]
   - Each action is executed for 3 physics steps

3. **Data Recording**
   - **Observations**: Joint positions/velocities, gripper state, tactile readings, block positions
   - **Actions**: 8D control commands from expert
   - **Rewards**: Height-based reward + tactile bonus
   - **Video**: Optional frame capture at 30 FPS

4. **Success Criteria**
   - Red block lifted > 5mm (during grasping)
   - Red block placed on blue block (final task)
   - Horizontal alignment < 5cm between blocks

## 💾 Output Data Structure

### HDF5 File Format:
```
expert_demonstrations.hdf5
├── metadata
│   ├── n_demos: 50
│   ├── successful_demos: 45
│   └── success_rate: 0.9
└── demo_0/
    ├── observations/
    │   ├── joint_pos: (275, 7)
    │   ├── joint_vel: (275, 7)
    │   ├── gripper_pos: (275,)
    │   ├── tactile: (275, 72)
    │   ├── target_block_pos: (275, 3)
    │   └── block2_pos: (275, 3)
    ├── actions: (275, 8)
    ├── rewards: (275,)
    ├── tactile_readings: (275, 72)
    └── success: True
```

## 🚀 Running Data Collection

### Quick Start:
```bash
cd tactile-rl/scripts
python create_expert_demos.py
```

### What Happens:
1. Creates 30-50 demonstrations
2. Each demo takes ~15 seconds
3. Saves to `../../datasets/expert_demonstrations.hdf5`
4. Creates a fast video of 3 sample demos
5. Reports success rate statistics

### Customization Options:
- `n_demos`: Number of demonstrations to collect
- `randomize`: Add small position variations to blocks
- `save_video`: Create video for first demo
- `max_steps`: Maximum steps per demo (default: 300)

## 🎮 Action Space Details

The expert provides 8D continuous actions:
- **Dimensions 0-6**: Joint velocity commands (rad/s)
  - Scaled by 2.0 rad/s maximum
  - Integrated to position targets
- **Dimension 7**: Gripper command
  - -1.0 = fully open
  - +1.0 = fully closed
  - Maps to tendon control (0-255)

## 🔧 Key Implementation Details

1. **Control Frequency**: 20Hz (0.05s timestep)
   - Matches real robot control rate
   - Each action executes for 3 physics steps

2. **Physics Integration**:
   - Velocity commands → position targets
   - Joint limits enforced
   - Collision detection active

3. **Tactile Sensing**:
   - 3x4 taxels per finger
   - 72D total tactile vector
   - Updates every physics step

4. **Reward Function**:
   - Lifting reward: 10 * height_increase
   - Tactile bonus: 0.1 * tactile_sum
   - Success bonus for stacking

## 🐛 Common Issues & Solutions

1. **"Block not reachable"**
   - Check robot home position
   - Verify block positions in XML
   - Ensure Joint 1 = π (faces workspace)

2. **"Gripper won't close"**
   - Check tendon control mapping
   - Verify control dimension (should be 8)
   - Test gripper limits (0-0.04m)

3. **"No tactile readings"**
   - Verify finger body names match
   - Check collision detection enabled
   - Ensure tactile sensor initialized

4. **"Slow data collection"**
   - Reduce physics steps per action
   - Disable video recording
   - Use parallel collection (future)