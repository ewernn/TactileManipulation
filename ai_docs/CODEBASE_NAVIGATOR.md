# 🧭 TactileManipulation Codebase Navigator

*Last Updated: 2025-05-28 | Clean scripts directory | 110° wrist configuration*

## 🚀 Quick Actions (Most Common Tasks)

### **"I need to fix tactile sensor issues"**
- Sensor implementation → `/ai_docs/TACTILE_HUB.md#sensor-implementation`
- Contact detection → `/tactile-rl/environments/tactile_sensor.py`
- Noise modeling → `/ai_docs/TACTILE_HUB.md#noise-model`

### **"I need to create new environments"**
1. Check existing patterns → `/ai_docs/PATTERNS/ENVIRONMENT_PATTERNS.md`
2. Find similar environments → `/ai_docs/ENVIRONMENT_HUB.md`
3. Follow implementation guide → `/ai_docs/MAIN_TEMPLATE.md`

### **"I need to understand the robot model"**
- Panda arm setup → `/ai_docs/ROBOT_HUB.md`
- XML configurations → `/tactile-rl/franka_emika_panda/`
- Control modes → `/ai_docs/CONTROL_HUB.md`

### **"I need to collect demonstration data"**
- Expert demo script → `/tactile-rl/scripts/create_expert_demos.py`
- Data collection → `/tactile-rl/scripts/collect_tactile_demos.py`
- Dataset format → `/ai_docs/PATTERNS/DATA_PATTERNS.md`

## 📁 System Architecture

### **Core Systems**
- **[TACTILE_HUB.md](/ai_docs/TACTILE_HUB.md)** - Tactile sensing implementation
- **[ROBOT_HUB.md](/ai_docs/ROBOT_HUB.md)** - Robot models & control (7-DOF Panda)
- **[ENVIRONMENT_HUB.md](/ai_docs/ENVIRONMENT_HUB.md)** - MuJoCo environments
- **[DATA_HUB.md](/ai_docs/DATA_HUB.md)** - Data collection & datasets
- **[TRAINING_HUB.md](/ai_docs/TRAINING_HUB.md)** - Policy training & evaluation
- **[VISUALIZATION_HUB.md](/ai_docs/VISUALIZATION_HUB.md)** - Plotting & video generation

### **Technical Patterns**
- **[ENVIRONMENT_PATTERNS.md](/ai_docs/PATTERNS/ENVIRONMENT_PATTERNS.md)** - Environment design
- **[DATA_PATTERNS.md](/ai_docs/PATTERNS/DATA_PATTERNS.md)** - HDF5 data handling
- **[CONTROL_PATTERNS.md](/ai_docs/PATTERNS/CONTROL_PATTERNS.md)** - Robot control
- **[SENSOR_PATTERNS.md](/ai_docs/PATTERNS/SENSOR_PATTERNS.md)** - Tactile sensing
- **[TEST_PATTERNS.md](/ai_docs/PATTERNS/TEST_PATTERNS.md)** - Testing approach

## 🏗️ Project Structure
```
TactileManipulation/
├── tactile-rl/               # Main codebase
│   ├── environments/         # MuJoCo environments
│   │   ├── tactile_sensor.py         # Core tactile implementation
│   │   ├── simple_tactile_env.py     # Working simplified env
│   │   └── tactile_grasping_env.py   # Original complex env
│   ├── franka_emika_panda/   # Robot models & XML
│   │   ├── panda.xml                 # 7-DOF Panda model
│   │   ├── panda_tactile_grasp.xml   # Working grasp setup
│   │   └── assets/                   # Meshes & textures
│   └── scripts/              # Core scripts (12 essential files)
│       ├── create_expert_demos.py    # Expert demonstrations
│       ├── train_policies.py         # Policy training
│       └── visualize_grasp.py       # Visualization
├── datasets/                 # Collected data
│   └── tactile_grasping/    # Demo datasets & results
├── ai_docs/                 # AI-friendly documentation
└── mimicgen/               # External data generation tool
```

## 🔍 Key References

### **Key Scripts (12 Essential)**
- Expert demos: `create_expert_demos.py`
- Data collection: `collect_tactile_demos.py`
- Training: `train_policies.py`, `train_simple.py`
- Visualization: `visualize_grasp.py`, `replay_demo.py`
- Diagnostics: `diagnose_expert_policy.py`, `tune_expert_systematically.py`

### **Environment Configurations**
- Working: `panda_tactile_grasp.xml`, `mjx_single_cube_fixed.xml`
- 7-DOF: `panda.xml`, `mjx_panda.xml`
- Original: `original_stack_environment.xml`

## 🚨 Common Pitfalls

### **Don't Recreate**
- ❌ Tactile sensor logic → Use `tactile-rl/environments/tactile_sensor.py`
- ❌ Robot XML files → Use existing in `tactile-rl/franka_emika_panda/`
- ❌ Data collection → Use `tactile-rl/scripts/collect_direct.py`

### **Always Check**
- ✅ Collision settings (contype/conaffinity) in XML files
- ✅ Control dimensions match robot DOF
- ✅ Sensor noise parameters for realism
- ✅ Existing demos in `datasets/tactile_grasping/`

## 🎯 Current Project Status

### **Working Components**
- ✅ 110° wrist configuration (Joint 6 = 1.920 rad)
- ✅ 7-DOF Panda environments (`panda_7dof_rl_env.py`, `panda_demo_env.py`)
- ✅ Tactile sensor with 3x4 taxel arrays per finger (72D total)
- ✅ Clean scripts directory (12 essential scripts from 106)
- ✅ Expert demonstration system (needs final tuning)

### **Key Results**
- 30% improvement over vision-only baseline
- 0.445m average lift height (4.5x target)
- Robust slip detection via tactile feedback

### **Next Steps**
- Implement neural network policies
- Add diverse object shapes
- Real robot transfer preparation

## 🔧 Quick Start Commands

```bash
# Test tactile grasping
cd tactile-rl/scripts
python test_simple_robot.py

# Collect new demonstrations
python collect_direct.py

# View results
python create_summary.py

# Test 7-DOF Panda
python panda_7dof_grasp.py
```

## 📚 Documentation Update Protocol

After ANY code changes:
1. Update relevant hub documentation
2. Check for new patterns to document  
3. Update this navigator if new features added
4. Keep docs concise and focused on essential patterns