# 🤖 AI-Friendly Documentation System

This directory contains structured documentation designed for AI agents to efficiently navigate and work with the TactileManipulation codebase.

## 🚀 Quick Start for AI Agents

Always start with:
```
@ai_docs/CODEBASE_NAVIGATOR.md
```

For implementation requests, use:
```
@ai_docs/MAIN_TEMPLATE.md
```

## 📁 Structure

### Core Documents
- **CODEBASE_NAVIGATOR.md** - Master index and entry point
- **MAIN_TEMPLATE.md** - Universal template for implementation requests

### Domain Hubs
- **TACTILE_HUB.md** - Tactile sensor implementation details
- **ROBOT_HUB.md** - Robot models and control (7-DOF Panda)
- **ENVIRONMENT_HUB.md** - MuJoCo environment setup
- **DATA_HUB.md** - Data collection and datasets
- **TRAINING_HUB.md** - Policy training approaches
- **VISUALIZATION_HUB.md** - Plotting and video generation

### Pattern Libraries (PATTERNS/)
- **ENVIRONMENT_PATTERNS.md** - Environment design patterns
- **DATA_PATTERNS.md** - Data handling patterns
- **CONTROL_PATTERNS.md** - Robot control patterns
- **SENSOR_PATTERNS.md** - Tactile sensor patterns
- **TEST_PATTERNS.md** - Testing approaches

## 🔧 Git Hooks Setup

To enable automatic reminders for documentation updates:
```bash
cd /path/to/TactileManipulation
bash ai_docs/setup_hooks.sh
```

## 💡 Usage Examples

### For AI Agents:
```
@ai_docs/MAIN_TEMPLATE.md

"Create slip detection (tactile time series, threshold-based, real-time) 
in tactile_sensor.py with grasp controller"
```

### For Developers:
1. Start with CODEBASE_NAVIGATOR.md
2. Navigate to relevant hub for your task
3. Check pattern libraries for implementation guidance
4. Update documentation after significant changes

## 📝 Maintenance

This is a **curated** documentation system. When adding major features:
1. Update the relevant hub file
2. Add new patterns if discovered
3. Keep CODEBASE_NAVIGATOR.md current

The focus is on **patterns and concepts**, not exhaustive file listings.