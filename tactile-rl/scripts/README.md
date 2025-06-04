# Tactile RL Scripts

This directory contains essential scripts for the tactile manipulation pipeline.

## 🎯 Core Training Scripts

- **`train_policies.py`** - Main training script for behavioral cloning and policy learning
- **`train_simple.py`** - Simplified training implementation for quick experiments

## 📊 Data Collection

- **`collect_tactile_demos.py`** - Collects demonstrations with tactile sensor data
- **`create_expert_demos.py`** - Generates expert demonstrations using scripted policy (110° wrist config)

## 🔧 Data Processing

- **`augment_dataset.py`** - Augments collected datasets for improved training

## 📹 Visualization & Analysis

- **`visualize_grasp.py`** - Visualizes grasping behaviors and tactile feedback
- **`replay_demo.py`** - Replays collected demonstrations for inspection
- **`generate_video.py`** - Creates videos from demonstration data
- **`explore_dataset.py`** - Explores and analyzes HDF5 datasets

## 🔍 Development Tools

- **`diagnose_expert_policy.py`** - Diagnoses and analyzes expert policy performance
- **`tune_expert_systematically.py`** - Systematic tuning of expert policy parameters

## 🧹 Utilities

- **`cleanup_scripts.py`** - Archives non-essential scripts (already executed)

---

## Quick Start

1. **Generate expert demonstrations:**
   ```bash
   python create_expert_demos.py
   ```

2. **Train a policy:**
   ```bash
   python train_policies.py --dataset ../datasets/expert_demonstrations.hdf5
   ```

3. **Visualize results:**
   ```bash
   python replay_demo.py --dataset ../datasets/expert_demonstrations.hdf5
   ```

## Notes

- All scripts are configured for the 110° wrist angle (Joint 6 = 1.920 rad)
- Archived scripts can be found in `_archive_[timestamp]/` directories