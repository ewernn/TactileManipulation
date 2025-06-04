#!/usr/bin/env python3
"""
Quick test to find the right configuration for vertical alignment above red block.
"""

import numpy as np
import mujoco

# Load model
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Get IDs
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")

# Reset and step once to get block positions
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_step(model, data)

red_pos = data.xpos[red_id].copy()
print(f"Red block position: {red_pos}")

# Test different configurations to get EE above red block
# Red block is at X=0.05, we need EE X to be close to 0.05
configs_to_test = [
    # [j0, j1, j2, j3, j4, j5, j6]
    [0, 0.5, 0, -2.5, 0, 2.0, 0.785],    # More positive j1
    [0, 0.6, 0, -2.3, 0, 1.7, 0.785],    # Even more j1, less j3
    [0, 0.7, 0, -2.1, 0, 1.4, 0.785],    # Continue trend
    [0, 0.8, 0, -1.9, 0, 1.1, 0.785],    # More extreme
    [0, 0.4, 0, -2.2, 0, 1.8, 0.785],    # Moderate attempt
]

print("\nTesting configurations to get vertically above red block:")
print("Target X position: 0.050m\n")

best_config = None
best_distance = float('inf')

for i, config in enumerate(configs_to_test):
    # Apply configuration
    for j, val in enumerate(config):
        data.qpos[14 + j] = val
        data.ctrl[j] = val
    
    # Forward dynamics
    mujoco.mj_forward(model, data)
    
    # Get end-effector position
    ee_pos = data.xpos[ee_id].copy()
    
    # Calculate distances
    x_diff = ee_pos[0] - red_pos[0]
    xy_dist = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
    
    print(f"Config {i+1}: j1={config[1]:.1f}, j3={config[3]:.1f}")
    print(f"  EE X: {ee_pos[0]:.3f} (diff: {x_diff:+.3f})")
    print(f"  EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  XY distance: {xy_dist:.3f}m")
    
    if abs(x_diff) < best_distance:
        best_distance = abs(x_diff)
        best_config = config.copy()
    
    print()

print(f"\nBest configuration found:")
print(f"Config: {best_config}")
print(f"X difference: {best_distance:.3f}m")