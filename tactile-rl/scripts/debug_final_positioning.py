#!/usr/bin/env python3
"""
Debug the final positioning - check joint 6 rotation and vertical alignment.
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt

# Load model
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Get IDs
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")

# Reset
mujoco.mj_resetDataKeyframe(model, data, 0)

# Test configurations
configs = [
    ("Default (J6=0)", [0, -0.5, 0, -2.0, 0, 1.57, 0]),
    ("J6=45° (0.785 rad)", [0, -0.5, 0, -2.0, 0, 1.57, 0.785]),
    ("J6=90° (1.57 rad)", [0, -0.5, 0, -2.0, 0, 1.57, 1.57]),
    ("J6=-45° (-0.785 rad)", [0, -0.5, 0, -2.0, 0, 1.57, -0.785]),
]

print("Block positions:")
print(f"Red block: {data.xpos[red_id]}")
print(f"Blue block: {data.xpos[blue_id]}")
print("\nTesting joint 6 rotations and positions:\n")

for name, config in configs:
    # Apply configuration
    for i, val in enumerate(config):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    
    mujoco.mj_forward(model, data)
    
    # Get end-effector info
    ee_pos = data.xpos[ee_id]
    
    # Get gripper orientation
    hand_quat = data.xquat[ee_id].copy()
    rotmat = np.zeros(9)
    mujoco.mju_quat2Mat(rotmat, hand_quat)
    rotmat = rotmat.reshape(3, 3)
    
    gripper_x = rotmat[:, 0]  # Right
    gripper_y = rotmat[:, 1]  # Up  
    gripper_z = rotmat[:, 2]  # Forward
    
    # Calculate distances
    red_pos = data.xpos[red_id]
    xy_dist = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
    z_dist = ee_pos[2] - red_pos[2]
    
    print(f"{name}:")
    print(f"  EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Gripper Forward (Z): [{gripper_z[0]:.3f}, {gripper_z[1]:.3f}, {gripper_z[2]:.3f}]")
    print(f"  Gripper Right (X): [{gripper_x[0]:.3f}, {gripper_x[1]:.3f}, {gripper_x[2]:.3f}]")
    print(f"  Distance to red: XY={xy_dist:.3f}m, Z={z_dist:.3f}m")
    print(f"  Joint 6 angle: {config[6]:.3f} rad ({np.degrees(config[6]):.1f}°)")
    print()

# Now test positions closer to red block
print("\nTesting positions closer to red block vertically:")
print("(Red block at X=0.05, Y=0.0)\n")

# Configurations to get closer to X=0.05
close_configs = [
    ("Closer attempt 1", [0, 0.2, 0, -2.6, 0, 2.2, 0.785]),
    ("Closer attempt 2", [0, 0.3, 0, -2.7, 0, 2.3, 0.785]),
    ("Closer attempt 3", [0, 0.4, 0, -2.8, 0, 2.4, 0.785]),
]

for name, config in close_configs:
    # Apply configuration
    for i, val in enumerate(config):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    
    mujoco.mj_forward(model, data)
    
    # Get end-effector info
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    
    xy_dist = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
    
    print(f"{name}:")
    print(f"  EE Position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  XY distance to red: {xy_dist:.3f}m")
    print(f"  X difference: {ee_pos[0] - red_pos[0]:.3f}m")
    print()

print("\nDEBUG INSIGHTS:")
print("1. Joint 6 at 0.785 rad (45°) IS rotating the gripper")
print("2. To get vertically above red block at X=0.05:")
print("   - Need positive joint 1 values (shoulder lift UP)")
print("   - Need more negative joint 3 (elbow bent MORE)")
print("   - This brings EE forward toward positive X")