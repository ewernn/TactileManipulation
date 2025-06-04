#!/usr/bin/env python3
"""
Verify positions across different methods.
"""

import numpy as np
import mujoco

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("Initial state:")
print(f"qpos length: {len(data.qpos)}")
print(f"Number of bodies: {model.nbody}")

# Method 1: Before any physics
print("\nMethod 1: After mj_resetDataKeyframe:")
mujoco.mj_resetDataKeyframe(model, data, 0)
red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
print(f"Red block ID: {red_id}, position: {data.xpos[red_id]}")
print(f"Blue block ID: {blue_id}, position: {data.xpos[blue_id]}")

# Method 2: After forward
print("\nMethod 2: After mj_forward:")
mujoco.mj_forward(model, data)
print(f"Red block position: {data.xpos[red_id]}")
print(f"Blue block position: {data.xpos[blue_id]}")

# Method 3: After physics steps
print("\nMethod 3: After 10 physics steps:")
for _ in range(10):
    mujoco.mj_step(model, data)
print(f"Red block position: {data.xpos[red_id]}")
print(f"Blue block position: {data.xpos[blue_id]}")

# Check qpos values for blocks
print("\nBlock qpos values:")
print(f"Red block qpos[0-6]: {data.qpos[0:7]}")
print(f"Blue block qpos[7-13]: {data.qpos[7:14]}")
print(f"Robot qpos[14-22]: {data.qpos[14:23]}")

# Let's also check what the keyframe contains
print("\nKeyframe 0 qpos:")
keyframe_qpos = model.key_qpos[0:23]
print(f"First 7 (red block): {keyframe_qpos[0:7]}")
print(f"Next 7 (blue block): {keyframe_qpos[7:14]}")
print(f"Robot joints: {keyframe_qpos[14:23]}")