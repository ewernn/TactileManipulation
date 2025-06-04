#!/usr/bin/env python3
"""
Debug and fix body ID issues.
"""

import numpy as np
import mujoco

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Reset
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

print("All bodies with IDs:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"  ID {i}: {name}")

print("\nSearching for blocks by name:")
for name in ["target_block", "block2", "table"]:
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    print(f"  {name}: ID = {id}")
    if id >= 0:
        print(f"    Position after forward: {data.xpos[id]}")

# Let's check qpos structure
print("\nFull qpos (length={})".format(len(data.qpos)))
for i in range(len(data.qpos)):
    print(f"  qpos[{i}] = {data.qpos[i]:.3f}")

# Now step physics and check again
print("\nAfter 10 physics steps:")
for _ in range(10):
    mujoco.mj_step(model, data)

# Find blocks by checking all body positions
print("\nAll body positions after physics:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and ("block" in name or "table" in name):
        print(f"  {name}: {data.xpos[i]}")