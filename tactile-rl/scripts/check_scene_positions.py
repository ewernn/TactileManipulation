#!/usr/bin/env python3
"""
Check exact positions of all objects in the scene.
"""

import numpy as np
import mujoco

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Reset to initial state
mujoco.mj_resetData(model, data)

print("\n" + "="*60)
print("SCENE POSITION CHECK")
print("="*60)

# Get all body names and positions
print("\nAll bodies and their positions:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name:
        pos = data.xpos[i]
        print(f"{name:20s}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")

print("\n" + "-"*60)

# Check specific objects
table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
red_block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
blue_block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")

print("\nKey objects:")
print(f"Table position:      {data.xpos[table_id]}")
print(f"Red block position:  {data.xpos[red_block_id]}")
print(f"Blue block position: {data.xpos[blue_block_id]}")

# Get table geometry info
print("\nTable geometry:")
table_geoms = []
for i in range(model.ngeom):
    geom_body = model.geom_bodyid[i]
    if geom_body == table_id:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        pos = data.geom_xpos[i]
        size = model.geom_size[i]
        print(f"  {name}: pos={pos}, size={size}")

# Check qpos values
print("\nBlock qpos values:")
print(f"Red block qpos[9-11]:   [{data.qpos[9]:.3f}, {data.qpos[10]:.3f}, {data.qpos[11]:.3f}]")
print(f"Blue block qpos[16-18]: [{data.qpos[16]:.3f}, {data.qpos[17]:.3f}, {data.qpos[18]:.3f}]")

# Run one physics step to see where things settle
print("\nAfter 1 physics step:")
mujoco.mj_step(model, data)
print(f"Red block position:  {data.xpos[red_block_id]}")
print(f"Blue block position: {data.xpos[blue_block_id]}")

# Run more steps
for i in range(50):
    mujoco.mj_step(model, data)

print("\nAfter 50 physics steps:")
print(f"Red block position:  {data.xpos[red_block_id]}")
print(f"Blue block position: {data.xpos[blue_block_id]}")
print(f"Red block qpos:   [{data.qpos[9]:.3f}, {data.qpos[10]:.3f}, {data.qpos[11]:.3f}]")
print(f"Blue block qpos:  [{data.qpos[16]:.3f}, {data.qpos[17]:.3f}, {data.qpos[18]:.3f}]")

# Check for contacts
if data.ncon > 0:
    print(f"\nActive contacts: {data.ncon}")
    for i in range(min(5, data.ncon)):
        c = data.contact[i]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "Unknown"
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "Unknown"
        print(f"  Contact {i}: {geom1} <-> {geom2}")