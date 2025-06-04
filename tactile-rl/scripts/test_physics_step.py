#!/usr/bin/env python3
"""
Test what happens during physics step.
"""

import numpy as np
import mujoco

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initialize
mujoco.mj_resetDataKeyframe(model, data, 0)

RED_BLOCK_QPOS = 9

print("Initial red block position:", data.qpos[RED_BLOCK_QPOS:RED_BLOCK_QPOS+3])

# Move block
data.qpos[RED_BLOCK_QPOS] = 0.2  # Move to X=0.2
data.qpos[RED_BLOCK_QPOS+2] = 0.5  # Move to Z=0.5

print("After manual move:", data.qpos[RED_BLOCK_QPOS:RED_BLOCK_QPOS+3])

# Forward dynamics only
mujoco.mj_forward(model, data)
print("After mj_forward:", data.qpos[RED_BLOCK_QPOS:RED_BLOCK_QPOS+3])

# One physics step
mujoco.mj_step(model, data)
print("After mj_step:", data.qpos[RED_BLOCK_QPOS:RED_BLOCK_QPOS+3])

# Multiple steps
for i in range(10):
    mujoco.mj_step(model, data)
    print(f"After step {i+2}:", data.qpos[RED_BLOCK_QPOS:RED_BLOCK_QPOS+3])

# Check what's in the model
print("\nModel info:")
print(f"Gravity: {model.opt.gravity}")
print(f"Number of equality constraints: {model.neq}")
print(f"Number of joints: {model.njnt}")

# List all joints
print("\nAll joints:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = model.jnt_type[i]
    type_names = ["FREE", "BALL", "SLIDE", "HINGE"]
    print(f"  Joint {i}: {name} (type: {type_names[jtype] if jtype < 4 else 'UNKNOWN'})")