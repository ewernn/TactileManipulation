#!/usr/bin/env python3
"""
Debug script to understand block joint addressing.
"""

import numpy as np
import mujoco

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initialize to keyframe position
mujoco.mj_resetDataKeyframe(model, data, 0)

print("\n" + "="*60)
print("DEBUGGING BLOCK JOINTS")
print("="*60)

# Get body IDs
target_block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
block2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")

print(f"\nBody IDs:")
print(f"  target_block: {target_block_id}")
print(f"  block2: {block2_id}")

# Get joint info
print(f"\nJoint addressing:")
print(f"  model.nq (total DOF): {model.nq}")
print(f"  qpos array length: {len(data.qpos)}")

# Find the joints
target_joint_name = "cube:joint"  # From XML
block2_joint_name = "block2:joint"  # From XML

target_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target_joint_name)
block2_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, block2_joint_name)

print(f"\nJoint IDs:")
print(f"  {target_joint_name}: {target_joint_id}")
print(f"  {block2_joint_name}: {block2_joint_id}")

if target_joint_id >= 0:
    target_qpos_addr = model.jnt_qposadr[target_joint_id]
    print(f"\nTarget block qpos address: {target_qpos_addr}")
    print(f"  Position: [{data.qpos[target_qpos_addr]:.3f}, {data.qpos[target_qpos_addr+1]:.3f}, {data.qpos[target_qpos_addr+2]:.3f}]")
    print(f"  Quaternion: [{data.qpos[target_qpos_addr+3]:.3f}, {data.qpos[target_qpos_addr+4]:.3f}, {data.qpos[target_qpos_addr+5]:.3f}, {data.qpos[target_qpos_addr+6]:.3f}]")

if block2_joint_id >= 0:
    block2_qpos_addr = model.jnt_qposadr[block2_joint_id]
    print(f"\nBlock2 qpos address: {block2_qpos_addr}")
    print(f"  Position: [{data.qpos[block2_qpos_addr]:.3f}, {data.qpos[block2_qpos_addr+1]:.3f}, {data.qpos[block2_qpos_addr+2]:.3f}]")
    print(f"  Quaternion: [{data.qpos[block2_qpos_addr+3]:.3f}, {data.qpos[block2_qpos_addr+4]:.3f}, {data.qpos[block2_qpos_addr+5]:.3f}, {data.qpos[block2_qpos_addr+6]:.3f}]")

# Check body positions directly
print(f"\nBody positions (xpos):")
print(f"  target_block: {data.xpos[target_block_id]}")
print(f"  block2: {data.xpos[block2_id]}")

# Print full qpos to see structure
print(f"\nFull qpos array:")
for i in range(min(30, len(data.qpos))):
    print(f"  qpos[{i}]: {data.qpos[i]:.3f}")

print("\nNOTE: Freejoint has 7 DOF: [x, y, z, qw, qx, qy, qz]")
print("      The interactive script was using body_jntadr which might be wrong!")