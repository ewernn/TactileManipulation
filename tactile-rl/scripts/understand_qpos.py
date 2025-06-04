#!/usr/bin/env python3
"""Understand the qpos structure."""

import numpy as np
import mujoco

def understand_qpos():
    # Load model
    xml_path = "/Users/ewern/Desktop/code/TactileManipulation/tactile-rl/franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    print("MODEL INFO:")
    print(f"nq (position dim): {model.nq}")
    print(f"nbody: {model.nbody}")
    print(f"njnt: {model.njnt}")
    
    print("\nJOINTS:")
    for i in range(model.njnt):
        joint = model.joint(i)
        qpos_start = model.jnt_qposadr[i]
        qpos_size = model.jnt_qpos0.shape[0] if i == model.njnt - 1 else model.jnt_qposadr[i+1] - qpos_start
        
        print(f"\nJoint {i}: {model.joint(i).name}")
        print(f"  type: {joint.type}")
        print(f"  qpos range: [{qpos_start}:{qpos_start + qpos_size}]")
        print(f"  qpos values: {data.qpos[qpos_start:qpos_start + qpos_size]}")
    
    print("\n\nKEYFRAME:")
    print(f"Full qpos from keyframe: {data.qpos}")
    
    # Find robot joints
    print("\n\nROBOT JOINT MAPPING:")
    robot_joints = []
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        if "joint" in joint_name and any(str(n) in joint_name for n in range(1, 8)):
            qpos_idx = model.jnt_qposadr[i]
            robot_joints.append((joint_name, qpos_idx))
            print(f"{joint_name} -> qpos[{qpos_idx}] = {data.qpos[qpos_idx]}")
    
    # Check actuator to qpos mapping
    print("\n\nACTUATOR TO QPOS MAPPING:")
    for i in range(model.nu):
        if i < 7:  # Robot joints
            joint_id = model.actuator_trnid[i, 0]
            joint_name = model.joint(joint_id).name
            qpos_idx = model.jnt_qposadr[joint_id]
            print(f"Actuator {i} -> {joint_name} -> qpos[{qpos_idx}] = {data.qpos[qpos_idx]}")

if __name__ == "__main__":
    understand_qpos()