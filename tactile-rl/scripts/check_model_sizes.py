#!/usr/bin/env python3
"""Check the size requirements for the MuJoCo model."""

import mujoco
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xml_path = os.path.join(base_dir, "franka_emika_panda", "panda_demo_scene_rl.xml")

print(f"Loading model from: {xml_path}")

try:
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    print(f"\nModel information:")
    print(f"  Total qpos size: {model.nq}")
    print(f"  Total qvel size: {model.nv}")
    print(f"  Number of joints: {model.njnt}")
    print(f"  Number of bodies: {model.nbody}")
    print(f"  Number of actuators: {model.nu}")
    
    # List all joints
    print(f"\nJoints:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        joint_qposadr = model.jnt_qposadr[i]
        joint_dofadr = model.jnt_dofadr[i]
        
        # Get number of DOF for this joint
        if i < model.njnt - 1:
            n_qpos = model.jnt_qposadr[i+1] - joint_qposadr
            n_dof = model.jnt_dofadr[i+1] - joint_dofadr
        else:
            n_qpos = model.nq - joint_qposadr
            n_dof = model.nv - joint_dofadr
            
        type_names = ["free", "ball", "slide", "hinge"]
        type_name = type_names[joint_type] if joint_type < len(type_names) else "unknown"
        
        print(f"  {i:2d}: {joint_name:20s} type={type_name:6s} qpos_adr={joint_qposadr:2d} n_qpos={n_qpos} dof_adr={joint_dofadr:2d} n_dof={n_dof}")
    
    print(f"\nActuators:")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {act_name}")
        
    # Check existing keyframe
    if model.nkey > 0:
        print(f"\nExisting keyframes: {model.nkey}")
        for i in range(model.nkey):
            key_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            print(f"  Key {i}: {key_name}")
            
except Exception as e:
    print(f"Error: {e}")
    
    # Try loading base panda.xml to compare
    print("\n\nTrying base panda.xml for comparison...")
    try:
        base_xml = os.path.join(base_dir, "franka_emika_panda", "panda.xml")
        base_model = mujoco.MjModel.from_xml_path(base_xml)
        print(f"Base panda.xml has nq={base_model.nq}, nv={base_model.nv}")
    except Exception as e2:
        print(f"Base model error: {e2}")