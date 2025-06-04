#!/usr/bin/env python3
"""
Simple test to understand the control issue.
"""

import numpy as np
import mujoco

def main():
    # Load model
    model = mujoco.MjModel.from_xml_path("../franka_emika_panda/panda_demo_scene.xml")
    data = mujoco.MjData(model)
    
    print("="*80)
    print("CONTROL SYSTEM DIAGNOSTIC")
    print("="*80)
    
    # Check actuator configuration
    print(f"\nModel Configuration:")
    print(f"  Number of actuators: {model.nu}")
    print(f"  Number of joints: {model.njnt}")
    print(f"  Timestep: {model.opt.timestep}")
    
    print(f"\nActuator Details:")
    for i in range(min(8, model.nu)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  Actuator {i}: {name}")
        
        # Get actuator info
        trntype = model.actuator_trntype[i]
        gaintype = model.actuator_gaintype[i]
        biastype = model.actuator_biastype[i]
        dyntype = model.actuator_dyntype[i]
        
        print(f"    Transmission: {trntype} (0=joint, 1=jointinparent, 2=slide, 3=site)")
        print(f"    Gain type: {gaintype} (0=fixed, 1=affine, 2=muscle, 3=user)")
        print(f"    Bias type: {biastype} (0=none, 1=affine, 2=muscle, 3=user)")
        print(f"    Dynamics: {dyntype} (0=none, 1=integrator, 2=filter, 3=muscle)")
        
        # Check gains
        gain_prm = model.actuator_gainprm[i]
        print(f"    Gain params: {gain_prm[:3]}")
        
        # Check control range
        ctrl_range = model.actuator_ctrlrange[i]
        print(f"    Control range: [{ctrl_range[0]:.3f}, {ctrl_range[1]:.3f}]")
    
    # Reset and test movement
    print("\n" + "="*80)
    print("MOVEMENT TEST")
    print("="*80)
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set initial position
    joint_vals = [0, -0.1, 0, -2.0, 0, 1.5, 0]
    for i, val in enumerate(joint_vals):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    
    mujoco.mj_forward(model, data)
    
    print(f"\nInitial state:")
    print(f"  Joint positions: {[f'{data.qpos[14+i]:.3f}' for i in range(7)]}")
    print(f"  Control values: {[f'{data.ctrl[i]:.3f}' for i in range(7)]}")
    
    # Test 1: Direct control
    print(f"\nTest 1: Setting control directly")
    data.ctrl[1] = -0.1 + 0.5  # Move joint 1 to position 0.4
    print(f"  Set ctrl[1] = {data.ctrl[1]:.3f}")
    
    # Step simulation
    for step in range(500):
        mujoco.mj_step(model, data)
        if step % 100 == 0:
            print(f"  Step {step}: pos={data.qpos[15]:.3f}, vel={data.qvel[15]:.3f}")
    
    # Test 2: Velocity control simulation
    print(f"\nTest 2: Simulating velocity control")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for i, val in enumerate(joint_vals):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    
    velocity_cmd = 0.5  # rad/s
    dt = model.opt.timestep
    
    print(f"  Velocity command: {velocity_cmd} rad/s")
    print(f"  Timestep: {dt}")
    
    for step in range(500):
        # Update control based on velocity
        current_pos = data.qpos[15]  # Joint 1
        target_pos = current_pos + velocity_cmd * dt
        data.ctrl[1] = target_pos
        
        mujoco.mj_step(model, data)
        
        if step % 100 == 0:
            print(f"  Step {step}: pos={data.qpos[15]:.3f}, vel={data.qvel[15]:.3f}, ctrl={data.ctrl[1]:.3f}")
    
    # Test 3: Check XML for clues
    print(f"\n" + "="*80)
    print("XML CONFIGURATION CHECK")
    print("="*80)
    
    # Look for any damping or other parameters
    print(f"\nJoint damping:")
    for i in range(7):
        joint_id = model.jnt_qposadr[i + 14]  # Skip non-panda joints
        damping = model.dof_damping[joint_id] if joint_id < model.nv else 0
        print(f"  Joint {i}: damping = {damping}")
    
    # Check for implicit damping
    print(f"\nIntegrator settings:")
    print(f"  integrator: {model.opt.integrator}")
    print(f"  impratio: {model.opt.impratio}")
    
    # Check solver settings
    print(f"\nSolver settings:")
    print(f"  solver: {model.opt.solver}")
    print(f"  iterations: {model.opt.iterations}")
    print(f"  tolerance: {model.opt.tolerance}")

if __name__ == "__main__":
    main()