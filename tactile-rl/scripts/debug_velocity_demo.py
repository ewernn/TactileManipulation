#!/usr/bin/env python3
"""
Debug velocity control by printing qpos values.
"""

import numpy as np
import mujoco

def main():
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    print("Debugging qpos values")
    print("=" * 70)
    print(f"Total qpos size: {model.nq}")
    print(f"Total controls: {model.nu}")
    print()
    
    # Find important indices
    finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    finger1_addr = model.jnt_qposadr[finger1_id]
    finger2_addr = model.jnt_qposadr[finger2_id]
    
    # Print initial qpos
    print("Initial qpos values:")
    print(f"qpos[0-6] (red block pose): {data.qpos[0:7]}")
    print(f"qpos[7-13] (blue block pose): {data.qpos[7:14]}")
    print(f"qpos[14-20] (robot joints 1-7): {data.qpos[14:21]}")
    print(f"qpos[{finger1_addr}] (finger1): {data.qpos[finger1_addr]:.4f}")
    print(f"qpos[{finger2_addr}] (finger2): {data.qpos[finger2_addr]:.4f}")
    print()
    
    # Get body positions
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    
    print("Initial body positions:")
    print(f"End effector: {data.xpos[ee_id]}")
    print(f"Red block: {data.xpos[red_id]}")
    print()
    
    # Test gripper control
    print("Testing gripper control (10 steps each):")
    print("-" * 50)
    
    # Open gripper
    print("\nSetting ctrl[7] = 255 (should open gripper)")
    for i in range(10):
        data.ctrl[7] = 255
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            print(f"  Step {i}: finger1={data.qpos[finger1_addr]:.4f}, finger2={data.qpos[finger2_addr]:.4f}")
    
    # Close gripper
    print("\nSetting ctrl[7] = 0 (should close gripper)")
    for i in range(10):
        data.ctrl[7] = 0
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            print(f"  Step {i}: finger1={data.qpos[finger1_addr]:.4f}, finger2={data.qpos[finger2_addr]:.4f}")
    
    # Check if blocks moved
    print("\nFinal positions:")
    print(f"Red block qpos: {data.qpos[0:7]}")
    print(f"Red block xpos: {data.xpos[red_id]}")
    
    # Test velocity control on joint 1
    print("\n" + "=" * 70)
    print("Testing velocity control on joint 1:")
    print("-" * 50)
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    dt = model.opt.timestep
    print(f"Timestep: {dt}")
    
    # Apply velocity to joint 1
    print("\nApplying velocity 0.1 rad/s to joint 1 for 20 steps:")
    joint1_start = data.qpos[14]
    print(f"Joint 1 start: {joint1_start:.4f}")
    
    for i in range(20):
        # Velocity control: integrate velocity
        current_pos = data.qpos[14]
        velocity = 0.1  # rad/s
        new_pos = current_pos + velocity * dt
        data.ctrl[0] = new_pos
        
        mujoco.mj_step(model, data)
        
        if i % 5 == 0:
            print(f"  Step {i}: joint1={data.qpos[14]:.4f}, ctrl[0]={data.ctrl[0]:.4f}")
    
    joint1_end = data.qpos[14]
    print(f"Joint 1 end: {joint1_end:.4f}")
    print(f"Total change: {joint1_end - joint1_start:.4f} rad")
    print(f"Expected change: {0.1 * dt * 20:.4f} rad")

if __name__ == "__main__":
    main()