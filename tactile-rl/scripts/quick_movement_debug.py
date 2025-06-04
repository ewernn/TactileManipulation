#!/usr/bin/env python3
"""Quick debug to check actuator configuration and control."""

import numpy as np
import mujoco

def quick_debug():
    # Load model
    xml_path = "/Users/ewern/Desktop/code/TactileManipulation/tactile-rl/franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    print("MODEL CONFIGURATION:")
    print(f"nu (control dim): {model.nu}")
    print(f"nq (position dim): {model.nq}")
    print(f"nv (velocity dim): {model.nv}")
    
    print("\nACTUATORS:")
    for i in range(model.nu):
        act = model.actuator(i)
        print(f"\nActuator {i}: {model.actuator(i).name}")
        print(f"  trntype: {act.trntype}")
        print(f"  ctrlrange: {act.ctrlrange}")
        
        # Check transmission
        if i < 7:  # Joint actuators
            joint_id = model.actuator_trnid[i, 0]
            joint_name = model.joint(joint_id).name
            print(f"  controls joint: {joint_name}")
    
    # Test simple movement
    print("\n\nTEST MOVEMENT:")
    
    # Initial positions
    hand_id = model.body("hand").id
    hand_pos_init = data.xpos[hand_id].copy()
    qpos_init = data.qpos[:7].copy()
    
    print(f"Initial hand pos: {hand_pos_init}")
    print(f"Initial joint pos: {qpos_init}")
    
    # Apply control
    print("\nApplying velocity control...")
    data.ctrl[:7] = [0.1, 0, 0, 0, 0, 0, 0]  # Move joint 0
    data.ctrl[7] = 255  # Gripper open
    
    # Step simulation
    for i in range(100):
        mujoco.mj_step(model, data)
    
    # Check final positions
    hand_pos_final = data.xpos[hand_id].copy()
    qpos_final = data.qpos[:7].copy()
    
    print(f"\nFinal hand pos: {hand_pos_final}")
    print(f"Hand moved: {hand_pos_final - hand_pos_init}")
    print(f"Final joint pos: {qpos_final}")
    print(f"Joint moved: {qpos_final - qpos_init}")

if __name__ == "__main__":
    quick_debug()