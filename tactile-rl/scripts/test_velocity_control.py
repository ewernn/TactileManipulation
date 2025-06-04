#!/usr/bin/env python3
"""Test velocity control with correct qpos mapping."""

import numpy as np
import mujoco

def test_velocity_control():
    # Load model
    xml_path = "/Users/ewern/Desktop/code/TactileManipulation/tactile-rl/franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    print("QPOS STRUCTURE:")
    print(f"qpos[0:7]: block1 (target_block) pose")
    print(f"qpos[7:14]: block2 pose")
    print(f"qpos[14:21]: robot joints 1-7")
    print(f"qpos[21:23]: gripper joints")
    
    # Get initial state
    robot_qpos_start = 14  # Robot joints start at index 14
    robot_qpos = data.qpos[robot_qpos_start:robot_qpos_start+7].copy()
    
    print(f"\nInitial robot joint positions: {robot_qpos}")
    
    # Get hand position in world coordinates
    hand_id = model.body("hand").id
    hand_pos_init = data.xpos[hand_id].copy()
    print(f"Initial hand world position: {hand_pos_init}")
    
    # Test 1: Position control (should work)
    print("\n--- TEST 1: POSITION CONTROL ---")
    new_pos = robot_qpos.copy()
    new_pos[0] += 0.1  # Move joint1 by 0.1 rad
    
    data.ctrl[:7] = new_pos
    data.ctrl[7] = 255  # Gripper open
    
    # Step simulation
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    hand_pos_after = data.xpos[hand_id].copy()
    robot_qpos_after = data.qpos[robot_qpos_start:robot_qpos_start+7].copy()
    
    print(f"After position control:")
    print(f"  Robot joints: {robot_qpos_after}")
    print(f"  Joint1 moved: {robot_qpos_after[0] - robot_qpos[0]:.4f} rad")
    print(f"  Hand position: {hand_pos_after}")
    print(f"  Hand moved: {hand_pos_after - hand_pos_init}")
    
    # Reset for test 2
    mujoco.mj_resetDataKeyframe(model, data, 0)
    robot_qpos = data.qpos[robot_qpos_start:robot_qpos_start+7].copy()
    hand_pos_init = data.xpos[hand_id].copy()
    
    # Test 2: Velocity control 
    print("\n--- TEST 2: VELOCITY CONTROL ---")
    print("NOTE: Panda XML uses position actuators, not velocity!")
    print("To use velocity control, we need velocity actuators in the XML.")
    
    # Check actuator type
    for i in range(min(3, model.nu)):
        act = model.actuator(i)
        print(f"\nActuator {i} ({model.actuator(i).name}):")
        print(f"  dyntype: {act.dyntype}")
        print(f"  gaintype: {act.gaintype}")
        print(f"  biastype: {act.biastype}")
        
    print("\nFor velocity control, actuators need:")
    print("- dyntype=1 (integrator)")
    print("- OR proper velocity servos")
    print("\nCurrent actuators appear to be position servos.")

if __name__ == "__main__":
    test_velocity_control()