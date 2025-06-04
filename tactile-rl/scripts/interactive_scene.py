#!/usr/bin/env python3
"""
Interactive scene viewer - move robot, table, and blocks freely without recordings.
"""

import numpy as np
import mujoco
import mujoco.viewer

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initialize to keyframe position
mujoco.mj_resetDataKeyframe(model, data, 0)  # Use "home" keyframe

print("\n" + "="*60)
print("INTERACTIVE SCENE VIEWER")
print("="*60)
print("\nControls:")
print("  Mouse: Left-drag to rotate, Right-drag to pan, Scroll to zoom")
print("  Space: Pause/unpause simulation")
print("  Tab: Show/hide UI panel")
print("  Ctrl+Left-click: Select and drag joints")
print("\nKeyboard shortcuts:")
print("  Q/A: Move red block left/right")
print("  W/S: Move red block forward/backward")
print("  E/D: Move red block up/down")
print("  1-7: Select joint to control")
print("  +/-: Increase/decrease selected joint angle")
print("  R: Reset to home position")
print("  P: Print current positions")
print("="*60 + "\n")

# Track selected joint
selected_joint = 0
joint_names = [f"joint{i+1}" for i in range(7)]

# Get body IDs
target_block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

# Get joint addresses for the red block
target_joint_id = model.body_jntadr[target_block_id]

def key_callback(keycode):
    global selected_joint
    
    if keycode == ord('R'):
        # Reset to home position
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print("Reset to home position")
    
    elif keycode == ord('P'):
        # Print current positions
        print("\n" + "-"*40)
        print("Current positions:")
        print(f"Robot joints (rad):")
        for i in range(7):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[i])
            if joint_id >= 0:
                addr = model.jnt_qposadr[joint_id]
                angle = data.qpos[addr]
                print(f"  {joint_names[i]}: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
        
        print(f"\nRed block position: [{data.qpos[target_joint_id]:.3f}, {data.qpos[target_joint_id+1]:.3f}, {data.qpos[target_joint_id+2]:.3f}]")
        
        # Get hand position and orientation
        hand_pos = data.xpos[hand_id]
        hand_quat = data.xquat[hand_id]
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, hand_quat)
        rotmat = rotmat.reshape(3, 3)
        gripper_direction = rotmat[:, 2]
        
        print(f"Hand position: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
        print(f"Gripper direction: [{gripper_direction[0]:.3f}, {gripper_direction[1]:.3f}, {gripper_direction[2]:.3f}]")
        print("-"*40)
    
    # Move red block
    elif keycode == ord('Q'):  # Left (-Y)
        data.qpos[target_joint_id + 1] -= 0.02
    elif keycode == ord('A'):  # Right (+Y)
        data.qpos[target_joint_id + 1] += 0.02
    elif keycode == ord('W'):  # Forward (+X)
        data.qpos[target_joint_id] += 0.02
    elif keycode == ord('S'):  # Backward (-X)
        data.qpos[target_joint_id] -= 0.02
    elif keycode == ord('E'):  # Up (+Z)
        data.qpos[target_joint_id + 2] += 0.02
    elif keycode == ord('D'):  # Down (-Z)
        data.qpos[target_joint_id + 2] -= 0.02
    
    # Select joint
    elif keycode >= ord('1') and keycode <= ord('7'):
        selected_joint = keycode - ord('1')
        print(f"Selected {joint_names[selected_joint]}")
    
    # Adjust selected joint
    elif keycode == ord('=') or keycode == ord('+'):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[selected_joint])
        if joint_id >= 0:
            addr = model.jnt_qposadr[joint_id]
            data.qpos[addr] += 0.1
            print(f"{joint_names[selected_joint]}: {data.qpos[addr]:.3f} rad ({np.degrees(data.qpos[addr]):.1f}°)")
    elif keycode == ord('-'):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[selected_joint])
        if joint_id >= 0:
            addr = model.jnt_qposadr[joint_id]
            data.qpos[addr] -= 0.1
            print(f"{joint_names[selected_joint]}: {data.qpos[addr]:.3f} rad ({np.degrees(data.qpos[addr]):.1f}°)")
    
    # Forward simulate to update
    mujoco.mj_forward(model, data)

# Launch interactive viewer
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    # Set camera to demo view
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat = [0.5, 0, 0.5]
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = -45
    
    # Run until closed
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()