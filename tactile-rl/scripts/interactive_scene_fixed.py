#!/usr/bin/env python3
"""
Fixed interactive scene viewer - properly handles pausing physics.
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

# Global pause state
paused = True  # Start paused so blocks don't fall immediately

print("\n" + "="*60)
print("INTERACTIVE SCENE VIEWER (FIXED)")
print("="*60)
print("\nControls:")
print("  Mouse: Left-drag to rotate, Right-drag to pan, Scroll to zoom")
print("  Space: Pause/unpause simulation (STARTS PAUSED)")
print("  Tab: Show/hide UI panel")
print("  Ctrl+Left-click: Select and drag joints")
print("\nKeyboard shortcuts:")
print("  Q/A: Move red block left/right (Y-axis)")
print("  W/S: Move red block forward/backward (X-axis)")
print("  E/D: Move red block up/down (Z-axis)")
print("  1-7: Select joint to control")
print("  +/-: Increase/decrease selected joint angle")
print("  R: Reset to home position")
print("  P: Print current positions")
print("="*60)
print("⚠️  SIMULATION STARTS PAUSED - Press Space to enable physics")
print("="*60 + "\n")

# Track selected joint
selected_joint = 0
joint_names = [f"joint{i+1}" for i in range(7)]

# Get body IDs
target_block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
block2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

# Get joint addresses for the blocks
target_joint_id = model.body_jntadr[target_block_id]
block2_joint_id = model.body_jntadr[block2_id]

def key_callback(keycode):
    global selected_joint, paused
    
    if keycode == ord(' '):
        # Toggle pause
        paused = not paused
        print(f"Simulation {'PAUSED' if paused else 'RUNNING'}")
    
    elif keycode == ord('R'):
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
        print(f"Blue block position: [{data.qpos[block2_joint_id]:.3f}, {data.qpos[block2_joint_id+1]:.3f}, {data.qpos[block2_joint_id+2]:.3f}]")
        
        # Get hand position and orientation
        hand_pos = data.xpos[hand_id]
        hand_quat = data.xquat[hand_id]
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, hand_quat)
        rotmat = rotmat.reshape(3, 3)
        gripper_direction = rotmat[:, 2]
        
        print(f"Hand position: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
        print(f"Gripper direction: [{gripper_direction[0]:.3f}, {gripper_direction[1]:.3f}, {gripper_direction[2]:.3f}]")
        print(f"Simulation: {'PAUSED' if paused else 'RUNNING'}")
        print("-"*40)
    
    # Move red block
    elif keycode == ord('Q'):  # Left (-Y)
        data.qpos[target_joint_id + 1] -= 0.05
        print(f"Red block Y: {data.qpos[target_joint_id + 1]:.3f}")
    elif keycode == ord('A'):  # Right (+Y)
        data.qpos[target_joint_id + 1] += 0.05
        print(f"Red block Y: {data.qpos[target_joint_id + 1]:.3f}")
    elif keycode == ord('W'):  # Forward (+X)
        data.qpos[target_joint_id] += 0.05
        print(f"Red block X: {data.qpos[target_joint_id]:.3f}")
    elif keycode == ord('S'):  # Backward (-X)
        data.qpos[target_joint_id] -= 0.05
        print(f"Red block X: {data.qpos[target_joint_id]:.3f}")
    elif keycode == ord('E'):  # Up (+Z)
        data.qpos[target_joint_id + 2] += 0.05
        print(f"Red block Z: {data.qpos[target_joint_id + 2]:.3f}")
    elif keycode == ord('D'):  # Down (-Z)
        data.qpos[target_joint_id + 2] -= 0.05
        print(f"Red block Z: {data.qpos[target_joint_id + 2]:.3f}")
    
    # Move blue block with shifted keys
    elif keycode == ord('I'):  # Blue Forward (+X)
        data.qpos[block2_joint_id] += 0.05
        print(f"Blue block X: {data.qpos[block2_joint_id]:.3f}")
    elif keycode == ord('K'):  # Blue Backward (-X)
        data.qpos[block2_joint_id] -= 0.05
        print(f"Blue block X: {data.qpos[block2_joint_id]:.3f}")
    elif keycode == ord('J'):  # Blue Left (-Y)
        data.qpos[block2_joint_id + 1] -= 0.05
        print(f"Blue block Y: {data.qpos[block2_joint_id + 1]:.3f}")
    elif keycode == ord('L'):  # Blue Right (+Y)
        data.qpos[block2_joint_id + 1] += 0.05
        print(f"Blue block Y: {data.qpos[block2_joint_id + 1]:.3f}")
    
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
    
    # Forward simulate to update positions
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
        # Only step physics if not paused
        if not paused:
            mujoco.mj_step(model, data)
        viewer.sync()