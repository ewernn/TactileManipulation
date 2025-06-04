#!/usr/bin/env python3
"""
Properly working interactive scene viewer with correct block control.
"""

import numpy as np
import mujoco
import mujoco.viewer

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initialize to keyframe position
mujoco.mj_resetDataKeyframe(model, data, 0)

# Global pause state
paused = True

print("\n" + "="*60)
print("INTERACTIVE BLOCKS - WORKING VERSION")
print("="*60)
print("\nControls:")
print("  Space: Pause/unpause simulation (STARTS PAUSED)")
print("  \nRed Block:")
print("  Q/A: Move left/right (Y-axis)")
print("  W/S: Move forward/backward (X-axis)")
print("  E/D: Move up/down (Z-axis)")
print("  \nBlue Block:")
print("  J/L: Move left/right (Y-axis)")
print("  I/K: Move forward/backward (X-axis)")
print("  U/H: Move up/down (Z-axis)")
print("  \nRobot:")
print("  1-7: Select joint")
print("  +/-: Adjust selected joint")
print("  \nOther:")
print("  R: Reset to home position")
print("  P: Print current positions")
print("="*60)
print("⚠️  STARTS PAUSED - Press Space to enable physics")
print("="*60 + "\n")

# Track selected joint
selected_joint = 0
joint_names = [f"joint{i+1}" for i in range(7)]

# Direct qpos addresses for blocks (from debug output)
RED_BLOCK_QPOS = 9   # Red block starts at qpos[9]
BLUE_BLOCK_QPOS = 16  # Blue block starts at qpos[16]

def key_callback(keycode):
    global selected_joint, paused
    
    if keycode == ord(' '):
        paused = not paused
        print(f"Simulation {'PAUSED' if paused else 'RUNNING'}")
    
    elif keycode == ord('R'):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print("Reset to home position")
    
    elif keycode == ord('P'):
        print("\n" + "-"*50)
        print("Current positions:")
        print(f"\nRobot joints (rad):")
        for i in range(7):
            print(f"  joint{i+1}: {data.qpos[i]:.3f} rad ({np.degrees(data.qpos[i]):.1f}°)")
        
        print(f"\nRed block:")
        print(f"  Position: [{data.qpos[RED_BLOCK_QPOS]:.3f}, {data.qpos[RED_BLOCK_QPOS+1]:.3f}, {data.qpos[RED_BLOCK_QPOS+2]:.3f}]")
        print(f"  Quaternion: [{data.qpos[RED_BLOCK_QPOS+3]:.3f}, {data.qpos[RED_BLOCK_QPOS+4]:.3f}, {data.qpos[RED_BLOCK_QPOS+5]:.3f}, {data.qpos[RED_BLOCK_QPOS+6]:.3f}]")
        
        print(f"\nBlue block:")
        print(f"  Position: [{data.qpos[BLUE_BLOCK_QPOS]:.3f}, {data.qpos[BLUE_BLOCK_QPOS+1]:.3f}, {data.qpos[BLUE_BLOCK_QPOS+2]:.3f}]")
        print(f"  Quaternion: [{data.qpos[BLUE_BLOCK_QPOS+3]:.3f}, {data.qpos[BLUE_BLOCK_QPOS+4]:.3f}, {data.qpos[BLUE_BLOCK_QPOS+5]:.3f}, {data.qpos[BLUE_BLOCK_QPOS+6]:.3f}]")
        
        print(f"\nSimulation: {'PAUSED' if paused else 'RUNNING'}")
        print("-"*50)
    
    # Red block controls
    elif keycode == ord('Q'):  # Left (-Y)
        data.qpos[RED_BLOCK_QPOS + 1] -= 0.05
        print(f"Red Y: {data.qpos[RED_BLOCK_QPOS + 1]:.3f}")
    elif keycode == ord('A'):  # Right (+Y)
        data.qpos[RED_BLOCK_QPOS + 1] += 0.05
        print(f"Red Y: {data.qpos[RED_BLOCK_QPOS + 1]:.3f}")
    elif keycode == ord('W'):  # Forward (+X)
        data.qpos[RED_BLOCK_QPOS] += 0.05
        print(f"Red X: {data.qpos[RED_BLOCK_QPOS]:.3f}")
    elif keycode == ord('S'):  # Backward (-X)
        data.qpos[RED_BLOCK_QPOS] -= 0.05
        print(f"Red X: {data.qpos[RED_BLOCK_QPOS]:.3f}")
    elif keycode == ord('E'):  # Up (+Z)
        data.qpos[RED_BLOCK_QPOS + 2] += 0.05
        print(f"Red Z: {data.qpos[RED_BLOCK_QPOS + 2]:.3f}")
    elif keycode == ord('D'):  # Down (-Z)
        data.qpos[RED_BLOCK_QPOS + 2] -= 0.05
        print(f"Red Z: {data.qpos[RED_BLOCK_QPOS + 2]:.3f}")
    
    # Blue block controls
    elif keycode == ord('J'):  # Left (-Y)
        data.qpos[BLUE_BLOCK_QPOS + 1] -= 0.05
        print(f"Blue Y: {data.qpos[BLUE_BLOCK_QPOS + 1]:.3f}")
    elif keycode == ord('L'):  # Right (+Y)
        data.qpos[BLUE_BLOCK_QPOS + 1] += 0.05
        print(f"Blue Y: {data.qpos[BLUE_BLOCK_QPOS + 1]:.3f}")
    elif keycode == ord('I'):  # Forward (+X)
        data.qpos[BLUE_BLOCK_QPOS] += 0.05
        print(f"Blue X: {data.qpos[BLUE_BLOCK_QPOS]:.3f}")
    elif keycode == ord('K'):  # Backward (-X)
        data.qpos[BLUE_BLOCK_QPOS] -= 0.05
        print(f"Blue X: {data.qpos[BLUE_BLOCK_QPOS]:.3f}")
    elif keycode == ord('U'):  # Up (+Z)
        data.qpos[BLUE_BLOCK_QPOS + 2] += 0.05
        print(f"Blue Z: {data.qpos[BLUE_BLOCK_QPOS + 2]:.3f}")
    elif keycode == ord('H'):  # Down (-Z)
        data.qpos[BLUE_BLOCK_QPOS + 2] -= 0.05
        print(f"Blue Z: {data.qpos[BLUE_BLOCK_QPOS + 2]:.3f}")
    
    # Robot joint controls
    elif keycode >= ord('1') and keycode <= ord('7'):
        selected_joint = keycode - ord('1')
        print(f"Selected joint{selected_joint + 1}")
    elif keycode == ord('=') or keycode == ord('+'):
        data.qpos[selected_joint] += 0.1
        print(f"joint{selected_joint + 1}: {data.qpos[selected_joint]:.3f} rad ({np.degrees(data.qpos[selected_joint]):.1f}°)")
    elif keycode == ord('-'):
        data.qpos[selected_joint] -= 0.1
        print(f"joint{selected_joint + 1}: {data.qpos[selected_joint]:.3f} rad ({np.degrees(data.qpos[selected_joint]):.1f}°)")
    
    # Update physics
    mujoco.mj_forward(model, data)

# Launch viewer
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat = [0.5, 0, 0.5]
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -20
    viewer.cam.azimuth = -45
    
    while viewer.is_running():
        if not paused:
            mujoco.mj_step(model, data)
        viewer.sync()