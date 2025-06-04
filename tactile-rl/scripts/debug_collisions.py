#!/usr/bin/env python3
"""
Debug script to check for collisions and constraints.
"""

import numpy as np
import mujoco
import mujoco.viewer

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initialize
mujoco.mj_resetDataKeyframe(model, data, 0)

# Force paused
paused = True

print("\n" + "="*60)
print("COLLISION DEBUGGER")
print("="*60)

# qpos addresses
RED_BLOCK_QPOS = 9
BLUE_BLOCK_QPOS = 16

def key_callback(keycode):
    global paused
    
    if keycode == ord(' '):
        paused = not paused
        print(f"\nSimulation {'PAUSED' if paused else 'RUNNING'}")
    
    elif keycode == ord('P'):
        print("\n" + "-"*50)
        print("Current state:")
        print(f"\nRed block qpos: [{data.qpos[RED_BLOCK_QPOS]:.3f}, {data.qpos[RED_BLOCK_QPOS+1]:.3f}, {data.qpos[RED_BLOCK_QPOS+2]:.3f}]")
        print(f"Blue block qpos: [{data.qpos[BLUE_BLOCK_QPOS]:.3f}, {data.qpos[BLUE_BLOCK_QPOS+1]:.3f}, {data.qpos[BLUE_BLOCK_QPOS+2]:.3f}]")
        
        # Check actual body positions
        red_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        blue_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        
        print(f"\nRed block xpos: {data.xpos[red_body_id]}")
        print(f"Blue block xpos: {data.xpos[blue_body_id]}")
        
        # Check for active contacts
        print(f"\nActive contacts: {data.ncon}")
        if data.ncon > 0:
            for i in range(min(5, data.ncon)):
                c = data.contact[i]
                geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                print(f"  Contact {i}: {geom1_name} <-> {geom2_name}")
        
        # Check constraints
        print(f"\nActive constraints: {data.nefc}")
        print("-"*50)
    
    # Test movements with detailed output
    elif keycode == ord('T'):  # Test red block movement
        print("\n--- Testing Red Block Movement ---")
        old_pos = data.qpos[RED_BLOCK_QPOS:RED_BLOCK_QPOS+3].copy()
        print(f"Before: {old_pos}")
        
        # Try to move it
        data.qpos[RED_BLOCK_QPOS] += 0.1
        print(f"Set to: {data.qpos[RED_BLOCK_QPOS]:.3f}")
        
        # Forward dynamics
        mujoco.mj_forward(model, data)
        print(f"After forward: {data.qpos[RED_BLOCK_QPOS]:.3f}")
        
        # One physics step
        if not paused:
            mujoco.mj_step(model, data)
            print(f"After step: {data.qpos[RED_BLOCK_QPOS]:.3f}")
    
    # Normal movement keys
    elif keycode == ord('W'):
        print(f"Moving red +X from {data.qpos[RED_BLOCK_QPOS]:.3f}", end="")
        data.qpos[RED_BLOCK_QPOS] += 0.05
        print(f" to {data.qpos[RED_BLOCK_QPOS]:.3f}")
    elif keycode == ord('S'):
        print(f"Moving red -X from {data.qpos[RED_BLOCK_QPOS]:.3f}", end="")
        data.qpos[RED_BLOCK_QPOS] -= 0.05
        print(f" to {data.qpos[RED_BLOCK_QPOS]:.3f}")
    elif keycode == ord('E'):
        print(f"Moving red +Z from {data.qpos[RED_BLOCK_QPOS+2]:.3f}", end="")
        data.qpos[RED_BLOCK_QPOS + 2] += 0.05
        print(f" to {data.qpos[RED_BLOCK_QPOS+2]:.3f}")
    
    # Check keyframe
    elif keycode == ord('K'):
        print("\n--- Keyframe Analysis ---")
        print(f"Keyframe 'home' expects qpos length: {model.nq}")
        print(f"Current qpos length: {len(data.qpos)}")
        print(f"Keyframe sets red block to: {model.key_qpos[0][RED_BLOCK_QPOS:RED_BLOCK_QPOS+3]}")
        print(f"Keyframe sets blue block to: {model.key_qpos[0][BLUE_BLOCK_QPOS:BLUE_BLOCK_QPOS+3]}")
    
    elif keycode == ord('R'):
        print("\nRESETTING to keyframe...")
        mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Always forward after changes
    mujoco.mj_forward(model, data)

print("\nKeys:")
print("  Space: Toggle physics")
print("  P: Print positions and contacts")
print("  T: Test red block movement")
print("  K: Analyze keyframe")
print("  W/S/E: Move red block X/X/Z")
print("  R: Reset to keyframe")
print("\n⚠️  Watch the console output!")

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