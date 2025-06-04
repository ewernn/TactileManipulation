#!/usr/bin/env python3
"""Debug version of geometry-aware demo to examine robot positions and velocities."""

# Skip OpenGL setup for debugging

import numpy as np
import mujoco
import cv2
import time
from pathlib import Path

def create_debug_demo():
    """Create a debug demo to examine why robot doesn't move."""
    
    # Load model
    xml_path = "/Users/ewern/Desktop/code/TactileManipulation/tactile-rl/franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Get block positions
    red_block_id = model.body("target_block").id
    blue_block_id = model.body("block2").id
    red_pos_init = data.xpos[red_block_id].copy()
    blue_pos_init = data.xpos[blue_block_id].copy()
    
    print(f"Initial positions:")
    print(f"Red block (target_block): {red_pos_init}")
    print(f"Blue block (block2): {blue_pos_init}")
    
    # Get hand position
    hand_id = model.body("hand").id
    hand_pos_init = data.xpos[hand_id].copy()
    print(f"Hand position: {hand_pos_init}")
    
    # Gripper geometry from analysis
    gripper_reach = 0.1026
    ideal_fingertip_x = red_pos_init[0] + 0.01
    ideal_hand_x = ideal_fingertip_x - gripper_reach
    
    print(f"\nTarget calculations:")
    print(f"Gripper reach: {gripper_reach}m")
    print(f"Ideal fingertip X: {ideal_fingertip_x}")
    print(f"Ideal hand X: {ideal_hand_x}")
    print(f"Current hand X: {hand_pos_init[0]}")
    print(f"Need to move: {ideal_hand_x - hand_pos_init[0]}m")
    
    # Simulation parameters
    fps = 30
    dt = 1.0 / fps
    steps_per_frame = int(dt / model.opt.timestep)
    
    print(f"\nSimulation parameters:")
    print(f"Model timestep: {model.opt.timestep}")
    print(f"Steps per frame: {steps_per_frame}")
    print(f"Control dim: {model.nu}")
    print(f"Actuator types: {[model.actuator(i).name for i in range(model.nu)]}")
    
    # Phase definitions
    phases = [
        ("init", 30),          # 1 second
        ("move_x", 60),        # 2 seconds to move in X
        ("grasp", 30),         # 1 second
        ("check", 30),         # 1 second
    ]
    
    frame_count = 0
    
    for phase, duration in phases:
        print(f"\n{'='*50}")
        print(f"PHASE: {phase} (duration: {duration} frames)")
        print(f"{'='*50}")
        
        for i in range(duration):
            # Get current state
            hand_pos = data.xpos[hand_id].copy()
            hand_vel = data.cvel[hand_id][:3].copy()  # Linear velocity
            qpos = data.qpos[:7].copy()  # Joint positions
            qvel = data.qvel[:7].copy()  # Joint velocities
            
            # Set control based on phase
            control = np.zeros(model.nu)
            
            if phase == "init":
                # Keep current position with gripper open
                control[:7] = qpos  # Position control
                control[7] = 255    # Gripper open
                
            elif phase == "move_x":
                # Try to move in X direction
                desired_vel = np.zeros(7)
                # Simple velocity in joint 1 (base rotation) to move X
                desired_vel[0] = -0.1  # Negative to move in +X direction
                control[:7] = desired_vel
                control[7] = 255
                
            elif phase == "grasp":
                # Close gripper
                control[:7] = qpos  # Hold position
                control[7] = 0      # Close gripper
                
            elif phase == "check":
                # Hold position
                control[:7] = qpos
                control[7] = 0
            
            # Apply control
            data.ctrl[:] = control
            
            # Print debug info every 10 frames
            if i % 10 == 0:
                print(f"\nFrame {i}/{duration}:")
                print(f"  Hand pos: X={hand_pos[0]:.4f}, Y={hand_pos[1]:.4f}, Z={hand_pos[2]:.4f}")
                print(f"  Hand vel: X={hand_vel[0]:.4f}, Y={hand_vel[1]:.4f}, Z={hand_vel[2]:.4f}")
                print(f"  Joint pos: {qpos}")
                print(f"  Joint vel: {qvel}")
                print(f"  Control: {control}")
                print(f"  Distance to target X: {ideal_hand_x - hand_pos[0]:.4f}m")
            
            # Step simulation
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            
            frame_count += 1
    
    # Final check
    print(f"\n{'='*50}")
    print(f"FINAL STATE:")
    print(f"{'='*50}")
    
    hand_pos_final = data.xpos[hand_id].copy()
    red_pos_final = data.xpos[red_block_id].copy()
    
    print(f"Hand final position: {hand_pos_final}")
    print(f"Hand moved: {hand_pos_final - hand_pos_init}")
    print(f"Red block final: {red_pos_final}")
    print(f"Red block moved: {red_pos_final - red_pos_init}")
    
    # Check actuator configuration
    print(f"\n{'='*50}")
    print(f"ACTUATOR CONFIGURATION:")
    print(f"{'='*50}")
    
    for i in range(model.nu):
        act = model.actuator(i)
        trntype = act.trntype
        print(f"\nActuator {i} ({model.actuator(i).name}):")
        print(f"  Type: {['none', 'joint', 'jointinparent', 'slider', 'site'][trntype]}")
        print(f"  Control range: {act.ctrlrange}")
        print(f"  Force range: {act.forcerange}")
        
        # Get transmission info
        if i < 7:  # Joint actuators
            joint_id = model.actuator_trnid[i, 0]
            joint_name = model.joint(joint_id).name
            print(f"  Controls joint: {joint_name}")

if __name__ == "__main__":
    create_debug_demo()