#!/usr/bin/env python3
"""Simple position control demo with debugging."""

import numpy as np
import mujoco
import cv2
import time
from pathlib import Path

def simple_position_demo():
    """Create a simple demo using position control."""
    
    # Create output directory
    timestamp = time.strftime("%I%M%p").lower().lstrip('0')
    output_dir = Path(f"/Users/ewern/Desktop/code/TactileManipulation/videos/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    xml_path = "/Users/ewern/Desktop/code/TactileManipulation/tactile-rl/franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Forward kinematics to get actual positions
    mujoco.mj_forward(model, data)
    
    # IMPORTANT: Robot joints start at qpos[14]
    robot_qpos_start = 14
    
    # Get block positions after forward kinematics
    red_block_id = model.body("target_block").id  
    blue_block_id = model.body("block2").id
    red_pos_init = data.xpos[red_block_id].copy()
    blue_pos_init = data.xpos[blue_block_id].copy()
    
    print(f"Red block position: {red_pos_init}")
    print(f"Blue block position: {blue_pos_init}")
    
    # Get hand position
    hand_id = model.body("hand").id
    hand_pos_init = data.xpos[hand_id].copy()
    print(f"Initial hand position: {hand_pos_init}")
    
    # Check available cameras
    print("\nAvailable cameras:")
    for i in range(model.ncam):
        print(f"  {i}: {model.camera(i).name}")
    
    # Create renderer with first camera
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Get initial robot configuration
    robot_qpos_init = data.qpos[robot_qpos_start:robot_qpos_start+7].copy()
    print(f"\nInitial robot joints: {robot_qpos_init}")
    
    # Simulation parameters
    fps = 30
    dt = 1.0 / fps
    steps_per_frame = int(dt / model.opt.timestep)
    
    # Simple trajectory: just move joint 1 back and forth
    frames = []
    
    for phase in ["forward", "back"]:
        print(f"\nPhase: {phase}")
        
        for i in range(60):  # 2 seconds
            # Set target position
            robot_qpos_target = robot_qpos_init.copy()
            
            if phase == "forward":
                # Move joint 1 gradually
                robot_qpos_target[0] = robot_qpos_init[0] + (i/60) * 0.3
            else:
                # Move back
                robot_qpos_target[0] = robot_qpos_init[0] + 0.3 - (i/60) * 0.3
            
            # Set control
            data.ctrl[:7] = robot_qpos_target
            data.ctrl[7] = 255  # Gripper open
            
            # Step simulation
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            
            # Every 20 frames, print debug info
            if i % 20 == 0:
                hand_pos = data.xpos[hand_id]
                print(f"  Frame {i}: hand pos = {hand_pos}, joint1 = {data.qpos[robot_qpos_start]:.3f}")
            
            # Render frame (try different camera if available)
            try:
                if model.ncam > 2:
                    renderer.update_scene(data, camera=2)  # Try third camera
                else:
                    renderer.update_scene(data)  # Use default
                    
                pixels = renderer.render()
                
                # Add text overlay
                frame = pixels.copy()
                cv2.putText(frame, f"POSITION CONTROL TEST", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Phase: {phase}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                frames.append(frame)
                
            except Exception as e:
                print(f"Render error: {e}")
                break
    
    # Save video if we got frames
    if frames:
        video_path = output_dir / "simple_position_test.mp4"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"\nVideo saved to: {video_path}")
    
    # Final positions
    mujoco.mj_forward(model, data)
    hand_pos_final = data.xpos[hand_id]
    print(f"\nFinal hand position: {hand_pos_final}")
    print(f"Hand moved: {hand_pos_final - hand_pos_init}")

if __name__ == "__main__":
    simple_position_demo()