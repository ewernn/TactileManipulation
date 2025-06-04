#!/usr/bin/env python3
"""Create a working grasping demo using position control."""

import numpy as np
import mujoco
import cv2
import time
from pathlib import Path

def create_grasping_demo():
    """Create a grasping and stacking demo."""
    
    # Create output directory
    timestamp = time.strftime("%I%M%p").lower().lstrip('0')
    output_dir = Path(f"/Users/ewern/Desktop/code/TactileManipulation/videos/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    xml_path = "/Users/ewern/Desktop/code/TactileManipulation/tactile-rl/franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset and forward kinematics
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Robot joints start at qpos[14]
    robot_qpos_start = 14
    
    # Get initial positions
    red_block_id = model.body("target_block").id  
    blue_block_id = model.body("block2").id
    hand_id = model.body("hand").id
    
    red_pos_init = data.xpos[red_block_id].copy()
    blue_pos_init = data.xpos[blue_block_id].copy()
    hand_pos_init = data.xpos[hand_id].copy()
    
    print(f"Red block: {red_pos_init}")
    print(f"Blue block: {blue_pos_init}")
    print(f"Hand: {hand_pos_init}")
    
    # Gripper geometry
    gripper_reach = 0.1026
    ideal_hand_x = red_pos_init[0] + 0.01 - gripper_reach
    
    print(f"\nTarget hand X for grasping: {ideal_hand_x}")
    print(f"Current hand X: {hand_pos_init[0]}")
    print(f"Need to move: {ideal_hand_x - hand_pos_init[0]}m in X")
    
    # Get initial robot configuration
    robot_qpos_init = data.qpos[robot_qpos_start:robot_qpos_start+7].copy()
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Simulation parameters
    fps = 30
    dt = 1.0 / fps
    steps_per_frame = int(dt / model.opt.timestep)
    
    # Define key configurations through experimentation
    # These values are tuned for the specific scene
    
    # Configuration 1: Approach position (move forward and prepare)
    approach_config = robot_qpos_init.copy()
    approach_config[0] = 0.25   # Rotate base to move hand forward in X
    approach_config[1] = -0.9   # Lower shoulder
    approach_config[2] = 0.1    # Slight elbow adjustment
    approach_config[3] = -1.6   # Elbow flex
    approach_config[4] = -0.4   # Wrist adjustment
    
    # Configuration 2: Grasp position (lower to block height)
    grasp_config = approach_config.copy()
    grasp_config[1] = -1.1      # Lower shoulder more
    grasp_config[3] = -1.4      # Adjust elbow for grasp
    grasp_config[5] = 1.4       # Slight wrist rotation
    
    # Configuration 3: Lift position
    lift_config = grasp_config.copy()
    lift_config[1] = -0.7       # Raise shoulder
    lift_config[3] = -1.7       # Elbow adjustment
    
    # Configuration 4: Move to blue block
    place_config = lift_config.copy()
    place_config[0] = -0.05     # Rotate base toward blue block
    place_config[2] = -0.1      # Elbow adjustment
    
    # Configuration 5: Lower onto blue block
    stack_config = place_config.copy()
    stack_config[1] = -0.95     # Lower shoulder
    stack_config[3] = -1.5      # Elbow down
    
    # Define phases with smooth transitions
    phases = [
        ("init", 30, robot_qpos_init, robot_qpos_init, 255),
        ("approach", 90, robot_qpos_init, approach_config, 255),
        ("align", 45, approach_config, grasp_config, 255),
        ("grasp", 30, grasp_config, grasp_config, 0),
        ("lift", 60, grasp_config, lift_config, 0),
        ("move_over", 90, lift_config, place_config, 0),
        ("descend", 45, place_config, stack_config, 0),
        ("release", 30, stack_config, stack_config, 255),
        ("retreat", 60, stack_config, robot_qpos_init, 255),
    ]
    
    # Render loop
    frames = []
    
    for phase_name, duration, start_config, end_config, gripper_val in phases:
        print(f"\n{'='*40}")
        print(f"Phase: {phase_name} ({duration} frames)")
        
        for i in range(duration):
            # Smooth interpolation
            if duration > 1:
                t = i / (duration - 1)
                # Use cosine interpolation for smoother motion
                alpha = 0.5 * (1 - np.cos(np.pi * t))
            else:
                alpha = 1
                
            robot_qpos_target = start_config + alpha * (end_config - start_config)
            
            # Set control
            data.ctrl[:7] = robot_qpos_target
            data.ctrl[7] = gripper_val
            
            # Step simulation
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            
            # Get current state
            hand_pos = data.xpos[hand_id].copy()
            red_pos = data.xpos[red_block_id].copy()
            
            # Render
            renderer.update_scene(data, camera="side_cam")
            pixels = renderer.render()
            
            # Add overlay
            frame = pixels.copy()
            cv2.putText(frame, "FRANKA BLOCK STACKING DEMO", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Phase: {phase_name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Gripper: {'OPEN' if gripper_val > 128 else 'CLOSED'}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Debug info
            cv2.putText(frame, f"Hand: X={hand_pos[0]:.3f} Y={hand_pos[1]:.3f} Z={hand_pos[2]:.3f}", 
                       (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Red: X={red_pos[0]:.3f} Y={red_pos[1]:.3f} Z={red_pos[2]:.3f}", 
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
            
            frames.append(frame)
            
            # Save key frames
            if i == 0 or (i == duration - 1 and phase_name in ["approach", "grasp", "lift", "release"]):
                cv2.imwrite(str(output_dir / f"frame_{phase_name}_{i}.png"), frame)
            
            # Print progress every 30 frames
            if i % 30 == 0:
                print(f"  Frame {i}: Hand X={hand_pos[0]:.3f}, Red block Z={red_pos[2]:.3f}")
    
    # Final state check
    mujoco.mj_forward(model, data)
    red_pos_final = data.xpos[red_block_id].copy()
    blue_pos_final = data.xpos[blue_block_id].copy()
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS:")
    print(f"{'='*50}")
    print(f"Red block final: {red_pos_final}")
    print(f"Blue block final: {blue_pos_final}")
    print(f"Red block height above blue: {red_pos_final[2] - blue_pos_final[2]:.3f}m")
    print(f"Horizontal alignment: X_diff={red_pos_final[0] - blue_pos_final[0]:.3f}, Y_diff={red_pos_final[1] - blue_pos_final[1]:.3f}")
    
    # Save video
    if frames:
        video_path = output_dir / "block_stacking_demo.mp4"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"\nVideo saved to: {video_path}")
        print(f"Output directory: {output_dir}")
    
    return str(output_dir)

if __name__ == "__main__":
    output_dir = create_grasping_demo()
    print(f"\nDemo complete!")