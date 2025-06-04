#!/usr/bin/env python3
"""Create a calibrated grasping demo with correct approach."""

import numpy as np
import mujoco
import cv2
import time
from pathlib import Path

def find_grasp_configuration(model, data, target_x, target_y, target_z):
    """Use iterative search to find joint config for target position."""
    robot_qpos_start = 14
    best_config = data.qpos[robot_qpos_start:robot_qpos_start+7].copy()
    best_dist = float('inf')
    
    hand_id = model.body("hand").id
    
    # Search parameters for each joint
    search_ranges = [
        (-0.5, 0.5, 0.1),     # joint1: base rotation
        (-1.5, -0.3, 0.2),    # joint2: shoulder
        (-0.5, 0.5, 0.2),     # joint3: upper arm
        (-2.5, -1.0, 0.2),    # joint4: elbow
        (-1.0, 0.0, 0.2),     # joint5: forearm
        (1.0, 2.0, 0.2),      # joint6: wrist 1
        (-0.5, 0.5, 0.2),     # joint7: wrist 2
    ]
    
    # Coarse search
    for j1 in np.arange(*search_ranges[0]):
        for j2 in np.arange(*search_ranges[1]):
            config = best_config.copy()
            config[0] = j1
            config[1] = j2
            
            data.ctrl[:7] = config
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            
            hand_pos = data.xpos[hand_id]
            dist = np.sqrt((hand_pos[0] - target_x)**2 + (hand_pos[1] - target_y)**2)
            
            if dist < best_dist:
                best_dist = dist
                best_config = config.copy()
                
    return best_config, best_dist

def create_calibrated_demo():
    """Create demo with calibrated approach."""
    
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
    
    # Get positions
    red_block_id = model.body("target_block").id  
    blue_block_id = model.body("block2").id
    hand_id = model.body("hand").id
    
    red_pos = data.xpos[red_block_id].copy()
    blue_pos = data.xpos[blue_block_id].copy()
    
    print(f"Red block: {red_pos}")
    print(f"Blue block: {blue_pos}")
    
    # Calculate approach positions
    gripper_reach = 0.1026
    approach_offset = 0.05  # Approach from slightly away
    
    # Target positions for different phases
    approach_x = red_pos[0] - gripper_reach - approach_offset
    approach_y = red_pos[1]
    approach_z = red_pos[2] + 0.15  # Above block
    
    grasp_x = red_pos[0] - gripper_reach + 0.01
    grasp_y = red_pos[1]
    grasp_z = red_pos[2]
    
    print(f"\nApproach target: X={approach_x:.3f}, Y={approach_y:.3f}, Z={approach_z:.3f}")
    print(f"Grasp target: X={grasp_x:.3f}, Y={grasp_y:.3f}, Z={grasp_z:.3f}")
    
    # Get initial config
    robot_qpos_init = data.qpos[robot_qpos_start:robot_qpos_start+7].copy()
    
    # Create key configurations based on the scene
    # These are manually tuned for this specific robot and scene
    
    # Config 1: Turn to face blocks
    face_config = robot_qpos_init.copy()
    face_config[0] = -0.3  # Negative rotation to face +X direction
    
    # Config 2: Approach position (high above red block)
    approach_config = face_config.copy()
    approach_config[1] = -0.6   # Shoulder forward
    approach_config[2] = 0.2    # Upper arm
    approach_config[3] = -2.2   # Elbow bent
    approach_config[4] = -0.5   # Forearm
    approach_config[5] = 1.8    # Wrist
    
    # Config 3: Grasp position (at block height)
    grasp_config = approach_config.copy()
    grasp_config[1] = -1.0      # Lower shoulder
    grasp_config[3] = -1.8      # Adjust elbow
    grasp_config[4] = -0.3      # Forearm adjustment
    
    # Config 4: Lift
    lift_config = grasp_config.copy()
    lift_config[1] = -0.5       # Raise shoulder
    
    # Config 5: Move to blue
    blue_config = lift_config.copy()
    blue_config[0] = -0.6       # More rotation toward blue
    blue_config[2] = -0.2       # Adjust arm
    
    # Config 6: Stack
    stack_config = blue_config.copy()
    stack_config[1] = -0.9      # Lower to place
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Simulation parameters
    fps = 30
    dt = 1.0 / fps
    steps_per_frame = int(dt / model.opt.timestep)
    
    # Phases
    phases = [
        ("init", 30, robot_qpos_init, robot_qpos_init, 255),
        ("turn_to_blocks", 60, robot_qpos_init, face_config, 255),
        ("approach", 90, face_config, approach_config, 255),
        ("descend", 60, approach_config, grasp_config, 255),
        ("grasp", 45, grasp_config, grasp_config, 0),
        ("lift", 60, grasp_config, lift_config, 0),
        ("move_to_blue", 90, lift_config, blue_config, 0),
        ("place", 60, blue_config, stack_config, 0),
        ("release", 30, stack_config, stack_config, 255),
        ("retreat", 60, stack_config, face_config, 255),
    ]
    
    # Render loop
    frames = []
    
    for phase_name, duration, start_config, end_config, gripper_val in phases:
        print(f"\n{'='*40}")
        print(f"Phase: {phase_name}")
        
        for i in range(duration):
            # Smooth interpolation
            if duration > 1:
                t = i / (duration - 1)
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
            red_pos_current = data.xpos[red_block_id].copy()
            
            # Render
            renderer.update_scene(data, camera="demo_cam")
            pixels = renderer.render()
            
            # Add overlay
            frame = pixels.copy()
            cv2.putText(frame, "CALIBRATED BLOCK STACKING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Phase: {phase_name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Gripper: {'OPEN' if gripper_val > 128 else 'CLOSED'}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Position info
            cv2.putText(frame, f"Hand: X={hand_pos[0]:.3f} Y={hand_pos[1]:.3f} Z={hand_pos[2]:.3f}", 
                       (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Target X: {grasp_x:.3f}", (10, 460), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            frames.append(frame)
            
            # Save key frames
            if i == duration - 1 and phase_name in ["turn_to_blocks", "approach", "grasp", "lift", "release"]:
                cv2.imwrite(str(output_dir / f"key_{phase_name}.png"), frame)
            
            # Progress
            if i % 30 == 0:
                dist_to_target = abs(hand_pos[0] - grasp_x)
                print(f"  Frame {i}: Hand X={hand_pos[0]:.3f}, Distance to target: {dist_to_target:.3f}m")
                if phase_name == "grasp":
                    print(f"    Red block: {red_pos_current}")
    
    # Check final state
    mujoco.mj_forward(model, data)
    red_final = data.xpos[red_block_id].copy()
    blue_final = data.xpos[blue_block_id].copy()
    
    print(f"\n{'='*50}")
    print("RESULTS:")
    print(f"{'='*50}")
    print(f"Red moved: {red_final - red_pos}")
    print(f"Red height above blue: {red_final[2] - blue_final[2]:.3f}m")
    print(f"Success: {'YES' if red_final[2] > blue_final[2] + 0.03 else 'NO'}")
    
    # Save video
    if frames:
        video_path = output_dir / "calibrated_stacking.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"\nVideo: {video_path}")
    
    return str(output_dir)

if __name__ == "__main__":
    output_dir = create_calibrated_demo()
    print("\nDone!")