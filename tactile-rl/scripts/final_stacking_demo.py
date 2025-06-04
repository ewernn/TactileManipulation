#!/usr/bin/env python3
"""Final working block stacking demo with correct orientation."""

import numpy as np
import mujoco
import cv2
import time
from pathlib import Path

def final_stacking_demo():
    """Create the final working demo."""
    
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
    
    # Get IDs
    red_block_id = model.body("target_block").id  
    blue_block_id = model.body("block2").id
    hand_id = model.body("hand").id
    
    # Get positions
    red_pos = data.xpos[red_block_id].copy()
    blue_pos = data.xpos[blue_block_id].copy()
    hand_pos_init = data.xpos[hand_id].copy()
    
    print(f"Initial positions:")
    print(f"  Robot hand: {hand_pos_init}")
    print(f"  Red block: {red_pos}")
    print(f"  Blue block: {blue_pos}")
    
    # Robot is at negative X, blocks are at positive X
    # Need positive base rotation to face blocks
    
    # Get initial config
    robot_qpos_init = data.qpos[robot_qpos_start:robot_qpos_start+7].copy()
    print(f"\nInitial joint config: {robot_qpos_init}")
    
    # Key configurations (manually tuned)
    
    # 1. Turn right to face blocks (positive rotation)
    face_blocks = robot_qpos_init.copy()
    face_blocks[0] = 0.8  # Positive rotation to face +X
    
    # 2. Extend arm toward red block
    reach_out = face_blocks.copy()
    reach_out[1] = -0.3   # Shoulder down/forward
    reach_out[2] = 0.3    # Upper arm out
    reach_out[3] = -2.3   # Elbow bent
    reach_out[4] = 0.0    # Forearm neutral
    reach_out[5] = 1.57   # Wrist 90 deg
    reach_out[6] = 0.8    # Wrist 2 adjust
    
    # 3. Move closer and lower
    approach_block = reach_out.copy()
    approach_block[0] = 1.0    # More base rotation
    approach_block[1] = -0.6   # Lower shoulder
    approach_block[3] = -2.0   # Adjust elbow
    
    # 4. Final grasp position
    grasp_pos = approach_block.copy()
    grasp_pos[0] = 1.1        # Fine tune base
    grasp_pos[1] = -0.8       # Lower more
    grasp_pos[3] = -1.8       # Elbow for grasp
    
    # 5. Lift configuration
    lift_pos = grasp_pos.copy()
    lift_pos[1] = -0.4        # Raise shoulder
    lift_pos[3] = -2.1        # Elbow up
    
    # 6. Move toward blue block
    over_blue = lift_pos.copy()
    over_blue[0] = 0.8        # Rotate toward blue
    over_blue[2] = 0.1        # Adjust upper arm
    
    # 7. Lower onto blue block
    stack_pos = over_blue.copy()
    stack_pos[1] = -0.7       # Lower shoulder
    stack_pos[3] = -1.9       # Elbow down
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Simulation parameters
    fps = 30
    dt = 1.0 / fps
    steps_per_frame = int(dt / model.opt.timestep)
    
    # Define phases
    phases = [
        ("Initial", 30, robot_qpos_init, robot_qpos_init, 255),
        ("Turn to face blocks", 60, robot_qpos_init, face_blocks, 255),
        ("Reach toward red", 60, face_blocks, reach_out, 255),
        ("Approach red block", 60, reach_out, approach_block, 255),
        ("Position for grasp", 45, approach_block, grasp_pos, 255),
        ("Close gripper", 30, grasp_pos, grasp_pos, 0),
        ("Lift red block", 60, grasp_pos, lift_pos, 0),
        ("Move over blue", 60, lift_pos, over_blue, 0),
        ("Lower to stack", 45, over_blue, stack_pos, 0),
        ("Release", 30, stack_pos, stack_pos, 255),
        ("Retreat", 60, stack_pos, face_blocks, 255),
    ]
    
    # Render loop
    frames = []
    
    for phase_name, duration, start_config, end_config, gripper_val in phases:
        print(f"\n{'='*50}")
        print(f"Phase: {phase_name}")
        
        for i in range(duration):
            # Smooth interpolation
            t = i / max(1, duration - 1)
            alpha = 0.5 * (1 - np.cos(np.pi * t))
            
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
            blue_pos_current = data.xpos[blue_block_id].copy()
            
            # Render from demo camera
            renderer.update_scene(data, camera="demo_cam")
            pixels = renderer.render()
            
            # Create frame with overlay
            frame = pixels.copy()
            
            # Title and phase
            cv2.putText(frame, "FRANKA PANDA BLOCK STACKING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"{phase_name}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Gripper state
            gripper_text = "OPEN" if gripper_val > 128 else "CLOSED"
            color = (0, 255, 255) if gripper_val > 128 else (0, 0, 255)
            cv2.putText(frame, f"Gripper: {gripper_text}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Positions
            cv2.putText(frame, f"Hand: [{hand_pos[0]:6.3f}, {hand_pos[1]:6.3f}, {hand_pos[2]:6.3f}]", 
                       (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Red:  [{red_pos_current[0]:6.3f}, {red_pos_current[1]:6.3f}, {red_pos_current[2]:6.3f}]", 
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            
            frames.append(frame)
            
            # Save key frames
            if (i == 0 and phase_name == "Initial") or \
               (i == duration - 1 and phase_name in ["Turn to face blocks", "Position for grasp", 
                                                     "Close gripper", "Lift red block", "Release"]):
                filename = f"frame_{phase_name.replace(' ', '_').lower()}.png"
                cv2.imwrite(str(output_dir / filename), frame)
            
            # Progress info
            if i % 20 == 0:
                print(f"  Frame {i:3d}: Hand X={hand_pos[0]:7.3f}, Red Z={red_pos_current[2]:7.3f}")
                
                # Extra info during grasp
                if "grasp" in phase_name.lower() or "close" in phase_name.lower():
                    dist_to_red = np.linalg.norm(hand_pos - red_pos_current)
                    print(f"            Distance to red: {dist_to_red:.3f}m")
    
    # Final check
    mujoco.mj_forward(model, data)
    red_final = data.xpos[red_block_id].copy()
    blue_final = data.xpos[blue_block_id].copy()
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Red block final position: {red_final}")
    print(f"Blue block final position: {blue_final}")
    print(f"Red height above blue: {red_final[2] - blue_final[2]:.4f}m")
    print(f"Horizontal offset: X={abs(red_final[0] - blue_final[0]):.4f}m, Y={abs(red_final[1] - blue_final[1]):.4f}m")
    
    # Check success
    success = red_final[2] > blue_final[2] + 0.03  # At least 3cm above
    print(f"\nStacking successful: {'YES!' if success else 'NO'}")
    
    # Save video
    if frames:
        video_path = output_dir / "block_stacking_final.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (640, 480))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        print(f"\nVideo saved: {video_path}")
        print(f"Key frames in: {output_dir}")
    
    return str(output_dir)

if __name__ == "__main__":
    output_dir = final_stacking_demo()
    print("\nDemo complete!")