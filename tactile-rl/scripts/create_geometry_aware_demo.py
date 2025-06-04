#!/usr/bin/env python3
"""
Create a demonstration that properly accounts for gripper geometry.
"""

import numpy as np
import mujoco
import cv2
import os

class GeometryAwareExpertPolicy:
    """Expert policy that accounts for gripper geometry."""
    
    def __init__(self):
        self.phase = "init"
        self.step_count = 0
        self.phase_steps = {
            "init": 30,          # Initial wait
            "lift_to_safe": 60,  # Lift to safe height
            "align_x": 80,       # Move to correct X position
            "align_y": 60,       # Center on block Y
            "descend": 100,      # Carefully descend
            "final_adjust": 40,  # Final positioning
            "grasp": 30,         # Close gripper
            "lift": 60,          # Lift block
            "move_to_blue": 80,  # Move to blue block
            "place": 60,         # Lower onto blue
            "release": 30        # Open gripper
        }
        
        # Critical gripper geometry from our analysis
        self.gripper_reach = 0.1026  # How far fingertips extend beyond hand center
        self.finger_span = 0.084     # Y-axis span when open
        
        # Block parameters
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        self.block_half_size = 0.025
        
        # Calculate ideal hand position for grasping
        # Want fingertips slightly past block center
        ideal_fingertip_x = self.red_block_pos[0] + 0.01
        self.ideal_hand_x = ideal_fingertip_x - self.gripper_reach
        
        print(f"Gripper geometry: reach={self.gripper_reach:.3f}m, span={self.finger_span:.3f}m")
        print(f"Ideal hand X position: {self.ideal_hand_x:.3f}m")
        
    def get_action(self, model, data):
        """Get action accounting for gripper geometry."""
        action = np.zeros(8)
        
        # Get current positions
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        
        ee_pos = data.xpos[ee_id].copy()
        red_pos = data.xpos[red_id].copy()
        
        # Calculate where fingertips currently are
        current_fingertip_x = ee_pos[0] + self.gripper_reach
        
        if self.phase == "init":
            # Just wait
            action[:7] = 0.0
            action[7] = 255  # Gripper open
            
        elif self.phase == "lift_to_safe":
            # Lift to safe height (0.55m)
            z_error = 0.55 - ee_pos[2]
            if abs(z_error) > 0.01:
                action[1] = np.clip(-2.0 * z_error, -0.5, 0.5)
                action[3] = np.clip(1.5 * z_error, -0.4, 0.4)
            
            # Maintain wrist orientation
            action[5] = 0.1  # Slight wrist pitch to keep gripper horizontal
            action[7] = 255
            
        elif self.phase == "align_x":
            # Move to ideal X position
            x_error = self.ideal_hand_x - ee_pos[0]
            
            # Use shoulder and elbow to move in X
            if abs(x_error) > 0.005:
                # Positive error = need to move forward (+X)
                # Scale up for position control (was designed for velocity control)
                action[1] = np.clip(30.0 * x_error, -2.0, 2.0)  # 10x larger
                action[3] = np.clip(20.0 * x_error, -1.5, 1.5)  # 10x larger
            
            # Maintain height
            z_error = 0.55 - ee_pos[2]
            action[1] += np.clip(-1.0 * z_error, -0.2, 0.2)
            
            # Wrist control to keep gripper properly oriented
            action[5] = 0.1  # Maintain horizontal gripper
            action[6] = -0.05  # Slight adjustment for better approach angle
            
            action[7] = 255
            
            # Debug print
            if self.step_count % 20 == 0:
                print(f"  X alignment: hand={ee_pos[0]:.3f}, target={self.ideal_hand_x:.3f}, error={x_error:.3f}")
                print(f"  Fingertips at X={current_fingertip_x:.3f}")
                
        elif self.phase == "align_y":
            # Center on block Y position
            y_error = self.red_block_pos[1] - ee_pos[1]
            
            if abs(y_error) > 0.005:
                action[2] = np.clip(3.0 * y_error, -0.3, 0.3)
            
            # Maintain X and Z
            x_error = self.ideal_hand_x - ee_pos[0]
            action[0] = np.clip(2.0 * x_error, -0.1, 0.1)
            
            # Continue wrist control
            action[5] = 0.1  # Keep gripper horizontal
            
            action[7] = 255
            
        elif self.phase == "descend":
            # Carefully descend to grasp height
            target_z = self.red_block_pos[2]
            z_error = target_z - ee_pos[2]
            
            # Slow descent
            if z_error < -0.01:
                action[1] = np.clip(1.5 * z_error, -0.3, 0.3)
                action[3] = np.clip(-1.0 * z_error, -0.2, 0.2)
            
            # Maintain X-Y position
            x_error = self.ideal_hand_x - ee_pos[0]
            y_error = self.red_block_pos[1] - ee_pos[1]
            action[0] = np.clip(1.0 * x_error, -0.05, 0.05)
            action[2] = np.clip(1.0 * y_error, -0.05, 0.05)
            
            # Adjust wrist for grasp approach
            action[5] = 0.15  # Slight pitch down for better grasp
            action[6] = -0.1  # Fine-tune approach angle
            
            action[7] = 255
            
            # Safety check
            if self.step_count % 30 == 0:
                if current_fingertip_x > self.red_block_pos[0] - self.block_half_size - 0.005:
                    print(f"  ⚠️  Fingertips approaching block! X={current_fingertip_x:.3f}")
                    
        elif self.phase == "final_adjust":
            # Fine positioning before grasp
            x_error = self.ideal_hand_x - ee_pos[0]
            y_error = self.red_block_pos[1] - ee_pos[1]
            
            action[0] = np.clip(0.5 * x_error, -0.02, 0.02)
            action[2] = np.clip(0.5 * y_error, -0.02, 0.02)
            
            # Final wrist adjustments for optimal grasp
            action[5] = 0.2   # Pitch for grasp
            action[6] = -0.15 # Final orientation
            
            action[7] = 255
            
        elif self.phase == "grasp":
            # Close gripper
            action[:7] = 0.0
            action[7] = 0  # Close
            
        elif self.phase == "lift":
            # Lift to safe height
            z_error = 0.6 - ee_pos[2]
            if abs(z_error) > 0.01:
                action[1] = np.clip(-2.0 * z_error, -0.5, 0.5)
                action[3] = np.clip(1.5 * z_error, -0.4, 0.4)
            
            # Maintain wrist position during lift
            action[5] = 0.1  # Return to more neutral position
            
            action[7] = 0  # Keep closed
            
        elif self.phase == "move_to_blue":
            # Move to blue block position
            target_x = self.blue_block_pos[0] + 0.01 - self.gripper_reach
            target_y = self.blue_block_pos[1]
            
            x_error = target_x - ee_pos[0]
            y_error = target_y - ee_pos[1]
            
            # Movement
            action[0] = np.clip(2.0 * x_error, -0.3, 0.3)
            action[2] = np.clip(3.0 * y_error, -0.4, 0.4)
            
            # Maintain height
            z_error = 0.6 - ee_pos[2]
            action[1] += np.clip(-1.0 * z_error, -0.2, 0.2)
            
            # Keep wrist stable during transport
            action[5] = 0.05
            
            action[7] = 0  # Keep closed
            
        elif self.phase == "place":
            # Lower onto blue block
            place_z = self.blue_block_pos[2] + 0.04  # Account for block heights
            z_error = place_z - ee_pos[2]
            
            if z_error < -0.01:
                action[1] = np.clip(1.5 * z_error, -0.3, 0.3)
                action[3] = np.clip(-1.0 * z_error, -0.2, 0.2)
            
            # Prepare wrist for release
            action[5] = 0.15
            
            action[7] = 0  # Keep closed
            
        elif self.phase == "release":
            # Open gripper and lift slightly
            action[1] = -0.2
            action[7] = 255  # Open
        
        # Phase transition
        self.step_count += 1
        if self.step_count >= self.phase_steps[self.phase]:
            phases = list(self.phase_steps.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.step_count = 0
                print(f"\n>>> Transitioning to: {self.phase}")
                
        return action

def main():
    """Create geometry-aware demonstration."""
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set to proper initial configuration with wrist control
    # IMPORTANT: Joint 0 = 0 to face +X (toward workspace)
    joint_vals = [
        0,        # Joint 0: Base rotation (0 = face +X workspace)
        -0.1,     # Joint 1: Shoulder lift (slight down for reach)
        0,        # Joint 2: Shoulder rotation
        -2.0,     # Joint 3: Elbow bent
        0,        # Joint 4: Wrist roll neutral
        1.5,      # Joint 5: Wrist pitch
        0         # Joint 6: Wrist rotation
    ]
    for i, val in enumerate(joint_vals):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    data.ctrl[7] = 255
    
    mujoco.mj_forward(model, data)
    
    # Initialize policy
    expert = GeometryAwareExpertPolicy()
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video frames
    frames = []
    video_fps = 30
    
    print("\nCreating geometry-aware demonstration...")
    print("=" * 70)
    
    # Get IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Run demo
    total_steps = sum(expert.phase_steps.values())
    dt = model.opt.timestep
    
    for step in range(total_steps):
        # Get action
        action = expert.get_action(model, data)
        
        # Apply action (POSITION CONTROL - actuators are position servos!)
        # The action contains velocity commands, but we need to integrate to positions
        for i in range(7):
            # Get current target position (not actual position!)
            current_target = data.ctrl[i]
            # Integrate velocity to get new target
            new_target = current_target + action[i] * dt
            # Apply control
            data.ctrl[i] = new_target
        data.ctrl[7] = action[7]
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Print status periodically with positions and velocities
        if step % 30 == 0:  # More frequent logging
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            blue_pos = data.xpos[blue_id]
            fingertip_x = ee_pos[0] + expert.gripper_reach
            
            # Get velocities (cvel is 6D: 3 linear + 3 angular)
            ee_vel = data.cvel[ee_id][:3]  # Linear velocity of end-effector
            red_vel = data.cvel[red_id][:3]  # Linear velocity of red block
            blue_vel = data.cvel[blue_id][:3]  # Linear velocity of blue block
            
            print(f"\n{'='*80}")
            print(f"Step {step:4d} | Phase: {expert.phase:15s} | Time: {step*dt:.2f}s")
            print(f"{'='*80}")
            
            print(f"End-Effector:")
            print(f"  Position:  [{ee_pos[0]:7.4f}, {ee_pos[1]:7.4f}, {ee_pos[2]:7.4f}]")
            print(f"  Velocity:  [{ee_vel[0]:7.4f}, {ee_vel[1]:7.4f}, {ee_vel[2]:7.4f}] m/s")
            print(f"  Fingertips X: {fingertip_x:.4f}")
            
            print(f"Red Block:")
            print(f"  Position:  [{red_pos[0]:7.4f}, {red_pos[1]:7.4f}, {red_pos[2]:7.4f}]")
            print(f"  Velocity:  [{red_vel[0]:7.4f}, {red_vel[1]:7.4f}, {red_vel[2]:7.4f}] m/s")
            
            print(f"Blue Block:")
            print(f"  Position:  [{blue_pos[0]:7.4f}, {blue_pos[1]:7.4f}, {blue_pos[2]:7.4f}]")
            print(f"  Velocity:  [{blue_vel[0]:7.4f}, {blue_vel[1]:7.4f}, {blue_vel[2]:7.4f}] m/s")
            
            # Joint positions
            joint_pos = [data.qpos[14 + i] for i in range(7)]
            print(f"Joint Positions: [{', '.join([f'{j:6.3f}' for j in joint_pos])}]")
            
            # Check if blocks moved
            if np.linalg.norm(red_pos[:2] - expert.red_block_pos[:2]) > 0.01:
                print(f"  ⚠️  Red block has moved!")
            if np.linalg.norm(red_vel) > 0.01:
                print(f"  ⚠️  Red block is moving! |v| = {np.linalg.norm(red_vel):.4f} m/s")
                
        # Render frame
        renderer.update_scene(data, camera="demo_cam")
        frame = renderer.render()
        
        # Overlays
        cv2.putText(frame, "GEOMETRY-AWARE GRASPING", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 2)
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Gripper info
        gripper_state = "OPEN" if action[7] > 100 else "CLOSED"
        color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
        cv2.putText(frame, f"Gripper: {gripper_state}", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Position info
        ee_pos = data.xpos[ee_id]
        fingertip_x = ee_pos[0] + expert.gripper_reach
        cv2.putText(frame, f"Hand X: {ee_pos[0]:.3f}, Fingertips X: {fingertip_x:.3f}", 
                    (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Progress bar
        progress = expert.step_count / expert.phase_steps[expert.phase]
        bar_width = int(400 * progress)
        cv2.rectangle(frame, (40, 280), (40 + bar_width, 320), (0, 255, 0), -1)
        cv2.rectangle(frame, (40, 280), (440, 320), (255, 255, 255), 3)
        
        frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    
    # Save video
    output_path = "../../videos/121pm/geometry_aware_grasping_demo.mp4"
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    print(f"\n✓ Video saved to: {output_path}")
    print(f"  Full path: {os.path.abspath(output_path)}")
    print(f"  Duration: {len(frames)/video_fps:.1f} seconds")
    print(f"  Resolution: {width}x{height}")
    
    # Also save key frames as images
    key_frames = {
        'start': frames[0],
        'aligned': frames[sum(list(expert.phase_steps.values())[:3])],  # After align_x
        'grasping': frames[sum(list(expert.phase_steps.values())[:6])],  # During grasp
        'lifted': frames[sum(list(expert.phase_steps.values())[:8])]   # After lift
    }
    
    for name, frame in key_frames.items():
        img_path = f"../../videos/121pm/key_frame_{name}.png"
        cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  Saved key frame: {name} -> {img_path}")

if __name__ == "__main__":
    main()