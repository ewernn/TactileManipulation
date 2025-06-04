#!/usr/bin/env python3
"""
Create balanced final demonstration video with proper joint angles.
- Joint 4: Small positive value to keep gripper at right height
- Joint 6: 45 degrees rotation for gripper orientation
"""

import numpy as np
import mujoco
import cv2
import os

class BalancedFinalDemoVideo:
    def __init__(self):
        # Load model
        self.xml_path = "../franka_emika_panda/panda_demo_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Renderer
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        
        # Get IDs
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.red_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        self.blue_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        
        # Table height + safety margin
        self.table_height = 0.4
        self.safety_margin = 0.02  # 2cm minimum
        self.min_z = self.table_height + self.safety_margin
        
        # Demonstration sequence with balanced configuration
        # Joint 4: 0.3 rad to maintain proper height
        # Joint 6: 0.785 rad (45°) for gripper rotation
        self.sequence = [
            # Phase, duration, [j0, j1, j2, j3, j4, j5, j6], gripper, description
            ("home", 60, [0, -0.4, 0, -2.1, 0.3, 1.65, 0.785], 255, "Home - Balanced Height"),
            ("approach", 80, [0, -0.2, 0, -2.35, 0.3, 1.85, 0.785], 255, "Approach Block"),
            ("align_x", 60, [0, -0.05, 0, -2.45, 0.3, 1.95, 0.785], 255, "Align X Position"),
            ("align_y", 60, [0, -0.05, 0, -2.45, 0.3, 1.95, 0.785], 255, "Fine Y Alignment"),
            ("descend_safe", 80, [0, 0.05, 0, -2.5, 0.3, 2.0, 0.785], 255, "Descend Safely"),
            ("grasp", 80, [0, 0.05, 0, -2.5, 0.3, 2.0, 0.785], 0, "Close Gripper"),
            ("lift_initial", 60, [0, -0.1, 0, -2.4, 0.3, 1.9, 0.785], 0, "Initial Lift"),
            ("lift_clear", 80, [0, -0.3, 0, -2.2, 0.3, 1.75, 0.785], 0, "Lift Clear"),
            ("rotate_base", 120, [-0.5, -0.3, 0, -2.2, 0.3, 1.75, 0.785], 0, "Rotate to Blue"),
            ("position_blue", 80, [-0.5, -0.05, 0, -2.45, 0.3, 1.95, 0.785], 0, "Above Blue Block"),
            ("descend_place", 80, [-0.5, 0.05, 0, -2.5, 0.3, 2.0, 0.785], 0, "Descend to Place"),
            ("release", 60, [-0.5, 0.05, 0, -2.5, 0.3, 2.0, 0.785], 255, "Release Block"),
            ("retreat", 80, [-0.5, -0.2, 0, -2.3, 0.3, 1.85, 0.785], 255, "Retreat"),
            ("return", 100, [0, -0.4, 0, -2.1, 0.3, 1.65, 0.785], 255, "Return Home"),
        ]
        
    def interpolate_configs(self, config1, config2, alpha):
        """Smooth interpolation with ease-in-out."""
        # Ease-in-out function
        if alpha < 0.5:
            t = 2 * alpha * alpha
        else:
            t = 1 - pow(-2 * alpha + 2, 2) / 2
        
        return [(1-t)*c1 + t*c2 for c1, c2 in zip(config1, config2)]
    
    def get_gripper_info(self):
        """Get comprehensive gripper information."""
        # Get gripper orientation
        hand_quat = self.data.xquat[self.ee_id].copy()
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, hand_quat)
        rotmat = rotmat.reshape(3, 3)
        
        # Gripper axes
        gripper_x = rotmat[:, 0]  # Right
        gripper_y = rotmat[:, 1]  # Up  
        gripper_z = rotmat[:, 2]  # Forward (pointing direction)
        
        # Get all gripper-related body positions to find lowest point
        gripper_bodies = ["hand", "left_finger", "right_finger"]
        lowest_z = float('inf')
        
        for body_name in gripper_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                z_pos = self.data.xpos[body_id][2]
                lowest_z = min(lowest_z, z_pos)
        
        # Table clearance
        table_clearance = lowest_z - self.table_height
        
        # Calculate gripper opening
        left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        
        if left_finger_id != -1 and right_finger_id != -1:
            gripper_width = np.linalg.norm(
                self.data.xpos[left_finger_id] - self.data.xpos[right_finger_id]
            )
        else:
            gripper_width = 0.08
        
        return gripper_z, table_clearance, gripper_width, lowest_z, gripper_x, gripper_y
    
    def run(self):
        """Create the balanced final demonstration video."""
        print("Creating BALANCED FINAL demonstration video...")
        print("- Joint 4: 0.3 rad for balanced height")
        print("- Joint 6: 0.785 rad (45°) for gripper rotation")
        print("- Maintaining safe clearance above table")
        print("=" * 70)
        
        # Reset simulation
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Set initial configuration
        initial_config = self.sequence[0][2]
        for i, val in enumerate(initial_config):
            self.data.qpos[14 + i] = val
            self.data.ctrl[i] = val
        self.data.ctrl[7] = 255
        
        mujoco.mj_forward(self.model, self.data)
        
        # Video settings
        frames = []
        fps = 30
        
        # Statistics tracking
        min_clearance_overall = float('inf')
        below_table_count = 0
        
        # Execute sequence
        for phase_idx, (phase_name, duration, target_config, gripper_cmd, description) in enumerate(self.sequence):
            print(f"\nPhase: {phase_name} - {description}")
            
            # Get current config
            current_config = [self.data.qpos[14 + i] for i in range(7)]
            
            # Interpolate over duration
            for frame in range(duration):
                # Calculate interpolation
                alpha = frame / duration
                
                # Interpolate joint positions
                if phase_idx == 0:
                    config = target_config
                else:
                    config = self.interpolate_configs(current_config, target_config, alpha)
                
                # Apply configuration
                for i, val in enumerate(config):
                    self.data.ctrl[i] = val
                self.data.ctrl[7] = gripper_cmd
                
                # Step physics
                mujoco.mj_step(self.model, self.data)
                
                # Get info
                ee_pos = self.data.xpos[self.ee_id]
                red_pos = self.data.xpos[self.red_id]
                blue_pos = self.data.xpos[self.blue_id]
                gripper_forward, table_clearance, gripper_width, lowest_z, gripper_x, gripper_y = self.get_gripper_info()
                
                # Track stats
                if table_clearance < 0:
                    below_table_count += 1
                min_clearance_overall = min(min_clearance_overall, table_clearance)
                
                # Render frame
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Add overlays
                # Title
                cv2.putText(frame_img, "PANDA ROBOT - BALANCED FINAL DEMONSTRATION", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Progress bar
                progress = (sum([s[1] for s in self.sequence[:phase_idx]]) + frame) / sum([s[1] for s in self.sequence])
                bar_width = int(1200 * progress)
                cv2.rectangle(frame_img, (40, 160), (40 + bar_width, 180), (0, 255, 0), -1)
                cv2.rectangle(frame_img, (40, 160), (1240, 180), (255, 255, 255), 2)
                
                # Info panel
                panel_y = 240
                line_height = 35
                
                # Positions
                cv2.putText(frame_img, "POSITIONS", (40, panel_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame_img, f"End-Effector: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]",
                           (40, panel_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Red Block:    [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]",
                           (40, panel_y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                
                # Gripper status
                cv2.putText(frame_img, "GRIPPER STATUS", (40, panel_y + 3*line_height + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Forward vector
                forward_color = (0, 255, 0) if gripper_forward[0] > 0.7 else (255, 255, 0)
                cv2.putText(frame_img, f"Forward Vector: [{gripper_forward[0]:5.2f}, {gripper_forward[1]:5.2f}, {gripper_forward[2]:5.2f}]",
                           (40, panel_y + 4*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, forward_color, 2)
                
                # Joint angles
                joint_config = [self.data.qpos[14 + i] for i in range(7)]
                cv2.putText(frame_img, f"J4: {joint_config[4]:5.3f} rad ({np.degrees(joint_config[4]):5.1f}°)  "
                           f"J6: {joint_config[6]:5.3f} rad ({np.degrees(joint_config[6]):5.1f}°)",
                           (40, panel_y + 5*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
                
                # Table clearance with visual warnings
                if table_clearance < 0:
                    clearance_color = (0, 0, 255)
                    warning = " ⚠️ BELOW TABLE!"
                    cv2.rectangle(frame_img, (10, 10), (frame_img.shape[1]-10, frame_img.shape[0]-10), 
                                (0, 0, 255), 5)
                elif table_clearance < 0.02:
                    clearance_color = (0, 0, 255)
                    warning = " ⚠️ TOO LOW!"
                elif table_clearance < 0.05:
                    clearance_color = (0, 165, 255)
                    warning = " ⚠️ CLOSE"
                else:
                    clearance_color = (0, 255, 0)
                    warning = " ✓ SAFE"
                
                cv2.putText(frame_img, f"Table Clearance: {table_clearance:6.3f}m{warning}",
                           (40, panel_y + 6*line_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, clearance_color, 2)
                
                # Gripper state
                gripper_state = "OPEN" if gripper_cmd > 100 else "CLOSED"
                state_color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
                cv2.putText(frame_img, f"Gripper: {gripper_state} (Width: {gripper_width:.3f}m)",
                           (40, panel_y + 7*line_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
                
                # Distance to target
                dist_to_red = np.linalg.norm(ee_pos[:2] - red_pos[:2])
                cv2.putText(frame_img, f"XY Distance to Red: {dist_to_red:.3f}m",
                           (40, panel_y + 8*line_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                frames.append(frame_img)
                
                # Print progress
                if frame % 20 == 0:
                    print(f"  Frame {frame:3d}/{duration}: "
                          f"Z={ee_pos[2]:.3f}, Clear={table_clearance:5.3f}m, "
                          f"J4={joint_config[4]:.3f}")
        
        # Save video
        print("\n" + "=" * 70)
        print("Saving balanced final video...")
        
        # Create output directory if needed
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_balanced_final_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ BALANCED FINAL VIDEO SAVED!")
            print(f"📹 Path: {output_path}")
            print(f"⏱️  Duration: {len(frames)/fps:.1f} seconds")
            print(f"📐 Resolution: {width}x{height}")
            print(f"🎬 Total frames: {len(frames)}")
            print(f"\n📊 Final Statistics:")
            print(f"  Joint 4: 0.300 rad (17.2°)")
            print(f"  Joint 6: 0.785 rad (45.0°)")
            print(f"  Minimum table clearance: {min_clearance_overall:.3f}m")
            print(f"  Frames below table: {below_table_count}")
            
            if min_clearance_overall < 0:
                print("\n⚠️  WARNING: Gripper went below table level!")
            elif min_clearance_overall < 0.02:
                print("\n⚠️  WARNING: Very low clearance detected")
            else:
                print("\n✅ Safe clearance maintained throughout!")
            
        print("\n🎉 Balanced final demonstration complete!")

if __name__ == "__main__":
    creator = BalancedFinalDemoVideo()
    creator.run()