#!/usr/bin/env python3
"""
Create final demonstration video with proper gripper orientation and table clearance.
"""

import numpy as np
import mujoco
import cv2
import os

class FinalDemoVideo:
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
        
        # Table height
        self.table_height = 0.4
        
        # Key positions with proper height and orientation
        # Note: These are carefully tuned to:
        # 1. Keep gripper above table (Z > 0.45 for clearance)
        # 2. Point gripper forward with proper wrist angles
        # 3. Reach blocks at X=0.05
        self.sequence = [
            # Phase, duration, [j0, j1, j2, j3, j4, j5, j6], gripper, description
            # j5 = wrist pitch (1.571 = horizontal), j6 = wrist roll (0.785 = 45°)
            ("home", 60, [0, -0.5, 0, -1.8, 0, 1.571, 0.785], 255, "Home - Gripper Forward & High"),
            ("approach_high", 100, [0, -0.3, 0, -2.2, 0, 1.8, 0.785], 255, "Approach from Above"),
            ("align_x", 80, [0, -0.1, 0, -2.35, 0, 1.9, 0.785], 255, "Align with Block X"),
            ("align_y", 60, [0, -0.1, 0, -2.35, 0, 1.9, 0.785], 255, "Fine Positioning"),
            ("descend_safe", 100, [0, 0.0, 0, -2.4, 0, 1.95, 0.785], 255, "Descend (Keep Clearance)"),
            ("grasp", 80, [0, 0.0, 0, -2.4, 0, 1.95, 0.785], 0, "Close Gripper"),
            ("lift_block", 100, [0, -0.3, 0, -2.2, 0, 1.8, 0.785], 0, "Lift with Block"),
            ("move_y", 120, [-0.4, -0.3, 0, -2.2, 0, 1.8, 0.785], 0, "Move to Blue Y"),
            ("align_blue", 80, [-0.4, -0.1, 0, -2.35, 0, 1.9, 0.785], 0, "Align Above Blue"),
            ("place_safe", 100, [-0.4, 0.0, 0, -2.4, 0, 1.95, 0.785], 0, "Lower to Place"),
            ("release", 60, [-0.4, 0.0, 0, -2.4, 0, 1.95, 0.785], 255, "Open Gripper"),
            ("retreat", 100, [-0.2, -0.4, 0, -2.0, 0, 1.7, 0.785], 255, "Retreat to Safe Position"),
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
        
        # Get finger positions
        left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        
        finger_positions = []
        if left_finger_id != -1:
            finger_positions.append(self.data.xpos[left_finger_id])
        if right_finger_id != -1:
            finger_positions.append(self.data.xpos[right_finger_id])
        
        # Calculate minimum clearance
        min_clearance = float('inf')
        for pos in finger_positions:
            clearance = pos[2] - self.table_height
            min_clearance = min(min_clearance, clearance)
        
        # Gripper opening
        if len(finger_positions) >= 2:
            gripper_width = np.linalg.norm(finger_positions[0] - finger_positions[1])
        else:
            gripper_width = 0.08  # Default
        
        return gripper_z, min_clearance, gripper_width
    
    def run(self):
        """Create the final demonstration video."""
        print("Creating FINAL demonstration video...")
        print("- Proper gripper orientation (pointing +X)")
        print("- Maintaining table clearance")
        print("- Smooth movements")
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
        max_x_reach = -float('inf')
        
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
                gripper_forward, table_clearance, gripper_width = self.get_gripper_info()
                
                # Update stats
                min_clearance_overall = min(min_clearance_overall, table_clearance)
                max_x_reach = max(max_x_reach, ee_pos[0])
                
                # Render frame
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Title and phase
                cv2.putText(frame_img, "PANDA ROBOT - FINAL DEMONSTRATION", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Progress bar
                progress = (sum([s[1] for s in self.sequence[:phase_idx]]) + frame) / sum([s[1] for s in self.sequence])
                bar_width = int(1200 * progress)
                cv2.rectangle(frame_img, (40, 160), (40 + bar_width, 180), (0, 255, 0), -1)
                cv2.rectangle(frame_img, (40, 160), (1240, 180), (255, 255, 255), 2)
                
                # Create info panel
                panel_y = 250
                line_height = 35
                
                # Positions
                cv2.putText(frame_img, "POSITIONS", (40, panel_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame_img, f"End-Effector: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]",
                           (40, panel_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Red Block:    [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]",
                           (40, panel_y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                
                # Gripper info
                cv2.putText(frame_img, "GRIPPER STATUS", (40, panel_y + 3*line_height + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Forward vector with visual indicator
                forward_color = (0, 255, 0) if gripper_forward[0] > 0.7 else (255, 255, 0)
                cv2.putText(frame_img, f"Forward Vector: [{gripper_forward[0]:5.2f}, {gripper_forward[1]:5.2f}, {gripper_forward[2]:5.2f}]",
                           (40, panel_y + 4*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, forward_color, 2)
                
                # Table clearance with warning
                clearance_color = (0, 255, 0) if table_clearance > 0.03 else (0, 0, 255)
                warning = " ⚠️ TOO LOW!" if table_clearance < 0.03 else ""
                cv2.putText(frame_img, f"Table Clearance: {table_clearance:6.3f}m{warning}",
                           (40, panel_y + 5*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clearance_color, 2)
                
                # Gripper state and width
                gripper_state = "OPEN" if gripper_cmd > 100 else "CLOSED"
                state_color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
                cv2.putText(frame_img, f"Gripper: {gripper_state} (Width: {gripper_width:.3f}m)",
                           (40, panel_y + 6*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
                
                # Distance to red block
                dist_to_red = np.linalg.norm(ee_pos - red_pos)
                cv2.putText(frame_img, f"Distance to Red: {dist_to_red:.3f}m",
                           (40, panel_y + 7*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                frames.append(frame_img)
                
                # Print progress
                if frame % 30 == 0:
                    print(f"  Frame {frame:3d}/{duration}: EE=[{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}], "
                          f"Clear={table_clearance:5.3f}m, Forward=[{gripper_forward[0]:4.2f}]")
        
        # Save video
        print("\n" + "=" * 70)
        print("Saving final video...")
        
        output_path = "../../videos/121pm/panda_final_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ FINAL VIDEO SAVED!")
            print(f"📹 Path: {output_path}")
            print(f"⏱️  Duration: {len(frames)/fps:.1f} seconds")
            print(f"📐 Resolution: {width}x{height}")
            print(f"🎬 Total frames: {len(frames)}")
            print(f"\n📊 Statistics:")
            print(f"  Minimum table clearance: {min_clearance_overall:.3f}m")
            print(f"  Maximum X reach: {max_x_reach:.3f}m")
            
        print("\n🎉 Final demonstration complete!")

if __name__ == "__main__":
    creator = FinalDemoVideo()
    creator.run()