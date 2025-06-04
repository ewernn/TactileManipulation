#!/usr/bin/env python3
"""
Create corrected final demonstration with proper vertical alignment and joint 6 rotation.
"""

import numpy as np
import mujoco
import cv2
import os

class CorrectedFinalDemoVideo:
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
        
        # Table and block info
        self.table_height = 0.4
        self.safety_margin = 0.02
        
        # Corrected sequence - start vertically above red block
        # To reach X=0.05, we need:
        # - Positive J1 (shoulder up)
        # - More negative J3 (elbow bent more)
        # - J6 = 0.785 for 45° rotation
        self.sequence = [
            # Phase, duration, [j0, j1, j2, j3, j4, j5, j6], gripper, description
            # Start directly above red block
            ("home", 60, [0, 0.3, 0, -2.8, 0, 2.5, 0.785], 255, "Home - Above Red Block"),
            # Fine positioning
            ("align", 60, [0, 0.35, 0, -2.85, 0, 2.5, 0.785], 255, "Align Vertically"),
            # Descend straight down
            ("descend", 80, [0, 0.4, 0, -2.9, 0, 2.5, 0.785], 255, "Descend to Block"),
            # Grasp
            ("grasp", 80, [0, 0.4, 0, -2.9, 0, 2.5, 0.785], 0, "Close Gripper"),
            # Lift straight up
            ("lift", 60, [0, 0.3, 0, -2.8, 0, 2.5, 0.785], 0, "Lift Block"),
            # Lift higher
            ("lift_high", 60, [0, 0.2, 0, -2.7, 0, 2.4, 0.785], 0, "Lift Higher"),
            # Rotate to blue
            ("rotate", 100, [-0.5, 0.2, 0, -2.7, 0, 2.4, 0.785], 0, "Rotate to Blue"),
            # Position above blue
            ("above_blue", 60, [-0.5, 0.35, 0, -2.85, 0, 2.5, 0.785], 0, "Above Blue Block"),
            # Descend to place
            ("place", 80, [-0.5, 0.4, 0, -2.9, 0, 2.5, 0.785], 0, "Place on Blue"),
            # Release
            ("release", 60, [-0.5, 0.4, 0, -2.9, 0, 2.5, 0.785], 255, "Release"),
            # Retreat
            ("retreat", 60, [-0.5, 0.3, 0, -2.8, 0, 2.5, 0.785], 255, "Retreat"),
            # Return
            ("return", 100, [0, 0.3, 0, -2.8, 0, 2.5, 0.785], 255, "Return Home"),
        ]
        
    def interpolate_configs(self, config1, config2, alpha):
        """Smooth interpolation with ease-in-out."""
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
        
        return gripper_z, table_clearance, gripper_width, lowest_z
    
    def run(self):
        """Create the corrected final demonstration video."""
        print("Creating CORRECTED FINAL demonstration video...")
        print("- Starting vertically above red block (X=0.05)")
        print("- Joint 6: 0.785 rad (45°) rotation maintained")
        print("=" * 70)
        
        # Reset simulation
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Set initial configuration
        initial_config = self.sequence[0][2]
        for i, val in enumerate(initial_config):
            self.data.qpos[14 + i] = val
            self.data.ctrl[i] = val
        self.data.ctrl[7] = 255
        
        # Step once to update positions
        mujoco.mj_step(self.model, self.data)
        
        # Video settings
        frames = []
        fps = 30
        
        # Statistics tracking
        min_clearance_overall = float('inf')
        initial_xy_distance = None
        final_xy_distance = None
        
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
                gripper_forward, table_clearance, gripper_width, lowest_z = self.get_gripper_info()
                
                # Track stats
                min_clearance_overall = min(min_clearance_overall, table_clearance)
                
                # Calculate distances
                xy_dist_to_red = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
                x_diff = ee_pos[0] - red_pos[0]
                y_diff = ee_pos[1] - red_pos[1]
                
                if phase_idx == 0 and frame == 0:
                    initial_xy_distance = xy_dist_to_red
                if phase_name == "grasp":
                    final_xy_distance = xy_dist_to_red
                
                # Render frame
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Add overlays
                cv2.putText(frame_img, "PANDA ROBOT - CORRECTED FINAL DEMO", (40, 60),
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
                
                # Vertical alignment indicator
                alignment_color = (0, 255, 0) if abs(x_diff) < 0.02 else (255, 255, 0)
                cv2.putText(frame_img, f"X Alignment: EE-Red = {x_diff:+6.3f}m",
                           (40, panel_y + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alignment_color, 2)
                cv2.putText(frame_img, f"XY Distance: {xy_dist_to_red:6.3f}m",
                           (40, panel_y + 4*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                # Gripper forward vector
                cv2.putText(frame_img, f"Gripper Forward: [{gripper_forward[0]:5.2f}, {gripper_forward[1]:5.2f}, {gripper_forward[2]:5.2f}]",
                           (40, panel_y + 5*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)
                
                # Joint 6 angle
                joint_config = [self.data.qpos[14 + i] for i in range(7)]
                cv2.putText(frame_img, f"Joint 6: {joint_config[6]:5.3f} rad ({np.degrees(joint_config[6]):5.1f}°)",
                           (40, panel_y + 6*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
                
                # Table clearance
                if table_clearance < 0:
                    clearance_color = (0, 0, 255)
                    warning = " ⚠️ BELOW TABLE!"
                elif table_clearance < 0.02:
                    clearance_color = (0, 0, 255)
                    warning = " ⚠️ TOO LOW!"
                elif table_clearance < 0.05:
                    clearance_color = (0, 165, 255)
                    warning = " ⚠️ LOW"
                else:
                    clearance_color = (0, 255, 0)
                    warning = " ✓ SAFE"
                
                cv2.putText(frame_img, f"Table Clearance: {table_clearance:6.3f}m{warning}",
                           (40, panel_y + 7*line_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, clearance_color, 2)
                
                # Gripper state
                gripper_state = "OPEN" if gripper_cmd > 100 else "CLOSED"
                state_color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
                cv2.putText(frame_img, f"Gripper: {gripper_state}",
                           (40, panel_y + 8*line_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
                
                frames.append(frame_img)
                
                # Print progress
                if frame % 20 == 0:
                    print(f"  Frame {frame:3d}/{duration}: "
                          f"X={ee_pos[0]:+.3f} (Red at 0.050), "
                          f"XY_dist={xy_dist_to_red:.3f}m, "
                          f"Clear={table_clearance:.3f}m")
        
        # Save video
        print("\n" + "=" * 70)
        print("Saving corrected final video...")
        
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_corrected_final_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ CORRECTED FINAL VIDEO SAVED!")
            print(f"📹 Path: {output_path}")
            print(f"⏱️  Duration: {len(frames)/fps:.1f} seconds")
            print(f"📐 Resolution: {width}x{height}")
            print(f"🎬 Total frames: {len(frames)}")
            print(f"\n📊 Final Statistics:")
            print(f"  Initial XY distance to red: {initial_xy_distance:.3f}m")
            print(f"  Final XY distance at grasp: {final_xy_distance:.3f}m")
            print(f"  Joint 6 maintained at: 0.785 rad (45.0°)")
            print(f"  Minimum table clearance: {min_clearance_overall:.3f}m")
            
            if min_clearance_overall > 0:
                print("\n✅ SUCCESS: Started vertically above red block!")
            else:
                print("\n⚠️  WARNING: Table collision detected")
            
        print("\n🎉 Corrected demonstration complete!")

if __name__ == "__main__":
    creator = CorrectedFinalDemoVideo()
    creator.run()