#!/usr/bin/env python3
"""
Create optimal final demonstration video with careful approach to avoid table collision.
- Joint 4: Adjusted for optimal reach
- Joint 6: 45 degrees rotation for gripper orientation
- Approach from higher angle to blocks at X=0.05m
"""

import numpy as np
import mujoco
import cv2
import os

class OptimalFinalDemoVideo:
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
        self.block_height = 0.445  # From XML
        self.safety_margin = 0.02
        
        # Optimal sequence - approach from higher angle, keep wrist more bent
        # Since blocks are at X=0.05 (very close), we need extreme joint configs
        self.sequence = [
            # Phase, duration, [j0, j1, j2, j3, j4, j5, j6], gripper, description
            # Start high with good clearance
            ("home", 60, [0, -0.6, 0, -1.8, 0.4, 1.4, 0.785], 255, "Home - Safe Height"),
            # Move above the block position first
            ("position_above", 80, [0, -0.4, 0, -2.0, 0.4, 1.6, 0.785], 255, "Position Above Target"),
            # Descend while maintaining angle
            ("descend_angled", 100, [0, -0.2, 0, -2.2, 0.4, 1.75, 0.785], 255, "Angled Descent"),
            # Final approach - keep some height
            ("final_approach", 80, [0, -0.05, 0, -2.35, 0.4, 1.85, 0.785], 255, "Final Approach"),
            # Grasp position - don't go too low
            ("grasp_position", 60, [0, 0, 0, -2.4, 0.4, 1.9, 0.785], 255, "Grasp Position"),
            ("close_gripper", 80, [0, 0, 0, -2.4, 0.4, 1.9, 0.785], 0, "Close Gripper"),
            # Lift straight up first
            ("lift_vertical", 60, [0, -0.2, 0, -2.2, 0.4, 1.75, 0.785], 0, "Lift Vertically"),
            # Then lift higher
            ("lift_clear", 80, [0, -0.4, 0, -2.0, 0.4, 1.6, 0.785], 0, "Lift Clear"),
            # Rotate to blue position
            ("rotate", 120, [-0.5, -0.4, 0, -2.0, 0.4, 1.6, 0.785], 0, "Rotate to Blue"),
            # Descend to blue
            ("descend_blue", 100, [-0.5, -0.2, 0, -2.2, 0.4, 1.75, 0.785], 0, "Descend to Blue"),
            ("place_position", 80, [-0.5, -0.05, 0, -2.35, 0.4, 1.85, 0.785], 0, "Place Position"),
            ("release", 60, [-0.5, -0.05, 0, -2.35, 0.4, 1.85, 0.785], 255, "Release"),
            # Retreat
            ("retreat_up", 60, [-0.5, -0.3, 0, -2.1, 0.4, 1.7, 0.785], 255, "Retreat Up"),
            ("return", 100, [0, -0.6, 0, -1.8, 0.4, 1.4, 0.785], 255, "Return Home"),
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
        """Create the optimal final demonstration video."""
        print("Creating OPTIMAL FINAL demonstration video...")
        print("- Blocks at X=0.05m (very close to robot)")
        print("- Joint 4: 0.4 rad for better reach geometry")
        print("- Joint 6: 0.785 rad (45°) for gripper rotation")
        print("- Approach from higher angle to maintain clearance")
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
        min_clearance_grasp = float('inf')  # Clearance during grasp phase
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
                if phase_name in ["grasp_position", "close_gripper"]:
                    min_clearance_grasp = min(min_clearance_grasp, table_clearance)
                
                # Render frame
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Add overlays
                # Title
                cv2.putText(frame_img, "PANDA ROBOT - OPTIMAL FINAL DEMONSTRATION", (40, 60),
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
                cv2.putText(frame_img, "POSITIONS (Blocks at X=0.05m)", (40, panel_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame_img, f"End-Effector: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]",
                           (40, panel_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Red Block:    [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]",
                           (40, panel_y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                cv2.putText(frame_img, f"Blue Block:   [{blue_pos[0]:6.3f}, {blue_pos[1]:6.3f}, {blue_pos[2]:6.3f}]",
                           (40, panel_y + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                
                # Gripper orientation
                forward_color = (0, 255, 0) if gripper_forward[0] > 0.6 else (255, 255, 0)
                cv2.putText(frame_img, f"Gripper Forward: [{gripper_forward[0]:5.2f}, {gripper_forward[1]:5.2f}, {gripper_forward[2]:5.2f}]",
                           (40, panel_y + 4*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, forward_color, 2)
                
                # Joint configuration
                joint_config = [self.data.qpos[14 + i] for i in range(7)]
                cv2.putText(frame_img, f"J4: {joint_config[4]:4.2f} rad  J6: {joint_config[6]:4.2f} rad",
                           (40, panel_y + 5*line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
                
                # Table clearance with strong visual indication
                if table_clearance < 0:
                    clearance_color = (0, 0, 255)
                    warning = " ⚠️ BELOW TABLE!"
                    cv2.rectangle(frame_img, (5, 5), (frame_img.shape[1]-5, frame_img.shape[0]-5), 
                                (0, 0, 255), 5)
                elif table_clearance < 0.02:
                    clearance_color = (0, 0, 255)
                    warning = " ⚠️ VERY LOW!"
                elif table_clearance < 0.05:
                    clearance_color = (0, 165, 255)
                    warning = " ⚠️ LOW"
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
                
                # Distance info
                dist_to_red = np.linalg.norm(ee_pos[:2] - red_pos[:2])
                height_above_block = ee_pos[2] - red_pos[2]
                cv2.putText(frame_img, f"Distance to Red: XY={dist_to_red:.3f}m, Z={height_above_block:+.3f}m",
                           (40, panel_y + 8*line_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                frames.append(frame_img)
                
                # Print progress
                if frame % 20 == 0:
                    print(f"  Frame {frame:3d}/{duration}: "
                          f"EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}], "
                          f"Clear={table_clearance:.3f}m")
        
        # Save video
        print("\n" + "=" * 70)
        print("Saving optimal final video...")
        
        # Create output directory if needed
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_optimal_final_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ OPTIMAL FINAL VIDEO SAVED!")
            print(f"📹 Path: {output_path}")
            print(f"⏱️  Duration: {len(frames)/fps:.1f} seconds")
            print(f"📐 Resolution: {width}x{height}")
            print(f"🎬 Total frames: {len(frames)}")
            print(f"\n📊 Final Statistics:")
            print(f"  Joint 4: 0.400 rad (22.9°)")
            print(f"  Joint 6: 0.785 rad (45.0°)")
            print(f"  Minimum table clearance overall: {min_clearance_overall:.3f}m")
            print(f"  Minimum clearance during grasp: {min_clearance_grasp:.3f}m")
            print(f"  Frames below table: {below_table_count}")
            
            if below_table_count == 0:
                print("\n✅ SUCCESS: Gripper stayed above table throughout!")
            elif below_table_count < 10:
                print("\n⚠️  Minor table contact detected")
            else:
                print("\n❌ WARNING: Significant table collision!")
            
        print("\n🎉 Optimal final demonstration complete!")

if __name__ == "__main__":
    creator = OptimalFinalDemoVideo()
    creator.run()