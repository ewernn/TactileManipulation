#!/usr/bin/env python3
"""
Create final demonstration with fixed positioning - holds above block before descending.
"""

import numpy as np
import mujoco
import cv2
import os

class FinalDemoFixedPositioning:
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
        self.block_height = 0.05
        self.safety_margin = 0.08  # 8cm minimum clearance
        
        # Fixed sequence - hold positions properly
        self.sequence = [
            # Phase, duration, [j0, j1, j2, j3, j4, j5, j6], gripper, description
            # Start high and safe
            ("home", 60, [0, -0.7, 0, -1.6, 0, 1.3, 0], 255, "Home - High Position"),
            
            # Approach from high
            ("approach_high", 80, [0, -0.5, 0, -2.0, 0, 1.5, 0], 255, "Approach from Above"),
            
            # Position above red block - HOLD HERE
            ("position_hold", 100, [0, -0.3, 0, -2.2, 0, 1.7, 0], 255, "HOLD Above Red Block"),
            
            # Now descend carefully
            ("descend_start", 60, [0, -0.2, 0, -2.3, 0, 1.8, 0], 255, "Begin Descent"),
            
            # Final approach - still maintaining clearance
            ("descend_final", 80, [0, -0.1, 0, -2.35, 0, 1.85, 0], 255, "Final Descent"),
            
            # Grasp position - minimum safe height
            ("grasp", 100, [0, -0.05, 0, -2.38, 0, 1.88, 0], 0, "Close Gripper"),
            
            # Lift straight up first
            ("lift_initial", 60, [0, -0.15, 0, -2.32, 0, 1.82, 0], 0, "Initial Lift"),
            
            # Continue lifting
            ("lift_more", 80, [0, -0.3, 0, -2.2, 0, 1.7, 0], 0, "Lift Higher"),
            
            # Lift to clearing height
            ("lift_clear", 80, [0, -0.5, 0, -2.0, 0, 1.5, 0], 0, "Lift to Clear Blue"),
            
            # Rotate while high
            ("rotate", 120, [-0.5, -0.5, 0, -2.0, 0, 1.5, 0], 0, "Rotate to Blue"),
            
            # Position above blue - accounting for block height
            ("above_blue_hold", 100, [-0.5, -0.3, 0, -2.2, 0, 1.7, 0], 0, "HOLD Above Blue"),
            
            # Descend to place - blue is higher than table
            ("place_descend", 80, [-0.5, -0.2, 0, -2.3, 0, 1.8, 0], 0, "Descend to Blue"),
            
            # Final placement
            ("place", 80, [-0.5, -0.15, 0, -2.32, 0, 1.82, 0], 0, "Place on Blue"),
            
            # Release
            ("release", 80, [-0.5, -0.15, 0, -2.32, 0, 1.82, 0], 255, "Release Block"),
            
            # Retreat up
            ("retreat", 100, [-0.5, -0.5, 0, -2.0, 0, 1.5, 0], 255, "Retreat Upward"),
            
            # Return
            ("return", 100, [0, -0.7, 0, -1.6, 0, 1.3, 0], 255, "Return Home"),
        ]
        
    def interpolate_configs(self, config1, config2, alpha):
        """Smooth interpolation."""
        if alpha < 0.5:
            t = 2 * alpha * alpha
        else:
            t = 1 - pow(-2 * alpha + 2, 2) / 2
        
        return [(1-t)*c1 + t*c2 for c1, c2 in zip(config1, config2)]
    
    def get_gripper_info(self):
        """Get gripper information."""
        # Get gripper orientation
        hand_quat = self.data.xquat[self.ee_id].copy()
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, hand_quat)
        rotmat = rotmat.reshape(3, 3)
        
        gripper_z = rotmat[:, 2]  # Forward
        
        # Get lowest point of gripper
        gripper_bodies = ["hand", "left_finger", "right_finger"]
        lowest_z = float('inf')
        
        for body_name in gripper_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                z_pos = self.data.xpos[body_id][2]
                lowest_z = min(lowest_z, z_pos)
        
        table_clearance = lowest_z - self.table_height
        
        # Gripper opening
        left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        
        if left_id != -1 and right_id != -1:
            gripper_width = np.linalg.norm(self.data.xpos[left_id] - self.data.xpos[right_id])
        else:
            gripper_width = 0.08
        
        return gripper_z, table_clearance, gripper_width, lowest_z
    
    def run(self):
        """Create the video."""
        print("Creating FINAL DEMO with FIXED POSITIONING...")
        print("- Holding positions properly before descending")
        print("- Maintaining safe table clearance")
        print("- Proper height for block stacking")
        print("=" * 70)
        
        # Reset
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Set initial configuration
        initial_config = self.sequence[0][2]
        for i, val in enumerate(initial_config):
            self.data.qpos[14 + i] = val
            self.data.ctrl[i] = val
        self.data.ctrl[7] = 255
        
        # Step to update
        mujoco.mj_step(self.model, self.data)
        
        # Video settings
        frames = []
        fps = 30
        
        # Stats
        min_clearance = float('inf')
        position_hold_clearance = None
        grasp_clearance = None
        
        # Execute sequence
        for phase_idx, (phase_name, duration, target_config, gripper_cmd, description) in enumerate(self.sequence):
            print(f"\nPhase: {phase_name} - {description}")
            
            current_config = [self.data.qpos[14 + i] for i in range(7)]
            
            for frame in range(duration):
                alpha = frame / duration
                
                if phase_idx == 0:
                    config = target_config
                else:
                    config = self.interpolate_configs(current_config, target_config, alpha)
                
                # Apply
                for i, val in enumerate(config):
                    self.data.ctrl[i] = val
                self.data.ctrl[7] = gripper_cmd
                
                # Step
                mujoco.mj_step(self.model, self.data)
                
                # Get positions
                ee_pos = self.data.xpos[self.ee_id]
                red_pos = self.data.xpos[self.red_id]
                blue_pos = self.data.xpos[self.blue_id]
                
                # Info
                gripper_forward, table_clearance, gripper_width, lowest_z = self.get_gripper_info()
                
                # Track stats
                min_clearance = min(min_clearance, table_clearance)
                if phase_name == "position_hold" and frame > duration/2:
                    position_hold_clearance = table_clearance
                if phase_name == "grasp":
                    grasp_clearance = table_clearance
                
                # Calculate distances
                xy_dist = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
                red_z = red_pos[2]
                blue_top = blue_pos[2] + self.block_height/2
                clearance_to_blue = red_z - self.block_height/2 - blue_top
                
                # Render
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Overlays
                cv2.putText(frame_img, "PANDA ROBOT - FIXED POSITIONING DEMO", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Progress bar
                progress = (sum([s[1] for s in self.sequence[:phase_idx]]) + frame) / sum([s[1] for s in self.sequence])
                bar_width = int(1200 * progress)
                cv2.rectangle(frame_img, (40, 160), (40 + bar_width, 180), (0, 255, 0), -1)
                cv2.rectangle(frame_img, (40, 160), (1240, 180), (255, 255, 255), 2)
                
                # Position info
                y = 220
                cv2.putText(frame_img, f"End-Effector: [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]",
                           (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Gripper Lowest: Z={lowest_z:.3f}m",
                           (40, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                # Phase indicator for HOLD phases
                if "HOLD" in description:
                    cv2.putText(frame_img, ">>> HOLDING POSITION <<<", (40, y + 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                
                # Critical clearance info
                if table_clearance < self.safety_margin:
                    clear_color = (0, 0, 255) if table_clearance < 0.03 else (255, 165, 0)
                    warning = " ⚠️ TOO LOW!" if table_clearance < 0.03 else " ⚠️ LOW"
                else:
                    clear_color = (0, 255, 0)
                    warning = " ✓ SAFE"
                
                cv2.putText(frame_img, f"TABLE CLEARANCE: {table_clearance:.3f}m{warning}",
                           (40, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, clear_color, 3)
                
                # Draw clearance line
                table_y_pixel = int(520 - (table_clearance * 1000))  # Scale for visualization
                cv2.line(frame_img, (300, table_y_pixel), (980, table_y_pixel), clear_color, 2)
                cv2.putText(frame_img, f"{table_clearance:.3f}m", (990, table_y_pixel + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, clear_color, 2)
                
                # Block clearance when lifting
                if phase_name in ["lift_clear", "rotate", "above_blue_hold"]:
                    cv2.putText(frame_img, f"Red-Blue Clearance: {clearance_to_blue:.3f}m",
                               (40, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                               (0, 255, 0) if clearance_to_blue > 0 else (255, 0, 0), 2)
                
                # Joint configuration
                joint_config = [self.data.qpos[14 + i] for i in range(7)]
                cv2.putText(frame_img, f"J1={joint_config[1]:+.2f} J3={joint_config[3]:+.2f} J5={joint_config[5]:+.2f}",
                           (40, y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
                
                # Gripper state
                state = "OPEN" if gripper_cmd > 100 else "CLOSED"
                state_color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
                cv2.putText(frame_img, f"Gripper: {state}",
                           (40, y + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
                
                frames.append(frame_img)
                
                if frame % 30 == 0:
                    print(f"  Frame {frame:3d}/{duration}: "
                          f"Clear={table_clearance:.3f}m, Z={lowest_z:.3f}")
        
        # Save
        print("\nSaving video...")
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_final_fixed_positioning.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ FINAL VIDEO WITH FIXED POSITIONING SAVED!")
            print(f"📹 Path: {output_path}")
            print(f"📊 Statistics:")
            print(f"  Minimum table clearance: {min_clearance:.3f}m")
            print(f"  Position hold clearance: {position_hold_clearance:.3f}m")
            print(f"  Grasp clearance: {grasp_clearance:.3f}m")
            print(f"  Safety margin: {self.safety_margin:.3f}m")
            
            if min_clearance > 0.03:
                print("\n✅ SUCCESS: Safe clearance maintained!")
            else:
                print("\n⚠️  WARNING: Clearance below safe threshold")

if __name__ == "__main__":
    creator = FinalDemoFixedPositioning()
    creator.run()