#!/usr/bin/env python3
"""
Create final SAFE demonstration maintaining table clearance.
Realistic approach to blocks at X=0.05m.
"""

import numpy as np
import mujoco
import cv2
import os

class FinalSafeDemo:
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
        self.block_height = 0.05  # 2.5cm * 2
        self.min_safe_clearance = 0.05  # 5cm minimum
        
        # Safe sequence - maintain clearance while getting as close as possible
        self.sequence = [
            # Start at safe height with 0.2m clearance
            ("home", 60, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Home - 0.2m Clearance"),
            
            # Approach carefully
            ("approach", 80, [0, -0.3, 0, -2.1, 0, 1.65, 0], 255, "Approach Position"),
            
            # Get closer but maintain height
            ("position", 80, [0, -0.15, 0, -2.25, 0, 1.75, 0], 255, "Position Near Block"),
            
            # Carefully descend maintaining clearance
            ("descend", 100, [0, -0.05, 0, -2.35, 0, 1.85, 0], 255, "Descend Carefully"),
            
            # Grasp at safe height
            ("grasp", 100, [0, -0.05, 0, -2.35, 0, 1.85, 0], 0, "Grasp at Safe Height"),
            
            # Lift straight up first
            ("lift_initial", 60, [0, -0.15, 0, -2.25, 0, 1.75, 0], 0, "Initial Lift"),
            
            # Continue lifting
            ("lift_high", 80, [0, -0.4, 0, -2.0, 0, 1.5, 0], 0, "Lift High"),
            
            # Rotate to blue position
            ("rotate", 120, [-0.35, -0.4, 0, -2.0, 0, 1.5, 0], 0, "Rotate to Blue"),
            
            # Position above blue block
            ("above_blue", 80, [-0.35, -0.15, 0, -2.25, 0, 1.75, 0], 0, "Above Blue Block"),
            
            # Descend to place (blue is 5cm higher than table)
            ("place", 100, [-0.35, -0.05, 0, -2.35, 0, 1.85, 0], 0, "Place on Blue"),
            
            # Release
            ("release", 80, [-0.35, -0.05, 0, -2.35, 0, 1.85, 0], 255, "Release Block"),
            
            # Retreat safely
            ("retreat", 80, [-0.35, -0.4, 0, -2.0, 0, 1.5, 0], 255, "Retreat Safely"),
            
            # Return home
            ("return", 100, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Return Home"),
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
        
        return gripper_z, table_clearance, gripper_width
    
    def run(self):
        """Create the video."""
        print("Creating FINAL SAFE DEMONSTRATION...")
        print("- Maintaining safe table clearance")
        print("- Getting as close as possible to blocks at X=0.05")
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
        min_xy_distance = float('inf')
        grasp_clearance = None
        grasp_xy_distance = None
        
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
                gripper_forward, table_clearance, gripper_width = self.get_gripper_info()
                
                # Calculate distances
                xy_dist_red = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
                xy_dist_blue = np.sqrt((ee_pos[0] - blue_pos[0])**2 + (ee_pos[1] - blue_pos[1])**2)
                
                # Track stats
                min_clearance = min(min_clearance, table_clearance)
                min_xy_distance = min(min_xy_distance, xy_dist_red)
                
                if phase_name == "grasp" and frame > 50:
                    grasp_clearance = table_clearance
                    grasp_xy_distance = xy_dist_red
                
                # Calculate block stacking clearance
                red_bottom = red_pos[2] - self.block_height/2
                blue_top = blue_pos[2] + self.block_height/2
                stack_clearance = red_bottom - blue_top
                
                # Render
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Overlays
                cv2.putText(frame_img, "PANDA ROBOT - FINAL SAFE DEMONSTRATION", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Progress bar
                progress = (sum([s[1] for s in self.sequence[:phase_idx]]) + frame) / sum([s[1] for s in self.sequence])
                bar_width = int(1200 * progress)
                cv2.rectangle(frame_img, (40, 160), (40 + bar_width, 180), (0, 255, 0), -1)
                cv2.rectangle(frame_img, (40, 160), (1240, 180), (255, 255, 255), 2)
                
                # Info panel
                y = 220
                cv2.putText(frame_img, f"End-Effector: [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]",
                           (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Red Block: [{red_pos[0]:+.3f}, {red_pos[1]:+.3f}, {red_pos[2]:+.3f}]",
                           (40, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                
                # Distance info
                cv2.putText(frame_img, f"Distance to Red: {xy_dist_red:.3f}m",
                           (40, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame_img, f"EE X position: {ee_pos[0]:+.3f} (Target: +0.050)",
                           (40, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                # Table clearance - CRITICAL
                if table_clearance < 0:
                    clear_color = (0, 0, 255)
                    warning = " ⚠️ BELOW TABLE!"
                elif table_clearance < self.min_safe_clearance:
                    clear_color = (255, 165, 0)
                    warning = " ⚠️ LOW!"
                else:
                    clear_color = (0, 255, 0)
                    warning = " ✓ SAFE"
                
                cv2.putText(frame_img, f"TABLE CLEARANCE: {table_clearance:.3f}m{warning}",
                           (40, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, clear_color, 3)
                
                # Stack clearance when lifting
                if phase_name in ["lift_high", "rotate", "above_blue"]:
                    stack_color = (0, 255, 0) if stack_clearance > 0 else (255, 0, 0)
                    cv2.putText(frame_img, f"Stack clearance: {stack_clearance:.3f}m",
                               (40, y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, stack_color, 2)
                
                # Gripper state
                state = "OPEN" if gripper_cmd > 100 else "CLOSED"
                state_color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
                cv2.putText(frame_img, f"Gripper: {state} (Width: {gripper_width:.3f}m)",
                           (40, y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
                
                frames.append(frame_img)
                
                # Debug every 30 frames
                if frame % 30 == 0:
                    print(f"  Frame {frame:3d}: EE_X={ee_pos[0]:+.3f}, Dist={xy_dist_red:.3f}, Clear={table_clearance:.3f}")
        
        # Save
        print("\nSaving video...")
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_final_safe_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ FINAL SAFE VIDEO SAVED!")
            print(f"📹 Path: {output_path}")
            print(f"\n📊 Final Statistics:")
            print(f"  Minimum table clearance: {min_clearance:.3f}m")
            print(f"  Minimum XY distance to red: {min_xy_distance:.3f}m")
            print(f"  Clearance during grasp: {grasp_clearance:.3f}m")
            print(f"  XY distance during grasp: {grasp_xy_distance:.3f}m")
            
            if min_clearance > 0:
                print("\n✅ SUCCESS: Safe clearance maintained throughout!")
            else:
                print("\n⚠️  WARNING: Went below table")

if __name__ == "__main__":
    creator = FinalSafeDemo()
    creator.run()