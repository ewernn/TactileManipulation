#!/usr/bin/env python3
"""
Create final demonstration with correct reach to blocks at X=0.05.
Target: 0.062m clearance for red block, proper XY alignment.
"""

import numpy as np
import mujoco
import cv2
import os

class FinalDemoCorrectReach:
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
        self.target_grasp_clearance = 0.062  # 6.2cm for red block
        self.initial_clearance = 0.2  # 20cm initial
        
        # To reach X=0.05, we need more positive joint values
        # Joint 1 (shoulder) needs to be MORE positive (lift up more)
        # Joint 3 (elbow) needs to be MORE negative (bend more)
        self.sequence = [
            # Start at 0.2m clearance
            ("home", 60, [0, -0.4, 0, -2.0, 0, 1.6, 0], 255, "Home - 0.2m Clearance"),
            
            # Approach with more forward reach
            ("approach", 80, [0, 0.2, 0, -2.6, 0, 2.2, 0], 255, "Approach Forward"),
            
            # Fine position above red
            ("position", 80, [0, 0.4, 0, -2.8, 0, 2.4, 0], 255, "Position Above Red"),
            
            # Hold at target clearance (0.062m)
            ("hold_62mm", 100, [0, 0.5, 0, -2.85, 0, 2.35, 0], 255, "Hold at 62mm Clearance"),
            
            # Grasp
            ("grasp", 100, [0, 0.5, 0, -2.85, 0, 2.35, 0], 0, "Grasp Block"),
            
            # Lift up
            ("lift", 80, [0, 0.2, 0, -2.6, 0, 2.2, 0], 0, "Lift Block"),
            
            # Lift higher
            ("lift_high", 60, [0, -0.2, 0, -2.2, 0, 1.8, 0], 0, "Lift Higher"),
            
            # Rotate to blue Y (-0.15)
            ("rotate", 120, [-0.4, -0.2, 0, -2.2, 0, 1.8, 0], 0, "Rotate to Blue"),
            
            # Position above blue
            ("above_blue", 80, [-0.4, 0.2, 0, -2.6, 0, 2.2, 0], 0, "Above Blue"),
            
            # Place (accounting for blue block height)
            ("place", 100, [-0.4, 0.4, 0, -2.8, 0, 2.4, 0], 0, "Place on Blue"),
            
            # Release
            ("release", 80, [-0.4, 0.4, 0, -2.8, 0, 2.4, 0], 255, "Release"),
            
            # Retreat
            ("retreat", 80, [-0.4, -0.2, 0, -2.2, 0, 1.8, 0], 255, "Retreat"),
            
            # Return
            ("return", 100, [0, -0.4, 0, -2.0, 0, 1.6, 0], 255, "Return Home"),
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
        
        return gripper_z, table_clearance, lowest_z
    
    def run(self):
        """Create the video."""
        print("Creating FINAL DEMO with CORRECT REACH...")
        print("Target: X=0.05 for red block, 62mm clearance")
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
        closest_xy_dist = float('inf')
        hold_clearance = None
        
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
                gripper_forward, table_clearance, lowest_z = self.get_gripper_info()
                
                # Track stats
                min_clearance = min(min_clearance, table_clearance)
                xy_dist = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
                closest_xy_dist = min(closest_xy_dist, xy_dist)
                
                if phase_name == "hold_62mm" and frame > 50:
                    hold_clearance = table_clearance
                
                # Render
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Overlays
                cv2.putText(frame_img, "FINAL DEMO - CORRECT REACH", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Progress bar
                progress = (sum([s[1] for s in self.sequence[:phase_idx]]) + frame) / sum([s[1] for s in self.sequence])
                bar_width = int(1200 * progress)
                cv2.rectangle(frame_img, (40, 160), (40 + bar_width, 180), (0, 255, 0), -1)
                cv2.rectangle(frame_img, (40, 160), (1240, 180), (255, 255, 255), 2)
                
                # Key info
                y = 220
                cv2.putText(frame_img, f"EE: [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]",
                           (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Red: [{red_pos[0]:+.3f}, {red_pos[1]:+.3f}, {red_pos[2]:+.3f}]",
                           (40, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                
                # XY alignment
                x_diff = ee_pos[0] - red_pos[0]
                align_color = (0, 255, 0) if abs(x_diff) < 0.02 else (255, 255, 0)
                cv2.putText(frame_img, f"X alignment: EE-Red = {x_diff:+.3f}m",
                           (40, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, align_color, 2)
                cv2.putText(frame_img, f"XY distance: {xy_dist:.3f}m",
                           (40, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                
                # Clearance with target
                clear_color = (0, 255, 0) if table_clearance > 0.05 else (255, 0, 0)
                cv2.putText(frame_img, f"TABLE CLEARANCE: {table_clearance:.3f}m",
                           (40, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, clear_color, 3)
                
                if phase_name == "hold_62mm":
                    target_color = (0, 255, 0) if abs(table_clearance - 0.062) < 0.01 else (255, 165, 0)
                    cv2.putText(frame_img, f"Target: 0.062m (Diff: {table_clearance - 0.062:+.3f}m)",
                               (40, y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, target_color, 2)
                
                # Joint config
                joint_config = [self.data.qpos[14 + i] for i in range(7)]
                cv2.putText(frame_img, f"J1={joint_config[1]:+.2f} J3={joint_config[3]:+.2f}",
                           (40, y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
                
                frames.append(frame_img)
                
                # Debug output
                if frame % 30 == 0:
                    print(f"  Frame {frame:3d}: X={ee_pos[0]:+.3f}, XY_dist={xy_dist:.3f}, Clear={table_clearance:.3f}")
        
        # Save
        print("\nSaving video...")
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_final_correct_reach.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ VIDEO SAVED: {output_path}")
            print(f"\nStatistics:")
            print(f"  Minimum clearance: {min_clearance:.3f}m")
            print(f"  Clearance during hold: {hold_clearance:.3f}m")
            print(f"  Closest XY distance: {closest_xy_dist:.3f}m")

if __name__ == "__main__":
    creator = FinalDemoCorrectReach()
    creator.run()