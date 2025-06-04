#!/usr/bin/env python3
"""
Create truly corrected demonstration - using extreme joint configs to reach X=0.05.
"""

import numpy as np
import mujoco
import cv2
import os

class TrulyCorrectedDemoVideo:
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
        
        # Extreme configuration to reach X=0.05
        # We need very high j1 and very negative j3 to push the arm forward
        self.sequence = [
            # Phase, duration, [j0, j1, j2, j3, j4, j5, j6], gripper, description
            # Start as close as possible to X=0.05
            ("home", 60, [0, 1.2, 0, -2.8, -0.5, 2.4, 0.785], 255, "Home - Near Red Block"),
            # Fine tune position
            ("align", 60, [0, 1.3, 0, -2.85, -0.5, 2.45, 0.785], 255, "Fine Alignment"),
            # Descend
            ("descend", 80, [0, 1.4, 0, -2.9, -0.5, 2.5, 0.785], 255, "Descend to Grasp"),
            # Grasp
            ("grasp", 80, [0, 1.4, 0, -2.9, -0.5, 2.5, 0.785], 0, "Close Gripper"),
            # Lift
            ("lift", 60, [0, 1.2, 0, -2.8, -0.5, 2.4, 0.785], 0, "Lift Block"),
            # Lift higher
            ("lift_high", 60, [0, 1.0, 0, -2.6, -0.5, 2.2, 0.785], 0, "Lift Higher"),
            # Rotate to blue
            ("rotate", 100, [-0.5, 1.0, 0, -2.6, -0.5, 2.2, 0.785], 0, "Rotate to Blue"),
            # Above blue
            ("above_blue", 60, [-0.5, 1.3, 0, -2.85, -0.5, 2.45, 0.785], 0, "Above Blue"),
            # Place
            ("place", 80, [-0.5, 1.4, 0, -2.9, -0.5, 2.5, 0.785], 0, "Place Block"),
            # Release
            ("release", 60, [-0.5, 1.4, 0, -2.9, -0.5, 2.5, 0.785], 255, "Release"),
            # Retreat
            ("retreat", 60, [-0.5, 1.2, 0, -2.8, -0.5, 2.4, 0.785], 255, "Retreat"),
            # Return
            ("return", 100, [0, 1.2, 0, -2.8, -0.5, 2.4, 0.785], 255, "Return Home"),
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
        
        # Get lowest point
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
        print("Creating TRULY CORRECTED demonstration...")
        print("- Using extreme joint configs to reach X=0.05")
        print("- Joint 6 at 0.785 rad (45°)")
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
        closest_x_dist = float('inf')
        
        # Execute sequence
        for phase_idx, (phase_name, duration, target_config, gripper_cmd, description) in enumerate(self.sequence):
            print(f"\nPhase: {phase_name}")
            
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
                
                # Info
                gripper_forward, table_clearance, gripper_width = self.get_gripper_info()
                
                # Track stats
                min_clearance = min(min_clearance, table_clearance)
                x_dist = abs(ee_pos[0] - red_pos[0])
                closest_x_dist = min(closest_x_dist, x_dist)
                
                # Render
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Overlays
                cv2.putText(frame_img, "TRULY CORRECTED FINAL DEMO", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Key info
                y_offset = 200
                cv2.putText(frame_img, f"EE X: {ee_pos[0]:+.3f} (Target: +0.050)",
                           (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame_img, f"X Distance: {x_dist:.3f}m",
                           (40, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                cv2.putText(frame_img, f"Joint 6: {config[6]:.3f} rad (45°)",
                           (40, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 255), 2)
                cv2.putText(frame_img, f"Table Clear: {table_clearance:.3f}m",
                           (40, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                           (0, 255, 0) if table_clearance > 0.02 else (255, 0, 0), 2)
                
                frames.append(frame_img)
                
                if frame % 30 == 0:
                    print(f"  EE_X={ee_pos[0]:+.3f}, X_dist={x_dist:.3f}, Clear={table_clearance:.3f}")
        
        # Save
        print("\nSaving video...")
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_truly_corrected_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ VIDEO SAVED: {output_path}")
            print(f"Closest X distance: {closest_x_dist:.3f}m")
            print(f"Min clearance: {min_clearance:.3f}m")

if __name__ == "__main__":
    creator = TrulyCorrectedDemoVideo()
    creator.run()