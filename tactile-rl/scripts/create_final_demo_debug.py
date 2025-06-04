#!/usr/bin/env python3
"""
Create final demonstration with debugging and proper clearances.
Target: 0.62m clearance for red block grasp, 0.2m initial clearance.
"""

import numpy as np
import mujoco
import cv2
import os

class FinalDemoDebug:
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
        self.block_size = 0.025  # Half-size from XML
        self.target_grasp_clearance = 0.062  # 6.2cm for red block
        self.initial_clearance = 0.2  # 20cm initial
        
        # Debug: Calculate target heights
        self.initial_gripper_z = self.table_height + self.initial_clearance  # 0.6m
        self.grasp_gripper_z = self.table_height + self.target_grasp_clearance  # 0.462m
        
        print(f"\nDEBUG: Target heights:")
        print(f"  Initial gripper Z: {self.initial_gripper_z:.3f}m")
        print(f"  Grasp gripper Z: {self.grasp_gripper_z:.3f}m")
        print(f"  Table height: {self.table_height:.3f}m")
        
        # Sequence designed for these clearances
        self.sequence = [
            # Start at 0.2m clearance (0.6m height)
            ("home", 60, [0, -0.4, 0, -2.0, 0, 1.6, 0], 255, "Home - 0.2m Clearance"),
            
            # Move toward red block position
            ("approach_xy", 80, [0, -0.2, 0, -2.2, 0, 1.7, 0], 255, "Approach XY Position"),
            
            # Hold above red block
            ("hold_above", 80, [0, -0.1, 0, -2.3, 0, 1.8, 0], 255, "Hold Above Red"),
            
            # Descend to 0.062m clearance
            ("descend", 100, [0, 0.05, 0, -2.4, 0, 1.9, 0], 255, "Descend to 6.2cm"),
            
            # Grasp
            ("grasp", 100, [0, 0.05, 0, -2.4, 0, 1.9, 0], 0, "Grasp Red Block"),
            
            # Lift
            ("lift", 80, [0, -0.2, 0, -2.2, 0, 1.7, 0], 0, "Lift Block"),
            
            # Rotate to blue Y position
            ("rotate", 120, [-0.3, -0.2, 0, -2.2, 0, 1.7, 0], 0, "Rotate to Blue Y"),
            
            # Position above blue
            ("above_blue", 80, [-0.3, -0.1, 0, -2.3, 0, 1.8, 0], 0, "Above Blue Block"),
            
            # Place (blue is 0.045m higher than table)
            ("place", 100, [-0.3, 0.0, 0, -2.35, 0, 1.85, 0], 0, "Place on Blue"),
            
            # Release
            ("release", 80, [-0.3, 0.0, 0, -2.35, 0, 1.85, 0], 255, "Release"),
            
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
        """Get gripper information with debugging."""
        # Get gripper orientation
        hand_quat = self.data.xquat[self.ee_id].copy()
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, hand_quat)
        rotmat = rotmat.reshape(3, 3)
        
        gripper_z = rotmat[:, 2]  # Forward
        
        # Get lowest point of gripper
        gripper_bodies = ["hand", "left_finger", "right_finger"]
        lowest_z = float('inf')
        body_positions = {}
        
        for body_name in gripper_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                z_pos = self.data.xpos[body_id][2]
                body_positions[body_name] = z_pos
                lowest_z = min(lowest_z, z_pos)
        
        table_clearance = lowest_z - self.table_height
        
        return gripper_z, table_clearance, lowest_z, body_positions
    
    def run(self):
        """Create the video with debugging."""
        print("\nCreating FINAL DEMO with DEBUG...")
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
        
        # Check initial positions
        print("\nINITIAL STATE:")
        ee_pos = self.data.xpos[self.ee_id]
        red_pos = self.data.xpos[self.red_id]
        blue_pos = self.data.xpos[self.blue_id]
        print(f"  End-effector: {ee_pos}")
        print(f"  Red block: {red_pos}")
        print(f"  Blue block: {blue_pos}")
        
        # Video settings
        frames = []
        fps = 30
        
        # Stats
        phase_stats = {}
        
        # Execute sequence
        for phase_idx, (phase_name, duration, target_config, gripper_cmd, description) in enumerate(self.sequence):
            print(f"\n{'='*60}")
            print(f"PHASE: {phase_name} - {description}")
            print(f"Target config: {target_config}")
            
            current_config = [self.data.qpos[14 + i] for i in range(7)]
            phase_clearances = []
            
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
                gripper_forward, table_clearance, lowest_z, body_positions = self.get_gripper_info()
                phase_clearances.append(table_clearance)
                
                # Calculate distances
                xy_dist_red = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
                xy_dist_blue = np.sqrt((ee_pos[0] - blue_pos[0])**2 + (ee_pos[1] - blue_pos[1])**2)
                
                # Debug output every 20 frames
                if frame % 20 == 0:
                    print(f"\n  Frame {frame}/{duration}:")
                    print(f"    Config: [{', '.join([f'{c:.2f}' for c in config])}]")
                    print(f"    EE pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                    print(f"    Gripper lowest: {lowest_z:.3f}m")
                    print(f"    Table clearance: {table_clearance:.3f}m")
                    print(f"    XY dist to red: {xy_dist_red:.3f}m")
                    print(f"    XY dist to blue: {xy_dist_blue:.3f}m")
                
                # Render
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Overlays
                cv2.putText(frame_img, "DEBUG: FINAL DEMO", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Detailed info
                y = 200
                cv2.putText(frame_img, f"EE: [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]",
                           (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Red: [{red_pos[0]:+.3f}, {red_pos[1]:+.3f}, {red_pos[2]:+.3f}]",
                           (40, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                cv2.putText(frame_img, f"Blue: [{blue_pos[0]:+.3f}, {blue_pos[1]:+.3f}, {blue_pos[2]:+.3f}]",
                           (40, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                
                # Clearance
                clear_color = (0, 255, 0) if table_clearance > 0.05 else (255, 0, 0)
                cv2.putText(frame_img, f"TABLE CLEARANCE: {table_clearance:.3f}m",
                           (40, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, clear_color, 3)
                
                # Target clearances
                if phase_name == "home":
                    cv2.putText(frame_img, f"Target: 0.200m (Actual: {table_clearance:.3f}m)",
                               (40, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                elif phase_name == "grasp":
                    cv2.putText(frame_img, f"Target: 0.062m (Actual: {table_clearance:.3f}m)",
                               (40, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Distances
                cv2.putText(frame_img, f"Dist to red: {xy_dist_red:.3f}m",
                           (40, y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(frame_img, f"Dist to blue: {xy_dist_blue:.3f}m",
                           (40, y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                frames.append(frame_img)
            
            # Phase summary
            phase_stats[phase_name] = {
                'min_clearance': min(phase_clearances),
                'max_clearance': max(phase_clearances),
                'final_clearance': phase_clearances[-1]
            }
            print(f"\nPhase Summary:")
            print(f"  Min clearance: {phase_stats[phase_name]['min_clearance']:.3f}m")
            print(f"  Max clearance: {phase_stats[phase_name]['max_clearance']:.3f}m")
            print(f"  Final clearance: {phase_stats[phase_name]['final_clearance']:.3f}m")
        
        # Save
        print("\n" + "="*70)
        print("Saving debug video...")
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_final_debug.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ DEBUG VIDEO SAVED: {output_path}")
            
            # Final summary
            print("\nFINAL SUMMARY:")
            print("Phase clearances:")
            for phase, stats in phase_stats.items():
                print(f"  {phase:15s}: min={stats['min_clearance']:.3f}m, final={stats['final_clearance']:.3f}m")

if __name__ == "__main__":
    creator = FinalDemoDebug()
    creator.run()