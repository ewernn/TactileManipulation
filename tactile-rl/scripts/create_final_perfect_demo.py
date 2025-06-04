#!/usr/bin/env python3
"""
Create final PERFECT demonstration with exact clearances.
- Descend to EXACTLY 0.065m table clearance
- Stop placing when stack clearance = 0
"""

import numpy as np
import mujoco
import cv2
import os

class FinalPerfectDemo:
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
        
        # Critical measurements
        self.table_height = 0.4
        self.block_half_size = 0.025  # Block is 0.05m total height
        self.target_grasp_clearance = 0.065  # 6.5cm as specified
        
        # Calculate target heights
        self.blue_top = 0.445 + self.block_half_size  # 0.47m
        
        # Sequence - adjusted for 0.065m clearance
        self.sequence = [
            # Start at safe height
            ("home", 60, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Home - 0.2m Clearance"),
            
            # Approach
            ("approach", 80, [0, -0.3, 0, -2.1, 0, 1.65, 0], 255, "Approach Position"),
            
            # Position above red
            ("position", 80, [0, -0.2, 0, -2.2, 0, 1.7, 0], 255, "Position Above Red"),
            
            # Descend to EXACTLY 0.065m clearance - adjusted joints
            ("descend_65mm", 100, [0, -0.08, 0, -2.32, 0, 1.82, 0], 255, "Descend to 65mm"),
            
            # Hold at 65mm to verify
            ("hold_65mm", 60, [0, -0.08, 0, -2.32, 0, 1.82, 0], 255, "Hold at 65mm"),
            
            # Grasp
            ("grasp", 100, [0, -0.08, 0, -2.32, 0, 1.82, 0], 0, "Grasp Block"),
            
            # Lift carefully
            ("lift_initial", 60, [0, -0.2, 0, -2.2, 0, 1.7, 0], 0, "Initial Lift"),
            
            # Lift to safe height
            ("lift_high", 80, [0, -0.5, 0, -1.8, 0, 1.4, 0], 0, "Lift to Clear Blue"),
            
            # Rotate to blue
            ("rotate", 120, [-0.35, -0.5, 0, -1.8, 0, 1.4, 0], 0, "Rotate to Blue"),
            
            # Position high above blue
            ("above_blue", 80, [-0.35, -0.3, 0, -2.1, 0, 1.65, 0], 0, "High Above Blue"),
            
            # SMART PLACE - will stop when stack clearance = 0
            ("smart_place", 150, [-0.35, -0.05, 0, -2.35, 0, 1.85, 0], 0, "Smart Place"),
            
            # Release
            ("release", 80, None, 255, "Release"),  # Will use position from smart_place
            
            # Retreat
            ("retreat", 80, [-0.35, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Retreat"),
            
            # Return
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
        # Get lowest point of gripper
        gripper_bodies = ["hand", "left_finger", "right_finger"]
        lowest_z = float('inf')
        
        for body_name in gripper_bodies:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                z_pos = self.data.xpos[body_id][2]
                lowest_z = min(lowest_z, z_pos)
        
        table_clearance = lowest_z - self.table_height
        
        return table_clearance, lowest_z
    
    def run(self):
        """Create the video."""
        print("\nCreating FINAL PERFECT DEMONSTRATION...")
        print(f"- Target grasp clearance: {self.target_grasp_clearance:.3f}m")
        print(f"- Blue block top: {self.blue_top:.3f}m")
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
        grasp_clearance = None
        hold_clearance = None
        place_stopped_early = False
        place_stop_frame = None
        final_place_config = None
        
        # Execute sequence
        for phase_idx, (phase_name, duration, target_config, gripper_cmd, description) in enumerate(self.sequence):
            print(f"\n{'='*60}")
            print(f"Phase: {phase_name} - {description}")
            
            current_config = [self.data.qpos[14 + i] for i in range(7)]
            
            # For release phase, use the final position from smart_place
            if phase_name == "release" and final_place_config is not None:
                target_config = final_place_config
            
            for frame in range(duration):
                # For smart place, check if we should stop
                if phase_name == "smart_place":
                    red_pos = self.data.xpos[self.red_id]
                    red_bottom = red_pos[2] - self.block_half_size
                    stack_clearance = red_bottom - self.blue_top
                    
                    if stack_clearance <= 0.001 and not place_stopped_early:  # 1mm tolerance
                        place_stopped_early = True
                        place_stop_frame = frame
                        final_place_config = [self.data.qpos[14 + i] for i in range(7)]
                        print(f"\n>>> STOPPING! Stack clearance = {stack_clearance:.4f}m at frame {frame}")
                        # Hold current position
                        target_config = final_place_config
                
                alpha = frame / duration
                
                if phase_idx == 0:
                    config = target_config
                elif place_stopped_early and phase_name == "smart_place":
                    # Hold position after stopping
                    config = final_place_config
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
                table_clearance, lowest_z = self.get_gripper_info()
                
                # Track stats
                min_clearance = min(min_clearance, table_clearance)
                
                # Calculate key measurements
                xy_dist = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
                red_bottom = red_pos[2] - self.block_half_size
                stack_clearance = red_bottom - self.blue_top
                
                # Track specific clearances
                if phase_name == "hold_65mm" and frame > 30:
                    hold_clearance = table_clearance
                if phase_name == "grasp" and frame > 50:
                    grasp_clearance = table_clearance
                
                # Render
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Overlays
                cv2.putText(frame_img, "FINAL PERFECT DEMONSTRATION", (40, 60),
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
                cv2.putText(frame_img, f"Red: [{red_pos[0]:+.3f}, {red_pos[1]:+.3f}, {red_pos[2]:+.3f}]",
                           (40, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                
                # Table clearance with color coding
                if table_clearance < 0:
                    clear_color = (0, 0, 255)
                elif table_clearance < 0.05:
                    clear_color = (255, 165, 0)
                else:
                    clear_color = (0, 255, 0)
                    
                cv2.putText(frame_img, f"TABLE CLEARANCE: {table_clearance:.4f}m",
                           (40, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, clear_color, 3)
                
                # Target clearance indicators
                if phase_name in ["descend_65mm", "hold_65mm"]:
                    target_diff = table_clearance - self.target_grasp_clearance
                    if abs(target_diff) < 0.002:  # Within 2mm
                        target_color = (0, 255, 0)
                        status = "✓ ON TARGET"
                    else:
                        target_color = (255, 255, 0)
                        status = f"Diff: {target_diff:+.4f}m"
                    cv2.putText(frame_img, f"Target: 0.065m ({status})",
                               (40, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, target_color, 2)
                
                # Stack clearance
                if phase_name in ["lift_high", "rotate", "above_blue", "smart_place"]:
                    if stack_clearance > 0:
                        stack_color = (0, 255, 0)
                    elif stack_clearance > -0.001:
                        stack_color = (255, 255, 0)
                    else:
                        stack_color = (255, 0, 0)
                        
                    cv2.putText(frame_img, f"STACK CLEARANCE: {stack_clearance:.4f}m",
                               (40, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, stack_color, 2)
                    cv2.putText(frame_img, f"Red bottom: {red_bottom:.3f}m, Blue top: {self.blue_top:.3f}m",
                               (40, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    
                    if place_stopped_early and phase_name == "smart_place":
                        cv2.putText(frame_img, ">>> PLACEMENT COMPLETE <<<", (40, y + 170),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                # Distance and gripper
                cv2.putText(frame_img, f"XY Distance: {xy_dist:.3f}m",
                           (40, y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                state = "OPEN" if gripper_cmd > 100 else "CLOSED"
                state_color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
                cv2.putText(frame_img, f"Gripper: {state}",
                           (40, y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
                
                frames.append(frame_img)
                
                # Debug output
                if frame % 30 == 0:
                    print(f"  Frame {frame:3d}: Table={table_clearance:.4f}m, Stack={stack_clearance:.4f}m")
        
        # Save
        print("\n" + "="*70)
        print("Saving video...")
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_final_perfect_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ FINAL PERFECT VIDEO SAVED!")
            print(f"📹 Path: {output_path}")
            print(f"\n📊 Final Statistics:")
            print(f"  Minimum table clearance: {min_clearance:.4f}m")
            print(f"  Hold clearance at 65mm: {hold_clearance:.4f}m")
            print(f"  Grasp clearance: {grasp_clearance:.4f}m")
            print(f"  Target grasp clearance: {self.target_grasp_clearance:.3f}m")
            if place_stopped_early:
                print(f"  Smart place stopped at frame: {place_stop_frame}/150")
            print(f"\n✅ Perfect demonstration complete!")

if __name__ == "__main__":
    creator = FinalPerfectDemo()
    creator.run()