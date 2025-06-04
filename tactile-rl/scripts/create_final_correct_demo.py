#!/usr/bin/env python3
"""
Create final demonstration with correct clearances and stack detection.
- Descend to 0.065m table clearance for grasp
- Stop placing when stack clearance reaches 0
"""

import numpy as np
import mujoco
import cv2
import os

class FinalCorrectDemo:
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
        print(f"Blue block top: {self.blue_top:.3f}m")
        print(f"Target grasp clearance: {self.target_grasp_clearance:.3f}m")
        
        # Sequence with correct clearances
        self.sequence = [
            # Start at safe height
            ("home", 60, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Home - 0.2m Clearance"),
            
            # Approach
            ("approach", 80, [0, -0.3, 0, -2.1, 0, 1.65, 0], 255, "Approach Position"),
            
            # Position above red
            ("position", 80, [0, -0.15, 0, -2.25, 0, 1.75, 0], 255, "Position Above Red"),
            
            # Descend to EXACTLY 0.065m clearance
            ("descend_65mm", 100, [0, -0.05, 0, -2.37, 0, 1.87, 0], 255, "Descend to 65mm"),
            
            # Grasp
            ("grasp", 100, [0, -0.05, 0, -2.37, 0, 1.87, 0], 0, "Grasp Block"),
            
            # Lift to safe height
            ("lift", 80, [0, -0.3, 0, -2.1, 0, 1.65, 0], 0, "Lift Block"),
            
            # Lift higher for clearance
            ("lift_high", 80, [0, -0.5, 0, -1.8, 0, 1.4, 0], 0, "Lift to Clear Blue"),
            
            # Rotate to blue
            ("rotate", 120, [-0.35, -0.5, 0, -1.8, 0, 1.4, 0], 0, "Rotate to Blue"),
            
            # Position above blue - high enough to clear
            ("above_blue", 80, [-0.35, -0.3, 0, -2.1, 0, 1.65, 0], 0, "Above Blue Block"),
            
            # SMART PLACE - will stop when stack clearance = 0
            ("smart_place", 120, [-0.35, -0.1, 0, -2.3, 0, 1.8, 0], 0, "Smart Place"),
            
            # Release
            ("release", 80, [-0.35, -0.1, 0, -2.3, 0, 1.8, 0], 255, "Release"),
            
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
        print("\nCreating FINAL CORRECT DEMONSTRATION...")
        print("- Descend to 0.065m table clearance")
        print("- Stop placing when stack clearance = 0")
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
        place_stopped_early = False
        place_stop_frame = None
        
        # Execute sequence
        for phase_idx, (phase_name, duration, target_config, gripper_cmd, description) in enumerate(self.sequence):
            print(f"\nPhase: {phase_name} - {description}")
            
            current_config = [self.data.qpos[14 + i] for i in range(7)]
            
            for frame in range(duration):
                # For smart place, check if we should stop
                if phase_name == "smart_place":
                    red_pos = self.data.xpos[self.red_id]
                    red_bottom = red_pos[2] - self.block_half_size
                    stack_clearance = red_bottom - self.blue_top
                    
                    if stack_clearance <= 0 and not place_stopped_early:
                        place_stopped_early = True
                        place_stop_frame = frame
                        print(f"  STOPPING PLACE! Stack clearance reached 0 at frame {frame}")
                        # Hold current position
                        target_config = [self.data.qpos[14 + i] for i in range(7)]
                
                alpha = frame / duration
                
                if phase_idx == 0:
                    config = target_config
                elif place_stopped_early and phase_name == "smart_place":
                    # Hold position after stopping
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
                
                # Calculate key measurements
                xy_dist = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
                red_bottom = red_pos[2] - self.block_half_size
                stack_clearance = red_bottom - self.blue_top
                
                # Track grasp clearance
                if phase_name == "grasp" and frame > 50:
                    grasp_clearance = table_clearance
                
                # Render
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Overlays
                cv2.putText(frame_img, "FINAL CORRECT DEMONSTRATION", (40, 60),
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
                
                # Table clearance
                clear_color = (0, 255, 0) if table_clearance > 0.05 else (255, 165, 0) if table_clearance > 0 else (255, 0, 0)
                cv2.putText(frame_img, f"TABLE CLEARANCE: {table_clearance:.3f}m",
                           (40, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, clear_color, 3)
                
                # Target clearance for descend phase
                if phase_name == "descend_65mm":
                    target_diff = table_clearance - self.target_grasp_clearance
                    target_color = (0, 255, 0) if abs(target_diff) < 0.005 else (255, 255, 0)
                    cv2.putText(frame_img, f"Target: 0.065m (Diff: {target_diff:+.3f}m)",
                               (40, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, target_color, 2)
                
                # Stack clearance for place phases
                if phase_name in ["lift_high", "rotate", "above_blue", "smart_place"]:
                    stack_color = (0, 255, 0) if stack_clearance > 0 else (255, 0, 0)
                    cv2.putText(frame_img, f"STACK CLEARANCE: {stack_clearance:.3f}m",
                               (40, y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, stack_color, 2)
                    cv2.putText(frame_img, f"Red bottom: {red_bottom:.3f}m, Blue top: {self.blue_top:.3f}m",
                               (40, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    
                    if place_stopped_early and phase_name == "smart_place":
                        cv2.putText(frame_img, ">>> PLACEMENT COMPLETE <<<", (40, y + 170),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Distance info
                cv2.putText(frame_img, f"XY Distance: {xy_dist:.3f}m",
                           (40, y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Gripper state
                state = "OPEN" if gripper_cmd > 100 else "CLOSED"
                state_color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
                cv2.putText(frame_img, f"Gripper: {state}",
                           (40, y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
                
                frames.append(frame_img)
                
                # Debug output
                if frame % 30 == 0:
                    print(f"  Frame {frame:3d}: Clear={table_clearance:.3f}m, Stack={stack_clearance:.3f}m")
        
        # Save
        print("\nSaving video...")
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_final_correct_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ FINAL CORRECT VIDEO SAVED!")
            print(f"📹 Path: {output_path}")
            print(f"\n📊 Final Statistics:")
            print(f"  Minimum table clearance: {min_clearance:.3f}m")
            print(f"  Grasp clearance: {grasp_clearance:.3f}m")
            print(f"  Target grasp clearance: {self.target_grasp_clearance:.3f}m")
            if place_stopped_early:
                print(f"  Smart place stopped at frame: {place_stop_frame}")
            print(f"\n✅ Demonstration complete with correct clearances!")

if __name__ == "__main__":
    creator = FinalCorrectDemo()
    creator.run()