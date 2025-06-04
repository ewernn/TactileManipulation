#!/usr/bin/env python3
"""
Create final demonstration with proper height accounting for block stacking.
"""

import numpy as np
import mujoco
import cv2
import os

class FinalDemoProperHeight:
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
        self.block_height = 0.05  # Each block is 5cm tall (0.025 * 2)
        self.gripper_clearance = 0.03  # Keep gripper 3cm above surfaces
        
        # Sequence with proper height management
        self.sequence = [
            # Phase, duration, [j0, j1, j2, j3, j4, j5, j6], gripper, description
            # Start higher to avoid table collision
            ("home", 60, [0, -0.6, 0, -1.8, 0, 1.4, 0], 255, "Home - Safe Height"),
            # Approach red block from higher position
            ("approach", 80, [0, -0.3, 0, -2.2, 0, 1.7, 0], 255, "Approach Red Block"),
            # Position above red block with clearance
            ("position", 80, [0, -0.1, 0, -2.35, 0, 1.85, 0], 255, "Position Above Red"),
            # Descend to grasp height (keep higher to avoid table)
            ("descend", 60, [0, -0.05, 0, -2.4, 0, 1.9, 0], 255, "Descend to Grasp"),
            # Grasp
            ("grasp", 80, [0, -0.05, 0, -2.4, 0, 1.9, 0], 0, "Close Gripper"),
            # Lift to safe height
            ("lift", 60, [0, -0.2, 0, -2.3, 0, 1.8, 0], 0, "Lift Block"),
            # Lift higher to clear blue block
            ("lift_high", 60, [0, -0.4, 0, -2.1, 0, 1.6, 0], 0, "Lift to Clear Blue"),
            # Rotate to blue position
            ("rotate", 100, [-0.5, -0.4, 0, -2.1, 0, 1.6, 0], 0, "Rotate to Blue"),
            # Position above blue block (accounting for its height)
            ("above_blue", 80, [-0.5, -0.2, 0, -2.3, 0, 1.8, 0], 0, "Above Blue Block"),
            # Descend to place (blue block height + clearance)
            ("place", 80, [-0.5, -0.1, 0, -2.35, 0, 1.85, 0], 0, "Place on Blue"),
            # Release
            ("release", 60, [-0.5, -0.1, 0, -2.35, 0, 1.85, 0], 255, "Release"),
            # Retreat upward
            ("retreat", 80, [-0.5, -0.4, 0, -2.1, 0, 1.6, 0], 255, "Retreat"),
            # Return home
            ("return", 100, [0, -0.6, 0, -1.8, 0, 1.4, 0], 255, "Return Home"),
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
        print("Creating FINAL DEMO with PROPER HEIGHT...")
        print("- Avoiding table collision")
        print("- Lifting red block high enough to clear blue block")
        print(f"- Blue block height: {self.block_height}m")
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
        min_grasp_clearance = float('inf')
        max_lift_height = 0
        
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
                if phase_name in ["descend", "grasp"]:
                    min_grasp_clearance = min(min_grasp_clearance, table_clearance)
                if phase_name in ["lift_high", "rotate"]:
                    max_lift_height = max(max_lift_height, red_pos[2])
                
                # Calculate key distances
                xy_dist = np.sqrt((ee_pos[0] - red_pos[0])**2 + (ee_pos[1] - red_pos[1])**2)
                red_above_blue = red_pos[2] - blue_pos[2]
                blue_top = blue_pos[2] + self.block_height/2
                red_bottom = red_pos[2] - self.block_height/2
                clearance_to_blue = red_bottom - blue_top
                
                # Render
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Overlays
                cv2.putText(frame_img, "PANDA ROBOT - PROPER HEIGHT DEMO", (40, 60),
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
                cv2.putText(frame_img, f"Red Block:    [{red_pos[0]:+.3f}, {red_pos[1]:+.3f}, {red_pos[2]:+.3f}]",
                           (40, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                cv2.putText(frame_img, f"Blue Block:   [{blue_pos[0]:+.3f}, {blue_pos[1]:+.3f}, {blue_pos[2]:+.3f}]",
                           (40, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                
                # Height information
                cv2.putText(frame_img, "HEIGHT INFO:", (40, y + 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(frame_img, f"Red above Blue: {red_above_blue:.3f}m",
                           (40, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(frame_img, f"Clearance to Blue top: {clearance_to_blue:.3f}m",
                           (40, y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                           (0, 255, 0) if clearance_to_blue > 0 else (255, 0, 0), 2)
                
                # Table clearance - CRITICAL
                clear_color = (0, 255, 0) if table_clearance > 0.03 else (255, 0, 0)
                cv2.putText(frame_img, f"TABLE CLEARANCE: {table_clearance:.3f}m",
                           (40, y + 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, clear_color, 3)
                
                # Joint 6 info
                joint_config = [self.data.qpos[14 + i] for i in range(7)]
                cv2.putText(frame_img, f"Joint 6: {joint_config[6]:+.3f} rad ({np.degrees(joint_config[6]):+.1f}°)",
                           (40, y + 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
                
                # Gripper state
                state = "OPEN" if gripper_cmd > 100 else "CLOSED"
                state_color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
                cv2.putText(frame_img, f"Gripper: {state}",
                           (40, y + 285), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
                
                frames.append(frame_img)
                
                if frame % 30 == 0:
                    print(f"  Table clear={table_clearance:.3f}m, "
                          f"Red-Blue clear={clearance_to_blue:.3f}m")
        
        # Save
        print("\nSaving video...")
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_final_proper_height.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ FINAL VIDEO WITH PROPER HEIGHT SAVED!")
            print(f"📹 Path: {output_path}")
            print(f"📊 Statistics:")
            print(f"  Minimum table clearance: {min_clearance:.3f}m")
            print(f"  Minimum clearance during grasp: {min_grasp_clearance:.3f}m")
            print(f"  Maximum lift height: {max_lift_height:.3f}m")
            print(f"  Block height: {self.block_height}m")
            
            if min_clearance > 0:
                print("\n✅ SUCCESS: Gripper stayed above table!")
            else:
                print("\n⚠️  WARNING: Table collision detected")

if __name__ == "__main__":
    creator = FinalDemoProperHeight()
    creator.run()