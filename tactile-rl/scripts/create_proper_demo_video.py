#!/usr/bin/env python3
"""
Create a proper demonstration video with correct gripper orientation.
"""

import numpy as np
import mujoco
import cv2
import os

class ProperDemoVideo:
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
        
        # Demonstration sequence with proper wrist orientation
        # Joint 6 controls the wrist rotation - needs to be rotated to point gripper forward
        self.sequence = [
            # Phase, duration, joint_config, gripper, description
            # Note: Joint 5 (wrist pitch) and Joint 6 (wrist rotation) are key for gripper orientation
            ("home", 60, [0, -0.5, 0, -2.0, 0, 1.571, 0.785], 255, "Initial Position - Gripper Forward"),
            ("lift_clear", 80, [0, -0.6, 0, -1.8, 0, 1.571, 0.785], 255, "Lift Clear of Table"),
            ("reach_high", 100, [0, -0.4, 0, -2.2, 0, 1.8, 0.785], 255, "Reach Forward (High)"),
            ("position_above", 80, [0, -0.2, 0, -2.4, 0, 1.9, 0.785], 255, "Position Above Block"),
            ("descend", 100, [0, 0.1, 0, -2.5, 0, 2.0, 0.785], 255, "Descend to Block"),
            ("grasp", 80, [0, 0.1, 0, -2.5, 0, 2.0, 0.785], 0, "Close Gripper"),
            ("lift_with_block", 100, [0, -0.3, 0, -2.2, 0, 1.8, 0.785], 0, "Lift Block"),
            ("rotate_to_blue", 120, [-0.5, -0.3, 0, -2.2, 0, 1.8, 0.785], 0, "Rotate to Blue"),
            ("position_above_blue", 80, [-0.5, -0.1, 0, -2.4, 0, 1.9, 0.785], 0, "Above Blue Block"),
            ("place", 100, [-0.5, 0.1, 0, -2.5, 0, 2.0, 0.785], 0, "Place on Blue"),
            ("release", 60, [-0.5, 0.1, 0, -2.5, 0, 2.0, 0.785], 255, "Release Block"),
            ("retreat_up", 80, [-0.5, -0.4, 0, -2.0, 0, 1.7, 0.785], 255, "Retreat Upward"),
            ("return_home", 100, [0, -0.5, 0, -2.0, 0, 1.571, 0.785], 255, "Return Home"),
        ]
        
    def interpolate_configs(self, config1, config2, alpha):
        """Linearly interpolate between configurations."""
        return [(1-alpha)*c1 + alpha*c2 for c1, c2 in zip(config1, config2)]
    
    def get_gripper_info(self):
        """Get gripper orientation and height info."""
        # Get gripper orientation
        hand_quat = self.data.xquat[self.ee_id].copy()
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, hand_quat)
        rotmat = rotmat.reshape(3, 3)
        
        # Gripper forward direction (Z-axis in hand frame)
        gripper_forward = rotmat[:, 2]
        
        # Get finger positions to check table clearance
        left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        
        min_finger_height = float('inf')
        if left_finger_id != -1:
            min_finger_height = min(min_finger_height, self.data.xpos[left_finger_id][2])
        if right_finger_id != -1:
            min_finger_height = min(min_finger_height, self.data.xpos[right_finger_id][2])
        
        # Table height is approximately 0.4m
        table_clearance = min_finger_height - 0.4
        
        return gripper_forward, table_clearance
    
    def run(self):
        """Create the demonstration video."""
        print("Creating proper demonstration video with correct gripper orientation...")
        print("=" * 70)
        
        # Reset simulation
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Set initial configuration
        initial_config = self.sequence[0][2]
        for i, val in enumerate(initial_config):
            self.data.qpos[14 + i] = val
            self.data.ctrl[i] = val
        self.data.ctrl[7] = 255  # Open gripper
        
        mujoco.mj_forward(self.model, self.data)
        
        # Video settings
        frames = []
        fps = 30
        
        # Execute sequence
        for phase_idx, (phase_name, duration, target_config, gripper, description) in enumerate(self.sequence):
            print(f"\nPhase: {phase_name} - {description}")
            
            # Get current config
            current_config = [self.data.qpos[14 + i] for i in range(7)]
            
            # Interpolate over duration
            for frame in range(duration):
                # Calculate interpolation factor
                alpha = frame / duration
                
                # Interpolate joint positions
                if phase_idx == 0:
                    config = target_config
                else:
                    config = self.interpolate_configs(current_config, target_config, alpha)
                
                # Apply configuration
                for i, val in enumerate(config):
                    self.data.ctrl[i] = val
                self.data.ctrl[7] = gripper
                
                # Step physics
                mujoco.mj_step(self.model, self.data)
                
                # Render frame
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Get gripper info
                gripper_forward, table_clearance = self.get_gripper_info()
                
                # Add overlays
                cv2.putText(frame_img, "PANDA ROBOT - PROPER GRIPPER ORIENTATION", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Progress bar
                progress = (sum([s[1] for s in self.sequence[:phase_idx]]) + frame) / sum([s[1] for s in self.sequence])
                bar_width = int(1200 * progress)
                cv2.rectangle(frame_img, (40, 160), (40 + bar_width, 180), (0, 255, 0), -1)
                cv2.rectangle(frame_img, (40, 160), (1240, 180), (255, 255, 255), 2)
                
                # Position info
                ee_pos = self.data.xpos[self.ee_id]
                red_pos = self.data.xpos[self.red_id]
                blue_pos = self.data.xpos[self.blue_id]
                
                info_y = 250
                cv2.putText(frame_img, f"End-Effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
                           (40, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Red Block: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]",
                           (40, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 100), 2)
                
                # Gripper orientation info
                cv2.putText(frame_img, f"Gripper Forward Vector: [{gripper_forward[0]:.2f}, {gripper_forward[1]:.2f}, {gripper_forward[2]:.2f}]",
                           (40, info_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 255), 2)
                
                # Table clearance warning
                clearance_color = (0, 255, 0) if table_clearance > 0.02 else (0, 0, 255)
                cv2.putText(frame_img, f"Table Clearance: {table_clearance:.3f}m",
                           (40, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, clearance_color, 2)
                
                # Gripper state
                gripper_state = "OPEN" if gripper > 100 else "CLOSED"
                color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
                cv2.putText(frame_img, f"Gripper: {gripper_state}",
                           (40, info_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                
                # Joint configuration display
                joint_config = [self.data.qpos[14 + i] for i in range(7)]
                joint_str = "Joints: [" + ", ".join([f"{j:.2f}" for j in joint_config]) + "]"
                cv2.putText(frame_img, joint_str,
                           (40, frame_img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                frames.append(frame_img)
                
                # Print progress occasionally
                if frame % 30 == 0:
                    print(f"  Frame {frame}/{duration}: EE at [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}], "
                          f"Clearance: {table_clearance:.3f}m")
        
        # Save video
        print("\n" + "=" * 70)
        print("Saving video...")
        
        output_path = "../../videos/121pm/panda_proper_gripper_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ Video saved successfully!")
            print(f"📹 Path: {output_path}")
            print(f"⏱️  Duration: {len(frames)/fps:.1f} seconds")
            print(f"📐 Resolution: {width}x{height}")
            print(f"🎬 Total frames: {len(frames)}")
            
            # Also save key frames
            key_frame_indices = {
                'initial': 0,
                'lifted': int(sum([s[1] for s in self.sequence[:2]])),
                'above_block': int(sum([s[1] for s in self.sequence[:4]])),
                'grasping': int(sum([s[1] for s in self.sequence[:6]])),
                'lifted_block': int(sum([s[1] for s in self.sequence[:7]])),
                'final': len(frames) - 1
            }
            
            for name, idx in key_frame_indices.items():
                if idx < len(frames):
                    key_path = f"../../videos/121pm/key_frame_{name}.png"
                    cv2.imwrite(key_path, cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
                    print(f"  Saved key frame: {name}")
            
        print("\n🎉 Video creation complete!")

if __name__ == "__main__":
    creator = ProperDemoVideo()
    creator.run()