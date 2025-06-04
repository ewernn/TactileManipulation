#!/usr/bin/env python3
"""
Create ULTIMATE final demonstration with EXACT 0.065m clearance.
"""

import numpy as np
import mujoco
import cv2
import os

class UltimateFinalDemo:
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
        self.block_half_size = 0.025
        self.target_grasp_clearance = 0.065
        self.blue_top = 0.47
        
        # Sequence - fine-tuned for exact 0.065m
        self.sequence = [
            # Start
            ("home", 20, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Home - 0.2m Clearance"),
            
            # Approach
            ("approach", 25, [0, -0.25, 0, -2.15, 0, 1.65, 0], 255, "Approach"),
            
            # Position - stay higher, don't go too low
            ("position", 25, [0, -0.20, 0, -2.20, 0, 1.70, 0], 255, "Position Above"),
            
            # Fast descend to intermediate position
            ("descend_fast", 60, [0, -0.18, 0, -2.22, 0, 1.72, 0], 255, "Descend Fast"),
            
            # Slow descend to EXACTLY 65mm - final calibration for 65mm
            ("descend_65mm", 80, [0, -0.181, 0, -2.219, 0, 1.719, 0], 255, "Descend to 65mm"),
            
            # Hold and verify - give more time to settle
            ("hold_65mm", 25, None, 255, "Hold at 65mm"),  # None = hold current position
            
            # Grasp
            ("grasp", 30, None, 0, "Grasp"),  # None = hold current position
            
            # Lift
            ("lift", 25, [0, -0.25, 0, -2.15, 0, 1.65, 0], 0, "Lift"),
            
            # Lift high
            ("lift_high", 25, [0, -0.5, 0, -1.8, 0, 1.4, 0], 0, "Lift High"),
            
            # Rotate - add 10 degree rotation on joint 7
            ("rotate", 35, [-0.35, -0.5, 0, -1.8, 0, 1.4, 0.174], 0, "Rotate"),  # 0.174 rad = 10 deg
            
            # Above blue - maintain rotation, adjust for -5mm Y offset
            ("above_blue", 25, [-0.37, -0.35, 0, -2.05, 0, 1.55, 0.174], 0, "Above Blue"),
            
            # Smart place - maintain alignment
            ("smart_place", 40, [-0.37, -0.15, 0, -2.25, 0, 1.75, 0.174], 0, "Smart Place"),
            
            # Release
            ("release", 25, None, 255, "Release"),
            
            # Retreat - remove rotation
            ("retreat", 25, [-0.35, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Retreat"),
            
            # Return
            ("return", 30, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Return"),
        ]
        
    def interpolate_configs(self, config1, config2, alpha, linear=False):
        if linear:
            t = alpha  # Linear interpolation
        else:
            # Ease-in/ease-out
            if alpha < 0.5:
                t = 2 * alpha * alpha
            else:
                t = 1 - pow(-2 * alpha + 2, 2) / 2
        return [(1-t)*c1 + t*c2 for c1, c2 in zip(config1, config2)]
    
    def compute_stacking_reward(self):
        """Compute reward for how well red block is stacked on blue block."""
        # Get block positions and orientations
        red_pos = self.data.xpos[self.red_id].copy()
        blue_pos = self.data.xpos[self.blue_id].copy()
        
        # Get block quaternions
        red_quat = self.data.xquat[self.red_id].copy()
        blue_quat = self.data.xquat[self.blue_id].copy()
        
        # 1. Vertical alignment - red should be directly above blue
        xy_offset = np.sqrt((red_pos[0] - blue_pos[0])**2 + (red_pos[1] - blue_pos[1])**2)
        alignment_reward = np.exp(-10 * xy_offset)  # Exponential decay, max 1.0
        
        # 2. Height reward - red bottom should be at blue top
        red_bottom = red_pos[2] - self.block_half_size
        blue_top = blue_pos[2] + self.block_half_size
        height_error = abs(red_bottom - blue_top)
        height_reward = np.exp(-20 * height_error)  # More sensitive to height
        
        # 3. Orientation alignment - blocks should have similar orientation
        # Convert quaternions to rotation matrices
        red_mat = np.zeros(9)
        blue_mat = np.zeros(9)
        mujoco.mju_quat2Mat(red_mat, red_quat)
        mujoco.mju_quat2Mat(blue_mat, blue_quat)
        red_mat = red_mat.reshape(3, 3)
        blue_mat = blue_mat.reshape(3, 3)
        
        # Compare Z axes (up direction)
        z_alignment = np.dot(red_mat[:, 2], blue_mat[:, 2])
        orientation_reward = (z_alignment + 1) / 2  # Map from [-1, 1] to [0, 1]
        
        # 4. Stability - red block should not be tilted
        red_z_axis = red_mat[:, 2]
        upright_score = red_z_axis[2]  # How much Z axis points up
        stability_reward = max(0, upright_score)
        
        # 5. Contact reward - bonus if blocks are touching
        contact_bonus = 1.0 if height_error < 0.005 else 0.0  # 5mm tolerance
        
        # Combine rewards with weights
        total_reward = (
            0.3 * alignment_reward +    # 30% for XY alignment
            0.3 * height_reward +       # 30% for correct height
            0.2 * orientation_reward +  # 20% for orientation match
            0.1 * stability_reward +    # 10% for stability
            0.1 * contact_bonus        # 10% for contact
        )
        
        # Return detailed metrics
        return {
            'total_reward': total_reward,
            'alignment_reward': alignment_reward,
            'height_reward': height_reward,
            'orientation_reward': orientation_reward,
            'stability_reward': stability_reward,
            'contact_bonus': contact_bonus,
            'xy_offset': xy_offset,
            'height_error': height_error,
            'z_alignment': z_alignment
        }
    
    def get_gripper_info(self):
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
        print("\nULTIMATE FINAL DEMO - EXACT 0.065m")
        print("=" * 70)
        
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        initial_config = self.sequence[0][2]
        for i, val in enumerate(initial_config):
            self.data.qpos[14 + i] = val
            self.data.ctrl[i] = val
        self.data.ctrl[7] = 255
        
        mujoco.mj_step(self.model, self.data)
        
        frames = []
        fps = 120  # 4x faster to make ~10 second video
        
        # Stats
        grasp_clearance = None
        hold_clearance = None
        place_stop_frame = None
        final_place_config = None
        
        for phase_idx, (phase_name, duration, target_config, gripper_cmd, description) in enumerate(self.sequence):
            print(f"\nPhase: {phase_name}")
            
            current_config = [self.data.qpos[14 + i] for i in range(7)]
            
            if phase_name == "release" and final_place_config:
                target_config = final_place_config
            
            # Handle None target configs (hold current position)
            if target_config is None:
                target_config = current_config
                print(f"  {phase_name}: Holding current position")
            
            for frame in range(duration):
                # Smart place logic
                if phase_name == "smart_place":
                    red_pos = self.data.xpos[self.red_id]
                    red_bottom = red_pos[2] - self.block_half_size
                    stack_clearance = red_bottom - self.blue_top
                    
                    if stack_clearance <= 0 and not place_stop_frame:
                        place_stop_frame = frame
                        final_place_config = [self.data.qpos[14 + i] for i in range(7)]
                        print(f">>> STOP! Stack clearance = {stack_clearance:.4f}m at frame {frame}")
                        target_config = final_place_config
                
                alpha = frame / duration
                
                if phase_idx == 0:
                    config = target_config
                elif place_stop_frame and phase_name == "smart_place":
                    config = final_place_config
                elif phase_name in ["hold_65mm", "grasp"]:
                    # For hold/grasp phases, use the target (which is the current position)
                    config = target_config
                else:
                    # Use linear interpolation for descend phases
                    use_linear = (phase_name in ["descend_fast", "descend_65mm"])
                    config = self.interpolate_configs(current_config, target_config, alpha, linear=use_linear)
                
                for i, val in enumerate(config):
                    self.data.ctrl[i] = val
                self.data.ctrl[7] = gripper_cmd
                
                mujoco.mj_step(self.model, self.data)
                
                ee_pos = self.data.xpos[self.ee_id]
                red_pos = self.data.xpos[self.red_id]
                
                table_clearance, lowest_z = self.get_gripper_info()
                
                red_bottom = red_pos[2] - self.block_half_size
                stack_clearance = red_bottom - self.blue_top
                
                # Track clearances
                if phase_name == "hold_65mm" and frame > 40:
                    hold_clearance = table_clearance
                if phase_name == "grasp" and frame > 50:
                    grasp_clearance = table_clearance
                
                # Render
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # UI
                cv2.putText(frame_img, "ULTIMATE FINAL DEMO", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Progress
                progress = (sum([s[1] for s in self.sequence[:phase_idx]]) + frame) / sum([s[1] for s in self.sequence])
                bar_width = int(1200 * progress)
                cv2.rectangle(frame_img, (40, 160), (40 + bar_width, 180), (0, 255, 0), -1)
                cv2.rectangle(frame_img, (40, 160), (1240, 180), (255, 255, 255), 2)
                
                # Info
                y = 220
                
                # Table clearance
                clear_color = (0, 255, 0) if table_clearance > 0.05 else (255, 165, 0) if table_clearance > 0 else (255, 0, 0)
                cv2.putText(frame_img, f"TABLE CLEARANCE: {table_clearance:.4f}m",
                           (40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, clear_color, 3)
                
                # 65mm target
                if phase_name in ["descend_65mm", "hold_65mm", "grasp"]:
                    diff = table_clearance - 0.065
                    if abs(diff) < 0.003:
                        color = (0, 255, 0)
                        text = "✓ PERFECT!"
                    else:
                        color = (255, 255, 0)
                        text = f"Diff: {diff:+.4f}m"
                    cv2.putText(frame_img, f"Target: 0.0650m ({text})",
                               (40, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Stack clearance
                if phase_name in ["lift_high", "rotate", "above_blue", "smart_place"]:
                    stack_color = (0, 255, 0) if stack_clearance > 0 else (255, 0, 0)
                    cv2.putText(frame_img, f"STACK CLEARANCE: {stack_clearance:.4f}m",
                               (40, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, stack_color, 2)
                    
                    if place_stop_frame and phase_name == "smart_place":
                        cv2.putText(frame_img, ">>> PLACED! <<<", (40, y + 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Positions
                cv2.putText(frame_img, f"EE: [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]",
                           (40, y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_img, f"Red: [{red_pos[0]:+.3f}, {red_pos[1]:+.3f}, {red_pos[2]:+.3f}]",
                           (40, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                
                # Compute and display stacking reward
                if phase_name in ["smart_place", "release", "retreat"]:
                    reward_info = self.compute_stacking_reward()
                    
                    # Total reward with color coding
                    reward_color = (0, 255, 0) if reward_info['total_reward'] > 0.8 else \
                                  (255, 255, 0) if reward_info['total_reward'] > 0.5 else (255, 0, 0)
                    cv2.putText(frame_img, f"STACKING REWARD: {reward_info['total_reward']:.3f}",
                               (40, y + 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, reward_color, 2)
                    
                    # Track final reward
                    if phase_name == "retreat" and frame == duration - 1:
                        self.final_reward = reward_info
                
                frames.append(frame_img)
                
                if frame % 10 == 0 or (phase_name in ["hold_65mm", "grasp"] and frame < 3):
                    print(f"  Frame {frame}: Table={table_clearance:.4f}m, Stack={stack_clearance:.4f}m")
                    if phase_name == "descend_65mm" and frame > 0 and frame % 10 == 0:
                        print(f"    Descent rate: {(self.last_clearance - table_clearance)*1000:.1f}mm/10frames")
                if phase_name == "descend_65mm":
                    self.last_clearance = table_clearance
        
        # Save
        print("\nSaving...")
        os.makedirs("../../videos/121pm", exist_ok=True)
        output_path = "../../videos/121pm/panda_ultimate_final_demo.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            
            print(f"\n✅ SAVED: {output_path}")
            print(f"\nStats:")
            if hold_clearance:
                print(f"  Hold clearance: {hold_clearance:.4f}m")
            if grasp_clearance:
                print(f"  Grasp clearance: {grasp_clearance:.4f}m")
                print(f"  Target: 0.0650m")
                print(f"  Error: {abs(grasp_clearance - 0.065):.4f}m")
            if place_stop_frame:
                print(f"  Place stopped at frame {place_stop_frame}")
            print(f"  Total duration: {sum([s[1] for s in self.sequence]) / fps:.1f} seconds")
            
            if hasattr(self, 'final_reward'):
                print(f"\nFinal Stacking Reward: {self.final_reward['total_reward']:.3f}")
                print(f"  Alignment: {self.final_reward['alignment_reward']:.3f} (XY offset: {self.final_reward['xy_offset']*1000:.1f}mm)")
                print(f"  Height: {self.final_reward['height_reward']:.3f} (error: {self.final_reward['height_error']*1000:.1f}mm)")
                print(f"  Orientation: {self.final_reward['orientation_reward']:.3f}")
                print(f"  Stability: {self.final_reward['stability_reward']:.3f}")
                print(f"  Contact: {self.final_reward['contact_bonus']:.1f}")

if __name__ == "__main__":
    creator = UltimateFinalDemo()
    creator.run()