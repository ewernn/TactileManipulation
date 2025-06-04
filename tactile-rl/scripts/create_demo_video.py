#\!/usr/bin/env python3
"""
Create a demonstration video showing current robot capabilities.
"""

import numpy as np
import mujoco
import cv2
import os

class DemoVideoCreator:
    def __init__(self):
        # Load model
        self.xml_path = "../franka_emika_panda/panda_demo_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Renderer with multiple camera angles
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        
        # Get IDs
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.red_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        self.blue_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        
        # Demonstration sequence
        self.sequence = [
            # Phase, duration, joint_config, gripper, description
            ("home", 60, [0, -0.1, 0, -2.0, 0, 1.5, 0], 255, "Initial Position"),
            ("reach1", 120, [0, 0.2, 0, -2.4, 0, 1.8, 0], 255, "Reaching Forward"),
            ("reach2", 120, [0, 0.4, 0, -2.6, 0, 2.0, 0], 255, "Approaching Blocks"),
            ("lower", 100, [0, 0.5, 0, -2.7, 0, 2.1, 0], 255, "Lowering to Grasp"),
            ("grasp", 80, [0, 0.5, 0, -2.7, 0, 2.1, 0], 0, "Closing Gripper"),
            ("lift", 100, [0, 0.3, 0, -2.5, 0, 2.0, 0], 0, "Attempting Lift"),
            ("rotate", 120, [-0.3, 0.3, 0, -2.5, 0, 2.0, 0], 0, "Rotating to Blue"),
            ("lower2", 100, [-0.3, 0.5, 0, -2.7, 0, 2.1, 0], 0, "Lowering to Place"),
            ("release", 80, [-0.3, 0.5, 0, -2.7, 0, 2.1, 0], 255, "Opening Gripper"),
            ("retreat", 100, [-0.1, 0.1, 0, -2.2, 0, 1.8, 0], 255, "Retreating"),
        ]
        
    def interpolate_configs(self, config1, config2, alpha):
        """Linearly interpolate between configurations."""
        return [(1-alpha)*c1 + alpha*c2 for c1, c2 in zip(config1, config2)]
    
    def create_multi_view_frame(self, views):
        """Create a frame with multiple camera views."""
        # Arrange views in a 2x2 grid
        top_row = np.hstack([views[0], views[1]])
        bottom_row = np.hstack([views[2], views[3]])
        combined = np.vstack([top_row, bottom_row])
        
        # Scale down to fit 1280x720
        scale = min(1280 / combined.shape[1], 720 / combined.shape[0])
        new_width = int(combined.shape[1] * scale)
        new_height = int(combined.shape[0] * scale)
        
        return cv2.resize(combined, (new_width, new_height))
    
    def run(self):
        """Create the demonstration video."""
        print("Creating demonstration video...")
        print("=" * 70)
        
        # Reset simulation
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Video settings
        frames = []
        fps = 30
        
        # Camera names to use
        cameras = ["demo_cam", "robot_cam", "side_cam", "top_cam"]
        
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
                
                # Render from multiple cameras
                views = []
                for cam_name in cameras:
                    try:
                        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                        self.renderer.update_scene(self.data, camera=cam_id)
                    except:
                        self.renderer.update_scene(self.data)
                    
                    view = self.renderer.render()
                    
                    # Add camera label
                    cv2.putText(view, cam_name.upper(), (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    views.append(view)
                
                # Create multi-view frame
                multi_view = self.create_multi_view_frame(views)
                
                # Add main title and info
                cv2.putText(multi_view, "PANDA ROBOT DEMONSTRATION", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 3)
                cv2.putText(multi_view, f"Phase: {description}", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Add position info
                ee_pos = self.data.xpos[self.ee_id]
                red_pos = self.data.xpos[self.red_id]
                info_y = multi_view.shape[0] - 100
                
                cv2.putText(multi_view, f"End-Effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
                           (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(multi_view, f"Red Block: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]",
                           (20, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
                
                # Gripper state
                gripper_state = "OPEN" if gripper > 100 else "CLOSED"
                color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
                cv2.putText(multi_view, f"Gripper: {gripper_state}",
                           (20, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                frames.append(multi_view)
                
                # Print progress occasionally
                if frame % 30 == 0:
                    print(f"  Frame {frame}/{duration}: EE at [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        
        # Save video
        print("\n" + "=" * 70)
        print("Saving video...")
        
        output_path = "../../videos/121pm/panda_demo_multi_view.mp4"
        
        if frames:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ Video saved successfully\!")
            print(f"📹 Path: {output_path}")
            print(f"⏱️  Duration: {len(frames)/fps:.1f} seconds")
            print(f"📐 Resolution: {width}x{height}")
            print(f"🎬 Total frames: {len(frames)}")
        
        # Also save a single camera view for easier viewing
        print("\nCreating single camera view...")
        
        # Reset and create single view
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        single_frames = []
        
        for phase_idx, (phase_name, duration, target_config, gripper, description) in enumerate(self.sequence):
            current_config = [self.data.qpos[14 + i] for i in range(7)]
            
            for frame in range(duration):
                alpha = frame / duration
                
                if phase_idx == 0:
                    config = target_config
                else:
                    config = self.interpolate_configs(current_config, target_config, alpha)
                
                for i, val in enumerate(config):
                    self.data.ctrl[i] = val
                self.data.ctrl[7] = gripper
                
                mujoco.mj_step(self.model, self.data)
                
                # Render single view
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                self.renderer.update_scene(self.data, camera=cam_id)
                frame_img = self.renderer.render()
                
                # Add overlays
                cv2.putText(frame_img, "PANDA ROBOT DEMONSTRATION", (40, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 3)
                cv2.putText(frame_img, f"{description}", (40, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Progress bar
                progress = (sum([s[1] for s in self.sequence[:phase_idx]]) + frame) / sum([s[1] for s in self.sequence])
                bar_width = int(1200 * progress)
                cv2.rectangle(frame_img, (40, 160), (40 + bar_width, 180), (0, 255, 0), -1)
                cv2.rectangle(frame_img, (40, 160), (1240, 180), (255, 255, 255), 2)
                
                single_frames.append(frame_img)
        
        # Save single view
        output_single = "../../videos/121pm/panda_demo_single_view.mp4"
        
        if single_frames:
            height, width = single_frames[0].shape[:2]
            out = cv2.VideoWriter(output_single, fourcc, fps, (width, height))
            
            for frame in single_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            print(f"\n✅ Single view saved\!")
            print(f"📹 Path: {output_single}")
            
        print("\n🎉 Video creation complete\!")

if __name__ == "__main__":
    creator = DemoVideoCreator()
    creator.run()
