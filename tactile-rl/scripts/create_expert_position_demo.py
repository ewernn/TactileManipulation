#!/usr/bin/env python3
"""
Expert demonstration using position control for clean, repeatable trajectories.
Designed to work with the Panda's position servo actuators.
"""

import numpy as np
import mujoco
import cv2
import os

class ExpertPositionPolicy:
    """Expert policy using position control for demonstrations."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.phase = "init"
        self.phase_step = 0
        
        # Phase durations (in timesteps)
        self.phase_durations = {
            "init": 200,
            "approach_xy": 500,
            "approach_z": 400,
            "grasp": 200,
            "lift": 400,
            "move_xy": 600,
            "place": 400,
            "release": 200,
            "retreat": 400
        }
        
        # Target positions
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        
        # Gripper geometry
        self.gripper_reach = 0.103
        self.grasp_offset = -0.01  # Slightly before block center
        
        # Get initial joint configuration
        self.home_joints = np.array([0, -0.1, 0, -2.0, 0, 1.5, 0])
        
        # Pre-compute target joint configurations using forward kinematics
        self._compute_target_configs()
        
    def _compute_target_configs(self):
        """Pre-compute joint configurations for key positions."""
        # These are carefully tuned for the Panda robot to reach the blocks
        # Starting from home position facing +X (blocks are at +X)
        
        self.configs = {
            "home": self.home_joints.copy(),
            
            # Above red block - conservative approach
            # Small changes from home to move forward and up
            "above_red": np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0]),
            
            # Grasp red block - lower down
            "grasp_red": np.array([0.0, -0.1, 0.0, -1.95, 0.0, 1.6, 0.0]),
            
            # Lift position - raise up
            "lift": np.array([0.0, -0.4, 0.0, -1.7, 0.0, 1.4, 0.0]),
            
            # Above blue block - rotate base left
            "above_blue": np.array([-0.4, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0]),
            
            # Place on blue - lower down
            "place_blue": np.array([-0.4, -0.1, 0.0, -1.95, 0.0, 1.6, 0.0]),
            
            # Final retreat - back to safe position
            "retreat": np.array([-0.2, -0.5, 0.0, -1.5, 0.0, 1.3, 0.0])
        }
        
    def get_target_joints(self):
        """Get target joint positions for current phase."""
        current_joints = np.array([self.data.qpos[14 + i] for i in range(7)])
        
        # Determine target configuration based on phase
        if self.phase == "init":
            target = self.configs["home"]
            
        elif self.phase == "approach_xy":
            # Interpolate to above_red position
            alpha = min(self.phase_step / self.phase_durations["approach_xy"], 1.0)
            target = (1 - alpha) * self.configs["home"] + alpha * self.configs["above_red"]
            
        elif self.phase == "approach_z":
            # Interpolate down to grasp
            alpha = min(self.phase_step / self.phase_durations["approach_z"], 1.0)
            target = (1 - alpha) * self.configs["above_red"] + alpha * self.configs["grasp_red"]
            
        elif self.phase == "grasp":
            target = self.configs["grasp_red"]
            
        elif self.phase == "lift":
            # Interpolate up
            alpha = min(self.phase_step / self.phase_durations["lift"], 1.0)
            target = (1 - alpha) * self.configs["grasp_red"] + alpha * self.configs["lift"]
            
        elif self.phase == "move_xy":
            # Interpolate to above blue
            alpha = min(self.phase_step / self.phase_durations["move_xy"], 1.0)
            target = (1 - alpha) * self.configs["lift"] + alpha * self.configs["above_blue"]
            
        elif self.phase == "place":
            # Interpolate down
            alpha = min(self.phase_step / self.phase_durations["place"], 1.0)
            target = (1 - alpha) * self.configs["above_blue"] + alpha * self.configs["place_blue"]
            
        elif self.phase == "release":
            target = self.configs["place_blue"]
            
        elif self.phase == "retreat":
            # Interpolate to retreat position
            alpha = min(self.phase_step / self.phase_durations["retreat"], 1.0)
            target = (1 - alpha) * self.configs["place_blue"] + alpha * self.configs["retreat"]
            
        else:
            target = current_joints
            
        return target
    
    def get_gripper_command(self):
        """Get gripper command for current phase."""
        if self.phase in ["grasp", "lift", "move_xy", "place"]:
            return 0  # Closed
        else:
            return 255  # Open
            
    def step(self):
        """Update phase and step counter."""
        self.phase_step += 1
        
        # Check for phase transition
        if self.phase_step >= self.phase_durations.get(self.phase, 100):
            phases = list(self.phase_durations.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.phase_step = 0
                return True  # Phase changed
        return False

def main():
    """Create expert demonstration with position control."""
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set initial configuration
    joint_vals = [0, -0.1, 0, -2.0, 0, 1.5, 0]
    for i, val in enumerate(joint_vals):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    data.ctrl[7] = 255
    
    mujoco.mj_forward(model, data)
    
    # Initialize expert policy
    expert = ExpertPositionPolicy(model, data)
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video setup
    frames = []
    video_fps = 60  # Higher FPS for smoother video
    frame_skip = int(1.0 / (video_fps * model.opt.timestep))
    
    print("\nCreating expert position control demonstration...")
    print("=" * 70)
    
    # Get IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Get finger IDs for gripper visualization
    finger_ids = []
    for name in ["left_finger", "right_finger"]:
        finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if finger_id != -1:
            finger_ids.append(finger_id)
    
    # Run demonstration
    total_steps = sum(expert.phase_durations.values()) + 100
    
    for step in range(total_steps):
        # Get control commands
        target_joints = expert.get_target_joints()
        gripper_cmd = expert.get_gripper_command()
        
        # Apply position control
        for i in range(7):
            data.ctrl[i] = target_joints[i]
        data.ctrl[7] = gripper_cmd
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Update expert policy
        phase_changed = expert.step()
        if phase_changed:
            print(f"\n>>> Phase: {expert.phase}")
        
        # Print status periodically
        if step % 200 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            blue_pos = data.xpos[blue_id]
            
            print(f"\nStep {step:4d} | Phase: {expert.phase:12s}")
            print(f"  EE: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]")
            print(f"  Red: [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]")
            
            # Check if red block is grasped (lifted)
            if red_pos[2] > 0.46:
                print(f"  ✓ Red block lifted! Height: {red_pos[2]:.3f}m")
            
            # Check gripper opening
            if finger_ids:
                left_pos = data.xpos[finger_ids[0]]
                right_pos = data.xpos[finger_ids[1]]
                gripper_width = np.linalg.norm(left_pos - right_pos)
                print(f"  Gripper width: {gripper_width:.3f}m")
        
        # Render frame
        if step % frame_skip == 0:
            renderer.update_scene(data, camera="demo_cam")
            frame = renderer.render()
            
            # Add overlays
            cv2.putText(frame, "EXPERT POSITION CONTROL", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 2)
            cv2.putText(frame, f"Phase: {expert.phase}", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Phase progress bar
            if expert.phase in expert.phase_durations:
                progress = expert.phase_step / expert.phase_durations[expert.phase]
                bar_width = int(400 * progress)
                cv2.rectangle(frame, (40, 140), (40 + bar_width, 170), (0, 255, 0), -1)
                cv2.rectangle(frame, (40, 140), (440, 170), (255, 255, 255), 2)
            
            # Gripper state
            gripper_state = "CLOSED" if gripper_cmd < 100 else "OPEN"
            color = (0, 0, 255) if gripper_state == "CLOSED" else (0, 255, 0)
            cv2.putText(frame, f"Gripper: {gripper_state}", (40, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Time
            time_sec = step * model.opt.timestep
            cv2.putText(frame, f"Time: {time_sec:.1f}s", (40, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    
    # Final status
    final_red_pos = data.xpos[red_id]
    final_blue_pos = data.xpos[blue_id]
    print(f"\nFinal positions:")
    print(f"  Red block: [{final_red_pos[0]:.3f}, {final_red_pos[1]:.3f}, {final_red_pos[2]:.3f}]")
    print(f"  Blue block: [{final_blue_pos[0]:.3f}, {final_blue_pos[1]:.3f}, {final_blue_pos[2]:.3f}]")
    
    # Check success
    if final_red_pos[2] > final_blue_pos[2] + 0.04:
        print("\n✅ SUCCESS: Red block stacked on blue block!")
    
    # Save video
    os.makedirs("../../videos/121pm", exist_ok=True)
    output_path = "../../videos/121pm/expert_position_demo.mp4"
    
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        print(f"\n✓ Video saved to: {output_path}")
        print(f"  Duration: {len(frames)/video_fps:.1f} seconds")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {video_fps}")

if __name__ == "__main__":
    main()