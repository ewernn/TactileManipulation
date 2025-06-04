#!/usr/bin/env python3
"""
Create a working expert demonstration with carefully tuned positions.
Based on analysis: blocks are very close to robot base, need different approach.
"""

import numpy as np
import mujoco
import cv2
import os

class WorkingExpertPolicy:
    """Expert policy with positions tuned for actual block locations."""
    
    def __init__(self):
        self.phase = "init"
        self.phase_step = 0
        
        # Phase durations (in timesteps)
        self.phase_durations = {
            "init": 100,
            "reach_forward": 400,
            "lower_to_grasp": 300,
            "close_gripper": 150,
            "lift_up": 300,
            "move_to_blue": 400,
            "lower_to_place": 300,
            "open_gripper": 150,
            "retreat": 200
        }
        
        # Block positions (from actual scene)
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        
        # Key insight: blocks are VERY close to robot (5cm forward)
        # Need to use primarily joint 3 (elbow) to reach forward
        
        # Hand-tuned configurations based on robot kinematics
        self.configs = {
            # Starting position
            "home": np.array([0.0, -0.1, 0.0, -2.0, 0.0, 1.5, 0.0]),
            
            # Reach forward by extending elbow (joint 3)
            # Joint 3 more negative = arm extends forward
            "reach_red": np.array([0.0, 0.4, 0.0, -2.6, 0.0, 2.0, 0.0]),
            
            # Lower down to grasp height
            "grasp_red": np.array([0.0, 0.6, 0.0, -2.7, 0.0, 2.1, 0.0]),
            
            # Lift configuration
            "lift": np.array([0.0, 0.3, 0.0, -2.5, 0.0, 2.0, 0.0]),
            
            # Rotate base to reach blue block
            "above_blue": np.array([-0.5, 0.4, 0.0, -2.6, 0.0, 2.0, 0.0]),
            
            # Lower to place
            "place_blue": np.array([-0.5, 0.6, 0.0, -2.7, 0.0, 2.1, 0.0]),
            
            # Final position
            "retreat": np.array([-0.2, 0.0, 0.0, -2.0, 0.0, 1.8, 0.0])
        }
        
    def get_interpolated_config(self, config_from, config_to, alpha):
        """Linearly interpolate between configurations."""
        return (1 - alpha) * config_from + alpha * config_to
    
    def get_action(self, phase_progress):
        """Get target joint configuration for current phase."""
        if self.phase == "init":
            return self.configs["home"], 255
            
        elif self.phase == "reach_forward":
            config = self.get_interpolated_config(
                self.configs["home"], 
                self.configs["reach_red"], 
                phase_progress
            )
            return config, 255
            
        elif self.phase == "lower_to_grasp":
            config = self.get_interpolated_config(
                self.configs["reach_red"],
                self.configs["grasp_red"],
                phase_progress
            )
            return config, 255
            
        elif self.phase == "close_gripper":
            return self.configs["grasp_red"], 0  # Close
            
        elif self.phase == "lift_up":
            config = self.get_interpolated_config(
                self.configs["grasp_red"],
                self.configs["lift"],
                phase_progress
            )
            return config, 0  # Keep closed
            
        elif self.phase == "move_to_blue":
            config = self.get_interpolated_config(
                self.configs["lift"],
                self.configs["above_blue"],
                phase_progress
            )
            return config, 0  # Keep closed
            
        elif self.phase == "lower_to_place":
            config = self.get_interpolated_config(
                self.configs["above_blue"],
                self.configs["place_blue"],
                phase_progress
            )
            return config, 0  # Keep closed
            
        elif self.phase == "open_gripper":
            return self.configs["place_blue"], 255  # Open
            
        elif self.phase == "retreat":
            config = self.get_interpolated_config(
                self.configs["place_blue"],
                self.configs["retreat"],
                phase_progress
            )
            return config, 255
            
        return self.configs["home"], 255
    
    def update(self):
        """Update phase and return progress."""
        self.phase_step += 1
        phase_progress = self.phase_step / self.phase_durations[self.phase]
        
        if self.phase_step >= self.phase_durations[self.phase]:
            # Transition to next phase
            phases = list(self.phase_durations.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.phase_step = 0
                print(f"\n>>> Transitioned to: {self.phase}")
                phase_progress = 0.0
                
        return phase_progress

def main():
    """Create working expert demonstration."""
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Initialize expert
    expert = WorkingExpertPolicy()
    
    # Set initial configuration
    initial_config = expert.configs["home"]
    for i, val in enumerate(initial_config):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    data.ctrl[7] = 255  # Open gripper
    
    mujoco.mj_forward(model, data)
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video settings
    frames = []
    video_fps = 60
    frame_skip = int(1.0 / (video_fps * model.opt.timestep))
    
    print("\nCreating working expert demonstration...")
    print("=" * 70)
    
    # Get IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # For gripper width
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    
    # Run demonstration
    total_steps = sum(expert.phase_durations.values()) + 100
    
    for step in range(total_steps):
        # Get expert action
        phase_progress = expert.update()
        target_config, gripper_cmd = expert.get_action(phase_progress)
        
        # Apply position control
        for i in range(7):
            data.ctrl[i] = target_config[i]
        data.ctrl[7] = gripper_cmd
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Print status periodically
        if step % 100 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            blue_pos = data.xpos[blue_id]
            
            # Calculate gripper width
            if left_finger_id != -1 and right_finger_id != -1:
                left_pos = data.xpos[left_finger_id]
                right_pos = data.xpos[right_finger_id]
                gripper_width = np.linalg.norm(left_pos - right_pos)
            else:
                gripper_width = 0.08 if gripper_cmd > 100 else 0.0
            
            print(f"\nStep {step:4d} | Phase: {expert.phase:15s} | Progress: {phase_progress:.1%}")
            print(f"  EE pos:  [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]")
            print(f"  Red pos: [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]")
            print(f"  Blue pos: [{blue_pos[0]:6.3f}, {blue_pos[1]:6.3f}, {blue_pos[2]:6.3f}]")
            print(f"  Gripper width: {gripper_width:.3f}m")
            
            # Check if red block is lifted
            if red_pos[2] > 0.46:
                print(f"  ✓ Red block lifted! Height: {red_pos[2]:.3f}m")
        
        # Render frame
        if step % frame_skip == 0:
            renderer.update_scene(data, camera="demo_cam")
            frame = renderer.render()
            
            # Add overlays
            cv2.putText(frame, "WORKING EXPERT DEMO", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 2)
            cv2.putText(frame, f"Phase: {expert.phase}", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # Progress bar
            bar_width = int(400 * phase_progress)
            cv2.rectangle(frame, (40, 150), (40 + bar_width, 180), (0, 255, 0), -1)
            cv2.rectangle(frame, (40, 150), (440, 180), (255, 255, 255), 2)
            
            # Gripper state
            gripper_state = "CLOSED" if gripper_cmd < 100 else "OPEN"
            color = (0, 0, 255) if gripper_state == "CLOSED" else (0, 255, 0)
            cv2.putText(frame, f"Gripper: {gripper_state}", (40, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Block status
            red_pos = data.xpos[red_id]
            if red_pos[2] > 0.46:
                cv2.putText(frame, "Block: GRASPED", (40, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    
    # Final check
    final_red = data.xpos[red_id]
    final_blue = data.xpos[blue_id]
    
    print(f"\nFinal positions:")
    print(f"  Red block: [{final_red[0]:.3f}, {final_red[1]:.3f}, {final_red[2]:.3f}]")
    print(f"  Blue block: [{final_blue[0]:.3f}, {final_blue[1]:.3f}, {final_blue[2]:.3f}]")
    
    success = final_red[2] > final_blue[2] + 0.03
    if success:
        print("\n✅ SUCCESS: Red block stacked on blue block!")
    else:
        print("\n❌ FAILED: Blocks not properly stacked")
    
    # Save video
    os.makedirs("../../videos/121pm", exist_ok=True)
    output_path = "../../videos/121pm/working_expert_demo.mp4"
    
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

if __name__ == "__main__":
    main()