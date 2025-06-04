#!/usr/bin/env python3
"""
Create precise expert demonstration with accurate positioning.
"""

import numpy as np
import mujoco
import cv2
import os

class PreciseExpertPolicy:
    """Expert policy with precise positioning for block stacking."""
    
    def __init__(self):
        self.phase = "approach"
        self.step_count = 0
        self.phase_steps = {
            "approach": 60,      # Move to above red block
            "descend": 40,       # Move down to grasp height
            "grasp": 30,         # Close gripper
            "lift": 40,          # Lift up
            "move_to_blue": 50,  # Move to above blue block
            "place": 40,         # Lower onto blue block
            "release": 20        # Open gripper
        }
        
        # Target positions
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        self.grasp_height_offset = 0.05  # How high above block to start descent
        self.lift_height = 0.15  # How high to lift
        
    def get_action(self, model, data):
        """Get expert action for current state."""
        # 7 joint velocities + 1 gripper command
        action = np.zeros(8)
        
        # Get current positions
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ee_pos = data.xpos[ee_id].copy()
        
        # Get current joint positions for better control
        joint_names = [f"joint{i}" for i in range(1, 8)]
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        current_joints = [data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]
        
        if self.phase == "approach":
            # Target: position above red block
            target = self.red_block_pos + np.array([0, 0, self.grasp_height_offset])
            diff = target - ee_pos
            
            # Use inverse kinematics approach
            # Primarily use shoulder and elbow to position
            action[0] = np.clip(2.0 * diff[0], -1, 1)  # Base rotation for X
            action[1] = np.clip(2.0 * diff[2], -1, 1)  # Shoulder for Z
            action[2] = np.clip(1.0 * diff[1], -1, 1)  # Shoulder pan for Y
            action[3] = np.clip(-1.0 * diff[2], -1, 1)  # Elbow for Z
            
            # Open gripper
            action[7] = 0
            
        elif self.phase == "descend":
            # Move straight down to grasp position
            target_z = self.red_block_pos[2]
            z_diff = target_z - ee_pos[2]
            
            action[1] = np.clip(3.0 * z_diff, -1, 1)  # Shoulder down
            action[3] = np.clip(-2.0 * z_diff, -1, 1)  # Elbow adjust
            
            # Keep gripper open
            action[7] = 0
            
        elif self.phase == "grasp":
            # Close gripper on block
            action[7] = 255  # Close gripper
            
            # Small adjustments to maintain position
            target = self.red_block_pos
            diff = target - ee_pos
            action[0] = np.clip(0.5 * diff[0], -0.2, 0.2)
            action[2] = np.clip(0.5 * diff[1], -0.2, 0.2)
            
        elif self.phase == "lift":
            # Lift straight up
            target_z = self.red_block_pos[2] + self.lift_height
            z_diff = target_z - ee_pos[2]
            
            action[1] = np.clip(-3.0 * z_diff, -1, 1)  # Shoulder up
            action[3] = np.clip(2.0 * z_diff, -1, 1)   # Elbow adjust
            
            # Keep gripper closed
            action[7] = 255
            
        elif self.phase == "move_to_blue":
            # Move to position above blue block
            target = self.blue_block_pos + np.array([0, 0, self.lift_height])
            diff = target - ee_pos
            
            # Move primarily in Y direction
            action[0] = np.clip(2.0 * diff[0], -1, 1)  # X adjustment
            action[2] = np.clip(3.0 * diff[1], -1, 1)  # Y movement (main)
            action[1] = np.clip(1.0 * diff[2], -1, 1)  # Maintain height
            
            # Keep gripper closed
            action[7] = 255
            
        elif self.phase == "place":
            # Lower onto blue block
            # Account for block heights: red is 0.025 tall, blue is 0.02 tall
            target_z = self.blue_block_pos[2] + 0.02 + 0.025  # Blue height + red height
            z_diff = target_z - ee_pos[2]
            
            action[1] = np.clip(2.0 * z_diff, -1, 1)  # Shoulder down
            action[3] = np.clip(-1.5 * z_diff, -1, 1)  # Elbow adjust
            
            # Keep gripper closed
            action[7] = 255
            
        elif self.phase == "release":
            # Open gripper to release block
            action[7] = 0
            
            # Small upward movement to clear
            action[1] = -0.3
        
        # Update phase
        self.step_count += 1
        if self.step_count >= self.phase_steps[self.phase]:
            phases = list(self.phase_steps.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.step_count = 0
                print(f"Transitioning to phase: {self.phase}")
                
        return action

def main():
    """Create video of precise Franka robot demonstration."""
    
    # Load the model and data
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Initialize expert policy
    expert = PreciseExpertPolicy()
    
    # Renderer setup
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video setup
    frames = []
    fps = 30
    
    print("Creating precise Franka Panda demonstration...")
    print("-" * 70)
    
    # Get body IDs for tracking
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Run demonstration
    total_steps = sum(expert.phase_steps.values())
    for step in range(total_steps):
        # Get expert action
        action = expert.get_action(model, data)
        
        # Apply action to actuators
        data.ctrl[:] = action
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Print detailed info every 20 steps
        if step % 20 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            blue_pos = data.xpos[blue_id]
            print(f"\nStep {step}: Phase={expert.phase}")
            print(f"  EE pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            print(f"  Red block: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
            print(f"  Distance to red: {np.linalg.norm(ee_pos[:2] - red_pos[:2]):.3f} m")
            if expert.phase in ["move_to_blue", "place"]:
                print(f"  Blue block: [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
                print(f"  Distance to blue: {np.linalg.norm(ee_pos[:2] - blue_pos[:2]):.3f} m")
        
        # Render frame
        # Update scene with main camera
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
        if cam_id >= 0:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)
            
        frame = renderer.render()
        
        # Add text overlay
        text_color = (255, 255, 255)
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)
        cv2.putText(frame, f"Step: {step}/{total_steps}", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)
        
        # Add phase progress bar
        phase_colors = {
            'approach': (0, 255, 0),      # Green
            'descend': (255, 255, 0),     # Yellow  
            'grasp': (255, 0, 0),         # Red
            'lift': (0, 0, 255),          # Blue
            'move_to_blue': (255, 0, 255), # Magenta
            'place': (0, 255, 255),       # Cyan
            'release': (128, 128, 128)    # Gray
        }
        color = phase_colors.get(expert.phase, (255, 255, 255))
        progress = expert.step_count / expert.phase_steps[expert.phase]
        bar_width = int(400 * progress)
        cv2.rectangle(frame, (40, 150), (40 + bar_width, 190), color, -1)
        cv2.rectangle(frame, (40, 150), (440, 190), text_color, 3)
        
        # Show gripper state
        gripper_closed = action[7] > 100
        gripper_text = "CLOSED" if gripper_closed else "OPEN"
        gripper_color = (0, 0, 255) if gripper_closed else (0, 255, 0)
        cv2.putText(frame, f"Gripper: {gripper_text}", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, gripper_color, 2)
        
        # Show position info
        ee_pos = data.xpos[ee_id]
        cv2.putText(frame, f"EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]", 
                    (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        frames.append(frame)
    
    print("\n" + "-" * 70)
    print("Demo complete!")
    
    # Save video
    if frames:
        output_path = "../../videos/franka_precise_stacking_demo.mp4"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"\nVideo saved to {output_path}")
        print(f"Duration: {len(frames)/fps:.1f} seconds")
        print(f"Resolution: {width}x{height}")
    else:
        print("No frames captured!")

if __name__ == "__main__":
    main()