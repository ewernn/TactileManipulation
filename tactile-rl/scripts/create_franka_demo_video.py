#!/usr/bin/env python3
"""
Create expert demonstration video with the actual Franka Panda robot.
"""

import numpy as np
import mujoco
import cv2
import os

class FrankaExpertPolicy:
    """Expert policy for Franka Panda robot."""
    
    def __init__(self):
        self.phase = "approach"
        self.step_count = 0
        self.phase_steps = {
            "approach": 50,
            "descend": 40,
            "grasp": 30,
            "lift": 40
        }
        
    def get_action(self, model, data):
        """Get expert action for current state - VELOCITY CONTROL."""
        # 7 joint velocities + 1 gripper command
        action = np.zeros(8)
        
        # Get end-effector and target positions
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        
        ee_pos = data.xpos[ee_id]
        target_pos = data.xpos[target_id]
        
        # Get current joint velocities to create velocity commands
        joint_vel = data.qvel[:7]  # First 7 are arm joints
        
        if self.phase == "approach":
            # Move towards target horizontally
            diff = target_pos - ee_pos
            # Joint velocities to move end-effector
            action[0] = 0.5 * diff[0]  # Base velocity for X
            action[1] = -0.3           # Shoulder velocity (raise)
            action[3] = -0.5           # Elbow velocity
            action[5] = 0.3            # Wrist velocity
            
        elif self.phase == "descend":
            # Move down to grasp
            action[1] = 0.5    # Shoulder velocity down
            action[3] = 0.3    # Elbow velocity extend
            action[5] = -0.2   # Wrist velocity adjust
            
        elif self.phase == "grasp":
            # Close gripper (keep joints still)
            action[:7] = 0.0   # Zero velocity for arm
            action[7] = 0      # Close gripper
            
        elif self.phase == "lift":
            # Lift up with velocity
            action[1] = -0.8   # Shoulder velocity up
            action[3] = -0.5   # Elbow velocity flex
            action[7] = 0      # Keep gripper closed
        
        # VELOCITY CONTROL: Scale velocities and integrate
        # The actuators will integrate these velocities to positions
        dt = model.opt.timestep
        for i in range(7):
            # Convert velocity to position change
            current_pos = data.qpos[i]
            target_pos = current_pos + action[i] * dt * 10.0  # Scale factor
            action[i] = target_pos
        
        # Update phase
        self.step_count += 1
        if self.step_count >= self.phase_steps[self.phase]:
            phases = list(self.phase_steps.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.step_count = 0
                
        return action

def main():
    """Create video of Franka robot expert demonstration."""
    
    # Load the model and data
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Initialize expert policy
    expert = FrankaExpertPolicy()
    
    # Renderer setup
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video setup
    frames = []
    fps = 30
    
    print("Creating Franka Panda expert demonstration...")
    print("-" * 50)
    
    # Run demonstration
    for step in range(160):
        # Get expert action
        action = expert.get_action(model, data)
        
        # Apply action to actuators
        data.ctrl[:] = action
        
        # Step physics
        mujoco.mj_step(model, data)
        
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
        cv2.putText(frame, f"Step: {step}", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2)
        
        # Add phase progress bar
        phase_colors = {
            'approach': (0, 255, 0),    # Green
            'descend': (255, 255, 0),   # Yellow  
            'grasp': (255, 0, 0),       # Red
            'lift': (0, 0, 255)         # Blue
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
        
        frames.append(frame)
        
        # Print progress
        if step % 40 == 0:
            print(f"Step {step}: Phase={expert.phase}")
            # Print end-effector position
            ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            ee_pos = data.xpos[ee_id]
            print(f"  End-effector position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    
    print("-" * 50)
    print("Demo complete!")
    
    # Save video
    if frames:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('franka_expert_demo.mp4', fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"\nVideo saved to franka_expert_demo.mp4")
        print(f"Duration: {len(frames)/fps:.1f} seconds")
        print(f"Resolution: {width}x{height}")
    else:
        print("No frames captured!")

if __name__ == "__main__":
    main()