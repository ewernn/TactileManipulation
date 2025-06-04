#\!/usr/bin/env python3
"""
Create a demonstration using POSITION CONTROL (which is what the Panda XML uses).
"""

import numpy as np
import mujoco
import cv2
import os

class PositionControlExpertPolicy:
    """Expert policy designed for position control actuators."""
    
    def __init__(self):
        self.phase = "init"
        self.step_count = 0
        self.phase_duration = {
            "init": 0.5,          # seconds
            "approach": 2.0,      # Move to above red block
            "descend": 1.5,       # Lower to grasp
            "grasp": 0.5,         # Close gripper
            "lift": 1.5,          # Lift block
            "move": 2.0,          # Move to blue block
            "place": 1.5,         # Lower onto blue
            "release": 0.5        # Open gripper
        }
        
        # Target positions for each phase
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        
        # Gripper parameters
        self.gripper_reach = 0.103
        self.grasp_offset = 0.01  # How far past block center
        
        print(f"Position Control Expert Policy initialized")
        
    def get_target_positions(self, model, data, dt):
        """Get target joint positions for current phase."""
        # Current joint positions
        current_joints = np.array([data.qpos[14 + i] for i in range(7)])
        
        # Get end-effector position
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ee_pos = data.xpos[ee_id].copy()
        
        # Initialize target as current position
        target_joints = current_joints.copy()
        gripper_cmd = 255  # Open by default
        
        if self.phase == "init":
            # Stay at initial position
            pass
            
        elif self.phase == "approach":
            # Move to position above red block
            # This requires inverse kinematics approximation
            target_x = self.red_block_pos[0] + self.grasp_offset - self.gripper_reach
            target_y = self.red_block_pos[1]
            target_z = 0.55  # Safe height
            
            # Simple IK using Jacobian (approximation)
            x_error = target_x - ee_pos[0]
            y_error = target_y - ee_pos[1]
            z_error = target_z - ee_pos[2]
            
            # Map errors to joint movements (empirically tuned)
            if abs(x_error) > 0.01:
                target_joints[1] += 0.8 * x_error  # Shoulder for X
                target_joints[3] += 0.6 * x_error  # Elbow for X
                
            if abs(y_error) > 0.01:
                target_joints[0] += 1.0 * y_error  # Base for Y
                target_joints[2] += 0.5 * y_error  # Upper arm for Y
                
            if abs(z_error) > 0.01:
                target_joints[1] -= 0.8 * z_error  # Shoulder for Z
                target_joints[3] += 0.6 * z_error  # Elbow for Z
                
            # Wrist orientation
            target_joints[5] = 1.5  # Wrist pitch
            
        elif self.phase == "descend":
            # Lower to grasp height
            target_z = self.red_block_pos[2]
            z_error = target_z - ee_pos[2]
            
            if z_error < -0.01:
                target_joints[1] -= 0.5 * z_error
                target_joints[3] += 0.4 * z_error
                
            # Fine-tune wrist
            target_joints[5] = 1.6
            target_joints[6] = 0.1
            
        elif self.phase == "grasp":
            # Close gripper
            gripper_cmd = 0
            
        elif self.phase == "lift":
            # Lift to safe height
            target_z = 0.6
            z_error = target_z - ee_pos[2]
            
            if abs(z_error) > 0.01:
                target_joints[1] -= 0.8 * z_error
                target_joints[3] += 0.6 * z_error
                
            gripper_cmd = 0  # Keep closed
            
        elif self.phase == "move":
            # Move to above blue block
            target_x = self.blue_block_pos[0] + self.grasp_offset - self.gripper_reach
            target_y = self.blue_block_pos[1]
            
            x_error = target_x - ee_pos[0]
            y_error = target_y - ee_pos[1]
            
            if abs(x_error) > 0.01:
                target_joints[1] += 0.6 * x_error
                target_joints[3] += 0.4 * x_error
                
            if abs(y_error) > 0.01:
                target_joints[0] += 0.8 * y_error
                target_joints[2] += 0.4 * y_error
                
            gripper_cmd = 0  # Keep closed
            
        elif self.phase == "place":
            # Lower onto blue block
            target_z = self.blue_block_pos[2] + 0.05  # Stack height
            z_error = target_z - ee_pos[2]
            
            if z_error < -0.01:
                target_joints[1] -= 0.5 * z_error
                target_joints[3] += 0.4 * z_error
                
            gripper_cmd = 0  # Keep closed
            
        elif self.phase == "release":
            # Open gripper
            gripper_cmd = 255
            # Lift slightly
            target_joints[1] += 0.05
        
        # Smooth the targets (limit max change per timestep)
        max_change = 0.1 * dt  # Max 0.1 rad per second
        for i in range(7):
            change = target_joints[i] - current_joints[i]
            if abs(change) > max_change:
                target_joints[i] = current_joints[i] + np.sign(change) * max_change
        
        return target_joints, gripper_cmd
    
    def update_phase(self, elapsed_time):
        """Check if should transition to next phase."""
        phase_list = list(self.phase_duration.keys())
        current_idx = phase_list.index(self.phase)
        
        if elapsed_time > self.phase_duration[self.phase]:
            if current_idx < len(phase_list) - 1:
                self.phase = phase_list[current_idx + 1]
                print(f"\n>>> Transitioning to: {self.phase}")
                return True
        return False

def main():
    """Create position control demonstration."""
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
    
    # Initialize policy
    expert = PositionControlExpertPolicy()
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video frames
    frames = []
    video_fps = 30
    
    print("\nCreating position control demonstration...")
    print("=" * 70)
    
    # Get IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Run demo
    dt = model.opt.timestep
    total_time = sum(expert.phase_duration.values())
    n_steps = int(total_time / dt)
    
    phase_start_time = 0
    current_time = 0
    
    for step in range(n_steps):
        current_time = step * dt
        phase_time = current_time - phase_start_time
        
        # Get target positions
        target_joints, gripper_cmd = expert.get_target_positions(model, data, dt)
        
        # Apply control (POSITION CONTROL)
        for i in range(7):
            data.ctrl[i] = target_joints[i]
        data.ctrl[7] = gripper_cmd
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Check phase transition
        if expert.update_phase(phase_time):
            phase_start_time = current_time
        
        # Print status periodically
        if step % int(0.5 / dt) == 0:  # Every 0.5 seconds
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            blue_pos = data.xpos[blue_id]
            
            print(f"\nTime: {current_time:.2f}s | Phase: {expert.phase}")
            print(f"  EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
            print(f"  Red block:   [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
            print(f"  Blue block:  [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
            
            # Joint positions
            joint_pos = [data.qpos[14 + i] for i in range(7)]
            print(f"  Joint positions: [{', '.join([f'{j:.2f}' for j in joint_pos])}]")
            print(f"  Joint targets:   [{', '.join([f'{data.ctrl[i]:.2f}' for i in range(7)])}]")
            
            # Check if red block moved
            if np.linalg.norm(red_pos[:2] - expert.red_block_pos[:2]) > 0.01:
                print(f"  ✓ Red block grasped\!")
                
        # Render frame every few steps
        if step % int(1.0 / (video_fps * dt)) == 0:
            renderer.update_scene(data, camera="demo_cam")
            frame = renderer.render()
            
            # Overlays
            cv2.putText(frame, "POSITION CONTROL DEMO", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 2)
            cv2.putText(frame, f"Phase: {expert.phase}", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            # Gripper info
            gripper_state = "OPEN" if gripper_cmd > 100 else "CLOSED"
            color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
            cv2.putText(frame, f"Gripper: {gripper_state}", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Time
            cv2.putText(frame, f"Time: {current_time:.1f}s", (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demo complete\!")
    
    # Save video
    os.makedirs("../../videos/121pm", exist_ok=True)
    output_path = "../../videos/121pm/position_control_demo.mp4"
    
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
