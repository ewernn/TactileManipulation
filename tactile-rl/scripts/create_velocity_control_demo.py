#!/usr/bin/env python3
"""
Create expert demonstration using TRUE velocity control.
"""

import numpy as np
import mujoco
import cv2

class VelocityControlExpert:
    """Expert policy using velocity control."""
    
    def __init__(self):
        self.phase = "approach"
        self.step_count = 0
        self.phase_steps = {
            "approach": 60,
            "descend": 50,
            "grasp": 30,
            "lift": 40,
            "move": 50,
            "place": 40,
            "release": 20
        }
        
    def get_action(self, model, data):
        """Get velocity commands for joints."""
        # 7 joint velocities + 1 gripper command
        velocities = np.zeros(8)
        
        # Get end-effector and target positions
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        
        ee_pos = data.xpos[ee_id]
        target_pos = data.xpos[target_id]
        blue_pos = data.xpos[blue_id]
        
        if self.phase == "approach":
            # Approach red block more gently
            diff = target_pos - ee_pos
            # Pure velocity commands (rad/s)
            velocities[0] = np.clip(1.0 * diff[0], -0.3, 0.3)  # Base rotation velocity
            velocities[1] = np.clip(-0.5 * diff[2], -0.2, 0.2) # Shoulder velocity
            velocities[2] = np.clip(1.0 * diff[1], -0.2, 0.2)  # Joint 3 for Y
            velocities[3] = -0.1  # Gentle elbow extension
            
            # Keep gripper open
            velocities[7] = 255
            
        elif self.phase == "descend":
            # Descend to grasp
            z_diff = target_pos[2] - ee_pos[2]
            velocities[1] = np.clip(3.0 * z_diff, -0.8, 0.8)   # Shoulder down
            velocities[3] = np.clip(-2.0 * z_diff, -0.6, 0.6)  # Elbow adjust
            
            # Fine X-Y adjustment
            xy_diff = target_pos[:2] - ee_pos[:2]
            velocities[0] = np.clip(1.0 * xy_diff[0], -0.3, 0.3)
            velocities[2] = np.clip(1.0 * xy_diff[1], -0.3, 0.3)
            
            # Keep gripper open
            velocities[7] = 255
            
        elif self.phase == "grasp":
            # Stop moving, close gripper
            velocities[:7] = 0.0  # Stop all joint velocities
            velocities[7] = 0     # Close gripper
            
        elif self.phase == "lift":
            # Lift up
            velocities[1] = -0.6  # Shoulder up velocity
            velocities[3] = 0.4   # Elbow flex velocity
            velocities[7] = 0     # Keep gripper closed
            
        elif self.phase == "move":
            # Move to blue block position
            y_diff = blue_pos[1] - ee_pos[1]
            velocities[2] = np.clip(2.0 * y_diff, -0.8, 0.8)  # Y movement
            velocities[1] = -0.1  # Slight upward velocity
            velocities[7] = 0     # Keep gripper closed
            
        elif self.phase == "place":
            # Lower onto blue block
            place_height = blue_pos[2] + 0.045  # Blue top + red height
            z_diff = place_height - ee_pos[2]
            velocities[1] = np.clip(2.0 * z_diff, -0.6, 0.6)
            velocities[3] = np.clip(-1.5 * z_diff, -0.4, 0.4)
            velocities[7] = 0  # Keep gripper closed
            
        elif self.phase == "release":
            # Open gripper and retreat
            velocities[1] = -0.3  # Move up
            velocities[7] = 255   # Open gripper
        
        # Update phase
        self.step_count += 1
        if self.step_count >= self.phase_steps[self.phase]:
            phases = list(self.phase_steps.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.step_count = 0
                print(f">>> Phase: {self.phase}")
                
        return velocities

def main():
    """Create velocity control demonstration."""
    
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Initialize expert
    expert = VelocityControlExpert()
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video setup
    frames = []
    fps = 30
    
    print("Creating TRUE velocity control demonstration...")
    print("=" * 70)
    print("Using velocity commands that are integrated by the simulator")
    print("=" * 70)
    
    # Get IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Initial state
    ee_pos = data.xpos[ee_id]
    print(f"\nInitial EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    
    # Run simulation
    total_steps = sum(expert.phase_steps.values())
    dt = model.opt.timestep
    
    for step in range(total_steps):
        # Get velocity commands
        velocities = expert.get_action(model, data)
        
        # VELOCITY CONTROL: Integrate velocities to get position targets
        # This simulates velocity control on top of position-controlled actuators
        for i in range(7):
            current_pos = data.qpos[i]
            # Integrate velocity to get new position
            new_pos = current_pos + velocities[i] * dt
            # Set position target
            data.ctrl[i] = new_pos
        
        # Gripper control
        data.ctrl[7] = velocities[7]
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Print status
        if step % 25 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            gripper_state = "OPEN" if velocities[7] > 100 else "CLOSED"
            
            print(f"\nStep {step:3d} | Phase: {expert.phase:8s} | Gripper: {gripper_state}")
            print(f"  EE:  [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]")
            print(f"  Red: [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]")
            print(f"  Velocities: [{velocities[0]:5.2f}, {velocities[1]:5.2f}, {velocities[2]:5.2f}] rad/s")
        
        # Render
        renderer.update_scene(data, camera="demo_cam")
        frame = renderer.render()
        
        # Overlay
        cv2.putText(frame, "VELOCITY CONTROL DEMO", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Gripper state
        gripper_state = "OPEN" if velocities[7] > 100 else "CLOSED"
        color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
        cv2.putText(frame, f"Gripper: {gripper_state}", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Velocity display
        vel_text = f"Vel: [{velocities[0]:4.1f}, {velocities[1]:4.1f}, {velocities[2]:4.1f}] rad/s"
        cv2.putText(frame, vel_text, (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
        
        # Progress
        progress = expert.step_count / expert.phase_steps[expert.phase]
        bar_width = int(400 * progress)
        cv2.rectangle(frame, (40, 280), (40 + bar_width, 320), (0, 255, 0), -1)
        cv2.rectangle(frame, (40, 280), (440, 320), (255, 255, 255), 3)
        
        frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    
    # Final positions
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    blue_pos = data.xpos[blue_id]
    
    print(f"\nFinal positions:")
    print(f"  EE:  [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Red: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"  Blue: [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
    
    # Save video
    if frames:
        output_path = "../../videos/franka_true_velocity_control.mp4"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"\nVideo saved to {output_path}")

if __name__ == "__main__":
    main()