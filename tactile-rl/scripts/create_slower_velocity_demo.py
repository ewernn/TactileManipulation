#!/usr/bin/env python3
"""
Create velocity control demo with proper speed and joint7 alignment.
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
            "approach": 120,   # Doubled for slower motion
            "descend": 100,
            "grasp": 60,
            "lift": 80,
            "move": 100,
            "place": 80,
            "release": 40
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
            # Approach red block slowly
            diff = target_pos - ee_pos
            # Much gentler velocity commands
            velocities[0] = np.clip(0.5 * diff[0], -0.15, 0.15)
            velocities[1] = np.clip(-0.3 * diff[2], -0.1, 0.1)
            velocities[2] = np.clip(0.5 * diff[1], -0.1, 0.1)
            velocities[3] = -0.05  # Very gentle elbow
            
            # Keep gripper open
            velocities[7] = 255
            
        elif self.phase == "descend":
            # Descend slowly
            z_diff = target_pos[2] - ee_pos[2]
            velocities[1] = np.clip(1.5 * z_diff, -0.4, 0.4)
            velocities[3] = np.clip(-1.0 * z_diff, -0.3, 0.3)
            
            # Fine X-Y adjustment
            xy_diff = target_pos[:2] - ee_pos[:2]
            velocities[0] = np.clip(0.5 * xy_diff[0], -0.1, 0.1)
            velocities[2] = np.clip(0.5 * xy_diff[1], -0.1, 0.1)
            
            # Keep gripper open
            velocities[7] = 255
            
        elif self.phase == "grasp":
            # Stop and close gripper
            velocities[:7] = 0.0
            velocities[7] = 0
            
        elif self.phase == "lift":
            # Lift slowly
            velocities[1] = -0.3  # Slower shoulder up
            velocities[3] = 0.2   # Slower elbow
            velocities[7] = 0     # Keep closed
            
        elif self.phase == "move":
            # Move to blue block
            y_diff = blue_pos[1] - ee_pos[1]
            velocities[2] = np.clip(1.0 * y_diff, -0.4, 0.4)
            velocities[1] = -0.05  # Slight up
            velocities[7] = 0
            
        elif self.phase == "place":
            # Lower slowly
            place_height = blue_pos[2] + 0.045
            z_diff = place_height - ee_pos[2]
            velocities[1] = np.clip(1.0 * z_diff, -0.3, 0.3)
            velocities[3] = np.clip(-0.8 * z_diff, -0.2, 0.2)
            velocities[7] = 0
            
        elif self.phase == "release":
            # Open and retreat
            velocities[1] = -0.15
            velocities[7] = 255
        
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
    
    # Renderer with proper FPS
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video setup
    frames = []
    video_fps = 30  # Video playback FPS
    sim_fps = 200   # Simulation FPS (5x slower than default 1000Hz)
    steps_per_frame = sim_fps // video_fps  # How many sim steps per video frame
    
    print("Creating velocity control demo with proper speed...")
    print("=" * 70)
    print(f"Simulation: {sim_fps} Hz, Video: {video_fps} FPS")
    print(f"Physics timestep: {model.opt.timestep:.4f}s")
    print("=" * 70)
    
    # Get IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    
    # Check initial joint7 angle
    joint7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint7")
    print(f"\nInitial joint7 angle: {data.qpos[joint7_id]:.3f} rad ({np.degrees(data.qpos[joint7_id]):.1f}°)")
    print("(This controls the gripper rotation)\n")
    
    # Run simulation
    total_steps = sum(expert.phase_steps.values())
    dt = model.opt.timestep
    
    for step in range(total_steps):
        # Get velocity commands
        velocities = expert.get_action(model, data)
        
        # Run multiple physics steps per control step for slower motion
        for _ in range(steps_per_frame):
            # Velocity control: integrate to positions
            for i in range(7):
                current_pos = data.qpos[i]
                new_pos = current_pos + velocities[i] * dt
                data.ctrl[i] = new_pos
            
            # Gripper
            data.ctrl[7] = velocities[7]
            
            # Step physics
            mujoco.mj_step(model, data)
        
        # Print status
        if step % 50 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            gripper_state = "OPEN" if velocities[7] > 100 else "CLOSED"
            
            print(f"\nStep {step:3d} | Phase: {expert.phase:8s} | Gripper: {gripper_state}")
            print(f"  EE:  [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]")
            print(f"  Red: [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]")
            
            # Check if red block moved significantly
            if np.linalg.norm(red_pos - np.array([0.05, 0.0, 0.445])) > 0.1:
                print("  ⚠️  Red block has moved significantly!")
        
        # Render frame
        renderer.update_scene(data, camera="demo_cam")
        frame = renderer.render()
        
        # Overlay
        cv2.putText(frame, "VELOCITY CONTROL (Proper Speed)", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Gripper state
        gripper_state = "OPEN" if velocities[7] > 100 else "CLOSED"
        color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
        cv2.putText(frame, f"Gripper: {gripper_state}", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Progress
        progress = expert.step_count / expert.phase_steps[expert.phase]
        bar_width = int(400 * progress)
        cv2.rectangle(frame, (40, 220), (40 + bar_width, 260), (0, 255, 0), -1)
        cv2.rectangle(frame, (40, 220), (440, 260), (255, 255, 255), 3)
        
        frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    
    # Save video
    if frames:
        output_path = "../../videos/franka_velocity_proper_speed.mp4"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"\nVideo saved to {output_path}")

if __name__ == "__main__":
    main()