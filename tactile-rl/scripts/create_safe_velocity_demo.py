#!/usr/bin/env python3
"""
Create velocity control demo that approaches from above to avoid collision.
"""

import numpy as np
import mujoco
import cv2

class SafeVelocityExpert:
    """Expert policy that avoids knocking blocks."""
    
    def __init__(self):
        self.phase = "lift_first"
        self.step_count = 0
        self.phase_steps = {
            "lift_first": 60,    # First lift up to safe height
            "position_xy": 100,   # Position above red block
            "descend": 80,       # Descend to grasp
            "grasp": 40,         # Close gripper
            "lift": 60,          # Lift with block
            "move_y": 80,        # Move to blue block
            "place": 60,         # Lower onto blue
            "release": 30        # Open gripper
        }
        
    def get_action(self, model, data):
        """Get velocity commands."""
        velocities = np.zeros(8)
        
        # Get positions
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        
        ee_pos = data.xpos[ee_id]
        red_pos = data.xpos[red_id]
        blue_pos = data.xpos[blue_id]
        
        # Target positions
        safe_height = 0.6  # Safe height above blocks
        grasp_height = red_pos[2] - 0.005  # Slightly below block center
        
        if self.phase == "lift_first":
            # First, lift to safe height
            z_error = safe_height - ee_pos[2]
            velocities[1] = np.clip(-1.0 * z_error, -0.3, 0.3)  # Shoulder up
            velocities[3] = np.clip(0.8 * z_error, -0.2, 0.2)   # Elbow
            velocities[7] = 255  # Keep open
            
        elif self.phase == "position_xy":
            # Move to position above red block
            target_xy = red_pos[:2]
            xy_error = target_xy - ee_pos[:2]
            
            # Gentle horizontal movement
            velocities[0] = np.clip(0.8 * xy_error[0], -0.2, 0.2)  # X
            velocities[2] = np.clip(0.8 * xy_error[1], -0.2, 0.2)  # Y
            
            # Maintain height
            z_error = safe_height - ee_pos[2]
            velocities[1] = np.clip(-0.5 * z_error, -0.1, 0.1)
            
            velocities[7] = 255  # Keep open
            
        elif self.phase == "descend":
            # Descend straight down to grasp
            z_error = grasp_height - ee_pos[2]
            velocities[1] = np.clip(1.0 * z_error, -0.25, 0.25)
            velocities[3] = np.clip(-0.8 * z_error, -0.2, 0.2)
            
            # Maintain X-Y position
            xy_error = red_pos[:2] - ee_pos[:2]
            velocities[0] = np.clip(0.3 * xy_error[0], -0.05, 0.05)
            velocities[2] = np.clip(0.3 * xy_error[1], -0.05, 0.05)
            
            velocities[7] = 255  # Still open
            
        elif self.phase == "grasp":
            # Stop and close gripper
            velocities[:7] = 0.0
            velocities[7] = 0  # Close
            
        elif self.phase == "lift":
            # Lift to safe height
            z_error = safe_height - ee_pos[2]
            velocities[1] = np.clip(-1.0 * z_error, -0.25, 0.25)
            velocities[3] = np.clip(0.8 * z_error, -0.2, 0.2)
            velocities[7] = 0  # Keep closed
            
        elif self.phase == "move_y":
            # Move to above blue block
            target_y = blue_pos[1]
            y_error = target_y - ee_pos[1]
            
            velocities[2] = np.clip(0.8 * y_error, -0.3, 0.3)  # Y movement
            
            # Maintain X and Z
            x_error = blue_pos[0] - ee_pos[0]
            velocities[0] = np.clip(0.5 * x_error, -0.1, 0.1)
            
            z_error = safe_height - ee_pos[2]
            velocities[1] = np.clip(-0.5 * z_error, -0.1, 0.1)
            
            velocities[7] = 0  # Keep closed
            
        elif self.phase == "place":
            # Descend to place on blue block
            place_height = blue_pos[2] + 0.04  # Account for block heights
            z_error = place_height - ee_pos[2]
            
            velocities[1] = np.clip(0.8 * z_error, -0.2, 0.2)
            velocities[3] = np.clip(-0.6 * z_error, -0.15, 0.15)
            
            # Fine position control
            xy_error = blue_pos[:2] - ee_pos[:2]
            velocities[0] = np.clip(0.3 * xy_error[0], -0.05, 0.05)
            velocities[2] = np.clip(0.3 * xy_error[1], -0.05, 0.05)
            
            velocities[7] = 0  # Keep closed
            
        elif self.phase == "release":
            # Open gripper and lift slightly
            velocities[1] = -0.1  # Gentle lift
            velocities[7] = 255   # Open
        
        # Phase update
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
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Expert
    expert = SafeVelocityExpert()
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video
    frames = []
    video_fps = 30
    
    print("Creating SAFE velocity control demo...")
    print("=" * 70)
    print("Strategy: Lift first, then approach from above")
    print("Joint7 set to 0° (no rotation)")
    print("=" * 70)
    
    # IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Run
    total_steps = sum(expert.phase_steps.values())
    dt = model.opt.timestep
    
    for step in range(total_steps):
        # Get velocities
        velocities = expert.get_action(model, data)
        
        # Integrate velocities to positions
        for i in range(7):
            current_pos = data.qpos[i]
            new_pos = current_pos + velocities[i] * dt
            data.ctrl[i] = new_pos
        
        data.ctrl[7] = velocities[7]
        
        # Step
        mujoco.mj_step(model, data)
        
        # Status
        if step % 40 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            
            print(f"\nStep {step:3d} | Phase: {expert.phase:12s}")
            print(f"  EE:  [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]")
            print(f"  Red: [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]")
            
            # Check if block moved
            if abs(red_pos[0] - 0.05) > 0.02 or abs(red_pos[1]) > 0.02:
                if abs(red_pos[2] - 0.445) < 0.01:  # Still on table
                    print("  ⚠️  Red block displaced on table")
                else:
                    print("  ✓ Red block lifted!")
        
        # Render
        renderer.update_scene(data, camera="demo_cam")
        frame = renderer.render()
        
        # Overlay
        cv2.putText(frame, "SAFE VELOCITY CONTROL", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 3)
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Gripper
        gripper_state = "OPEN" if velocities[7] > 100 else "CLOSED"
        color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
        cv2.putText(frame, f"Gripper: {gripper_state}", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Progress
        progress = expert.step_count / expert.phase_steps[expert.phase]
        bar_width = int(400 * progress)
        cv2.rectangle(frame, (40, 220), (40 + bar_width, 260), (0, 255, 0), -1)
        cv2.rectangle(frame, (40, 220), (440, 260), (255, 255, 255), 3)
        
        # Position
        ee_pos = data.xpos[ee_id]
        pos_text = f"EE: [{ee_pos[0]:5.2f}, {ee_pos[1]:5.2f}, {ee_pos[2]:5.2f}]"
        cv2.putText(frame, pos_text, (40, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    
    # Final check
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    blue_pos = data.xpos[blue_id]
    
    print(f"\nFinal positions:")
    print(f"  EE:   [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Red:  [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"  Blue: [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
    
    if abs(red_pos[0] - blue_pos[0]) < 0.05 and abs(red_pos[1] - blue_pos[1]) < 0.05:
        if red_pos[2] > blue_pos[2] + 0.02:
            print("\n✓✓✓ SUCCESS: Red block stacked on blue!")
    
    # Save
    if frames:
        output_path = "../../videos/franka_safe_velocity_demo.mp4"
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