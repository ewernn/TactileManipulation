#!/usr/bin/env python3
"""
Create working velocity control demo with better initial approach.
"""

import numpy as np
import mujoco
import cv2

class WorkingVelocityExpert:
    """Expert policy that actually works."""
    
    def __init__(self):
        self.phase = "wait"
        self.step_count = 0
        self.phase_steps = {
            "wait": 30,          # Wait a bit at start
            "open_wide": 40,     # Ensure gripper is fully open
            "lift_clear": 60,    # Lift to clear height first
            "align_x": 80,       # Align X position from above
            "align_y": 60,       # Align Y position from above
            "descend_slow": 100, # Slowly descend to grasp
            "grasp": 40,         # Close gripper
            "lift": 80,          # Lift with block
            "move_y": 100,       # Move to blue block Y
            "place": 80,         # Lower onto blue
            "release": 40        # Release
        }
        
        # Store initial block position
        self.red_block_init = np.array([0.05, 0.0, 0.445])
        self.blue_block_init = np.array([0.05, -0.15, 0.445])
        
    def get_action(self, model, data):
        """Get velocity commands."""
        velocities = np.zeros(8)
        
        # Get positions
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        
        ee_pos = data.xpos[ee_id].copy()
        red_pos = data.xpos[red_id].copy()
        blue_pos = data.xpos[blue_id].copy()
        
        # Safe heights
        clear_height = 0.55
        approach_height = 0.50
        grasp_height = 0.445  # Same as block height
        
        if self.phase == "wait":
            # Just wait, no movement
            velocities[:7] = 0.0
            velocities[7] = 255  # Gripper open (255 = open, 0 = close)
            
        elif self.phase == "open_wide":
            # Make sure gripper is fully open before moving
            velocities[:7] = 0.0
            velocities[7] = 255  # Full open
            
        elif self.phase == "lift_clear":
            # Lift straight up to clear any collision
            z_error = clear_height - ee_pos[2]
            if z_error > 0.01:
                velocities[1] = -0.3  # Shoulder up
                velocities[3] = 0.2   # Elbow adjust
            velocities[7] = 255
            
        elif self.phase == "align_x":
            # Move forward by extending arm
            x_error = self.red_block_init[0] - ee_pos[0]
            if x_error > 0.05:  # Need to move forward
                velocities[1] = 0.2   # Shoulder forward
                velocities[3] = 0.15  # Elbow extend
            elif x_error < -0.05:  # Too far forward
                velocities[1] = -0.2
                velocities[3] = -0.15
            
            # Maintain height
            z_error = clear_height - ee_pos[2]
            velocities[1] += np.clip(-0.3 * z_error, -0.1, 0.1)
            velocities[7] = 255
            
        elif self.phase == "align_y":
            # Align Y position
            y_error = self.red_block_init[1] - ee_pos[1]
            velocities[2] = np.clip(1.0 * y_error, -0.15, 0.15)
            
            # Start lowering slightly
            z_error = approach_height - ee_pos[2]
            velocities[1] = np.clip(1.0 * z_error, -0.1, 0.1)
            velocities[7] = 255
            
        elif self.phase == "descend_slow":
            # Very slowly descend to grasp height
            z_error = grasp_height - ee_pos[2]
            
            # Slow descent
            if z_error < -0.01:
                velocities[1] = 0.15   # Gentle shoulder down
                velocities[3] = -0.1   # Gentle elbow
            
            # Maintain X-Y position (use initial block position, not current)
            xy_error = self.red_block_init[:2] - ee_pos[:2]
            velocities[0] = np.clip(0.5 * xy_error[0], -0.05, 0.05)
            velocities[2] = np.clip(0.5 * xy_error[1], -0.05, 0.05)
            
            velocities[7] = 255  # Stay open
            
        elif self.phase == "grasp":
            # Stop moving and close gripper
            velocities[:7] = 0.0
            velocities[7] = 0  # Close
            
        elif self.phase == "lift":
            # Lift up with block
            z_error = clear_height - ee_pos[2]
            if z_error > 0.01:
                velocities[1] = -0.25
                velocities[3] = 0.2
            velocities[7] = 0  # Stay closed
            
        elif self.phase == "move_y":
            # Move to blue block Y position
            y_error = self.blue_block_init[1] - ee_pos[1]
            velocities[2] = np.clip(0.8 * y_error, -0.25, 0.25)
            
            # Maintain height and X
            z_error = clear_height - ee_pos[2]
            velocities[1] = np.clip(-0.5 * z_error, -0.1, 0.1)
            
            x_error = self.blue_block_init[0] - ee_pos[0]
            velocities[0] = np.clip(0.5 * x_error, -0.1, 0.1)
            
            velocities[7] = 0  # Stay closed
            
        elif self.phase == "place":
            # Lower to place on blue block
            place_height = self.blue_block_init[2] + 0.045
            z_error = place_height - ee_pos[2]
            
            if z_error < -0.01:
                velocities[1] = 0.15
                velocities[3] = -0.1
                
            # Fine positioning
            xy_error = self.blue_block_init[:2] - ee_pos[:2]
            velocities[0] = np.clip(0.3 * xy_error[0], -0.05, 0.05)
            velocities[2] = np.clip(0.3 * xy_error[1], -0.05, 0.05)
            
            velocities[7] = 0  # Stay closed
            
        elif self.phase == "release":
            # Open gripper and lift away
            velocities[1] = -0.15
            velocities[7] = 255  # Open
        
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
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Set gripper to be fully open at start
    finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    data.qpos[finger1_id] = 0.04  # Max open
    data.qpos[finger2_id] = 0.04  # Max open
    mujoco.mj_forward(model, data)
    
    # Expert
    expert = WorkingVelocityExpert()
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video
    frames = []
    video_fps = 30
    
    print("Creating WORKING velocity control demo...")
    print("=" * 70)
    print("Strategy: Wait, open gripper, lift first, then approach from above")
    print("=" * 70)
    
    # IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Run
    total_steps = sum(expert.phase_steps.values())
    dt = model.opt.timestep
    
    block_moved = False
    
    for step in range(total_steps):
        # Get velocities
        velocities = expert.get_action(model, data)
        
        # Integrate to positions
        for i in range(7):
            current_pos = data.qpos[i]
            new_pos = current_pos + velocities[i] * dt
            data.ctrl[i] = new_pos
        
        data.ctrl[7] = velocities[7]
        
        # Step
        mujoco.mj_step(model, data)
        
        # Check block status
        red_pos = data.xpos[red_id].copy()
        if not block_moved and np.linalg.norm(red_pos[:2] - expert.red_block_init[:2]) > 0.1:
            block_moved = True
            print(f"\n!!! Block knocked at step {step} in phase {expert.phase}")
        
        # Status
        if step % 50 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            
            print(f"\nStep {step:3d} | Phase: {expert.phase:12s}")
            print(f"  EE:  [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]")
            print(f"  Red: [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]")
            
            # Gripper opening
            gripper_open = data.qpos[finger1_id]
            print(f"  Gripper opening: {gripper_open:.3f}m")
            
            # Check success
            if red_pos[2] > 0.5:
                print("  ✓ Red block lifted!")
        
        # Render
        renderer.update_scene(data, camera="demo_cam")
        frame = renderer.render()
        
        # Overlay
        cv2.putText(frame, "WORKING VELOCITY CONTROL", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 3)
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Gripper
        gripper_state = "OPEN" if velocities[7] > 100 else "CLOSED"
        color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
        cv2.putText(frame, f"Gripper: {gripper_state}", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Block status
        if block_moved:
            cv2.putText(frame, "BLOCK KNOCKED!", (500, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    
    # Final
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    blue_pos = data.xpos[blue_id]
    
    print(f"\nFinal positions:")
    print(f"  EE:   [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Red:  [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"  Blue: [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
    
    # Save
    if frames:
        output_path = "../../videos/franka_working_velocity_demo.mp4"
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