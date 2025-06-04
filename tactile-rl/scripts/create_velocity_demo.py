#!/usr/bin/env python3
"""
Create expert demonstration using velocity control.
Properly controls the robot to reach and grasp blocks.
"""

import numpy as np
import mujoco
import cv2

class VelocityExpertPolicy:
    """Expert policy using velocity control with proper timing."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.phase = "approach_x"
        self.step_count = 0
        self.phase_steps = {
            "approach_x": 80,     # Move forward in X to get closer to blocks
            "approach_y": 40,     # Adjust Y position to align with red block
            "descend": 60,        # Descend to grasp height (gripper OPEN)
            "grasp": 30,          # Close gripper
            "lift": 50,           # Lift block up
            "move_y": 60,         # Move to blue block Y position
            "place": 50,          # Lower to place on blue block
            "release": 20         # Open gripper and retreat
        }
        
        # Target positions
        self.red_block_x = 0.05
        self.red_block_y = 0.0
        self.blue_block_y = -0.15
        self.grasp_height = 0.445
        self.lift_height = 0.6
        
    def get_action(self, model, data):
        """Get velocity control action."""
        action = np.zeros(8)
        
        # Get current positions
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ee_pos = data.xpos[ee_id].copy()
        
        # Get block positions
        red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        red_pos = data.xpos[red_id].copy()
        
        if self.phase == "approach_x":
            # Need to move forward (+X) to reach blocks
            # Robot starts at X ~ -0.8, blocks are at X = 0.05
            x_error = self.red_block_x - ee_pos[0]
            
            # Use base rotation and arm extension to move forward
            if x_error > 0.1:  # Still far from block
                # Strong forward movement
                action[1] = 0.8    # Shoulder forward
                action[3] = 0.6    # Elbow extend
            else:  # Getting close
                # Slower approach
                action[1] = 0.3
                action[3] = 0.2
            
            # Keep gripper OPEN
            action[7] = 255
            
        elif self.phase == "approach_y":
            # Fine tune Y position to align with red block
            y_error = self.red_block_y - ee_pos[1]
            
            # Use joint 3 for Y adjustment
            action[2] = np.clip(3.0 * y_error, -0.5, 0.5)
            
            # Maintain X position
            x_error = self.red_block_x - ee_pos[0]
            if abs(x_error) > 0.05:
                action[1] = np.clip(2.0 * x_error, -0.3, 0.3)
                action[3] = np.clip(1.5 * x_error, -0.3, 0.3)
            
            # Keep gripper OPEN
            action[7] = 255
            
        elif self.phase == "descend":
            # Move down to grasp height
            z_error = self.grasp_height - ee_pos[2]
            
            # Descend using shoulder and elbow
            if z_error < -0.05:  # Need to go down
                action[1] = 0.6    # Shoulder down
                action[3] = -0.4   # Elbow adjust
            else:
                # Small adjustments
                action[1] = np.clip(3.0 * z_error, -0.3, 0.3)
            
            # Maintain X-Y position near red block
            x_error = red_pos[0] - ee_pos[0]
            y_error = red_pos[1] - ee_pos[1]
            action[0] = np.clip(2.0 * x_error, -0.2, 0.2)  # Small X correction
            action[2] = np.clip(2.0 * y_error, -0.2, 0.2)  # Small Y correction
            
            # IMPORTANT: Keep gripper OPEN while descending
            action[7] = 255
            
        elif self.phase == "grasp":
            # Close gripper to grasp block
            action[7] = 0  # CLOSE gripper
            
            # Hold position steady
            x_error = red_pos[0] - ee_pos[0]
            y_error = red_pos[1] - ee_pos[1]
            action[0] = np.clip(1.0 * x_error, -0.1, 0.1)
            action[2] = np.clip(1.0 * y_error, -0.1, 0.1)
            
        elif self.phase == "lift":
            # Lift straight up
            z_error = self.lift_height - ee_pos[2]
            
            if z_error > 0.05:  # Need to go up
                action[1] = -0.8   # Shoulder up
                action[3] = 0.6    # Elbow adjust
            else:
                # Small adjustments
                action[1] = np.clip(-3.0 * z_error, -0.3, 0.3)
            
            # Keep gripper CLOSED
            action[7] = 0
            
        elif self.phase == "move_y":
            # Move to blue block Y position
            y_error = self.blue_block_y - ee_pos[1]
            
            # Strong Y movement
            action[2] = np.clip(3.0 * y_error, -0.8, 0.8)
            
            # Maintain height and X position
            z_error = self.lift_height - ee_pos[2]
            action[1] = np.clip(-2.0 * z_error, -0.3, 0.3)
            
            # Keep gripper CLOSED
            action[7] = 0
            
        elif self.phase == "place":
            # Lower to place on blue block
            # Target height: blue block top + red block height
            place_height = 0.44 + 0.02 + 0.025  # table + blue + half red
            z_error = place_height - ee_pos[2]
            
            if z_error < -0.05:  # Need to go down
                action[1] = 0.6    # Shoulder down
                action[3] = -0.4   # Elbow adjust
            else:
                action[1] = np.clip(3.0 * z_error, -0.3, 0.3)
            
            # Maintain X-Y position over blue block
            x_error = self.red_block_x - ee_pos[0]
            y_error = self.blue_block_y - ee_pos[1]
            action[0] = np.clip(2.0 * x_error, -0.2, 0.2)
            action[2] = np.clip(2.0 * y_error, -0.2, 0.2)
            
            # Keep gripper CLOSED
            action[7] = 0
            
        elif self.phase == "release":
            # Open gripper
            action[7] = 255
            
            # Move up slightly
            action[1] = -0.4
            action[3] = 0.3
        
        # Update phase
        self.step_count += 1
        if self.step_count >= self.phase_steps[self.phase]:
            phases = list(self.phase_steps.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.step_count = 0
                print(f"\n>>> Transitioning to: {self.phase}")
                
        return action

def main():
    """Create demonstration with velocity control."""
    
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Initialize policy
    expert = VelocityExpertPolicy(model, data)
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video setup
    frames = []
    fps = 30
    
    print("Creating Franka demo with velocity control...")
    print("=" * 70)
    print("Starting from proper position facing workspace")
    print("Gripper will stay OPEN until positioned around block")
    print("=" * 70)
    
    # Get IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Print initial state
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    print(f"\nInitial positions:")
    print(f"  EE:  [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Red: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"  Distance: {np.linalg.norm(ee_pos - red_pos):.3f}m")
    
    # Run demo
    total_steps = sum(expert.phase_steps.values())
    for step in range(total_steps):
        # Get action
        action = expert.get_action(model, data)
        
        # Apply action
        data.ctrl[:] = action
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Print status
        if step % 30 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            blue_pos = data.xpos[blue_id]
            gripper_state = "OPEN" if action[7] > 100 else "CLOSED"
            
            print(f"\nStep {step:3d} | Phase: {expert.phase:12s} | Gripper: {gripper_state}")
            print(f"  EE:  [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]")
            print(f"  Red: [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]")
            
            # Distance to target
            if expert.phase in ["approach_x", "approach_y", "descend", "grasp"]:
                dist = np.linalg.norm(ee_pos - red_pos)
                print(f"  Distance to red: {dist:.3f}m")
            elif expert.phase in ["move_y", "place"]:
                dist_y = abs(ee_pos[1] - expert.blue_block_y)
                print(f"  Y distance to blue: {dist_y:.3f}m")
                
            # Check if block is lifted
            if red_pos[2] > 0.48:
                print(f"  ✓ Red block lifted! Height: {red_pos[2]:.3f}m")
        
        # Render
        renderer.update_scene(data, camera="demo_cam")
        frame = renderer.render()
        
        # Overlays
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Gripper state
        gripper_state = "OPEN" if action[7] > 100 else "CLOSED"
        color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
        cv2.putText(frame, f"Gripper: {gripper_state}", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Progress
        progress = expert.step_count / expert.phase_steps[expert.phase]
        bar_width = int(400 * progress)
        cv2.rectangle(frame, (40, 180), (40 + bar_width, 220), (0, 255, 0), -1)
        cv2.rectangle(frame, (40, 180), (440, 220), (255, 255, 255), 3)
        
        # Positions
        ee_pos = data.xpos[ee_id]
        red_pos = data.xpos[red_id]
        cv2.putText(frame, f"EE:  [{ee_pos[0]:5.2f}, {ee_pos[1]:5.2f}, {ee_pos[2]:5.2f}]",
                    (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Red: [{red_pos[0]:5.2f}, {red_pos[1]:5.2f}, {red_pos[2]:5.2f}]",
                    (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
        
        frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    
    # Final check
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    blue_pos = data.xpos[blue_id]
    
    print(f"\nFinal positions:")
    print(f"  End-effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Red block:    [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"  Blue block:   [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
    
    # Success check
    x_aligned = abs(red_pos[0] - blue_pos[0]) < 0.05
    y_aligned = abs(red_pos[1] - blue_pos[1]) < 0.05
    stacked = red_pos[2] > blue_pos[2] + 0.03
    
    if x_aligned and y_aligned and stacked:
        print("\n✓✓✓ SUCCESS: Red block stacked on blue block!")
    else:
        print("\n✗ Stacking result:")
        print(f"  X alignment: {abs(red_pos[0] - blue_pos[0]):.3f}m")
        print(f"  Y alignment: {abs(red_pos[1] - blue_pos[1]):.3f}m")
        print(f"  Height diff: {red_pos[2] - blue_pos[2]:.3f}m")
    
    # Save video
    if frames:
        output_path = "../../videos/franka_velocity_demo.mp4"
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