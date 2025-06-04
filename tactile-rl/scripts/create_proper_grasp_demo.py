#!/usr/bin/env python3
"""
Create expert demonstration with proper gripper control.
Gripper stays OPEN until positioned around the block.
"""

import numpy as np
import mujoco
import cv2

class ProperGraspExpertPolicy:
    """Expert policy with correct gripper timing."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.phase = "approach"
        self.step_count = 0
        self.phase_steps = {
            "approach": 80,      # Move to above red block
            "descend": 60,       # Move down with OPEN gripper
            "position": 30,      # Fine position around block
            "grasp": 20,         # NOW close gripper
            "lift": 50,          # Lift up
            "move_to_blue": 60,  # Move to above blue block
            "place": 50,         # Lower onto blue block
            "release": 20        # Open gripper
        }
        
        # Target positions
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        
        # Control parameters
        self.approach_height = 0.12  # Height above block for approach
        self.grasp_height = 0.445    # Same as block center height
        self.lift_height = 0.6       # Height when lifted
        
    def get_action(self, model, data):
        """Get expert action with proper gripper control."""
        action = np.zeros(8)
        
        # Get current positions
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ee_pos = data.xpos[ee_id].copy()
        
        # Get block positions
        red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        red_pos = data.xpos[red_id].copy()
        
        if self.phase == "approach":
            # Move to position above red block
            target = np.array([self.red_block_pos[0], self.red_block_pos[1], 
                              self.red_block_pos[2] + self.approach_height])
            diff = target - ee_pos
            
            # Joint velocities for positioning
            action[0] = np.clip(2.0 * diff[0], -1, 1)  # Base rotation for X
            action[1] = np.clip(-1.5 * diff[2], -1, 1)  # Shoulder for Z
            action[2] = np.clip(2.0 * diff[1], -1, 1)  # Shoulder pan for Y
            action[3] = np.clip(-1.0 * diff[2], -1, 1)  # Elbow for Z
            
            # IMPORTANT: Keep gripper OPEN during approach
            action[7] = 255  # OPEN gripper
            
        elif self.phase == "descend":
            # Move straight down to block height
            target = np.array([red_pos[0], red_pos[1], self.grasp_height])
            diff = target - ee_pos
            
            # Mainly vertical movement
            action[1] = np.clip(3.0 * diff[2], -1, 1)   # Shoulder down
            action[3] = np.clip(-2.0 * diff[2], -1, 1)  # Elbow adjust
            
            # Small X-Y corrections
            action[0] = np.clip(1.0 * diff[0], -0.3, 0.3)
            action[2] = np.clip(1.0 * diff[1], -0.3, 0.3)
            
            # IMPORTANT: Keep gripper OPEN while descending
            action[7] = 255  # OPEN gripper
            
        elif self.phase == "position":
            # Fine positioning to ensure gripper is around block
            target = red_pos  # Go to exact block position
            diff = target - ee_pos
            
            # Very small adjustments
            action[0] = np.clip(0.5 * diff[0], -0.1, 0.1)
            action[2] = np.clip(0.5 * diff[1], -0.1, 0.1)
            action[1] = np.clip(0.5 * diff[2], -0.1, 0.1)
            
            # STILL keep gripper OPEN
            action[7] = 255  # OPEN gripper
            
        elif self.phase == "grasp":
            # NOW close the gripper
            action[7] = 0  # CLOSE gripper
            
            # Hold position steady
            target = red_pos
            diff = target - ee_pos
            action[0] = np.clip(0.2 * diff[0], -0.05, 0.05)
            action[2] = np.clip(0.2 * diff[1], -0.05, 0.05)
            
        elif self.phase == "lift":
            # Lift straight up
            target_z = self.lift_height
            z_diff = target_z - ee_pos[2]
            
            action[1] = np.clip(-3.0 * z_diff, -1, 1)  # Shoulder up
            action[3] = np.clip(2.0 * z_diff, -1, 1)   # Elbow adjust
            
            # Keep gripper CLOSED
            action[7] = 0  # CLOSED gripper
            
        elif self.phase == "move_to_blue":
            # Move to position above blue block
            target = np.array([self.blue_block_pos[0], self.blue_block_pos[1], self.lift_height])
            diff = target - ee_pos
            
            # Move mainly in Y direction
            action[0] = np.clip(1.5 * diff[0], -0.5, 0.5)  # X correction
            action[2] = np.clip(3.0 * diff[1], -1, 1)      # Y movement (main)
            action[1] = np.clip(1.0 * diff[2], -0.3, 0.3)  # Maintain height
            
            # Keep gripper CLOSED
            action[7] = 0  # CLOSED gripper
            
        elif self.phase == "place":
            # Lower onto blue block
            # Blue block height + half red block height
            target_z = self.blue_block_pos[2] + 0.02 + 0.025
            z_diff = target_z - ee_pos[2]
            
            action[1] = np.clip(2.5 * z_diff, -1, 1)   # Shoulder down
            action[3] = np.clip(-2.0 * z_diff, -1, 1)  # Elbow adjust
            
            # Maintain X-Y position
            target_xy = self.blue_block_pos[:2]
            xy_diff = target_xy - ee_pos[:2]
            action[0] = np.clip(1.0 * xy_diff[0], -0.2, 0.2)
            action[2] = np.clip(1.0 * xy_diff[1], -0.2, 0.2)
            
            # Keep gripper CLOSED until placed
            action[7] = 0  # CLOSED gripper
            
        elif self.phase == "release":
            # Open gripper to release block
            action[7] = 255  # OPEN gripper
            
            # Small upward movement
            action[1] = -0.3
        
        # Update phase
        self.step_count += 1
        if self.step_count >= self.phase_steps[self.phase]:
            phases = list(self.phase_steps.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.step_count = 0
                print(f"\n>>> Phase transition: {self.phase}")
                
        return action

def main():
    """Create video with proper gripper control."""
    
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Initialize policy
    expert = ProperGraspExpertPolicy(model, data)
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video setup
    frames = []
    fps = 30
    
    print("Creating Franka demo with proper gripper control...")
    print("=" * 70)
    print("Gripper will stay OPEN until positioned around block")
    print("=" * 70)
    
    # Get IDs for monitoring
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    
    # Run demo
    total_steps = sum(expert.phase_steps.values())
    for step in range(total_steps):
        # Get action
        action = expert.get_action(model, data)
        
        # Apply action
        data.ctrl[:] = action
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Print status every 20 steps
        if step % 20 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            blue_pos = data.xpos[blue_id]
            gripper_pos = data.qpos[finger1_id]
            gripper_state = "OPEN" if action[7] > 100 else "CLOSED"
            
            print(f"\nStep {step:3d} | Phase: {expert.phase:15s} | Gripper: {gripper_state:6s}")
            print(f"  EE: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]")
            print(f"  Red: [{red_pos[0]:6.3f}, {red_pos[1]:6.3f}, {red_pos[2]:6.3f}]")
            print(f"  Distance to red: {np.linalg.norm(ee_pos - red_pos):.3f}m")
            
        # Render frame
        renderer.update_scene(data, camera="demo_cam")
        frame = renderer.render()
        
        # Add overlays
        # Phase
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Gripper state with color
        gripper_state = "OPEN" if action[7] > 100 else "CLOSED"
        color = (0, 255, 0) if gripper_state == "OPEN" else (0, 0, 255)
        cv2.putText(frame, f"Gripper: {gripper_state}", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Progress bar
        progress = expert.step_count / expert.phase_steps[expert.phase]
        bar_width = int(400 * progress)
        phase_colors = {
            'approach': (0, 255, 0),
            'descend': (255, 255, 0),
            'position': (255, 128, 0),
            'grasp': (255, 0, 0),
            'lift': (0, 0, 255),
            'move_to_blue': (255, 0, 255),
            'place': (0, 255, 255),
            'release': (128, 128, 128)
        }
        bar_color = phase_colors.get(expert.phase, (255, 255, 255))
        cv2.rectangle(frame, (40, 180), (40 + bar_width, 220), bar_color, -1)
        cv2.rectangle(frame, (40, 180), (440, 220), (255, 255, 255), 3)
        
        # Position info
        ee_pos = data.xpos[ee_id]
        cv2.putText(frame, f"Position: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]", 
                    (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Step counter
        cv2.putText(frame, f"Step: {step}/{total_steps}", (40, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    
    # Final status
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    blue_pos = data.xpos[blue_id]
    print(f"\nFinal positions:")
    print(f"  End-effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Red block: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"  Blue block: [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
    
    # Check if stacking was successful
    if abs(red_pos[0] - blue_pos[0]) < 0.05 and abs(red_pos[1] - blue_pos[1]) < 0.05:
        if red_pos[2] > blue_pos[2] + 0.03:
            print("\n✓ SUCCESS: Red block is stacked on blue block!")
        else:
            print("\n✗ Red block not properly stacked")
    else:
        print("\n✗ Blocks not aligned")
    
    # Save video
    if frames:
        output_path = "../../videos/franka_proper_grasp_demo.mp4"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"\nVideo saved to {output_path}")
        print(f"Duration: {len(frames)/fps:.1f} seconds")

if __name__ == "__main__":
    main()