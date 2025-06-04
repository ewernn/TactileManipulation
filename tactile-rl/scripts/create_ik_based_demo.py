#!/usr/bin/env python3
"""
Create expert demonstration using inverse kinematics approach.
"""

import numpy as np
import mujoco
import cv2

class IKBasedExpertPolicy:
    """Expert policy using IK-style positioning."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.phase = "pre_approach"
        self.step_count = 0
        self.phase_steps = {
            "pre_approach": 40,      # Get to good starting position
            "approach_x": 50,        # Move in X to align with block
            "approach_y": 30,        # Fine tune Y position
            "descend": 40,           # Move down to grasp
            "grasp": 20,             # Close gripper
            "lift": 40,              # Lift up
            "move_y": 50,            # Move to Y position of blue block
            "place": 40,             # Lower onto blue block
            "release": 20            # Open gripper
        }
        
        # Target positions
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        
        # Get joint IDs
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        
        # Joint limits for safety
        self.joint_limits = []
        for jid in self.joint_ids:
            self.joint_limits.append(model.jnt_range[jid])
            
    def get_ee_target_for_phase(self):
        """Get target end-effector position for current phase."""
        if self.phase == "pre_approach":
            # Position somewhat close but safe
            return self.red_block_pos + np.array([-0.15, 0, 0.1])
        elif self.phase == "approach_x":
            # Move closer in X, maintaining Y and Z
            return self.red_block_pos + np.array([-0.05, 0, 0.08])
        elif self.phase == "approach_y":
            # Fine tune Y position
            return self.red_block_pos + np.array([0, 0, 0.08])
        elif self.phase == "descend":
            # Move down to grasp height
            return self.red_block_pos + np.array([0, 0, 0.01])
        elif self.phase in ["grasp", "lift"]:
            if self.phase == "lift":
                # Lift position
                return self.red_block_pos + np.array([0, 0, 0.15])
            else:
                # Stay at grasp position
                return self.red_block_pos + np.array([0, 0, 0.01])
        elif self.phase == "move_y":
            # Move to above blue block
            return self.blue_block_pos + np.array([0, 0, 0.15])
        elif self.phase == "place":
            # Lower to stack position (blue height + half red height)
            return self.blue_block_pos + np.array([0, 0, 0.045])
        else:  # release
            # Move up slightly
            return self.blue_block_pos + np.array([0, 0, 0.1])
            
    def get_action(self, model, data):
        """Get expert action using pseudo-IK approach."""
        action = np.zeros(8)
        
        # Get current end-effector position
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ee_pos = data.xpos[ee_id].copy()
        
        # Get target position for current phase
        target_pos = self.get_ee_target_for_phase()
        pos_error = target_pos - ee_pos
        
        # Get current joint positions
        current_joints = np.array([data.qpos[jid] for jid in self.joint_ids])
        
        # Simple jacobian-based control
        # For Franka, approximate joint contributions to end-effector motion
        if self.phase in ["pre_approach", "approach_x", "approach_y"]:
            # Use base rotation for X-Y positioning
            action[0] = np.clip(3.0 * pos_error[0], -1, 1)  # Joint 1 for X
            action[2] = np.clip(3.0 * pos_error[1], -1, 1)  # Joint 3 for Y
            
            # Use shoulder/elbow for height
            if pos_error[2] > 0:  # Need to go up
                action[1] = np.clip(-2.0 * pos_error[2], -1, 1)
                action[3] = np.clip(-1.5 * pos_error[2], -1, 1)
            else:  # Need to go down
                action[1] = np.clip(-2.0 * pos_error[2], -1, 1)
                action[3] = np.clip(-1.0 * pos_error[2], -1, 1)
                
        elif self.phase == "descend":
            # Precise vertical movement
            action[1] = np.clip(4.0 * pos_error[2], -1, 1)
            action[3] = np.clip(-3.0 * pos_error[2], -1, 1)
            # Maintain X-Y position
            action[0] = np.clip(1.0 * pos_error[0], -0.3, 0.3)
            action[2] = np.clip(1.0 * pos_error[1], -0.3, 0.3)
            
        elif self.phase == "grasp":
            # Close gripper
            action[7] = 255
            # Small position corrections
            action[0] = np.clip(0.5 * pos_error[0], -0.1, 0.1)
            action[2] = np.clip(0.5 * pos_error[1], -0.1, 0.1)
            
        elif self.phase == "lift":
            # Vertical movement
            action[1] = np.clip(-4.0 * pos_error[2], -1, 1)
            action[3] = np.clip(3.0 * pos_error[2], -1, 1)
            # Keep gripper closed
            action[7] = 255
            
        elif self.phase == "move_y":
            # Move primarily in Y direction
            action[2] = np.clip(4.0 * pos_error[1], -1, 1)  # Strong Y movement
            action[0] = np.clip(2.0 * pos_error[0], -0.5, 0.5)  # X correction
            action[1] = np.clip(-1.0 * pos_error[2], -0.5, 0.5)  # Maintain height
            # Keep gripper closed
            action[7] = 255
            
        elif self.phase == "place":
            # Precise lowering
            action[1] = np.clip(3.0 * pos_error[2], -1, 1)
            action[3] = np.clip(-2.5 * pos_error[2], -1, 1)
            # Maintain X-Y
            action[0] = np.clip(1.0 * pos_error[0], -0.2, 0.2)
            action[2] = np.clip(1.0 * pos_error[1], -0.2, 0.2)
            # Keep gripper closed
            action[7] = 255
            
        elif self.phase == "release":
            # Open gripper
            action[7] = 0
            # Small upward movement
            action[1] = -0.5
            
        # Safety: clip actions based on joint limits
        for i in range(7):
            current_pos = current_joints[i]
            if action[i] > 0 and current_pos >= self.joint_limits[i][1] - 0.1:
                action[i] = 0
            elif action[i] < 0 and current_pos <= self.joint_limits[i][0] + 0.1:
                action[i] = 0
        
        # Update phase
        self.step_count += 1
        if self.step_count >= self.phase_steps[self.phase]:
            phases = list(self.phase_steps.keys())
            current_idx = phases.index(self.phase)
            if current_idx < len(phases) - 1:
                self.phase = phases[current_idx + 1]
                self.step_count = 0
                print(f"\n==> Transitioning to phase: {self.phase}")
                
        return action

def main():
    """Create video with IK-based demonstration."""
    
    # Load model and data
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Initialize policy
    expert = IKBasedExpertPolicy(model, data)
    
    # Renderer setup
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video frames
    frames = []
    fps = 30
    
    print("Creating IK-based Franka demonstration...")
    print("=" * 70)
    
    # Get body IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Run demo
    total_steps = sum(expert.phase_steps.values())
    for step in range(total_steps):
        # Get action
        action = expert.get_action(model, data)
        
        # Apply action
        data.ctrl[:] = action
        
        # Step physics
        mujoco.mj_step(model, data)
        
        # Log progress
        if step % 10 == 0:
            ee_pos = data.xpos[ee_id]
            red_pos = data.xpos[red_id]
            target_pos = expert.get_ee_target_for_phase()
            error = np.linalg.norm(target_pos - ee_pos)
            
            print(f"Step {step:3d} | Phase: {expert.phase:12s} | "
                  f"EE: [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}] | "
                  f"Error: {error:.3f}m")
            
            # Also print gripper state
            gripper_opening = data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
            print(f"         | Gripper: {gripper_opening:.3f}m | "
                  f"Red block: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
        
        # Render
        renderer.update_scene(data, camera="demo_cam")
        frame = renderer.render()
        
        # Overlay
        cv2.putText(frame, f"Phase: {expert.phase}", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Phase progress
        progress = expert.step_count / expert.phase_steps[expert.phase]
        bar_width = int(400 * progress)
        cv2.rectangle(frame, (40, 100), (40 + bar_width, 140), (0, 255, 0), -1)
        cv2.rectangle(frame, (40, 100), (440, 140), (255, 255, 255), 3)
        
        # Gripper state
        gripper_text = "CLOSED" if action[7] > 100 else "OPEN"
        color = (0, 0, 255) if action[7] > 100 else (0, 255, 0)
        cv2.putText(frame, f"Gripper: {gripper_text}", (40, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        
        # Position info
        ee_pos = data.xpos[ee_id]
        cv2.putText(frame, f"X:{ee_pos[0]:.2f} Y:{ee_pos[1]:.2f} Z:{ee_pos[2]:.2f}", 
                    (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    
    # Check final positions
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    blue_pos = data.xpos[blue_id]
    print(f"\nFinal positions:")
    print(f"  End-effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Red block: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"  Blue block: [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
    
    # Save video
    if frames:
        output_path = "../../videos/franka_ik_stacking_demo.mp4"
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