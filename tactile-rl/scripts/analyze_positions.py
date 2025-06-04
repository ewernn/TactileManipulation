#!/usr/bin/env python3
"""
Analyze positions and dimensions of end effector and blocks.
"""

import numpy as np
import mujoco

def main():
    """Analyze positions throughout the demonstration."""
    
    # Load the model and data
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Get body IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    # Get geom IDs for dimensions
    red_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_block_geom")
    blue_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block2_geom")
    
    # Get finger positions
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    
    print("="*70)
    print("INITIAL POSITIONS AND DIMENSIONS")
    print("="*70)
    
    # Print block dimensions
    red_size = model.geom_size[red_geom_id]
    blue_size = model.geom_size[blue_geom_id]
    print(f"\nRed block dimensions: {2*red_size[0]:.3f} x {2*red_size[1]:.3f} x {2*red_size[2]:.3f} m")
    print(f"Blue block dimensions: {2*blue_size[0]:.3f} x {2*blue_size[1]:.3f} x {2*blue_size[2]:.3f} m")
    
    # Print initial positions
    ee_pos = data.xpos[ee_id].copy()
    red_pos = data.xpos[red_block_id].copy()
    blue_pos = data.xpos[blue_block_id].copy()
    
    print(f"\nEnd effector (hand) position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"Red block center position: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"Blue block center position: [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
    
    # Calculate distances
    ee_to_red = red_pos - ee_pos
    print(f"\nDistance from EE to red block: [{ee_to_red[0]:.3f}, {ee_to_red[1]:.3f}, {ee_to_red[2]:.3f}]")
    print(f"Horizontal distance (X-Y plane): {np.linalg.norm(ee_to_red[:2]):.3f} m")
    
    # Gripper opening
    gripper_opening = data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
    print(f"\nGripper opening: {gripper_opening:.3f} m")
    print(f"Max gripper opening: {model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'finger_joint1')][1]:.3f} m")
    
    # Finger positions
    left_finger_pos = data.xpos[left_finger_id].copy()
    right_finger_pos = data.xpos[right_finger_id].copy()
    print(f"\nLeft finger position: [{left_finger_pos[0]:.3f}, {left_finger_pos[1]:.3f}, {left_finger_pos[2]:.3f}]")
    print(f"Right finger position: [{right_finger_pos[0]:.3f}, {right_finger_pos[1]:.3f}, {right_finger_pos[2]:.3f}]")
    print(f"Distance between fingers: {np.linalg.norm(left_finger_pos - right_finger_pos):.3f} m")
    
    print("\n" + "="*70)
    print("SIMULATING EXPERT POLICY PHASES")
    print("="*70)
    
    # Simulate some movement phases
    phases = ["approach", "descend", "grasp", "lift"]
    phase_durations = [50, 40, 30, 40]
    
    step = 0
    for phase_idx, (phase, duration) in enumerate(zip(phases, phase_durations)):
        print(f"\n--- Phase: {phase.upper()} ---")
        
        # Simulate simple movements
        for i in range(duration):
            if phase == "approach":
                # Move towards red block in X-Y plane
                data.ctrl[0] = 0.5  # Shoulder pan
                data.ctrl[1] = -0.3  # Shoulder lift
                data.ctrl[3] = -0.5  # Elbow
            elif phase == "descend":
                # Move down
                data.ctrl[1] = 0.5
                data.ctrl[3] = 0.3
            elif phase == "grasp":
                # Close gripper
                data.ctrl[7] = 255
            elif phase == "lift":
                # Lift up
                data.ctrl[1] = -0.8
                data.ctrl[3] = -0.5
                data.ctrl[7] = 255  # Keep closed
            
            mujoco.mj_step(model, data)
            step += 1
            
            # Print positions at key moments
            if i == duration - 1:  # End of phase
                ee_pos = data.xpos[ee_id].copy()
                red_pos = data.xpos[red_block_id].copy()
                ee_to_red = red_pos - ee_pos
                
                print(f"  End of {phase}:")
                print(f"    EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                print(f"    Red block: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
                print(f"    EE to red: [{ee_to_red[0]:.3f}, {ee_to_red[1]:.3f}, {ee_to_red[2]:.3f}]")
                print(f"    X-Y distance: {np.linalg.norm(ee_to_red[:2]):.3f} m")
                
                # Check gripper state
                gripper_pos = data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
                print(f"    Gripper opening: {2*gripper_pos:.3f} m")
                
                # For grasp phase, check if we can grab the block
                if phase == "grasp":
                    left_pos = data.xpos[left_finger_id].copy()
                    right_pos = data.xpos[right_finger_id].copy()
                    finger_dist = np.linalg.norm(left_pos - right_pos)
                    print(f"    Finger separation: {finger_dist:.3f} m")
                    print(f"    Block width: {2*red_size[0]:.3f} m")
                    print(f"    Can grasp: {'YES' if finger_dist > 2*red_size[0] else 'NO'}")
    
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print("\nKey issues:")
    print("1. End effector needs to move more in +X direction to reach red block")
    print("2. Need to account for gripper finger positions, not just hand center")
    print("3. After grasping, need to move in -Y direction to position over blue block")
    print("4. Blue block is at Y=-0.15, so need significant -Y movement")
    
    print("\nRecommended approach vector:")
    print(f"- Red block at: X={0.05:.3f}, Y={0.0:.3f}")
    print(f"- Gripper should approach from: X≈{0.05-0.08:.3f}, Y=0.0 (accounting for gripper length)")
    print(f"- Then move to: X={0.05:.3f}, Y=0.0 for grasping")
    print(f"- After lifting, move to: X={0.05:.3f}, Y={-0.15:.3f} to stack on blue")

if __name__ == "__main__":
    main()