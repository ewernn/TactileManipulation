#!/usr/bin/env python3
"""
Test robot orientations to understand coordinate frames.
"""

import numpy as np
import mujoco
import cv2

def test_configuration(name, joint_vals, model, data):
    """Test a specific joint configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Joint values: {joint_vals}")
    print(f"{'='*60}")
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set joint positions
    for i, val in enumerate(joint_vals):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    data.ctrl[7] = 255  # Gripper open
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Get positions
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    
    ee_pos = data.xpos[ee_id].copy()
    red_pos = data.xpos[red_id].copy()
    table_pos = data.xpos[table_id].copy()
    
    # Get gripper orientation
    hand_quat = data.xquat[ee_id].copy()
    rotmat = np.zeros(9)
    mujoco.mju_quat2Mat(rotmat, hand_quat)
    rotmat = rotmat.reshape(3, 3)
    
    # Gripper forward direction (Z-axis of hand frame)
    gripper_forward = rotmat[:, 2]
    
    print(f"End-effector position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"Red block position: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"Table position: [{table_pos[0]:.3f}, {table_pos[1]:.3f}, {table_pos[2]:.3f}]")
    print(f"Distance to red block: {np.linalg.norm(ee_pos - red_pos):.3f}m")
    
    print(f"\nGripper forward direction: [{gripper_forward[0]:.3f}, {gripper_forward[1]:.3f}, {gripper_forward[2]:.3f}]")
    
    # Check if gripper is pointing toward blocks
    to_block = red_pos - ee_pos
    to_block_normalized = to_block / np.linalg.norm(to_block)
    alignment = np.dot(gripper_forward, to_block_normalized)
    print(f"Alignment with block direction: {alignment:.3f} (1.0 = perfect, -1.0 = opposite)")
    
    # Render a frame
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="demo_cam")
    frame = renderer.render()
    
    # Add text
    cv2.putText(frame, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Alignment: {alignment:.2f}", 
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame, alignment

def main():
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Test different configurations
    configs = [
        # Original from the script (before my change)
        ("Original (lower_j2_neg01)", [0, -0.1, 0, -2.0, 0, 1.5, 0]),
        
        # What I changed it to (from MAIN_TEMPLATE)
        ("MAIN_TEMPLATE home", [3.14159, -0.785, 0, -2.356, 0, 1.920, 0.785]),
        
        # Variations to understand Joint 0
        ("Joint0=0 (default)", [0, -0.785, 0, -2.356, 0, 1.920, 0.785]),
        ("Joint0=π/2", [1.571, -0.785, 0, -2.356, 0, 1.920, 0.785]),
        ("Joint0=π", [3.14159, -0.785, 0, -2.356, 0, 1.920, 0.785]),
        ("Joint0=-π/2", [-1.571, -0.785, 0, -2.356, 0, 1.920, 0.785]),
        
        # Test with original's Joint 1 value
        ("Joint0=0, Joint1=-0.1", [0, -0.1, 0, -2.356, 0, 1.920, 0.785]),
        ("Joint0=π, Joint1=-0.1", [3.14159, -0.1, 0, -2.356, 0, 1.920, 0.785]),
    ]
    
    frames = []
    results = []
    
    for name, joint_vals in configs:
        frame, alignment = test_configuration(name, joint_vals, model, data)
        frames.append(frame)
        results.append((name, alignment))
    
    # Create comparison image
    rows = []
    for i in range(0, len(frames), 2):
        if i + 1 < len(frames):
            row = np.hstack([frames[i], frames[i+1]])
        else:
            row = np.hstack([frames[i], np.zeros_like(frames[i])])
        rows.append(row)
    
    comparison = np.vstack(rows)
    
    # Save comparison
    output_path = "robot_orientation_comparison.png"
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"\n✓ Saved comparison image to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Alignment scores (higher is better):")
    print("="*60)
    for name, alignment in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:30s} : {alignment:+.3f}")
    
    print("\n🎯 CONCLUSION:")
    print("The original configuration had Joint 0 = 0 (facing +X)")
    print("I incorrectly changed it to Joint 0 = π (facing -X)")
    print("This rotated the robot 180° away from the workspace!")

if __name__ == "__main__":
    main()