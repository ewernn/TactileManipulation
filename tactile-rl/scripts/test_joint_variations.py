#!/usr/bin/env python3
"""
Test various joint configurations and save videos/images.
"""

import numpy as np
import mujoco
import cv2
import os

def test_configuration(xml_path, joint_values, config_name, output_dir):
    """Test a specific joint configuration and save video/image."""
    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set joint values
    for i, val in enumerate(joint_values):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    
    # Set gripper open
    data.ctrl[7] = 255
    
    mujoco.mj_forward(model, data)
    
    # Get positions
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Render from multiple views
    views = ["demo_cam", "side_cam", "overhead_cam"]
    frames = []
    
    for view in views:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view)
        if cam_id >= 0:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)
        
        frame = renderer.render()
        
        # Add text overlay
        cv2.putText(frame, f"Config: {config_name}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, f"View: {view}", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
        cv2.putText(frame, f"EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]", 
                    (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Height above table: {ee_pos[2] - 0.42:.3f}m", 
                    (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Joint info
        joint_text = f"J2:{joint_values[1]:.1f} J4:{joint_values[3]:.1f} J5:{joint_values[4]:.1f} J6:{joint_values[5]:.1f}"
        cv2.putText(frame, joint_text, (40, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
        
        frames.append(frame)
    
    # Save main view as image
    img_path = os.path.join(output_dir, f"{config_name}_main.png")
    cv2.imwrite(img_path, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    
    # Create short video showing all views
    video_path = os.path.join(output_dir, f"{config_name}.mp4")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 1, (width, height))  # 1 FPS
    
    for frame in frames:
        # Show each view for 1 second
        for _ in range(30):
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    
    print(f"Config: {config_name}")
    print(f"  EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Height above table: {ee_pos[2] - 0.42:.3f}m")
    print(f"  Distance to red block: {np.linalg.norm(ee_pos - red_pos):.3f}m")
    print()

def main():
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    output_dir = "../../videos/731pm"
    
    print("Testing various joint configurations...")
    print("=" * 70)
    
    # Base configuration
    base = [0, -0.5, 0, -2.0, 0, 1.5, 0]
    
    # Test configurations
    configs = [
        # Current with joint5 = -0.8
        ([0, -0.5, 0, -2.0, -0.8, 1.5, 0], "current_j5_neg08"),
        
        # Flip joint5 to positive
        ([0, -0.5, 0, -2.0, 0.8, 1.5, 0], "j5_pos08"),
        
        # Lower by adjusting joint2
        ([0, -0.2, 0, -2.0, 0, 1.5, 0], "lower_j2_neg02"),
        ([0, -0.1, 0, -2.0, 0, 1.5, 0], "lower_j2_neg01"),
        
        # Lower by adjusting joint4
        ([0, -0.5, 0, -2.3, 0, 1.5, 0], "lower_j4_neg23"),
        ([0, -0.5, 0, -2.5, 0, 1.5, 0], "lower_j4_neg25"),
        
        # Combination: lower + joint5
        ([0, -0.2, 0, -2.2, 0.5, 1.5, 0], "combo_low_j5_pos"),
        ([0, -0.2, 0, -2.2, -0.5, 1.5, 0], "combo_low_j5_neg"),
        
        # Adjust joint6 for less tilt
        ([0, -0.3, 0, -2.1, 0, 1.2, 0], "j6_12_lower"),
        ([0, -0.3, 0, -2.1, 0, 1.8, 0], "j6_18_lower"),
        
        # Best guess combination
        ([0, -0.2, 0, -2.3, 0.3, 1.4, 0], "best_guess"),
    ]
    
    for joint_vals, name in configs:
        test_configuration(xml_path, joint_vals, name, output_dir)
    
    print(f"All configurations saved to {output_dir}")
    print("\nSummary of configurations tested:")
    print("- Joint5 variations: -0.8, 0, 0.8")
    print("- Lowering via joint2: -0.5 → -0.1")
    print("- Lowering via joint4: -2.0 → -2.5")
    print("- Joint6 adjustments: 1.2, 1.5, 1.8")
    print("- Various combinations")

if __name__ == "__main__":
    main()