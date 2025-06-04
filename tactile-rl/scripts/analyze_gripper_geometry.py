#!/usr/bin/env python3
"""
Analyze gripper geometry in detail to understand grasping.
"""

import numpy as np
import mujoco
import cv2

def analyze_gripper_geometry():
    """Detailed analysis of gripper geometry."""
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set to lower_j2_neg01 configuration
    joint_vals = [0, -0.1, 0, -2.0, 0, 1.5, 0]
    for i, val in enumerate(joint_vals):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    
    # Test with gripper open
    data.ctrl[7] = 255
    mujoco.mj_forward(model, data)
    
    print("GRIPPER GEOMETRY ANALYSIS")
    print("=" * 80)
    
    # Get all relevant body IDs
    bodies = {
        'hand': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand"),
        'left_finger': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
        'right_finger': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
        'target_block': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block"),
        'wrist_camera': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_camera")
    }
    
    # Get positions
    positions = {}
    for name, body_id in bodies.items():
        if body_id >= 0:
            positions[name] = data.xpos[body_id].copy()
            print(f"{name:12s}: [{positions[name][0]:7.3f}, {positions[name][1]:7.3f}, {positions[name][2]:7.3f}]")
    
    print("\nGRIPPER GEOMETRY:")
    print("-" * 80)
    
    # Calculate finger offsets from hand
    if 'hand' in positions and 'left_finger' in positions:
        left_offset = positions['left_finger'] - positions['hand']
        right_offset = positions['right_finger'] - positions['hand']
        print(f"Left finger offset from hand:  [{left_offset[0]:7.3f}, {left_offset[1]:7.3f}, {left_offset[2]:7.3f}]")
        print(f"Right finger offset from hand: [{right_offset[0]:7.3f}, {right_offset[1]:7.3f}, {right_offset[2]:7.3f}]")
        
        # Finger separation
        finger_sep = np.linalg.norm(positions['left_finger'] - positions['right_finger'])
        print(f"\nFinger separation: {finger_sep:.3f}m")
    
    # Get finger joint positions
    finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    finger1_pos = data.qpos[model.jnt_qposadr[finger1_id]]
    finger2_pos = data.qpos[model.jnt_qposadr[finger2_id]]
    print(f"\nFinger joint positions:")
    print(f"  finger_joint1: {finger1_pos:.4f}m")
    print(f"  finger_joint2: {finger2_pos:.4f}m")
    
    # Get all gripper-related geoms
    print("\nGRIPPER GEOMS:")
    print("-" * 80)
    
    # Find finger geoms
    finger_geoms = []
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and ('finger' in geom_name or 'fingertip' in geom_name or 'hand' in geom_name):
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            geom_type = model.geom_type[i]
            geom_size = model.geom_size[i]
            geom_pos = model.geom_pos[i]
            
            # Get world position
            world_pos = data.geom_xpos[i]
            
            print(f"\nGeom: {geom_name}")
            print(f"  Body: {body_name}")
            print(f"  Type: {['plane', 'hfield', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box', 'mesh'][geom_type]}")
            print(f"  Size: {geom_size[:3]}")
            print(f"  Local pos: [{geom_pos[0]:.4f}, {geom_pos[1]:.4f}, {geom_pos[2]:.4f}]")
            print(f"  World pos: [{world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}]")
            
            finger_geoms.append((geom_name, world_pos.copy()))
    
    # Find extreme points
    print("\nGRIPPER EXTREME POINTS:")
    print("-" * 80)
    
    if finger_geoms:
        x_coords = [pos[0] for _, pos in finger_geoms]
        y_coords = [pos[1] for _, pos in finger_geoms]
        z_coords = [pos[2] for _, pos in finger_geoms]
        
        print(f"X range: [{min(x_coords):.4f}, {max(x_coords):.4f}] (width: {max(x_coords)-min(x_coords):.4f}m)")
        print(f"Y range: [{min(y_coords):.4f}, {max(y_coords):.4f}] (width: {max(y_coords)-min(y_coords):.4f}m)")
        print(f"Z range: [{min(z_coords):.4f}, {max(z_coords):.4f}] (height: {max(z_coords)-min(z_coords):.4f}m)")
        
        # Most forward point (max X)
        forward_geom = max(finger_geoms, key=lambda x: x[1][0])
        print(f"\nMost forward point: {forward_geom[0]} at X={forward_geom[1][0]:.4f}")
        
        # Leftmost and rightmost for gripper width
        left_geom = min(finger_geoms, key=lambda x: x[1][1])
        right_geom = max(finger_geoms, key=lambda x: x[1][1])
        print(f"Leftmost point: {left_geom[0]} at Y={left_geom[1][1]:.4f}")
        print(f"Rightmost point: {right_geom[0]} at Y={right_geom[1][1]:.4f}")
    
    # Calculate approach distances
    print("\nAPPROACH CALCULATIONS:")
    print("-" * 80)
    
    red_pos = positions['target_block']
    hand_pos = positions['hand']
    
    # Block dimensions
    red_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_block_geom")
    red_size = model.geom_size[red_geom_id]
    print(f"Red block size: {2*red_size[0]:.3f} x {2*red_size[1]:.3f} x {2*red_size[2]:.3f}m")
    
    # Current distances
    hand_to_block = red_pos - hand_pos
    print(f"\nHand to block center: [{hand_to_block[0]:.4f}, {hand_to_block[1]:.4f}, {hand_to_block[2]:.4f}]")
    
    # Calculate safe approach distance
    if finger_geoms:
        forward_point_x = max([pos[0] for _, pos in finger_geoms])
        gripper_extension = forward_point_x - hand_pos[0]
        print(f"\nGripper extends {gripper_extension:.4f}m forward from hand center")
        
        # Safe approach position (gripper tip should be at block edge - small margin)
        margin = 0.01  # 1cm safety margin
        block_edge_x = red_pos[0] - red_size[0]  # Front edge of block
        safe_hand_x = block_edge_x - gripper_extension - margin
        
        print(f"\nFor safe approach:")
        print(f"  Block front edge at X={block_edge_x:.4f}")
        print(f"  Hand should be at X={safe_hand_x:.4f}")
        print(f"  Current hand X={hand_pos[0]:.4f}")
        print(f"  Need to move {safe_hand_x - hand_pos[0]:.4f}m in X")
    
    # Gripper opening requirements
    print("\nGRASPING REQUIREMENTS:")
    print("-" * 80)
    
    block_width = 2 * red_size[0]
    current_finger_sep = finger1_pos + finger2_pos  # Both fingers move symmetrically
    print(f"Block width: {block_width:.4f}m")
    print(f"Current gripper opening: {current_finger_sep:.4f}m")
    print(f"Need at least {block_width + 0.01:.4f}m opening to grasp")
    
    if current_finger_sep > block_width + 0.01:
        print("✓ Gripper is open enough to grasp")
    else:
        print("✗ Gripper needs to open more")
    
    # Visualization
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Render with annotations
    renderer.update_scene(data, camera="side_cam")
    frame = renderer.render()
    
    # Add text
    cv2.putText(frame, "GRIPPER GEOMETRY ANALYSIS", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame, f"Hand pos: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]", 
                (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Gripper extends: {gripper_extension:.3f}m forward", 
                (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Finger separation: {current_finger_sep:.3f}m", 
                (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Save analysis image
    cv2.imwrite("../../videos/731pm/gripper_geometry_analysis.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"\nAnalysis image saved to videos/731pm/gripper_geometry_analysis.png")

if __name__ == "__main__":
    analyze_gripper_geometry()