#!/usr/bin/env python3
"""
Analyze gripper geometry in detail - improved version.
"""

import numpy as np
import mujoco
import cv2

def get_all_gripper_points(model, data):
    """Get all points that make up the gripper geometry."""
    gripper_points = []
    gripper_info = []
    
    # Bodies that are part of the gripper
    gripper_bodies = ['hand', 'left_finger', 'right_finger']
    
    for body_name in gripper_bodies:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
            
        # Get all geoms attached to this body
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == body_id:
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                
                # Skip visual-only geoms
                if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
                    continue
                
                # Get geom info
                geom_type = model.geom_type[i]
                geom_size = model.geom_size[i].copy()
                geom_pos = model.geom_pos[i].copy()
                
                # Get world transform
                geom_xpos = data.geom_xpos[i].copy()
                geom_xmat = data.geom_xmat[i].reshape(3, 3)
                
                # Calculate bounding points based on geom type
                if geom_type == 6:  # Box
                    # 8 corners of the box
                    corners = []
                    for dx in [-1, 1]:
                        for dy in [-1, 1]:
                            for dz in [-1, 1]:
                                local_point = np.array([dx * geom_size[0], 
                                                       dy * geom_size[1], 
                                                       dz * geom_size[2]])
                                world_point = geom_xpos + geom_xmat @ local_point
                                corners.append(world_point)
                                gripper_points.append(world_point)
                    
                    # Find most forward corner
                    forward_corner = max(corners, key=lambda p: p[0])
                    gripper_info.append({
                        'name': geom_name,
                        'body': body_name,
                        'type': 'box',
                        'center': geom_xpos,
                        'forward_point': forward_corner,
                        'size': geom_size
                    })
                    
                elif geom_type == 7:  # Mesh
                    # For mesh, use center and add approximate bounds
                    gripper_points.append(geom_xpos)
                    gripper_info.append({
                        'name': geom_name,
                        'body': body_name,
                        'type': 'mesh',
                        'center': geom_xpos
                    })
                else:
                    # For other types, use center
                    gripper_points.append(geom_xpos)
                    gripper_info.append({
                        'name': geom_name,
                        'body': body_name,
                        'type': geom_type,
                        'center': geom_xpos
                    })
    
    return np.array(gripper_points), gripper_info

def main():
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset and set configuration
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joint_vals = [0, -0.1, 0, -2.0, 0, 1.5, 0]  # lower_j2_neg01 config
    for i, val in enumerate(joint_vals):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    data.ctrl[7] = 255  # Open gripper
    mujoco.mj_forward(model, data)
    
    print("DETAILED GRIPPER GEOMETRY ANALYSIS")
    print("=" * 80)
    
    # Get body positions
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    
    hand_pos = data.xpos[hand_id]
    left_finger_pos = data.xpos[left_finger_id]
    right_finger_pos = data.xpos[right_finger_id]
    red_pos = data.xpos[red_id]
    
    print(f"Hand center:  [{hand_pos[0]:7.3f}, {hand_pos[1]:7.3f}, {hand_pos[2]:7.3f}]")
    print(f"Left finger:  [{left_finger_pos[0]:7.3f}, {left_finger_pos[1]:7.3f}, {left_finger_pos[2]:7.3f}]")
    print(f"Right finger: [{right_finger_pos[0]:7.3f}, {right_finger_pos[1]:7.3f}, {right_finger_pos[2]:7.3f}]")
    print(f"Red block:    [{red_pos[0]:7.3f}, {red_pos[1]:7.3f}, {red_pos[2]:7.3f}]")
    
    # Get gripper opening
    finger1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    finger2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
    finger1_pos = data.qpos[model.jnt_qposadr[finger1_id]]
    finger2_pos = data.qpos[model.jnt_qposadr[finger2_id]]
    
    print(f"\nGripper opening: {finger1_pos + finger2_pos:.4f}m")
    print(f"Finger separation: {np.linalg.norm(left_finger_pos - right_finger_pos):.4f}m")
    
    # Get all gripper collision geoms
    print("\nCOLLISION GEOMETRY:")
    print("-" * 80)
    
    # Find fingertip pads specifically
    fingertip_geoms = []
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and 'fingertip_pad' in geom_name:
            geom_xpos = data.geom_xpos[i]
            body_id = model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"{geom_name:30s} at [{geom_xpos[0]:7.4f}, {geom_xpos[1]:7.4f}, {geom_xpos[2]:7.4f}] (on {body_name})")
            fingertip_geoms.append((geom_name, geom_xpos, body_name))
    
    # Find extreme points
    if fingertip_geoms:
        print("\nFINGERTIP EXTREMES:")
        print("-" * 80)
        
        # Group by finger
        left_tips = [(n, p) for n, p, b in fingertip_geoms if 'left' in b]
        right_tips = [(n, p) for n, p, b in fingertip_geoms if 'right' in b]
        
        if left_tips:
            left_forward = max(left_tips, key=lambda x: x[1][0])
            print(f"Left finger most forward:  {left_forward[0]} at X={left_forward[1][0]:.4f}")
            
        if right_tips:
            right_forward = max(right_tips, key=lambda x: x[1][0])
            print(f"Right finger most forward: {right_forward[0]} at X={right_forward[1][0]:.4f}")
        
        # Overall most forward point
        all_x = [p[0] for _, p, _ in fingertip_geoms]
        max_x = max(all_x)
        print(f"\nMost forward fingertip point: X={max_x:.4f}")
        print(f"Hand center X: {hand_pos[0]:.4f}")
        print(f"Gripper extends: {max_x - hand_pos[0]:.4f}m beyond hand center")
    
    # Calculate approach strategy
    print("\nAPPROACH STRATEGY:")
    print("-" * 80)
    
    # Block info
    red_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_block_geom")
    block_size = model.geom_size[red_geom_id][0]  # Cube, so all sides equal
    
    print(f"Block center: X={red_pos[0]:.4f}")
    print(f"Block front edge: X={red_pos[0] - block_size:.4f}")
    print(f"Block back edge: X={red_pos[0] + block_size:.4f}")
    print(f"Block width: {2*block_size:.4f}m")
    
    # Safe approach calculations
    if fingertip_geoms:
        gripper_reach = max_x - hand_pos[0]
        safety_margin = 0.005  # 5mm safety
        
        # For grasping, fingertips should be past block center
        grasp_hand_x = red_pos[0] - gripper_reach + block_size + safety_margin
        
        print(f"\nFor proper grasp:")
        print(f"  Hand should be at X={grasp_hand_x:.4f}")
        print(f"  This puts fingertips at X={grasp_hand_x + gripper_reach:.4f}")
        print(f"  Which is {grasp_hand_x + gripper_reach - red_pos[0]:.4f}m past block center")
        
        current_error = grasp_hand_x - hand_pos[0]
        print(f"\nCurrent position error: {current_error:.4f}m")
        if current_error > 0:
            print(f"  → Need to move {current_error:.4f}m forward (+X)")
        else:
            print(f"  → Need to move {-current_error:.4f}m backward (-X)")
    
    # Height alignment
    print("\nHEIGHT ALIGNMENT:")
    print("-" * 80)
    print(f"Hand Z: {hand_pos[2]:.4f}")
    print(f"Block Z: {red_pos[2]:.4f}")
    print(f"Height difference: {hand_pos[2] - red_pos[2]:.4f}m")
    
    # Y alignment for grasping
    print("\nY-AXIS ALIGNMENT:")
    print("-" * 80)
    print(f"Hand Y: {hand_pos[1]:.4f}")
    print(f"Block Y: {red_pos[1]:.4f}")
    print(f"Y offset: {hand_pos[1] - red_pos[1]:.4f}m")
    
    # Gripper orientation check
    hand_quat = data.xquat[hand_id]
    hand_mat = np.zeros(9)
    mujoco.mju_quat2Mat(hand_mat, hand_quat)
    hand_mat = hand_mat.reshape(3, 3)
    
    print("\nGRIPPER ORIENTATION:")
    print("-" * 80)
    print(f"Gripper X-axis (forward): [{hand_mat[0,0]:6.3f}, {hand_mat[1,0]:6.3f}, {hand_mat[2,0]:6.3f}]")
    print(f"Gripper Y-axis (left):    [{hand_mat[0,1]:6.3f}, {hand_mat[1,1]:6.3f}, {hand_mat[2,1]:6.3f}]")
    print(f"Gripper Z-axis (up):      [{hand_mat[0,2]:6.3f}, {hand_mat[1,2]:6.3f}, {hand_mat[2,2]:6.3f}]")
    
    # Visual output
    renderer = mujoco.Renderer(model, height=720, width=1280)
    renderer.update_scene(data, camera="side_cam")
    frame = renderer.render()
    
    cv2.putText(frame, "GRIPPER GEOMETRY", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    info_y = 120
    if fingertip_geoms:
        cv2.putText(frame, f"Gripper reach: {gripper_reach:.3f}m", 
                    (40, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        info_y += 40
        cv2.putText(frame, f"Position error: {current_error:.3f}m", 
                    (40, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
    
    cv2.imwrite("../../videos/731pm/gripper_geometry_detailed.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"\nImage saved to videos/731pm/gripper_geometry_detailed.png")

if __name__ == "__main__":
    main()