#!/usr/bin/env python3
"""
Expert gripper geometry analysis with proper fingertip calculations.
"""

import numpy as np
import mujoco
import cv2

def analyze_complete_gripper_geometry():
    """Complete gripper geometry analysis including fingertip pads."""
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset and configure
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joint_vals = [0, -0.1, 0, -2.0, 0, 1.5, 0]  # lower_j2_neg01
    for i, val in enumerate(joint_vals):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    data.ctrl[7] = 255  # Open gripper
    mujoco.mj_forward(model, data)
    
    print("COMPLETE GRIPPER GEOMETRY FOR GRASPING")
    print("=" * 80)
    
    # Get key positions
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    
    hand_pos = data.xpos[hand_id]
    left_finger_pos = data.xpos[left_finger_id]
    right_finger_pos = data.xpos[right_finger_id]
    red_pos = data.xpos[red_id]
    
    # According to the XML, fingertip pads are defined relative to finger body:
    # fingertip_pad_collision_1: size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445"
    # fingertip_pad_collision_2: size="0.003 0.002 0.003" pos="0.0055 0.002 0.05"
    # fingertip_pad_collision_3: size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05"
    # fingertip_pad_collision_4: size="0.003 0.002 0.0035" pos="0.0055 0.002 0.0395"
    # fingertip_pad_collision_5: size="0.003 0.002 0.0035" pos="-0.0055 0.002 0.0395"
    
    # The most forward points are from pads 2 and 3 at Z=0.05 (50mm forward)
    fingertip_forward_offset = 0.05  # From the collision geometry definitions
    fingertip_y_offset = 0.002  # Y offset of the pads
    
    # Calculate actual fingertip positions
    # Need to transform from finger frame to world frame
    left_finger_quat = data.xquat[left_finger_id]
    right_finger_quat = data.xquat[right_finger_id]
    
    # Convert quaternions to rotation matrices
    left_mat = np.zeros(9)
    right_mat = np.zeros(9)
    mujoco.mju_quat2Mat(left_mat, left_finger_quat)
    mujoco.mju_quat2Mat(right_mat, right_finger_quat)
    left_mat = left_mat.reshape(3, 3)
    right_mat = right_mat.reshape(3, 3)
    
    # Calculate world positions of fingertip contact points
    # Most forward pad center in local frame
    local_tip = np.array([0, fingertip_y_offset, fingertip_forward_offset])
    
    # Transform to world
    left_tip_world = left_finger_pos + left_mat @ local_tip
    right_tip_world = right_finger_pos + right_mat @ local_tip
    
    # Also calculate the forward edge of the fingertip pads
    pad_half_size = 0.003  # Half size of forward pads in Z
    local_tip_edge = np.array([0, fingertip_y_offset, fingertip_forward_offset + pad_half_size])
    left_tip_edge = left_finger_pos + left_mat @ local_tip_edge
    right_tip_edge = right_finger_pos + right_mat @ local_tip_edge
    
    print("FINGERTIP POSITIONS:")
    print("-" * 80)
    print(f"Left finger body:      [{left_finger_pos[0]:7.4f}, {left_finger_pos[1]:7.4f}, {left_finger_pos[2]:7.4f}]")
    print(f"Left fingertip center: [{left_tip_world[0]:7.4f}, {left_tip_world[1]:7.4f}, {left_tip_world[2]:7.4f}]")
    print(f"Left fingertip edge:   [{left_tip_edge[0]:7.4f}, {left_tip_edge[1]:7.4f}, {left_tip_edge[2]:7.4f}]")
    print()
    print(f"Right finger body:      [{right_finger_pos[0]:7.4f}, {right_finger_pos[1]:7.4f}, {right_finger_pos[2]:7.4f}]")
    print(f"Right fingertip center: [{right_tip_world[0]:7.4f}, {right_tip_world[1]:7.4f}, {right_tip_world[2]:7.4f}]")
    print(f"Right fingertip edge:   [{right_tip_edge[0]:7.4f}, {right_tip_edge[1]:7.4f}, {right_tip_edge[2]:7.4f}]")
    
    # Maximum forward reach
    max_forward_x = max(left_tip_edge[0], right_tip_edge[0])
    print(f"\nMaximum forward reach: X={max_forward_x:.4f}")
    print(f"Hand center: X={hand_pos[0]:.4f}")
    print(f"Total gripper extension: {max_forward_x - hand_pos[0]:.4f}m")
    
    # Block analysis
    red_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_block_geom")
    block_half_size = model.geom_size[red_geom_id][0]
    
    print("\nBLOCK GEOMETRY:")
    print("-" * 80)
    print(f"Block center: [{red_pos[0]:.4f}, {red_pos[1]:.4f}, {red_pos[2]:.4f}]")
    print(f"Block size: {2*block_half_size:.4f} x {2*block_half_size:.4f} x {2*block_half_size:.4f}m")
    print(f"Block front edge: X={red_pos[0] - block_half_size:.4f}")
    print(f"Block back edge: X={red_pos[0] + block_half_size:.4f}")
    
    # Grasp planning
    print("\nGRASP PLANNING:")
    print("-" * 80)
    
    # For a good grasp, fingertips should be:
    # 1. Past the block center (to get good grip)
    # 2. Not too far (to avoid hitting the back edge)
    
    # Ideal: fingertips at block center or slightly past
    ideal_fingertip_x = red_pos[0] + 0.01  # 1cm past center
    required_hand_x = ideal_fingertip_x - (max_forward_x - hand_pos[0])
    
    print(f"For ideal grasp:")
    print(f"  Want fingertips at X={ideal_fingertip_x:.4f} (slightly past block center)")
    print(f"  Hand should be at X={required_hand_x:.4f}")
    print(f"  Current hand X={hand_pos[0]:.4f}")
    print(f"  Need to move: {required_hand_x - hand_pos[0]:.4f}m in X")
    
    # Check if current position would hit block
    if max_forward_x > red_pos[0] - block_half_size:
        penetration = max_forward_x - (red_pos[0] - block_half_size)
        print(f"\n⚠️  WARNING: Fingertips penetrate block by {penetration:.4f}m!")
        print(f"   Fingertip X={max_forward_x:.4f}, Block front X={red_pos[0] - block_half_size:.4f}")
    else:
        gap = (red_pos[0] - block_half_size) - max_forward_x
        print(f"\n✓ Fingertips clear of block by {gap:.4f}m")
    
    # Y-axis alignment
    print("\nY-AXIS ALIGNMENT FOR GRASP:")
    print("-" * 80)
    finger_y_span = abs(left_tip_world[1] - right_tip_world[1])
    print(f"Finger span (Y-axis): {finger_y_span:.4f}m")
    print(f"Block width: {2*block_half_size:.4f}m")
    print(f"Clearance: {finger_y_span - 2*block_half_size:.4f}m")
    
    # Height check
    print("\nHEIGHT ALIGNMENT:")
    print("-" * 80)
    avg_tip_z = (left_tip_world[2] + right_tip_world[2]) / 2
    print(f"Average fingertip height: {avg_tip_z:.4f}")
    print(f"Block center height: {red_pos[2]:.4f}")
    print(f"Height difference: {avg_tip_z - red_pos[2]:.4f}m")
    
    # Create visualization
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Render from side view
    renderer.update_scene(data, camera="side_cam")
    frame = renderer.render()
    
    # Add annotations
    cv2.putText(frame, "GRIPPER GEOMETRY ANALYSIS", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    y_pos = 120
    cv2.putText(frame, f"Gripper reach: {max_forward_x - hand_pos[0]:.4f}m", 
                (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    y_pos += 40
    
    if max_forward_x > red_pos[0] - block_half_size:
        cv2.putText(frame, f"WARNING: Penetrating block by {penetration:.4f}m!", 
                    (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"Clear of block by {gap:.4f}m", 
                    (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    y_pos += 40
    
    cv2.putText(frame, f"Need to move: {required_hand_x - hand_pos[0]:.4f}m", 
                (40, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)
    
    cv2.imwrite("../../videos/731pm/gripper_complete_analysis.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Also render overhead view
    renderer.update_scene(data, camera="overhead_cam")
    frame_overhead = renderer.render()
    cv2.putText(frame_overhead, "OVERHEAD VIEW - Y ALIGNMENT", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.imwrite("../../videos/731pm/gripper_overhead.png", cv2.cvtColor(frame_overhead, cv2.COLOR_RGB2BGR))
    
    print(f"\nAnalysis images saved to videos/731pm/")
    
    return {
        'hand_pos': hand_pos,
        'max_reach': max_forward_x - hand_pos[0],
        'required_hand_x': required_hand_x,
        'current_error': required_hand_x - hand_pos[0],
        'fingertip_positions': {
            'left': left_tip_world,
            'right': right_tip_world,
            'left_edge': left_tip_edge,
            'right_edge': right_tip_edge
        }
    }

if __name__ == "__main__":
    analyze_complete_gripper_geometry()