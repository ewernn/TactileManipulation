#!/usr/bin/env python3
"""
Check starting configuration and positions.
"""

import numpy as np
import mujoco

def main():
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    print("=" * 70)
    print("CURRENT STARTING CONFIGURATION")
    print("=" * 70)
    
    # Get joint names and positions
    joint_names = [f"joint{i}" for i in range(1, 8)]
    print("\nJoint positions (from keyframe):")
    for i, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_pos = data.qpos[joint_id]
        print(f"  {name}: {joint_pos:.4f} rad ({np.degrees(joint_pos):.1f}°)")
    
    # Get control values
    print("\nControl values (from keyframe):")
    for i in range(8):
        print(f"  ctrl[{i}]: {data.ctrl[i]:.4f}")
    
    # Get body positions
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")
    
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    blue_pos = data.xpos[blue_id]
    
    print("\nBody positions:")
    print(f"  End effector: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"  Red block:    [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"  Blue block:   [{blue_pos[0]:.3f}, {blue_pos[1]:.3f}, {blue_pos[2]:.3f}]")
    
    print("\nRelative positions:")
    ee_to_red = red_pos - ee_pos
    print(f"  EE to red block: [{ee_to_red[0]:.3f}, {ee_to_red[1]:.3f}, {ee_to_red[2]:.3f}]")
    print(f"  Distance: {np.linalg.norm(ee_to_red):.3f} m")
    
    # Get gripper orientation
    hand_quat = data.xquat[ee_id]
    rotmat = np.zeros(9)
    mujoco.mju_quat2Mat(rotmat, hand_quat)
    rotmat = rotmat.reshape(3, 3)
    
    print("\nGripper orientation:")
    print(f"  X-axis: [{rotmat[0,0]:.3f}, {rotmat[1,0]:.3f}, {rotmat[2,0]:.3f}]")
    print(f"  Y-axis: [{rotmat[0,1]:.3f}, {rotmat[1,1]:.3f}, {rotmat[2,1]:.3f}]")
    print(f"  Z-axis: [{rotmat[0,2]:.3f}, {rotmat[1,2]:.3f}, {rotmat[2,2]:.3f}]")
    
    # Keyframe raw values
    print("\n" + "=" * 70)
    print("KEYFRAME RAW VALUES (from XML)")
    print("=" * 70)
    
    # Read keyframe from XML
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    keyframe = root.find(".//keyframe/key[@name='home']")
    if keyframe is not None:
        qpos = keyframe.get('qpos')
        ctrl = keyframe.get('ctrl')
        print(f"\nqpos: {qpos}")
        print(f"\nctrl: {ctrl}")
        
        # Parse joint values
        qpos_vals = [float(x) for x in qpos.split()]
        print("\nJoint values from keyframe:")
        # First 7 values after the block positions/orientations (14 values)
        joint_start = 14
        for i in range(7):
            print(f"  joint{i+1}: {qpos_vals[joint_start + i]:.4f} rad ({np.degrees(qpos_vals[joint_start + i]):.1f}°)")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("\nTo adjust starting position:")
    print("1. Edit keyframe in panda_demo_scene.xml")
    print("2. Joint adjustments needed:")
    print("   - To move DOWN: increase joint2 (shoulder) from current -0.785")
    print("   - To move LEFT (+Y): decrease joint3 from current 0.0")
    print("   - To reduce tilt: adjust joint6 (wrist) from current 1.920")
    print("\nBest approach: Modify keyframe first, then velocity control will start from new position")

if __name__ == "__main__":
    main()