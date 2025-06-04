#!/usr/bin/env python3
"""
Debug why blocks are moving when robot moves.
"""

import numpy as np
import mujoco

def main():
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    print("Debugging block movement")
    print("=" * 70)
    
    # Get IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    
    # Track initial positions
    red_start = data.xpos[red_id].copy()
    ee_start = data.xpos[ee_id].copy()
    
    print(f"Initial red block: {red_start}")
    print(f"Initial EE: {ee_start}")
    print(f"Distance: {np.linalg.norm(ee_start - red_start):.3f}m")
    print()
    
    # Just step without any control changes
    print("Test 1: Step 50 times with NO control changes")
    for i in range(50):
        mujoco.mj_step(model, data)
        
        if i % 10 == 0:
            red_pos = data.xpos[red_id]
            moved = np.linalg.norm(red_pos - red_start)
            print(f"  Step {i}: Red block at {red_pos}, moved {moved:.6f}m")
    
    # Reset and try moving gripper
    print("\nTest 2: Open/close gripper")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    red_start = data.xpos[red_id].copy()
    
    for i in range(50):
        if i < 25:
            data.ctrl[7] = 255  # Open
        else:
            data.ctrl[7] = 0    # Close
            
        mujoco.mj_step(model, data)
        
        if i % 10 == 0:
            red_pos = data.xpos[red_id]
            moved = np.linalg.norm(red_pos - red_start)
            finger_pos = data.qpos[21]
            print(f"  Step {i}: Red moved {moved:.6f}m, finger at {finger_pos:.4f}")
    
    # Check contacts
    print(f"\nNumber of contacts: {data.ncon}")
    if data.ncon > 0:
        print("First 5 contacts:")
        for i in range(min(5, data.ncon)):
            c = data.contact[i]
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            print(f"  Contact {i}: {geom1_name} <-> {geom2_name}")
    
    # Reset and apply small joint movement
    print("\nTest 3: Small joint 1 movement")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    red_start = data.xpos[red_id].copy()
    
    for i in range(50):
        # Try to move joint 1 slightly
        current_j1 = data.qpos[14]
        data.ctrl[0] = current_j1 + 0.001  # Very small movement
        
        mujoco.mj_step(model, data)
        
        if i % 10 == 0:
            red_pos = data.xpos[red_id]
            moved = np.linalg.norm(red_pos - red_start)
            j1_pos = data.qpos[14]
            print(f"  Step {i}: Red moved {moved:.6f}m, joint1 at {j1_pos:.4f}")

if __name__ == "__main__":
    main()