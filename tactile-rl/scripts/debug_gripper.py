"""
Debug why the gripper isn't picking up blocks.
"""

import numpy as np
import sys
import os

sys.path.append('../environments')
from panda_demo_env import PandaDemoEnv
import mujoco


def debug_gripper():
    """Debug gripper and block interaction."""
    
    print("🔍 Debugging Gripper-Block Interaction...")
    
    env = PandaDemoEnv()
    
    # Reset environment
    obs = env.reset(randomize=False)
    
    print(f"\nInitial state:")
    print(f"  Red block pos: {obs['target_block_pos']}")
    print(f"  Blue block pos: {obs['block2_pos']}")
    print(f"  Gripper opening: {obs['gripper_pos']}")
    
    # Get end-effector position
    ee_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    ee_pos = env.data.xpos[ee_id]
    print(f"  End-effector pos: {ee_pos}")
    
    # Check gripper finger positions
    left_finger_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    left_pos = env.data.xpos[left_finger_id]
    right_pos = env.data.xpos[right_finger_id]
    print(f"  Left finger pos: {left_pos}")
    print(f"  Right finger pos: {right_pos}")
    
    # Move robot to be above red block
    print("\n📍 Moving to above red block...")
    
    # Approach motion
    for i in range(100):
        if i < 50:
            action = np.array([0.0, 0.3, 0.25, -0.35, 0.15, -0.1, 0.0, -1.0])
        else:
            action = np.array([0.0, 0.2, 0.3, -0.3, -0.2, -0.05, 0.0, -1.0])
        
        obs = env.step(action, steps=5)
        
        if i % 25 == 0:
            ee_pos = env.data.xpos[ee_id]
            red_pos = obs['target_block_pos']
            dist = np.linalg.norm(ee_pos[:2] - red_pos[:2])
            print(f"  Step {i}: EE-Block distance = {dist:.3f}, EE height = {ee_pos[2]:.3f}")
    
    # Check final position
    ee_pos = env.data.xpos[ee_id]
    red_pos = obs['target_block_pos']
    print(f"\nAfter approach:")
    print(f"  End-effector: {ee_pos}")
    print(f"  Red block: {red_pos}")
    print(f"  Horizontal distance: {np.linalg.norm(ee_pos[:2] - red_pos[:2]):.3f}")
    print(f"  Vertical distance: {ee_pos[2] - red_pos[2]:.3f}")
    
    # Close gripper
    print("\n🤏 Closing gripper...")
    initial_block_pos = red_pos.copy()
    
    for i in range(40):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        obs = env.step(action, steps=5)
        
        if i % 10 == 0:
            print(f"  Step {i}: Gripper={obs['gripper_pos']:.3f}, Tactile={np.sum(obs['tactile']):.1f}")
    
    # Check if block moved
    final_block_pos = obs['target_block_pos']
    block_moved = np.linalg.norm(final_block_pos - initial_block_pos) > 0.001
    print(f"\nBlock moved: {block_moved}")
    print(f"  Initial: {initial_block_pos}")
    print(f"  Final: {final_block_pos}")
    
    # Try to lift
    print("\n⬆️ Attempting to lift...")
    
    for i in range(40):
        action = np.array([0.0, -0.3, -0.3, 0.35, 0.15, 0.0, 0.0, 1.0])
        obs = env.step(action, steps=5)
        
        if i % 10 == 0:
            red_pos = obs['target_block_pos']
            ee_pos = env.data.xpos[ee_id]
            print(f"  Step {i}: Block height={red_pos[2]:.3f}, EE height={ee_pos[2]:.3f}")
    
    # Analyze results
    print("\n📊 Analysis:")
    
    # Check contact forces
    contact_forces = []
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        # Check if contact involves red block
        geom1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        
        if geom1_name and geom2_name:
            if "target_block" in geom1_name or "target_block" in geom2_name:
                if "finger" in geom1_name or "finger" in geom2_name:
                    force = np.linalg.norm(env.data.efc_force[i])
                    contact_forces.append(force)
                    print(f"  Contact: {geom1_name} <-> {geom2_name}, Force: {force:.2f}")
    
    if len(contact_forces) > 0:
        print(f"  Total contact forces: {sum(contact_forces):.2f}")
    else:
        print("  ⚠️  No gripper-block contacts detected!")
    
    # Check gripper actuator
    gripper_ctrl_id = 7  # Last actuator
    print(f"\n  Gripper control: {env.data.ctrl[gripper_ctrl_id]}")
    print(f"  Gripper actuator force: {env.data.actuator_force[gripper_ctrl_id]}")
    
    # Save a frame for visual inspection
    frame = env.render()
    import imageio
    imageio.imwrite("debug_gripper_position.png", frame)
    print(f"\n💾 Saved debug image to: debug_gripper_position.png")


if __name__ == "__main__":
    debug_gripper()