#!/usr/bin/env python3
"""
Test script to understand coordinate system and robot positioning.
"""

import numpy as np
import sys
import os
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments'))
from environments.panda_demo_env import PandaDemoEnv


def test_coordinates():
    """Test coordinate system and robot reach."""
    
    env = PandaDemoEnv(camera_name="demo_cam")
    obs = env.reset()
    
    print("\n" + "="*60)
    print("COORDINATE SYSTEM TEST")
    print("="*60)
    
    # Get all relevant positions
    ee_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    base_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
    
    print("\n📍 Positions:")
    print(f"  Base (link0): {env.data.xpos[base_id]}")
    print(f"  Hand body: {env.data.xpos[hand_id]}")
    if ee_site_id >= 0:
        print(f"  EE site: {env.data.site_xpos[ee_site_id]}")
    print(f"  Target block: {obs['target_block_pos']}")
    
    # Calculate distances
    if ee_site_id >= 0:
        ee_pos = env.data.site_xpos[ee_site_id]
    else:
        ee_pos = env.data.xpos[hand_id]
    
    distance = np.linalg.norm(ee_pos - obs['target_block_pos'])
    xy_distance = np.linalg.norm(ee_pos[:2] - obs['target_block_pos'][:2])
    
    print(f"\n📏 Distances:")
    print(f"  Total distance: {distance:.3f}m")
    print(f"  XY distance: {xy_distance:.3f}m")
    print(f"  X gap: {obs['target_block_pos'][0] - ee_pos[0]:.3f}m")
    print(f"  Y gap: {obs['target_block_pos'][1] - ee_pos[1]:.3f}m")
    print(f"  Z gap: {obs['target_block_pos'][2] - ee_pos[2]:.3f}m")
    
    # Joint positions
    print(f"\n🦾 Joint positions:")
    for i, pos in enumerate(obs['joint_pos']):
        print(f"  Joint {i+1}: {pos:.3f} rad ({np.degrees(pos):.1f}°)")
    
    # Test if the robot is facing the right direction
    print(f"\n🎯 Orientation check:")
    # Get hand orientation
    hand_quat = env.data.xquat[hand_id]
    rotmat = np.zeros(9)
    mujoco.mju_quat2Mat(rotmat, hand_quat)
    rotmat = rotmat.reshape(3, 3)
    
    # The gripper's pointing direction is typically the Z axis of its frame
    gripper_direction = rotmat[:, 2]
    print(f"  Gripper pointing direction: [{gripper_direction[0]:.3f}, {gripper_direction[1]:.3f}, {gripper_direction[2]:.3f}]")
    print(f"  Gripper X-component: {gripper_direction[0]:.3f} (should be > 0 to point forward)")
    
    # Test maximum reach
    print(f"\n🤸 Testing maximum forward reach...")
    
    # Move all joints to reach forward
    test_action = np.zeros(8)
    test_action[0] = 0.5   # Joint 1 - base rotation
    test_action[1] = 0.5   # Joint 2 - shoulder
    test_action[2] = 0.5   # Joint 3 - elbow  
    test_action[3] = -0.5  # Joint 4 - forearm
    test_action[7] = -1.0  # Keep gripper open
    
    # Apply action for several steps
    for _ in range(50):
        obs = env.step(test_action)
    
    # Check new position
    if ee_site_id >= 0:
        new_ee_pos = env.data.site_xpos[ee_site_id]
    else:
        new_ee_pos = env.data.xpos[hand_id]
    
    print(f"\n  After 50 steps of forward motion:")
    print(f"  New EE position: {new_ee_pos}")
    print(f"  Movement: {new_ee_pos - ee_pos}")
    print(f"  New distance to target: {np.linalg.norm(new_ee_pos - obs['target_block_pos']):.3f}m")
    
    # Check if blocks are reachable
    print(f"\n❓ Reachability analysis:")
    typical_reach = 0.855  # Panda's typical max reach
    print(f"  Typical Panda reach: {typical_reach}m")
    print(f"  Current distance: {xy_distance:.3f}m")
    if xy_distance > typical_reach:
        print(f"  ❌ Blocks are likely OUT OF REACH!")
        print(f"  Need to move blocks {xy_distance - typical_reach:.3f}m closer")
    else:
        print(f"  ✅ Blocks should be reachable")


if __name__ == "__main__":
    test_coordinates()