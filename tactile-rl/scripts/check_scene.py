"""Check the actual scene configuration and block positions."""

import numpy as np
import mujoco
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.panda_demo_env import PandaDemoEnv


def check_scene():
    """Check scene configuration."""
    
    print("🔍 Checking Scene Configuration")
    print("=" * 50)
    
    # Create environment
    env = PandaDemoEnv()
    obs = env.reset(randomize=False)
    
    # Check block positions from XML keyframe
    print("\n📍 Block Positions from Observation:")
    if 'target_block_pos' in obs:
        print(f"   Red block (target): {obs['target_block_pos']}")
    if 'block2_pos' in obs:
        print(f"   Blue block (block2): {obs['block2_pos']}")
    
    # Check from body positions
    print("\n📍 Body Positions:")
    bodies = ["target_block", "block2", "hand", "left_finger", "right_finger"]
    for body_name in bodies:
        try:
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            pos = env.data.xpos[body_id]
            print(f"   {body_name:15s}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
        except:
            print(f"   {body_name:15s}: Not found")
    
    # Check joint ranges and current positions
    print("\n🔧 Arm Joint Configuration:")
    for i, (joint_id, joint_name) in enumerate(zip(env.arm_joint_ids, env.arm_joint_names)):
        if joint_id is not None:
            addr = env.model.jnt_qposadr[joint_id]
            pos = env.data.qpos[addr]
            range_min = env.model.jnt_range[joint_id][0]
            range_max = env.model.jnt_range[joint_id][1]
            print(f"   {joint_name}: {pos:6.3f} (range: [{range_min:6.3f}, {range_max:6.3f}])")
    
    # Check keyframe if exists
    print("\n🎯 Checking Keyframes:")
    if env.model.nkey > 0:
        for i in range(env.model.nkey):
            key_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_KEY, i)
            if key_name:
                print(f"\n   Keyframe '{key_name}':")
                key_qpos = env.model.key_qpos[i]
                # First few are freejoint positions for blocks
                print(f"     Block positions in keyframe:")
                print(f"       Pos 1: [{key_qpos[0]:6.3f}, {key_qpos[1]:6.3f}, {key_qpos[2]:6.3f}]")
                print(f"       Pos 2: [{key_qpos[7]:6.3f}, {key_qpos[8]:6.3f}, {key_qpos[9]:6.3f}]")
    
    # Test a specific joint configuration
    print("\n🧪 Testing reach to red block position:")
    
    # Set joints to reach toward red block area
    test_config = [0.3, 0.2, 0.0, -1.5, 0.0, 1.7, 0.785]  # Adjusted for reaching forward
    
    for i, (joint_id, pos) in enumerate(zip(env.arm_joint_ids, test_config)):
        if joint_id is not None:
            addr = env.model.jnt_qposadr[joint_id]
            env.data.qpos[addr] = pos
    
    # Forward to update positions
    mujoco.mj_forward(env.model, env.data)
    
    # Check new hand position
    hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    hand_pos = env.data.xpos[hand_id]
    print(f"\n   Hand position after test config: [{hand_pos[0]:7.3f}, {hand_pos[1]:7.3f}, {hand_pos[2]:7.3f}]")
    
    # Estimate gripper position (hand + offset)
    hand_quat = env.data.xquat[hand_id]
    hand_mat = np.zeros((3, 3))
    mujoco.mju_quat2Mat(hand_mat.flatten(), hand_quat)
    hand_mat = hand_mat.reshape((3, 3))
    gripper_offset = hand_mat @ np.array([0, 0, -0.1])
    gripper_pos = hand_pos + gripper_offset
    print(f"   Estimated gripper: [{gripper_pos[0]:7.3f}, {gripper_pos[1]:7.3f}, {gripper_pos[2]:7.3f}]")
    
    # Distance to red block
    if 'target_block_pos' in obs:
        dist = np.linalg.norm(gripper_pos - obs['target_block_pos'])
        print(f"   Distance to red block: {dist:.3f}m")


if __name__ == "__main__":
    check_scene()