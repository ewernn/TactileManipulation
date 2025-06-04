"""
Diagnose the environment setup to understand joint ranges and positions.
"""

import numpy as np
import mujoco
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.panda_demo_env import PandaDemoEnv


def diagnose_environment():
    """Diagnose the environment to understand joint ranges and positions."""
    
    print("🔍 Diagnosing Panda Demo Environment")
    print("=" * 60)
    
    # Create environment
    env = PandaDemoEnv()
    obs = env.reset(randomize=False)
    
    print("\n📊 Model Information:")
    print(f"   Total joints: {env.model.njnt}")
    print(f"   Total actuators: {env.model.nu}")
    print(f"   Total bodies: {env.model.nbody}")
    
    print("\n🔧 Joint Information:")
    for i in range(env.model.njnt):
        joint_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = env.model.jnt_type[i]
        joint_range = env.model.jnt_range[i]
        joint_addr = env.model.jnt_qposadr[i]
        joint_pos = env.data.qpos[joint_addr]
        
        type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        type_str = type_names.get(joint_type, "unknown")
        
        if joint_name:
            print(f"   Joint {i:2d}: {joint_name:20s} | Type: {type_str:6s} | "
                  f"Range: [{joint_range[0]:7.3f}, {joint_range[1]:7.3f}] | "
                  f"Pos: {joint_pos:7.3f}")
        else:
            print(f"   Joint {i:2d}: <unnamed>            | Type: {type_str:6s}")
    
    print("\n🎮 Actuator Information:")
    for i in range(env.model.nu):
        actuator_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        ctrl_range = env.model.actuator_ctrlrange[i]
        ctrl_value = env.data.ctrl[i]
        
        if actuator_name:
            print(f"   Actuator {i}: {actuator_name:20s} | "
                  f"Range: [{ctrl_range[0]:7.1f}, {ctrl_range[1]:7.1f}] | "
                  f"Ctrl: {ctrl_value:7.1f}")
        else:
            print(f"   Actuator {i}: <unnamed>            | "
                  f"Range: [{ctrl_range[0]:7.1f}, {ctrl_range[1]:7.1f}] | "
                  f"Ctrl: {ctrl_value:7.1f}")
    
    print("\n🤖 Body Positions:")
    important_bodies = ["hand", "left_finger", "right_finger", "target_block", "block2"]
    for body_name in important_bodies:
        try:
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            body_pos = env.data.xpos[body_id]
            print(f"   {body_name:15s}: [{body_pos[0]:7.3f}, {body_pos[1]:7.3f}, {body_pos[2]:7.3f}]")
        except:
            print(f"   {body_name:15s}: Not found")
    
    print("\n🔍 Gripper Analysis:")
    if env.gripper_joint1_id is not None:
        gripper_addr1 = env.model.jnt_qposadr[env.gripper_joint1_id]
        gripper_addr2 = env.model.jnt_qposadr[env.gripper_joint2_id]
        gripper_pos1 = env.data.qpos[gripper_addr1]
        gripper_pos2 = env.data.qpos[gripper_addr2]
        
        print(f"   Finger 1 position: {gripper_pos1:.4f}")
        print(f"   Finger 2 position: {gripper_pos2:.4f}")
        print(f"   Normalized (pos/0.04): {gripper_pos1/0.04:.4f}")
    
    # Test gripper control
    print("\n🧪 Testing Gripper Control:")
    
    # Test closing
    action = np.zeros(8)
    action[7] = 1.0  # Close command
    obs = env.step(action, steps=50)
    print(f"   After close command: gripper_pos = {obs['gripper_pos']:.4f}")
    
    # Test opening
    action[7] = -1.0  # Open command
    obs = env.step(action, steps=50)
    print(f"   After open command: gripper_pos = {obs['gripper_pos']:.4f}")
    
    # Check tendon control
    print("\n🔗 Tendon Information:")
    for i in range(env.model.ntendon):
        tendon_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_TENDON, i)
        print(f"   Tendon {i}: {tendon_name}")


if __name__ == "__main__":
    diagnose_environment()