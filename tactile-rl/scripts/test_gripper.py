"""Test gripper control to understand the tendon mechanism."""

import numpy as np
import mujoco
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.panda_demo_env import PandaDemoEnv


def test_gripper():
    """Test gripper control mechanism."""
    
    print("🔧 Testing Gripper Control")
    print("=" * 50)
    
    # Create environment
    env = PandaDemoEnv()
    obs = env.reset(randomize=False)
    
    # Function to get gripper state
    def get_gripper_state():
        finger1_addr = env.model.jnt_qposadr[env.gripper_joint1_id]
        finger2_addr = env.model.jnt_qposadr[env.gripper_joint2_id]
        finger1_pos = env.data.qpos[finger1_addr]
        finger2_pos = env.data.qpos[finger2_addr]
        return finger1_pos, finger2_pos
    
    print("\n📊 Initial gripper state:")
    f1, f2 = get_gripper_state()
    print(f"   Finger 1: {f1:.4f}m")
    print(f"   Finger 2: {f2:.4f}m")
    print(f"   Actuator 8 ctrl: {env.data.ctrl[7]:.1f}")
    
    # Test different control values
    test_values = [
        (255, "Fully open"),
        (200, "Mostly open"),
        (127.5, "Middle"),
        (50, "Mostly closed"),
        (0, "Fully closed")
    ]
    
    print("\n🧪 Testing control values:")
    for ctrl_value, desc in test_values:
        # Reset to open position first
        action = np.zeros(8)
        action[7] = -1.0  # Open command
        for _ in range(50):
            env.step(action)
        
        # Apply test control
        env.data.ctrl[7] = ctrl_value
        
        # Step simulation
        for _ in range(100):
            mujoco.mj_step(env.model, env.data)
        
        # Check result
        f1, f2 = get_gripper_state()
        width = (f1 + f2)
        print(f"\n   Control: {ctrl_value:5.1f} ({desc})")
        print(f"   Finger 1: {f1:.4f}m")
        print(f"   Finger 2: {f2:.4f}m") 
        print(f"   Total width: {width:.4f}m")
    
    # Test action mapping
    print("\n\n🎮 Testing action mapping:")
    
    # Reset
    obs = env.reset(randomize=False)
    
    print("\n   Action = -1.0 (should open):")
    action = np.zeros(8)
    action[7] = -1.0
    for _ in range(50):
        obs = env.step(action)
    f1, f2 = get_gripper_state()
    print(f"   Fingers: {f1:.4f}, {f2:.4f} | Width: {f1+f2:.4f}m")
    print(f"   Ctrl value: {env.data.ctrl[7]:.1f}")
    
    print("\n   Action = 1.0 (should close):")
    action[7] = 1.0
    for _ in range(50):
        obs = env.step(action)
    f1, f2 = get_gripper_state()
    print(f"   Fingers: {f1:.4f}, {f2:.4f} | Width: {f1+f2:.4f}m")
    print(f"   Ctrl value: {env.data.ctrl[7]:.1f}")
    
    # Check gripper observation
    print(f"\n   Gripper observation: {obs['gripper_pos']:.4f}")
    print(f"   (This is finger1_pos / 0.04 = {f1} / 0.04)")


if __name__ == "__main__":
    test_gripper()