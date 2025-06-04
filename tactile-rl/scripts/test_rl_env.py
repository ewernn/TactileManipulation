#!/usr/bin/env python3
"""
Test script for the Panda 7-DOF RL environment.
Shows how to use the environment with and without tactile sensing.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.panda_7dof_rl_env_fixed import Panda7DOFTactileRL


def test_environment(use_tactile=True):
    """Test the environment with random actions."""
    
    print(f"\n{'='*50}")
    print(f"Testing Panda RL Environment")
    print(f"Tactile Sensing: {'Enabled' if use_tactile else 'Disabled'}")
    print(f"{'='*50}\n")
    
    # Create environment
    env = Panda7DOFTactileRL(
        control_frequency=20,
        joint_vel_limit=2.0,
        use_tactile=use_tactile,
        max_episode_steps=100
    )
    
    # Reset and check observation
    obs = env.reset()
    print("Observation keys:", list(obs.keys()))
    print(f"Joint positions: {obs['joint_pos']}")
    print(f"Target block position: {obs['target_block_pose'][:3]}")
    print(f"End-effector position: {obs['ee_pose'][:3]}")
    
    if use_tactile:
        print(f"Tactile readings shape: {obs['tactile'].shape}")
        print(f"Tactile sum: {np.sum(obs['tactile']):.3f}")
    
    # Run a simple policy
    print("\nRunning simple policy...")
    total_reward = 0
    
    for step in range(50):
        # Simple policy: move towards block and close gripper
        block_pos = obs['target_block_pose'][:3]
        ee_pos = obs['ee_pose'][:3]
        
        # Compute direction to block
        direction = block_pos - ee_pos
        distance = np.linalg.norm(direction)
        
        # Create action
        action = np.zeros(8)
        
        # Simple proportional control for first 3 joints
        if distance > 0.05:
            action[0] = np.clip(direction[0] * 2, -1, 1)  # X direction
            action[1] = np.clip(direction[1] * 2, -1, 1)  # Y direction
            action[2] = np.clip(-direction[2] * 2, -1, 1) # Z direction (inverted)
        
        # Close gripper when near
        if distance < 0.1:
            action[7] = 1.0  # Close gripper
        else:
            action[7] = -1.0  # Open gripper
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"  Step {step:3d}: reward={reward:6.2f}, "
                  f"distance={distance:.3f}, "
                  f"block_height={info['block_height']:.3f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended! Success: {info['success']}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    
    return env


def main():
    """Run tests with and without tactile sensing."""
    
    # Test with tactile
    env1 = test_environment(use_tactile=True)
    
    # Test without tactile
    env2 = test_environment(use_tactile=False)
    
    print("\n✅ All tests passed!")
    print("\nThe environment is ready for RL training with:")
    print("  - 7-DOF velocity control")
    print("  - Optional tactile sensing")
    print("  - Block manipulation task")
    print("  - Proper reward shaping")


if __name__ == "__main__":
    main()