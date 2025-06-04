#!/usr/bin/env python3
"""
Create a working demonstration with blocks placed where the robot can actually reach them.
"""

import numpy as np
import sys
import os
import mujoco
import imageio
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments'))
from environments.panda_demo_env import PandaDemoEnv


def create_reachable_demo():
    """Create a demo with blocks placed behind the robot where it can reach."""
    
    env = PandaDemoEnv()
    obs = env.reset()
    
    # Manually move blocks to reachable positions (behind robot)
    # Since gripper faces -X, place blocks at negative X
    target_block_id = env.target_block_id
    block2_id = env.block2_id
    block3_id = env.block3_id
    
    # Set block positions behind robot
    if target_block_id >= 0:
        block_qpos_addr = env.model.jnt_qposadr[env.model.body_jntadr[target_block_id]]
        env.data.qpos[block_qpos_addr:block_qpos_addr+3] = [-0.45, 0.0, 0.44]  # Behind robot
        env.data.qpos[block_qpos_addr+3:block_qpos_addr+7] = [1, 0, 0, 0]  # Quaternion
    
    if block2_id >= 0:
        block_qpos_addr = env.model.jnt_qposadr[env.model.body_jntadr[block2_id]]
        env.data.qpos[block_qpos_addr:block_qpos_addr+3] = [-0.35, 0.1, 0.44]
        env.data.qpos[block_qpos_addr+3:block_qpos_addr+7] = [1, 0, 0, 0]
    
    if block3_id >= 0:
        block_qpos_addr = env.model.jnt_qposadr[env.model.body_jntadr[block3_id]]
        env.data.qpos[block_qpos_addr:block_qpos_addr+3] = [-0.35, -0.1, 0.44]
        env.data.qpos[block_qpos_addr+3:block_qpos_addr+7] = [1, 0, 0, 0]
    
    # Forward simulation to update
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_observation()
    
    print("\n" + "="*60)
    print("CREATING REACHABLE DEMONSTRATION")
    print("="*60)
    
    # Get positions
    hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    hand_pos = env.data.xpos[hand_id].copy()
    target_pos = env.data.xpos[target_block_id].copy()
    
    print(f"\nSetup complete:")
    print(f"  Hand position: {hand_pos}")
    print(f"  Target block: {target_pos}")
    print(f"  Distance: {np.linalg.norm(hand_pos - target_pos):.3f}m")
    
    frames = []
    observations = []
    actions = []
    rewards = []
    
    # Initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    initial_block_z = target_pos[2]
    
    # Phase 1: Approach (40 steps)
    print("\n📍 Phase 1: Approach")
    for step in range(40):
        hand_pos = env.data.xpos[hand_id].copy()
        target_pos = env.data.xpos[target_block_id].copy()
        
        error = target_pos - hand_pos
        error[2] += 0.05  # Aim above
        
        action = np.zeros(8)
        action[0] = np.clip(error[0] * 2.0, -1, 1)
        action[1] = np.clip(error[1] * 2.0, -1, 1)
        action[2] = np.clip(error[2] * 1.5, -1, 1)
        action[7] = -1.0  # Open
        
        observations.append(obs)
        actions.append(action)
        
        obs = env.step(action)
        
        # Simple reward
        distance = np.linalg.norm(hand_pos[:2] - target_pos[:2])
        rewards.append(1.0 / (1.0 + distance))
        
        if step % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if step % 10 == 0:
            print(f"  Step {step}: distance = {distance:.3f}m")
    
    # Phase 2: Descend (30 steps)
    print("\n📍 Phase 2: Descend")
    for step in range(30):
        action = np.zeros(8)
        action[2] = -0.3
        action[7] = -1.0
        
        observations.append(obs)
        actions.append(action)
        obs = env.step(action)
        rewards.append(5.0)  # Small reward for descending
        
        if step % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    
    # Phase 3: Grasp (20 steps)
    print("\n📍 Phase 3: Grasp")
    for step in range(20):
        action = np.zeros(8)
        action[7] = 1.0  # Close
        
        observations.append(obs)
        actions.append(action)
        obs = env.step(action)
        
        tactile_sum = np.sum(obs['tactile'])
        rewards.append(10.0 + tactile_sum * 0.1)
        
        if step % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if step == 19:
            print(f"  Tactile sum: {tactile_sum:.1f}")
    
    # Phase 4: Lift (40 steps)
    print("\n📍 Phase 4: Lift")
    for step in range(40):
        action = np.zeros(8)
        action[2] = 0.5  # Lift
        action[7] = 1.0  # Keep closed
        
        observations.append(obs)
        actions.append(action)
        obs = env.step(action)
        
        current_z = env.data.xpos[target_block_id][2]
        lift_height = current_z - initial_block_z
        rewards.append(100.0 * lift_height)
        
        if step % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if step % 10 == 0:
            print(f"  Step {step}: lift = {lift_height*1000:.1f}mm")
    
    # Final frames
    for _ in range(10):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    
    # Results
    final_z = env.data.xpos[target_block_id][2]
    total_lift = final_z - initial_block_z
    
    print(f"\n📊 Results:")
    print(f"  Total steps: {len(actions)}")
    print(f"  Total reward: {sum(rewards):.1f}")
    print(f"  Lift height: {total_lift*1000:.1f}mm")
    print(f"  Success: {'YES' if total_lift > 0.005 else 'NO'}")
    
    # Save video
    if frames:
        output_path = "../../datasets/reachable_demonstration.mp4"
        print(f"\n💾 Saving video ({len(frames)} frames)...")
        imageio.mimsave(output_path, frames, fps=30, quality=8, codec='libx264')
        print(f"Video saved to: {output_path}")
        
        gif_path = output_path.replace('.mp4', '.gif')
        gif_frames = frames[::3]
        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
        print(f"GIF saved to: {gif_path}")
    
    # Save data
    save_demonstration(observations, actions, rewards, "../../datasets/reachable_demo.hdf5")
    
    return total_lift > 0.005


def save_demonstration(observations, actions, rewards, output_path):
    """Save demonstration data."""
    with h5py.File(output_path, 'w') as f:
        demo = f.create_group('demo_0')
        
        # Actions and rewards
        demo.create_dataset('actions', data=np.array(actions))
        demo.create_dataset('rewards', data=np.array(rewards))
        
        # Observations
        obs_group = demo.create_group('observations')
        n_steps = len(observations)
        
        # Extract each observation component
        for key in observations[0].keys():
            first_val = observations[0][key]
            if isinstance(first_val, np.ndarray):
                shape = (n_steps,) + first_val.shape
            else:
                shape = (n_steps,)
            
            data = np.zeros(shape)
            for i, obs in enumerate(observations):
                data[i] = obs[key]
            
            obs_group.create_dataset(key, data=data)
        
        # Tactile separately for compatibility
        tactile_data = np.array([obs['tactile'] for obs in observations])
        demo.create_dataset('tactile_readings', data=tactile_data)
    
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    success = create_reachable_demo()
    
    if success:
        print("\n✅ Successfully created a working demonstration!")
        print("The robot can reach blocks when they're placed behind it.")
    else:
        print("\n❌ Demonstration failed")