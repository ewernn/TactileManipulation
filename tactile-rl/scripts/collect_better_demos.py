#!/usr/bin/env python3
"""
Collect better expert demonstrations with proper gripper control and visual feedback.
"""

import numpy as np
import sys
import os
import mujoco
import h5py
import imageio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments'))
from environments.panda_demo_env import PandaDemoEnv


class ImprovedExpertPolicy:
    """
    Improved expert policy that properly reaches and grasps blocks.
    """
    
    def __init__(self):
        self.phase = "init"
        self.step_count = 0
        self.grasp_started = False
        self.lift_started = False
        
    def reset(self):
        self.phase = "init"
        self.step_count = 0
        self.grasp_started = False
        self.lift_started = False
    
    def get_action(self, env, observation):
        """
        Generate expert action based on environment state.
        """
        # Get end effector position
        ee_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        if ee_site_id >= 0:
            ee_pos = env.data.site_xpos[ee_site_id].copy()
        else:
            hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            ee_pos = env.data.xpos[hand_id].copy()
        
        # Get target block position
        target_pos = observation['target_block_pos']
        
        # Calculate error
        error = target_pos - ee_pos
        error[2] += 0.05  # Aim slightly above block initially
        
        action = np.zeros(8)
        
        # Determine phase based on position
        distance = np.linalg.norm(error[:2])  # XY distance
        height_diff = ee_pos[2] - target_pos[2]
        
        if self.phase == "init":
            print(f"\nStarting demonstration:")
            print(f"  EE position: {ee_pos}")
            print(f"  Target position: {target_pos}")
            print(f"  Distance: {distance:.3f}m")
            self.phase = "approach"
        
        if self.phase == "approach":
            # Approach the block from above
            if distance > 0.05:  # Far from target
                # Move towards target position
                action[0] = np.clip(error[0] * 2.0, -1, 1)  # X
                action[1] = np.clip(error[1] * 2.0, -1, 1)  # Y
                action[2] = np.clip(error[2] * 1.5, -1, 1)  # Z
                action[7] = -1.0  # Keep gripper open
                
                if self.step_count % 20 == 0:
                    print(f"  Approaching: distance={distance:.3f}m")
            else:
                print(f"  Reached above target at step {self.step_count}")
                self.phase = "descend"
                self.step_count = 0
        
        elif self.phase == "descend":
            # Descend to grasp height
            if height_diff > 0.02:  # Still above block
                action[0] = np.clip(error[0] * 1.0, -0.2, 0.2)  # Small X correction
                action[1] = np.clip(error[1] * 1.0, -0.2, 0.2)  # Small Y correction
                action[2] = -0.3  # Descend
                action[7] = -1.0  # Keep gripper open
                
                if self.step_count % 10 == 0:
                    print(f"  Descending: height_diff={height_diff:.3f}m")
            else:
                print(f"  Reached grasp height at step {self.step_count}")
                self.phase = "grasp"
                self.step_count = 0
                self.grasp_started = True
        
        elif self.phase == "grasp":
            # Close gripper
            if self.step_count < 20:
                action[:7] = 0  # No joint movement
                action[7] = 1.0  # Close gripper
                
                if self.step_count == 0:
                    print(f"  Starting grasp")
                elif self.step_count == 19:
                    print(f"  Grasp complete")
            else:
                self.phase = "lift"
                self.step_count = 0
                self.lift_started = True
        
        elif self.phase == "lift":
            # Lift the block
            if self.step_count < 40:
                action[:7] = 0
                action[2] = 0.5  # Lift up
                action[7] = 1.0  # Keep gripper closed
                
                if self.step_count % 10 == 0:
                    current_height = env.data.xpos[env.target_block_id][2]
                    print(f"  Lifting: block_height={current_height:.3f}m")
            else:
                self.phase = "done"
                print(f"  Lift complete")
        
        else:  # done
            action[:7] = 0
            action[7] = 1.0  # Keep gripper closed
        
        self.step_count += 1
        return action


def collect_demonstration(env, policy, demo_idx=0, max_steps=150, save_video=True):
    """Collect a single demonstration."""
    
    obs = env.reset()
    policy.reset()
    
    # Storage
    observations = []
    actions = []
    rewards = []
    frames = []
    
    # Get initial block height
    initial_block_z = obs['target_block_pos'][2]
    
    print(f"\n{'='*60}")
    print(f"Collecting demonstration {demo_idx}")
    print(f"{'='*60}")
    
    for step in range(max_steps):
        # Get expert action
        action = policy.get_action(env, obs)
        
        # Store data
        observations.append(obs)
        actions.append(action)
        
        # Step environment
        next_obs = env.step(action)
        
        # Calculate reward (simplified)
        block_height = env.data.xpos[env.target_block_id][2]
        lift_height = block_height - initial_block_z
        
        if policy.lift_started:
            reward = 100.0 * lift_height  # Reward for lifting
        elif policy.grasp_started:
            reward = 10.0  # Small reward for grasping
        else:
            # Reward for getting closer to target
            ee_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
            if ee_site_id >= 0:
                ee_pos = env.data.site_xpos[ee_site_id]
            else:
                ee_pos = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")]
            
            distance = np.linalg.norm(ee_pos[:2] - obs['target_block_pos'][:2])
            reward = 1.0 / (1.0 + distance)
        
        rewards.append(reward)
        
        # Render frame
        if save_video and step % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        obs = next_obs
        
        # Check success
        if lift_height > 0.01:  # 10mm lift
            print(f"\n✅ Success! Block lifted {lift_height*1000:.1f}mm at step {step}")
            break
    
    # Final statistics
    total_reward = sum(rewards)
    final_lift = env.data.xpos[env.target_block_id][2] - initial_block_z
    max_tactile = max([np.sum(obs['tactile']) for obs in observations])
    
    print(f"\n📊 Demo {demo_idx} Summary:")
    print(f"  Steps: {len(actions)}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final lift: {final_lift*1000:.1f}mm")
    print(f"  Max tactile: {max_tactile:.1f}")
    
    return {
        'observations': observations,
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'frames': frames
    }


def save_demonstrations(demos, output_path):
    """Save demonstrations to HDF5 file."""
    
    with h5py.File(output_path, 'w') as f:
        for demo_idx, demo in enumerate(demos):
            demo_group = f.create_group(f'demo_{demo_idx}')
            
            # Save actions and rewards
            demo_group.create_dataset('actions', data=demo['actions'])
            demo_group.create_dataset('rewards', data=demo['rewards'])
            
            # Save observations
            obs_group = demo_group.create_group('observations')
            
            # Extract observation components
            n_steps = len(demo['observations'])
            obs_keys = demo['observations'][0].keys()
            
            for key in obs_keys:
                # Get shape from first observation
                first_val = demo['observations'][0][key]
                if isinstance(first_val, np.ndarray):
                    shape = (n_steps,) + first_val.shape
                else:
                    shape = (n_steps,)
                
                # Create dataset
                data = np.zeros(shape)
                for i, obs in enumerate(demo['observations']):
                    data[i] = obs[key]
                
                obs_group.create_dataset(key, data=data)
            
            # Save tactile readings separately for compatibility
            tactile_data = np.array([obs['tactile'] for obs in demo['observations']])
            demo_group.create_dataset('tactile_readings', data=tactile_data)
            
            print(f"  Saved demo_{demo_idx}: {len(demo['actions'])} steps")
    
    print(f"\n✅ Saved {len(demos)} demonstrations to {output_path}")


def create_demo_video(demos, output_path):
    """Create a compilation video of all demonstrations."""
    
    all_frames = []
    
    for demo_idx, demo in enumerate(demos):
        frames = demo['frames']
        if frames:
            # Add title frame
            title_frame = frames[0].copy()
            # (Would need PIL to add text, skip for now)
            all_frames.extend(frames)
    
    if all_frames:
        print(f"\nSaving compilation video with {len(all_frames)} frames...")
        imageio.mimsave(output_path, all_frames, fps=30, quality=8, codec='libx264')
        print(f"Video saved to: {output_path}")


def main():
    """Collect multiple expert demonstrations."""
    
    # Initialize environment
    env = PandaDemoEnv(camera_name="demo_cam")
    policy = ImprovedExpertPolicy()
    
    # Collect demonstrations
    n_demos = 5
    demos = []
    
    for i in range(n_demos):
        demo = collect_demonstration(env, policy, demo_idx=i, save_video=(i==0))
        demos.append(demo)
    
    # Save demonstrations
    output_path = "../../datasets/better_expert_demos.hdf5"
    save_demonstrations(demos, output_path)
    
    # Create video from first demo
    if demos[0]['frames']:
        video_path = "../../datasets/better_expert_demo.mp4"
        create_demo_video([demos[0]], video_path)


if __name__ == "__main__":
    main()