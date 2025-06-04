"""
Create expert demonstrations for imitation learning and fast videos.
"""

import numpy as np
import sys
import os
import imageio
import h5py
from tqdm import tqdm

sys.path.append('../environments')
from panda_demo_env import PandaDemoEnv

class ExpertPolicy:
    """
    Expert policy for stacking blocks - picks up red block and places on blue block.
    """
    
    def __init__(self):
        self.phase = "approach"
        self.step_count = 0
        self.target_reached = False
        self.blue_block_pos = None
        
    def reset(self):
        """Reset policy state."""
        self.phase = "approach"
        self.step_count = 0
        self.target_reached = False
    
    def get_action(self, observation):
        """
        Generate expert action for block stacking demonstration.
        """
        
        # Get current state
        joint_pos = observation['joint_pos']
        target_pos = observation['target_block_pos']  # Red block
        gripper_pos = observation['gripper_pos']
        tactile_sum = np.sum(observation['tactile'])
        
        # Store blue block position if available
        if 'block2_pos' in observation and self.blue_block_pos is None:
            self.blue_block_pos = observation['block2_pos'].copy()
        
        action = np.zeros(8)  # 7 joints + gripper
        
        if self.phase == "approach":
            # Approach red block position
            if self.step_count < 40:
                # Move towards red block with controlled motion
                action = np.array([0.0, 0.2, 0.15, -0.25, 0.1, -0.05, 0.0, -1.0])
            else:
                self.phase = "pre_grasp"
                self.step_count = 0
        
        elif self.phase == "pre_grasp":
            # Fine positioning above red block
            if self.step_count < 30:
                # Small adjustments to align with block
                action = np.array([0.0, 0.1, 0.1, -0.1, 0.05, -0.05, 0.0, -1.0])
            else:
                self.phase = "descend"
                self.step_count = 0
        
        elif self.phase == "descend":
            # Descend to grasp red block
            if self.step_count < 25:
                # Controlled descent
                action = np.array([0.0, 0.15, 0.2, -0.15, -0.1, -0.05, 0.0, -1.0])
            else:
                self.phase = "grasp"
                self.step_count = 0
        
        elif self.phase == "grasp":
            # Close gripper on red block
            if self.step_count < 20:
                # Close gripper firmly
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            else:
                self.phase = "lift"
                self.step_count = 0
        
        elif self.phase == "lift":
            # Gentle lift to clear table
            if self.step_count < 30:
                # Lift straight up
                action = np.array([0.0, -0.15, -0.15, 0.15, 0.05, 0.0, 0.0, 1.0])
            else:
                self.phase = "move_to_blue"
                self.step_count = 0
        
        elif self.phase == "move_to_blue":
            # Move towards blue block position
            if self.step_count < 40:
                # Lateral movement to blue block
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 1.0])
            else:
                self.phase = "position_above_blue"
                self.step_count = 0
        
        elif self.phase == "position_above_blue":
            # Fine positioning above blue block
            if self.step_count < 20:
                # Small adjustments
                action = np.array([0.0, 0.05, 0.05, -0.05, 0.0, 0.05, 0.0, 1.0])
            else:
                self.phase = "place"
                self.step_count = 0
        
        elif self.phase == "place":
            # Lower red block onto blue block
            if self.step_count < 30:
                # Controlled descent
                action = np.array([0.0, 0.1, 0.1, -0.1, -0.05, 0.0, 0.0, 1.0])
            else:
                self.phase = "release"
                self.step_count = 0
        
        elif self.phase == "release":
            # Release red block
            if self.step_count < 15:
                # Open gripper
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            else:
                self.phase = "retreat"
                self.step_count = 0
        
        elif self.phase == "retreat":
            # Move away from stacked blocks
            if self.step_count < 25:
                # Retreat upward and backward
                action = np.array([0.0, -0.1, -0.1, 0.1, 0.0, -0.1, 0.0, -1.0])
            else:
                self.phase = "done"
                self.step_count = 0
        
        else:  # done
            # Hold final position
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        self.step_count += 1
        return action

def collect_expert_demonstration(env, expert_policy, max_steps=300, save_video=False):
    """Collect a single expert demonstration."""
    
    # Reset environment and policy
    obs = env.reset(randomize=True)
    expert_policy.reset()
    
    # Storage for demonstration data
    demo_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'tactile_readings': [],
        'success': False
    }
    
    frames = [] if save_video else None
    
    for step in range(max_steps):
        # Get expert action
        action = expert_policy.get_action(obs)
        
        # Store current state
        demo_data['observations'].append(obs.copy())
        demo_data['actions'].append(action.copy())
        demo_data['tactile_readings'].append(obs['tactile'].copy())
        
        # Execute action
        next_obs = env.step(action, steps=3)  # 3 physics steps per action
        
        # Simple reward: height of target block
        initial_height = 0.44
        current_height = next_obs['target_block_pos'][2]
        reward = max(0, current_height - initial_height) * 10  # Positive for lifting
        
        # Add tactile bonus
        tactile_bonus = np.sum(next_obs['tactile']) * 0.1
        reward += tactile_bonus
        
        demo_data['rewards'].append(reward)
        
        # Check success - red block is on top of blue block
        if 'block2_pos' in next_obs:
            red_pos = next_obs['target_block_pos']
            blue_pos = next_obs['block2_pos']
            # Success if red block is above blue block and close horizontally
            if (red_pos[2] > blue_pos[2] + 0.03 and 
                np.linalg.norm(red_pos[:2] - blue_pos[:2]) < 0.05):
                demo_data['success'] = True
        
        # Save frame if creating video
        if save_video:
            frame = env.render()
            frames.append(frame)
        
        obs = next_obs
        
        # Early termination if done
        if expert_policy.phase == "done" and expert_policy.step_count > 20:
            break
    
    # Convert lists to numpy arrays
    for key in ['observations', 'actions', 'rewards', 'tactile_readings']:
        if key == 'observations':
            # Handle dict observations
            obs_arrays = {}
            for obs_key in demo_data['observations'][0].keys():
                obs_arrays[obs_key] = np.array([obs[obs_key] for obs in demo_data['observations']])
            demo_data['observations'] = obs_arrays
        else:
            demo_data[key] = np.array(demo_data[key])
    
    return demo_data, frames

def create_expert_dataset(n_demos=50, save_path="../../datasets/expert_demonstrations.hdf5", success_threshold=0.01):
    """Create a dataset of expert demonstrations for imitation learning."""
    
    print(f"🤖 Creating {n_demos} expert demonstrations...")
    
    env = PandaDemoEnv()
    expert = ExpertPolicy()
    
    all_demos = []
    successful_demos = 0
    
    for i in tqdm(range(n_demos), desc="Collecting demos"):
        demo, _ = collect_expert_demonstration(env, expert, max_steps=250)
        
        all_demos.append(demo)
        
        if demo['success']:
            successful_demos += 1
        
        if i % 10 == 0:
            success_rate = successful_demos / (i + 1) * 100
            print(f"  Demo {i+1}: Success rate so far: {success_rate:.1f}%")
    
    # Save to HDF5 format
    print(f"💾 Saving {len(all_demos)} demonstrations to {save_path}...")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        # Metadata
        f.attrs['n_demos'] = len(all_demos)
        f.attrs['successful_demos'] = successful_demos
        f.attrs['success_rate'] = successful_demos / len(all_demos)
        
        # Create groups for each demonstration
        for i, demo in enumerate(all_demos):
            demo_group = f.create_group(f'demo_{i}')
            
            # Save observations (handle dict structure)
            obs_group = demo_group.create_group('observations')
            for key, value in demo['observations'].items():
                obs_group.create_dataset(key, data=value)
            
            # Save other data
            demo_group.create_dataset('actions', data=demo['actions'])
            demo_group.create_dataset('rewards', data=demo['rewards'])
            demo_group.create_dataset('tactile_readings', data=demo['tactile_readings'])
            demo_group.attrs['success'] = demo['success']
            demo_group.attrs['length'] = len(demo['actions'])
    
    print(f"✅ Expert dataset created!")
    print(f"   📊 Total demonstrations: {len(all_demos)}")
    print(f"   🎯 Successful: {successful_demos} ({successful_demos/len(all_demos)*100:.1f}%)")
    print(f"   💾 Saved to: {save_path}")
    
    return save_path

def create_fast_demo_video():
    """Create a fast demonstration video showing expert behavior."""
    
    print("🎬 Creating fast expert demonstration video...")
    
    env = PandaDemoEnv()
    expert = ExpertPolicy()
    
    all_frames = []
    
    # Create 3 different demonstration episodes
    for episode in range(3):
        print(f"  📹 Recording episode {episode + 1}/3")
        
        demo, frames = collect_expert_demonstration(
            env, expert, max_steps=250, save_video=True
        )
        
        # Add frames with some padding between episodes
        all_frames.extend(frames)
        
        if episode < 2:  # Add pause between episodes
            # Hold last frame for a moment
            for _ in range(10):
                all_frames.append(frames[-1])
    
    # Save fast video
    output_path = "../../datasets/expert_demo_fast.mp4"
    
    print(f"💾 Saving fast video with {len(all_frames)} frames...")
    
    imageio.mimsave(
        output_path,
        all_frames,
        fps=30,  # Fast 30 FPS
        quality=8,
        macro_block_size=1
    )
    
    print(f"✅ Fast demo video created!")
    print(f"   📹 Location: {output_path}")
    print(f"   ⏱️  Duration: {len(all_frames)/30:.1f} seconds")
    print(f"   🚀 Speed: 30 FPS (much faster!)")
    
    return output_path

def load_expert_dataset(dataset_path):
    """Example of how to load the expert dataset for training."""
    
    print(f"📖 Loading expert dataset from {dataset_path}...")
    
    with h5py.File(dataset_path, 'r') as f:
        n_demos = f.attrs['n_demos']
        success_rate = f.attrs['success_rate']
        
        print(f"   📊 Dataset contains {n_demos} demonstrations")
        print(f"   🎯 Success rate: {success_rate*100:.1f}%")
        
        # Example: Load first successful demonstration
        for i in range(n_demos):
            demo_group = f[f'demo_{i}']
            if demo_group.attrs['success']:
                print(f"   📋 Loading successful demo {i}...")
                
                # Load observations
                observations = {}
                obs_group = demo_group['observations']
                for key in obs_group.keys():
                    observations[key] = obs_group[key][:]
                
                # Load other data
                actions = demo_group['actions'][:]
                rewards = demo_group['rewards'][:]
                tactile = demo_group['tactile_readings'][:]
                
                print(f"      🔢 Length: {len(actions)} steps")
                print(f"      🎮 Action shape: {actions.shape}")
                print(f"      🤲 Tactile shape: {tactile.shape}")
                print(f"      🏆 Total reward: {np.sum(rewards):.2f}")
                
                break
    
    return True

if __name__ == "__main__":
    print("🤖 Expert Demonstration Generator")
    print("=" * 50)
    
    try:
        # Create fast demo video with fixed block positions
        fast_video = create_fast_demo_video()
        
        # Create expert dataset for imitation learning
        dataset_path = create_expert_dataset(n_demos=30)  # Start with 30 demos
        
        # Demonstrate loading
        load_expert_dataset(dataset_path)
        
        print("\n🎉 Expert demonstration system ready!")
        print(f"🎥 Fast video: {fast_video}")
        print(f"📊 Dataset: {dataset_path}")
        print("\n🚀 Ready for imitation learning training!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()