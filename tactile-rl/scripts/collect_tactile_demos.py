"""
Collect demonstration data for tactile-enhanced grasping.
Uses a scripted policy with tactile feedback to generate successful grasps.
"""

import numpy as np
import h5py
import os
import sys
import time
import argparse
import mujoco
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.tactile_grasping_env import TactileGraspingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TactileGraspingExpert:
    """
    Expert policy for collecting demonstrations.
    Uses tactile feedback to create robust grasps.
    """
    
    def __init__(
        self,
        tactile_threshold: float = 0.5,
        approach_speed: float = 0.3,
        grasp_force_target: float = 2.0,
        use_tactile_feedback: bool = True
    ):
        self.tactile_threshold = tactile_threshold
        self.approach_speed = approach_speed
        self.grasp_force_target = grasp_force_target
        self.use_tactile_feedback = use_tactile_feedback
        
        # State machine states
        self.states = ["approach", "descend", "grasp", "lift", "done"]
        self.current_state = "approach"
        
    def reset(self):
        """Reset the expert to initial state."""
        self.current_state = "approach"
        self.grasp_attempts = 0
        self.contact_detected = False
        
    def get_action(self, obs: Dict[str, np.ndarray], env: TactileGraspingEnv) -> np.ndarray:
        """
        Get expert action based on current observation.
        
        Returns action vector: [joint_velocities(7), gripper_action(1)]
        """
        action = np.zeros(8)
        
        # Extract relevant observations
        joint_pos = obs['proprio'][:7]
        gripper_width = obs['proprio'][14]
        cube_pos = obs['object_state'][:3]
        tactile_reading = obs['tactile']
        
        # Get end-effector position (approximate)
        ee_pos = self._get_ee_position(env)
        
        # State machine logic
        if self.current_state == "approach":
            # Move above the cube
            target_pos = cube_pos.copy()
            target_pos[2] = cube_pos[2] + 0.1  # 10cm above cube
            
            # Simple position control (would be better with IK)
            error = target_pos - ee_pos
            
            if np.linalg.norm(error[:2]) < 0.02:  # Close enough in x,y
                self.current_state = "descend"
            else:
                # Move towards target (simplified)
                action[0] = np.clip(error[0] * 2, -1, 1)  # x movement
                action[1] = np.clip(error[1] * 2, -1, 1)  # y movement
                
        elif self.current_state == "descend":
            # Descend towards cube
            if self.use_tactile_feedback and np.sum(tactile_reading) > self.tactile_threshold:
                # Contact detected via tactile
                self.contact_detected = True
                self.current_state = "grasp"
            elif ee_pos[2] < cube_pos[2] + 0.02:  # Close to cube
                self.current_state = "grasp"
            else:
                # Move down slowly
                action[2] = -self.approach_speed
                
        elif self.current_state == "grasp":
            # Close gripper with tactile feedback
            if self.use_tactile_feedback:
                # Adjust grip based on tactile symmetry
                left_force = np.sum(tactile_reading[:12])
                right_force = np.sum(tactile_reading[12:])
                
                if left_force + right_force > self.grasp_force_target:
                    # Sufficient grip achieved
                    self.current_state = "lift"
                else:
                    # Close gripper more
                    action[7] = 1.0  # Close gripper
            else:
                # Simple time-based closing
                action[7] = 1.0
                if gripper_width < 0.02:  # Gripper mostly closed
                    self.current_state = "lift"
                    
        elif self.current_state == "lift":
            # Lift the cube
            if ee_pos[2] > 0.15:  # Lifted high enough
                self.current_state = "done"
            else:
                action[2] = 0.5  # Lift up
                action[7] = 1.0  # Keep gripper closed
                
        # Add small noise for diversity
        if self.current_state not in ["done"]:
            action[:7] += np.random.normal(0, 0.05, 7)
            
        return np.clip(action, -1, 1)
        
    def _get_ee_position(self, env: TactileGraspingEnv) -> np.ndarray:
        """Get approximate end-effector position."""
        if env.ee_site_id is not None:
            return env.data.site_xpos[env.ee_site_id].copy()
        else:
            # Use the hand body position as approximation
            hand_body_id = None
            try:
                hand_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
                return env.data.xpos[hand_body_id].copy()
            except:
                # Final fallback
                return np.array([0.5, 0, 0.4])


def collect_demonstrations(
    num_episodes: int,
    output_path: str,
    use_tactile: bool = True,
    render: bool = False,
    save_video: bool = False
):
    """
    Collect demonstration episodes using the expert policy.
    """
    # Create environment
    env = TactileGraspingEnv(use_tactile=use_tactile)
    expert = TactileGraspingExpert(use_tactile_feedback=use_tactile)
    
    # Storage for demonstrations
    demonstrations = []
    successful_episodes = 0
    
    # Progress bar
    pbar = tqdm(total=num_episodes, desc="Collecting demonstrations")
    
    while successful_episodes < num_episodes:
        # Reset environment and expert
        obs = env.reset()
        expert.reset()
        
        # Episode storage
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'tactile_readings': [],
            'success': False
        }
        
        # Run episode
        done = False
        truncated = False
        total_reward = 0
        
        while not (done or truncated):
            # Get expert action
            action = expert.get_action(obs, env)
            
            # Store data
            episode_data['observations'].append(obs.copy())
            episode_data['actions'].append(action.copy())
            episode_data['tactile_readings'].append(obs['tactile'].copy())
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_data['rewards'].append(reward)
            total_reward += reward
            
            # Check if episode is successful
            if info.get('success', False):
                episode_data['success'] = True
                
        # Only save successful episodes
        if episode_data['success'] or (total_reward > 5.0 and expert.current_state == "done"):
            demonstrations.append(episode_data)
            successful_episodes += 1
            pbar.update(1)
            logger.info(f"Episode {successful_episodes}/{num_episodes} - Success! Total reward: {total_reward:.2f}")
        else:
            logger.debug(f"Episode failed - State: {expert.current_state}, Reward: {total_reward:.2f}")
            
    pbar.close()
    
    # Save demonstrations to HDF5
    save_demonstrations(demonstrations, output_path, use_tactile)
    
    logger.info(f"Collected {num_episodes} successful demonstrations")
    logger.info(f"Saved to: {output_path}")
    
    return demonstrations


def save_demonstrations(demonstrations: List[Dict], output_path: str, use_tactile: bool):
    """Save collected demonstrations to HDF5 file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create groups
        f.attrs['num_episodes'] = len(demonstrations)
        f.attrs['use_tactile'] = use_tactile
        f.attrs['env_name'] = 'TactileGraspingEnv'
        f.attrs['date_collected'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save each episode
        for i, episode in enumerate(demonstrations):
            episode_group = f.create_group(f'episode_{i}')
            
            # Convert lists to arrays
            observations = {key: np.array([obs[key] for obs in episode['observations']]) 
                          for key in episode['observations'][0].keys()}
            actions = np.array(episode['actions'])
            rewards = np.array(episode['rewards'])
            tactile = np.array(episode['tactile_readings'])
            
            # Save data
            obs_group = episode_group.create_group('observations')
            for key, value in observations.items():
                obs_group.create_dataset(key, data=value, compression='gzip')
                
            episode_group.create_dataset('actions', data=actions, compression='gzip')
            episode_group.create_dataset('rewards', data=rewards, compression='gzip')
            episode_group.create_dataset('tactile_readings', data=tactile, compression='gzip')
            episode_group.attrs['success'] = episode['success']
            episode_group.attrs['episode_length'] = len(actions)
            episode_group.attrs['total_reward'] = float(np.sum(rewards))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect tactile grasping demonstrations")
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of demonstrations to collect')
    parser.add_argument('--output_dir', type=str, default='../datasets/tactile_grasping', help='Output directory')
    parser.add_argument('--use_tactile', action='store_true', default=True, help='Use tactile feedback')
    parser.add_argument('--no_tactile', dest='use_tactile', action='store_false', help='Disable tactile feedback')
    parser.add_argument('--render', action='store_true', help='Render episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Determine output filename
    tactile_suffix = 'tactile' if args.use_tactile else 'no_tactile'
    output_path = os.path.join(args.output_dir, f'demonstrations_{tactile_suffix}.hdf5')
    
    # Collect demonstrations
    collect_demonstrations(
        num_episodes=args.num_episodes,
        output_path=output_path,
        use_tactile=args.use_tactile,
        render=args.render
    )