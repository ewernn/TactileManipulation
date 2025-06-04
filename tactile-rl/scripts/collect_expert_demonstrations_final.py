#!/usr/bin/env python3
"""
Collect expert demonstrations using the EXACT approach from the working ultimate demo.
This version replicates the successful demonstration with HDF5 recording.
"""

import numpy as np
import mujoco
import h5py
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import cv2


class ExpertDemonstrationCollector:
    def __init__(self, save_video: bool = False, control_frequency: float = 20.0):
        """Initialize collector with exact parameters from working demo."""
        # Load model
        self.xml_path = "../franka_emika_panda/panda_demo_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Control parameters
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.save_video = save_video
        
        # Get IDs (exact same as ultimate demo)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.red_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        self.blue_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        
        # Get joint IDs
        self.arm_joint_ids = []
        for i in range(7):
            joint_name = f"joint{i+1}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.arm_joint_ids.append(joint_id)
        
        # Critical measurements (exact from ultimate demo)
        self.table_height = 0.4
        self.block_half_size = 0.025
        self.target_grasp_clearance = 0.065
        self.blue_top = 0.47
        
        # Renderer if needed
        if save_video:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
    
    def get_sequence(self) -> List[Tuple]:
        """Get the EXACT sequence from the ultimate demo that works."""
        return [
            # Start
            ("home", 20, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Home - 0.2m Clearance"),
            
            # Approach
            ("approach", 25, [0, -0.25, 0, -2.15, 0, 1.65, 0], 255, "Approach"),
            
            # Position - stay higher, don't go too low
            ("position", 25, [0, -0.20, 0, -2.20, 0, 1.70, 0], 255, "Position Above"),
            
            # Fast descend to intermediate position
            ("descend_fast", 60, [0, -0.18, 0, -2.22, 0, 1.72, 0], 255, "Descend Fast"),
            
            # Slow descend to EXACTLY 65mm - final calibration for 65mm
            ("descend_65mm", 80, [0, -0.181, 0, -2.219, 0, 1.719, 0], 255, "Descend to 65mm"),
            
            # Hold and verify - give more time to settle
            ("hold_65mm", 25, None, 255, "Hold at 65mm"),  # None = hold current position
            
            # Grasp
            ("grasp", 30, None, 0, "Grasp"),  # None = hold current position
            
            # Lift
            ("lift", 25, [0, -0.25, 0, -2.15, 0, 1.65, 0], 0, "Lift"),
            
            # Lift high
            ("lift_high", 25, [0, -0.5, 0, -1.8, 0, 1.4, 0], 0, "Lift High"),
            
            # Rotate - add 10 degree rotation on joint 7
            ("rotate", 35, [-0.35, -0.5, 0, -1.8, 0, 1.4, 0.174], 0, "Rotate"),  # 0.174 rad = 10 deg
            
            # Above blue - maintain rotation, adjust for -5mm Y offset
            ("above_blue", 25, [-0.37, -0.35, 0, -2.05, 0, 1.55, 0.174], 0, "Above Blue"),
            
            # Smart place - maintain alignment
            ("smart_place", 40, [-0.37, -0.15, 0, -2.25, 0, 1.75, 0.174], 0, "Smart Place"),
            
            # Release
            ("release", 25, None, 255, "Release"),
            
            # Retreat - remove rotation
            ("retreat", 25, [-0.35, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Retreat"),
            
            # Return
            ("return", 30, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Return"),
        ]
    
    def interpolate_configs(self, config1, config2, alpha, linear=False):
        """Exact interpolation from ultimate demo."""
        if linear:
            t = alpha  # Linear interpolation
        else:
            # Ease-in/ease-out
            if alpha < 0.5:
                t = 2 * alpha * alpha
            else:
                t = 1 - pow(-2 * alpha + 2, 2) / 2
        return [(1-t)*c1 + t*c2 for c1, c2 in zip(config1, config2)]
    
    def compute_stacking_reward(self):
        """Exact reward computation from ultimate demo."""
        # Get block positions and orientations
        red_pos = self.data.xpos[self.red_id].copy()
        blue_pos = self.data.xpos[self.blue_id].copy()
        
        # Get block quaternions
        red_quat = self.data.xquat[self.red_id].copy()
        blue_quat = self.data.xquat[self.blue_id].copy()
        
        # 1. Vertical alignment - red should be directly above blue
        xy_offset = np.sqrt((red_pos[0] - blue_pos[0])**2 + (red_pos[1] - blue_pos[1])**2)
        alignment_reward = np.exp(-10 * xy_offset)  # Exponential decay, max 1.0
        
        # 2. Height reward - red bottom should be at blue top
        red_bottom = red_pos[2] - self.block_half_size
        blue_top = blue_pos[2] + self.block_half_size
        height_error = abs(red_bottom - blue_top)
        height_reward = np.exp(-20 * height_error)  # More sensitive to height
        
        # 3. Orientation alignment - blocks should have similar orientation
        # Convert quaternions to rotation matrices
        red_mat = np.zeros(9)
        blue_mat = np.zeros(9)
        mujoco.mju_quat2Mat(red_mat, red_quat)
        mujoco.mju_quat2Mat(blue_mat, blue_quat)
        red_mat = red_mat.reshape(3, 3)
        blue_mat = blue_mat.reshape(3, 3)
        
        # Compare Z axes (up direction)
        z_alignment = np.dot(red_mat[:, 2], blue_mat[:, 2])
        orientation_reward = (z_alignment + 1) / 2  # Map from [-1, 1] to [0, 1]
        
        # 4. Stability - red block should not be tilted
        red_z_axis = red_mat[:, 2]
        upright_score = red_z_axis[2]  # How much Z axis points up
        stability_reward = max(0, upright_score)
        
        # 5. Contact reward - bonus if blocks are touching
        contact_bonus = 1.0 if height_error < 0.005 else 0.0  # 5mm tolerance
        
        # Combine rewards with weights
        total_reward = (
            0.3 * alignment_reward +    # 30% for XY alignment
            0.3 * height_reward +       # 30% for correct height
            0.2 * orientation_reward +  # 20% for orientation match
            0.1 * stability_reward +    # 10% for stability
            0.1 * contact_bonus        # 10% for contact
        )
        
        # Return detailed metrics
        return {
            'total_reward': total_reward,
            'alignment_reward': alignment_reward,
            'height_reward': height_reward,
            'orientation_reward': orientation_reward,
            'stability_reward': stability_reward,
            'contact_bonus': contact_bonus,
            'xy_offset': xy_offset,
            'height_error': height_error,
            'z_alignment': z_alignment
        }
    
    def get_tactile_readings(self) -> np.ndarray:
        """Get 24-dimensional tactile readings."""
        # Simple tactile simulation based on contacts
        tactile = np.zeros(24)
        
        # Check contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get contact force magnitude
            force_magnitude = np.linalg.norm(contact.frame[:3])
            
            # Simple mapping to tactile arrays
            if force_magnitude > 0:
                # Distribute force randomly across sensors
                left_force = np.random.rand(12) * min(force_magnitude * 0.1, 1.0)
                right_force = np.random.rand(12) * min(force_magnitude * 0.1, 1.0)
                
                tactile[:12] = np.maximum(tactile[:12], left_force)
                tactile[12:] = np.maximum(tactile[12:], right_force)
        
        return np.clip(tactile, 0, 1)
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get complete observation for RL."""
        obs = {}
        
        # Joint positions and velocities
        joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] 
                             for jid in self.arm_joint_ids])
        joint_vel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]] 
                             for jid in self.arm_joint_ids])
        
        obs['joint_pos'] = joint_pos
        obs['joint_vel'] = joint_vel
        obs['proprio'] = np.concatenate([joint_pos, joint_vel])
        
        # Gripper state (normalized)
        obs['gripper_pos'] = np.array([self.data.ctrl[7] / 255.0])
        
        # End-effector pose
        obs['ee_pos'] = self.data.xpos[self.ee_id].copy()
        obs['ee_quat'] = self.data.xquat[self.ee_id].copy()
        
        # Object poses
        obs['target_block_pos'] = self.data.xpos[self.red_id].copy()
        obs['target_block_quat'] = self.data.xquat[self.red_id].copy()
        obs['block2_pos'] = self.data.xpos[self.blue_id].copy()
        obs['block2_quat'] = self.data.xquat[self.blue_id].copy()
        
        # Combined object state
        obs['object_state'] = np.concatenate([
            obs['target_block_pos'], obs['target_block_quat'],
            obs['block2_pos'], obs['block2_quat']
        ])
        
        # Tactile readings
        obs['tactile'] = self.get_tactile_readings()
        
        return obs
    
    def compute_action(self, current_config: List[float], target_config: List[float], 
                      gripper_cmd: float) -> np.ndarray:
        """Compute velocity-based action for RL compatibility."""
        # Velocity = position difference * gain
        config_diff = np.array(target_config) - np.array(current_config)
        joint_vels = np.clip(config_diff * 2.0, -2.0, 2.0)  # Max 2 rad/s
        
        # Gripper: map 0/255 to -1/1
        gripper_action = -1.0 if gripper_cmd > 128 else 1.0
        
        return np.concatenate([joint_vels, [gripper_action]])
    
    def execute_demonstration(self, demo_idx: int = 0) -> Tuple[Dict, Optional[List]]:
        """Execute a single demonstration using the exact ultimate demo approach."""
        print(f"\nExecuting demo {demo_idx}...")
        
        # Reset
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Get sequence
        sequence = self.get_sequence()
        
        # Set initial configuration
        initial_config = sequence[0][2]
        for i, val in enumerate(initial_config):
            self.data.qpos[14 + i] = val  # Robot joints start at qpos[14]
            self.data.ctrl[i] = val
        self.data.ctrl[7] = 255  # Gripper open
        
        mujoco.mj_step(self.model, self.data)
        
        # Storage for demonstration data
        observations = []
        actions = []
        rewards = []
        tactile_readings = []
        frames = [] if self.save_video else None
        
        # Stats
        grasp_clearance = None
        final_place_config = None
        place_stop_frame = None
        
        # Control frequency tracking
        sim_steps_per_control = int(500 / self.control_frequency)  # 500Hz sim
        
        # Execute sequence
        for phase_idx, (phase_name, duration, target_config, gripper_cmd, description) in enumerate(sequence):
            print(f"  Phase: {phase_name}")
            
            current_config = [self.data.qpos[14 + i] for i in range(7)]
            
            if phase_name == "release" and final_place_config:
                target_config = final_place_config
            
            # Handle None target configs (hold current position)
            if target_config is None:
                target_config = current_config
            
            for frame in range(duration):
                # Smart place logic (exact from ultimate demo)
                if phase_name == "smart_place":
                    red_pos = self.data.xpos[self.red_id]
                    red_bottom = red_pos[2] - self.block_half_size
                    stack_clearance = red_bottom - self.blue_top
                    
                    if stack_clearance <= 0 and not place_stop_frame:
                        place_stop_frame = frame
                        final_place_config = [self.data.qpos[14 + i] for i in range(7)]
                        target_config = final_place_config
                
                alpha = frame / duration
                
                if phase_idx == 0:
                    config = target_config
                elif place_stop_frame and phase_name == "smart_place":
                    config = final_place_config
                elif phase_name in ["hold_65mm", "grasp"]:
                    config = target_config
                else:
                    # Use linear interpolation for descend phases
                    use_linear = (phase_name in ["descend_fast", "descend_65mm"])
                    config = self.interpolate_configs(current_config, target_config, alpha, linear=use_linear)
                
                # Apply control
                for i, val in enumerate(config):
                    self.data.ctrl[i] = val
                self.data.ctrl[7] = gripper_cmd
                
                # Step physics
                mujoco.mj_step(self.model, self.data)
                
                # Record data at control frequency
                if frame % sim_steps_per_control == 0:
                    # Get observation
                    obs = self.get_observation()
                    
                    # Compute action (for RL training)
                    action = self.compute_action(current_config, config, gripper_cmd)
                    
                    # Store
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(0.0)  # Will compute task reward later
                    tactile_readings.append(obs['tactile'])
                
                # Track grasp clearance
                if phase_name == "grasp" and frame > 20:
                    gripper_bodies = ["hand", "left_finger", "right_finger"]
                    lowest_z = float('inf')
                    for body_name in gripper_bodies:
                        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                        if body_id != -1:
                            z_pos = self.data.xpos[body_id][2]
                            lowest_z = min(lowest_z, z_pos)
                    grasp_clearance = lowest_z - self.table_height
                
                # Render video frame
                if self.save_video and frame % 5 == 0:
                    cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                    self.renderer.update_scene(self.data, camera=cam_id)
                    frame_img = self.renderer.render()
                    
                    # Add text
                    cv2.putText(frame_img, f"Demo {demo_idx}: {description}", 
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    frames.append(frame_img)
        
        # Compute final reward
        final_reward = self.compute_stacking_reward()
        
        print(f"  Final reward: {final_reward['total_reward']:.3f}")
        print(f"  Grasp clearance: {grasp_clearance:.4f}m" if grasp_clearance else "")
        
        # Package demo data
        demo_data = {
            'observations': observations,
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'tactile_readings': np.array(tactile_readings),
            'success': final_reward['total_reward'] > 0.8,
            'final_reward': final_reward['total_reward'],
            'episode_length': len(actions),
            'reward_components': final_reward
        }
        
        return demo_data, frames
    
    def save_demonstrations(self, demos: List[Dict], output_path: str):
        """Save demonstrations to HDF5 file."""
        with h5py.File(output_path, 'w') as f:
            # Metadata
            f.attrs['num_demos'] = len(demos)
            f.attrs['env_name'] = 'PandaStackBlocks'
            f.attrs['use_tactile'] = True
            f.attrs['control_frequency'] = self.control_frequency
            f.attrs['date_collected'] = datetime.now().isoformat()
            
            # Save each demonstration
            for idx, demo in enumerate(demos):
                grp = f.create_group(f'demo_{idx}')
                
                # Observations
                obs_grp = grp.create_group('observations')
                for key in demo['observations'][0].keys():
                    data = np.array([obs[key] for obs in demo['observations']])
                    obs_grp.create_dataset(key, data=data)
                
                # Actions and rewards
                grp.create_dataset('actions', data=demo['actions'])
                grp.create_dataset('rewards', data=demo['rewards'])
                grp.create_dataset('tactile_readings', data=demo['tactile_readings'])
                
                # Metadata
                grp.attrs['success'] = demo['success']
                grp.attrs['episode_length'] = demo['episode_length']
                grp.attrs['final_reward'] = demo['final_reward']
                
                # Reward components
                for key, val in demo['reward_components'].items():
                    grp.attrs[f'reward_{key}'] = val
        
        print(f"\nSaved {len(demos)} demonstrations to {output_path}")
    
    def collect_demonstrations(self, num_demos: int = 50, output_dir: str = "../datasets/expert"):
        """Collect multiple demonstrations."""
        print(f"\nCollecting {num_demos} expert demonstrations...")
        print("="*60)
        
        demonstrations = []
        all_frames = []
        
        for demo_idx in range(num_demos):
            # Execute demonstration
            demo_data, frames = self.execute_demonstration(demo_idx)
            
            # Only save successful demos
            if demo_data['success']:
                demonstrations.append(demo_data)
                if frames:
                    all_frames.extend(frames)
                print(f"  ✅ Demo {demo_idx}: SUCCESS")
            else:
                print(f"  ❌ Demo {demo_idx}: FAILED (reward={demo_data['final_reward']:.3f})")
        
        # Save demonstrations
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"expert_demos_{timestamp}.hdf5")
        
        if demonstrations:
            self.save_demonstrations(demonstrations, output_path)
        
        # Save video compilation
        if self.save_video and all_frames:
            video_path = output_path.replace('.hdf5', '.mp4')
            self.save_video_compilation(all_frames, video_path)
        
        print(f"\n{'='*60}")
        print(f"✅ Collected {len(demonstrations)}/{num_demos} successful demonstrations")
        if demonstrations:
            avg_reward = np.mean([d['final_reward'] for d in demonstrations])
            print(f"   Average reward: {avg_reward:.3f}")
        
        return output_path
    
    def save_video_compilation(self, frames: List[np.ndarray], output_path: str):
        """Save video of all demonstrations."""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Saved video to {output_path}")


def main():
    """Main function to collect expert demonstrations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations")
    parser.add_argument("--output_dir", type=str, default="../datasets/expert", help="Output directory")
    parser.add_argument("--save_video", action="store_true", help="Save video compilation")
    parser.add_argument("--control_freq", type=float, default=20.0, help="Control frequency (Hz)")
    
    args = parser.parse_args()
    
    # Create collector
    collector = ExpertDemonstrationCollector(
        save_video=args.save_video,
        control_frequency=args.control_freq
    )
    
    # Collect demonstrations
    output_path = collector.collect_demonstrations(
        num_demos=args.num_demos,
        output_dir=args.output_dir
    )
    
    print(f"\n🎉 Done! Output: {output_path}")


if __name__ == "__main__":
    main()