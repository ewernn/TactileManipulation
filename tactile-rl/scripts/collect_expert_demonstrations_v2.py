#!/usr/bin/env python3
"""
Collect expert demonstrations using the exact waypoint approach from the working demo.
"""

import numpy as np
import mujoco
import h5py
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import cv2

class ExpertDemonstrationCollector:
    def __init__(self, 
                 xml_path: str = "../franka_emika_panda/panda_demo_scene.xml",
                 save_video: bool = True,
                 control_frequency: float = 20.0):
        """Initialize the expert demonstration collector."""
        self.xml_path = xml_path
        self.save_video = save_video
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get IDs
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.red_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        self.blue_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        
        # Finger IDs for tactile sensing
        self.left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        
        # Joint IDs
        self.arm_joint_ids = []
        for i in range(7):
            joint_name = f"joint{i+1}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.arm_joint_ids.append(joint_id)
        
        # Critical measurements
        self.table_height = 0.4
        self.block_half_size = 0.025
        self.target_grasp_clearance = 0.065
        
        # Working waypoint configurations from the successful demo
        self.base_waypoints = [
            # Start
            ("home", 20, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Home - 0.2m Clearance"),
            
            # Approach
            ("approach", 25, [0, -0.25, 0, -2.15, 0, 1.65, 0], 255, "Approach"),
            
            # Position - stay higher
            ("position", 25, [0, -0.20, 0, -2.20, 0, 1.70, 0], 255, "Position Above"),
            
            # Fast descend
            ("descend_fast", 60, [0, -0.18, 0, -2.22, 0, 1.72, 0], 255, "Descend Fast"),
            
            # Slow descend to grasp height
            ("descend_slow", 80, [0, -0.181, 0, -2.219, 0, 1.719, 0], 255, "Descend to 65mm"),
            
            # Hold
            ("hold", 25, None, 255, "Hold at 65mm"),
            
            # Grasp
            ("grasp", 30, None, 0, "Grasp"),
            
            # Lift
            ("lift", 25, [0, -0.25, 0, -2.15, 0, 1.65, 0], 0, "Lift"),
            
            # Lift high
            ("lift_high", 25, [0, -0.5, 0, -1.8, 0, 1.4, 0], 0, "Lift High"),
            
            # Rotate and move to blue
            ("rotate", 35, None, 0, "Rotate"),  # Will be computed
            
            # Above blue
            ("above_blue", 25, None, 0, "Above Blue"),  # Will be computed
            
            # Smart place
            ("smart_place", 40, None, 0, "Smart Place"),  # Will be computed
            
            # Release
            ("release", 25, None, 255, "Release"),
            
            # Retreat
            ("retreat", 25, None, 255, "Retreat"),  # Will be computed
            
            # Return
            ("return", 30, [0, -0.5, 0, -1.8, 0, 1.4, 0], 255, "Return"),
        ]
        
        if save_video:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
    
    def set_block_positions(self, red_pos: np.ndarray, blue_pos: np.ndarray):
        """Set block positions in the simulation."""
        red_body_id = self.red_id
        blue_body_id = self.blue_id
        
        # Find joint IDs from body IDs
        red_joint_id = -1
        blue_joint_id = -1
        
        for i in range(self.model.njnt):
            if self.model.jnt_bodyid[i] == red_body_id:
                red_joint_id = i
            elif self.model.jnt_bodyid[i] == blue_body_id:
                blue_joint_id = i
        
        # Set positions
        if red_joint_id != -1:
            addr = self.model.jnt_qposadr[red_joint_id]
            self.data.qpos[addr:addr+3] = red_pos
            self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
        
        if blue_joint_id != -1:
            addr = self.model.jnt_qposadr[blue_joint_id]
            self.data.qpos[addr:addr+3] = blue_pos
            self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
    
    def compute_adaptive_waypoints(self, red_pos: np.ndarray, blue_pos: np.ndarray) -> List[Dict]:
        """Compute waypoints adapted to block positions."""
        waypoints = []
        
        # Calculate offsets from default positions
        # Default: red at (0.55, 0.0), blue at (0.55, -0.1)
        red_offset = red_pos - np.array([0.55, 0.0, 0.425])
        blue_offset = blue_pos - np.array([0.55, -0.1, 0.425])
        
        # Base rotation adjustments
        red_angle_adjust = np.arctan2(red_offset[1], red_offset[0]) if np.any(red_offset[:2]) else 0
        blue_angle_adjust = np.arctan2(blue_offset[1], blue_offset[0]) if np.any(blue_offset[:2]) else 0
        
        for wp in self.base_waypoints:
            name, duration, config, gripper, desc = wp
            
            if config is None:
                # Compute configs for adaptive waypoints
                if name == "rotate":
                    # Rotate toward blue block
                    base_config = [-0.35, -0.5, 0, -1.8, 0, 1.4, 0.174]
                    base_config[0] += blue_angle_adjust
                    config = base_config
                
                elif name == "above_blue":
                    # Position above blue block
                    config = [-0.37, -0.35, 0, -2.05, 0, 1.55, 0.174]
                    config[0] += blue_angle_adjust
                
                elif name == "smart_place":
                    # Place position
                    config = [-0.37, -0.15, 0, -2.25, 0, 1.75, 0.174]
                    config[0] += blue_angle_adjust
                
                elif name == "retreat":
                    # Retreat with rotation removed
                    config = [-0.35, -0.5, 0, -1.8, 0, 1.4, 0]
                    config[0] += blue_angle_adjust * 0.5
            
            else:
                # Adjust existing configs for red block approach
                if name in ["approach", "position", "descend_fast", "descend_slow"]:
                    config = config.copy()
                    config[0] += red_angle_adjust
            
            waypoints.append({
                "name": name,
                "duration": duration,
                "config": config,
                "gripper": gripper,
                "description": desc
            })
        
        return waypoints
    
    def interpolate_configs(self, config1: List[float], config2: List[float], 
                          alpha: float, linear: bool = False) -> List[float]:
        """Interpolate between two configurations."""
        if linear:
            t = alpha
        else:
            # Ease-in/ease-out
            if alpha < 0.5:
                t = 2 * alpha * alpha
            else:
                t = 1 - pow(-2 * alpha + 2, 2) / 2
        
        return [(1-t)*c1 + t*c2 for c1, c2 in zip(config1, config2)]
    
    def get_tactile_readings(self) -> np.ndarray:
        """Get tactile sensor readings (24D)."""
        tactile = np.zeros(24)
        
        # Simple contact-based tactile simulation
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.linalg.norm(contact.frame[:3])
            
            # Check which finger
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Distribute force across tactile array
            if force > 0:
                # Random distribution for now
                tactile += np.random.rand(24) * min(force * 0.1, 1.0)
        
        return np.clip(tactile, 0, 1)
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation dictionary."""
        obs = {}
        
        # Joint positions and velocities
        joint_pos = np.array([self.data.qpos[self.model.jnt_qposadr[jid]] 
                             for jid in self.arm_joint_ids])
        joint_vel = np.array([self.data.qvel[self.model.jnt_dofadr[jid]] 
                             for jid in self.arm_joint_ids])
        
        obs['joint_pos'] = joint_pos
        obs['joint_vel'] = joint_vel
        obs['proprio'] = np.concatenate([joint_pos, joint_vel])
        
        # Gripper state
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
        """Compute action as velocity commands."""
        config_diff = np.array(target_config) - np.array(current_config)
        joint_vels = np.clip(config_diff * 2.0, -2.0, 2.0)
        gripper_action = -1.0 if gripper_cmd > 128 else 1.0
        
        action = np.concatenate([joint_vels, [gripper_action]])
        return action
    
    def execute_demonstration(self, demo_idx: int, red_pos: np.ndarray, 
                            blue_pos: np.ndarray) -> Tuple[Dict, Optional[List]]:
        """Execute a single demonstration."""
        # Reset and set block positions
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.set_block_positions(red_pos, blue_pos)
        
        # Get adapted waypoints
        waypoints = self.compute_adaptive_waypoints(red_pos, blue_pos)
        
        # Initialize storage
        observations = []
        actions = []
        rewards = []
        tactile_readings = []
        frames = [] if self.save_video else None
        
        # Set initial configuration
        initial_config = waypoints[0]["config"]
        for i, val in enumerate(initial_config):
            joint_addr = self.model.jnt_qposadr[self.arm_joint_ids[i]]
            self.data.qpos[joint_addr] = val
            self.data.ctrl[i] = val
        self.data.ctrl[7] = waypoints[0]["gripper"]
        
        mujoco.mj_forward(self.model, self.data)
        
        # Execute waypoints
        place_stopped = False
        final_place_config = None
        
        for wp_idx, waypoint in enumerate(waypoints):
            current_config = [self.data.ctrl[i] for i in range(7)]
            target_config = waypoint["config"] if waypoint["config"] is not None else current_config
            
            # Use saved config if place was stopped
            if waypoint["name"] == "release" and final_place_config:
                target_config = final_place_config
            
            for frame in range(waypoint["duration"]):
                # Get observation
                obs = self.get_observation()
                
                # Smart placement
                if waypoint["name"] == "smart_place" and not place_stopped:
                    red_pos = self.data.xpos[self.red_id]
                    red_bottom = red_pos[2] - self.block_half_size
                    blue_top = self.data.xpos[self.blue_id][2] + self.block_half_size
                    stack_clearance = red_bottom - blue_top
                    
                    if stack_clearance <= 0.001:
                        place_stopped = True
                        final_place_config = [self.data.ctrl[i] for i in range(7)]
                        target_config = final_place_config
                
                # Interpolate
                alpha = frame / waypoint["duration"]
                
                if waypoint["name"] in ["hold", "grasp", "release"] or place_stopped:
                    config = current_config if waypoint["name"] != "release" else target_config
                else:
                    linear = waypoint["name"] in ["descend_fast", "descend_slow", "smart_place"]
                    config = self.interpolate_configs(current_config, target_config, alpha, linear)
                
                # Apply control
                for i, val in enumerate(config):
                    self.data.ctrl[i] = val
                self.data.ctrl[7] = waypoint["gripper"]
                
                # Compute action
                action = self.compute_action(current_config, config, waypoint["gripper"])
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Store data at control frequency
                if frame % int(500 / self.control_frequency) == 0:
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(0.0)
                    tactile_readings.append(obs['tactile'])
                
                # Render
                if self.save_video and frame % 5 == 0:
                    cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                    self.renderer.update_scene(self.data, camera=cam_id)
                    frame_img = self.renderer.render()
                    
                    cv2.putText(frame_img, f"Demo {demo_idx}: {waypoint['description']}", 
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add metrics
                    red_current = self.data.xpos[self.red_id]
                    blue_current = self.data.xpos[self.blue_id]
                    dist = np.linalg.norm(red_current[:2] - blue_current[:2])
                    
                    cv2.putText(frame_img, f"Block dist: {dist:.3f}m", 
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    frames.append(frame_img)
        
        # Compute final reward
        final_reward = self.compute_stacking_reward()
        
        # Package data
        demo_data = {
            'observations': observations,
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'tactile_readings': np.array(tactile_readings),
            'success': final_reward['total_reward'] > 0.7,  # Lower threshold for variations
            'final_reward': final_reward['total_reward'],
            'episode_length': len(actions),
            'red_initial_pos': red_pos,
            'blue_initial_pos': blue_pos
        }
        
        return demo_data, frames
    
    def compute_stacking_reward(self) -> Dict[str, float]:
        """Compute stacking quality metrics."""
        red_pos = self.data.xpos[self.red_id].copy()
        blue_pos = self.data.xpos[self.blue_id].copy()
        
        xy_offset = np.linalg.norm(red_pos[:2] - blue_pos[:2])
        alignment_reward = np.exp(-10 * xy_offset)
        
        red_bottom = red_pos[2] - self.block_half_size
        blue_top = blue_pos[2] + self.block_half_size
        height_error = abs(red_bottom - blue_top)
        height_reward = np.exp(-20 * height_error)
        
        total_reward = 0.5 * alignment_reward + 0.5 * height_reward
        
        return {
            'total_reward': total_reward,
            'alignment_reward': alignment_reward,
            'height_reward': height_reward,
            'xy_offset': xy_offset,
            'height_error': height_error
        }
    
    def collect_demonstrations(self, num_demos: int = 10):
        """Collect demonstrations with default block positions for now."""
        print(f"\nCollecting {num_demos} expert demonstrations...")
        print("="*60)
        
        demonstrations = []
        all_frames = []
        
        # For now, use fixed positions that we know work
        # Later can add randomization
        positions = [
            (np.array([0.55, 0.0, 0.425]), np.array([0.55, -0.1, 0.425])),  # Default
            (np.array([0.50, 0.0, 0.425]), np.array([0.60, -0.1, 0.425])),  # Spread out
            (np.array([0.55, 0.05, 0.425]), np.array([0.55, -0.15, 0.425])), # Y offset
            (np.array([0.52, -0.05, 0.425]), np.array([0.58, 0.05, 0.425])), # Diagonal
        ]
        
        for demo_idx in range(num_demos):
            # Cycle through positions
            red_pos, blue_pos = positions[demo_idx % len(positions)]
            
            print(f"\nDemo {demo_idx + 1}/{num_demos}:")
            print(f"  Red: {red_pos}")
            print(f"  Blue: {blue_pos}")
            
            demo_data, frames = self.execute_demonstration(demo_idx, red_pos, blue_pos)
            
            print(f"  Success: {demo_data['success']}")
            print(f"  Reward: {demo_data['final_reward']:.3f}")
            print(f"  Length: {demo_data['episode_length']} steps")
            
            if demo_data['success']:
                demonstrations.append(demo_data)
                if frames:
                    all_frames.extend(frames)
        
        # Save demonstrations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("../datasets/expert", exist_ok=True)
        output_path = f"../datasets/expert/expert_demos_{timestamp}.hdf5"
        
        self.save_demonstrations(demonstrations, output_path)
        
        # Save video
        if self.save_video and all_frames:
            self.save_video_compilation(all_frames, output_path.replace('.hdf5', '.mp4'))
        
        print(f"\n✅ Collected {len(demonstrations)} successful demonstrations")
        return output_path
    
    def save_demonstrations(self, demos: List[Dict], output_path: str):
        """Save demonstrations to HDF5."""
        with h5py.File(output_path, 'w') as f:
            f.attrs['num_demos'] = len(demos)
            f.attrs['env_name'] = 'PandaStackBlocks'
            f.attrs['use_tactile'] = True
            f.attrs['control_frequency'] = self.control_frequency
            f.attrs['date_collected'] = datetime.now().isoformat()
            
            for idx, demo in enumerate(demos):
                grp = f.create_group(f'demo_{idx}')
                
                # Observations
                obs_grp = grp.create_group('observations')
                for key in demo['observations'][0].keys():
                    data = np.array([obs[key] for obs in demo['observations']])
                    obs_grp.create_dataset(key, data=data)
                
                # Other data
                grp.create_dataset('actions', data=demo['actions'])
                grp.create_dataset('rewards', data=demo['rewards'])
                grp.create_dataset('tactile_readings', data=demo['tactile_readings'])
                
                # Metadata
                grp.attrs['success'] = demo['success']
                grp.attrs['episode_length'] = demo['episode_length']
                grp.attrs['final_reward'] = demo['final_reward']
        
        print(f"Saved to {output_path}")
    
    def save_video_compilation(self, frames: List[np.ndarray], output_path: str):
        """Save video compilation."""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Video saved to {output_path}")


def main():
    """Collect expert demonstrations."""
    collector = ExpertDemonstrationCollector(save_video=True)
    output_path = collector.collect_demonstrations(num_demos=10)
    print(f"\n🎉 Done! Output: {output_path}")


if __name__ == "__main__":
    main()