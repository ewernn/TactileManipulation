#!/usr/bin/env python3
"""
Collect multiple expert demonstrations with parameterized initial conditions.
Generates high-quality position-controlled trajectories for block stacking.
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
        """Initialize the expert demonstration collector.
        
        Args:
            xml_path: Path to MuJoCo XML file
            save_video: Whether to save video of demonstrations
            control_frequency: Control frequency in Hz (for data saving)
        """
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
        
        # Workspace bounds for block placement
        self.workspace_bounds = {
            'x': [0.45, 0.65],  # Forward reach
            'y': [-0.15, 0.15], # Side to side
            'z': [0.425, 0.425] # On table
        }
        
        if save_video:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
    
    def randomize_block_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Randomize initial block positions within workspace.
        
        Returns:
            red_pos: Red block position
            blue_pos: Blue block position
        """
        # Sample positions ensuring blocks don't overlap
        min_separation = 0.08  # 8cm minimum between blocks
        
        while True:
            red_x = np.random.uniform(self.workspace_bounds['x'][0], self.workspace_bounds['x'][1])
            red_y = np.random.uniform(self.workspace_bounds['y'][0], self.workspace_bounds['y'][1])
            red_pos = np.array([red_x, red_y, self.workspace_bounds['z'][0]])
            
            blue_x = np.random.uniform(self.workspace_bounds['x'][0], self.workspace_bounds['x'][1])
            blue_y = np.random.uniform(self.workspace_bounds['y'][0], self.workspace_bounds['y'][1])
            blue_pos = np.array([blue_x, blue_y, self.workspace_bounds['z'][0]])
            
            # Check separation
            if np.linalg.norm(red_pos[:2] - blue_pos[:2]) >= min_separation:
                break
        
        return red_pos, blue_pos
    
    def set_block_positions(self, red_pos: np.ndarray, blue_pos: np.ndarray):
        """Set block positions in the simulation."""
        # Blocks have freejoint - 7 DOF each (3 pos + 4 quat)
        # Find the qpos indices for each block's freejoint
        
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
        
        # Set positions using qpos addresses
        if red_joint_id != -1:
            addr = self.model.jnt_qposadr[red_joint_id]
            self.data.qpos[addr:addr+3] = red_pos
            # Keep existing quaternion (1, 0, 0, 0)
            self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
        
        if blue_joint_id != -1:
            addr = self.model.jnt_qposadr[blue_joint_id]
            self.data.qpos[addr:addr+3] = blue_pos
            # Keep existing quaternion
            self.data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
    
    def compute_waypoints(self, red_pos: np.ndarray, blue_pos: np.ndarray) -> List[Dict]:
        """Compute waypoints for block stacking based on block positions.
        
        Args:
            red_pos: Red block position
            blue_pos: Blue block position
            
        Returns:
            List of waypoint dictionaries
        """
        # Base joint configurations (from original demo)
        home_config = [0, -0.5, 0, -1.8, 0, 1.4, 0]
        
        # Compute approach angles based on red block position
        red_angle = np.arctan2(red_pos[1], red_pos[0])
        
        # Adjust base rotation (joint 0) based on block position
        base_rotation = red_angle * 0.3  # Partial rotation toward block
        
        # Compute waypoints with adaptive positioning
        waypoints = [
            # Home
            {
                "name": "home",
                "duration": 20,
                "config": home_config.copy(),
                "gripper": 255,
                "description": "Home Position"
            },
            
            # Approach - rotate base toward red block
            {
                "name": "approach",
                "duration": 25,
                "config": [base_rotation, -0.25, 0, -2.15, 0, 1.65, 0],
                "gripper": 255,
                "description": "Approach Red Block"
            },
            
            # Position above red block
            {
                "name": "position",
                "duration": 25,
                "config": self._compute_position_config(red_pos, height_offset=0.15),
                "gripper": 255,
                "description": "Position Above Red"
            },
            
            # Descend to grasp height
            {
                "name": "descend",
                "duration": 60,
                "config": self._compute_position_config(red_pos, height_offset=0.065),
                "gripper": 255,
                "description": "Descend to Grasp"
            },
            
            # Grasp
            {
                "name": "grasp",
                "duration": 30,
                "config": None,  # Hold position
                "gripper": 0,
                "description": "Grasp Block"
            },
            
            # Lift
            {
                "name": "lift",
                "duration": 25,
                "config": self._compute_position_config(red_pos, height_offset=0.15),
                "gripper": 0,
                "description": "Lift Block"
            },
            
            # Move to above blue block
            {
                "name": "transport",
                "duration": 35,
                "config": self._compute_position_config(blue_pos, height_offset=0.15),
                "gripper": 0,
                "description": "Move to Blue"
            },
            
            # Place on blue block
            {
                "name": "place",
                "duration": 40,
                "config": self._compute_position_config(blue_pos, height_offset=0.05),
                "gripper": 0,
                "description": "Place on Blue"
            },
            
            # Release
            {
                "name": "release",
                "duration": 25,
                "config": None,  # Hold position
                "gripper": 255,
                "description": "Release"
            },
            
            # Retreat
            {
                "name": "retreat",
                "duration": 30,
                "config": home_config.copy(),
                "gripper": 255,
                "description": "Return Home"
            }
        ]
        
        return waypoints
    
    def _compute_position_config(self, target_pos: np.ndarray, height_offset: float) -> List[float]:
        """Compute joint configuration to reach a target position.
        
        Uses configurations from the working demonstration, adapted for position.
        """
        # For now, use fixed configurations from working demo
        # In practice, you'd use proper IK or learned mappings
        
        # Base configurations that work well
        if height_offset > 0.1:  # High position
            base_config = [0, -0.25, 0, -2.15, 0, 1.65, 0]
        elif height_offset > 0.06:  # Medium position (grasp approach)
            base_config = [0, -0.20, 0, -2.20, 0, 1.70, 0]
        else:  # Low position (grasp/place)
            base_config = [0, -0.181, 0, -2.219, 0, 1.719, 0]
        
        # Adjust base rotation for target position
        # Red block default is at ~(0.55, 0.0), blue at ~(0.55, -0.1)
        dx = target_pos[0] - 0.55
        dy = target_pos[1] - 0.0
        
        # Simple linear mapping for base rotation
        base_rotation = dy * 2.0  # Scale lateral offset to rotation
        
        # Adjust forward reach based on x distance
        reach_adjustment = (dx - 0.0) * 0.5
        
        joint_config = base_config.copy()
        joint_config[0] += base_rotation  # Adjust base rotation
        joint_config[1] += reach_adjustment * 0.2  # Adjust shoulder
        joint_config[3] += reach_adjustment * 0.3  # Adjust elbow
        
        return joint_config
    
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
        """Get tactile sensor readings from contact forces.
        
        Returns:
            24-dimensional array of tactile readings (12 per finger)
        """
        # Simplified tactile reading based on contact forces
        tactile_readings = np.zeros(24)
        
        # Check contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Check if contact involves fingers
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if geom1_name and geom2_name:
                # Simple force mapping to tactile array
                force_magnitude = np.linalg.norm(contact.frame[:3])
                
                if "left_finger" in geom1_name or "left_finger" in geom2_name:
                    tactile_readings[:12] += force_magnitude * 0.1 * np.random.rand(12)
                if "right_finger" in geom1_name or "right_finger" in geom2_name:
                    tactile_readings[12:] += force_magnitude * 0.1 * np.random.rand(12)
        
        return np.clip(tactile_readings, 0, 1)
    
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
        obs['gripper_pos'] = np.array([self.data.ctrl[7] / 255.0])  # Normalize to [0,1]
        
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
        """Compute action as velocity commands based on position difference.
        
        This simulates what an RL policy would output.
        """
        # Compute velocity commands (simplified)
        config_diff = np.array(target_config) - np.array(current_config)
        joint_vels = np.clip(config_diff * 2.0, -2.0, 2.0)  # Max 2 rad/s
        
        # Gripper action: map 0/255 to -1/1
        gripper_action = -1.0 if gripper_cmd > 128 else 1.0
        
        action = np.concatenate([joint_vels, [gripper_action]])
        return action
    
    def execute_demonstration(self, demo_idx: int, red_pos: np.ndarray, 
                            blue_pos: np.ndarray) -> Tuple[Dict, Optional[List]]:
        """Execute a single demonstration.
        
        Returns:
            demo_data: Dictionary containing all demonstration data
            frames: List of video frames if save_video is True
        """
        # Reset simulation
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Set block positions
        self.set_block_positions(red_pos, blue_pos)
        
        # Get waypoints
        waypoints = self.compute_waypoints(red_pos, blue_pos)
        
        # Initialize data storage
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
        
        # Run forward to update positions
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
                # Get observation before action
                obs = self.get_observation()
                
                # Smart placement logic
                if waypoint["name"] == "place" and not place_stopped:
                    red_pos = self.data.xpos[self.red_id]
                    red_bottom = red_pos[2] - self.block_half_size
                    blue_top = self.data.xpos[self.blue_id][2] + self.block_half_size
                    stack_clearance = red_bottom - blue_top
                    
                    if stack_clearance <= 0.001:  # Contact or very close
                        place_stopped = True
                        final_place_config = [self.data.ctrl[i] for i in range(7)]
                        target_config = final_place_config
                
                # Compute interpolated configuration
                alpha = frame / waypoint["duration"]
                
                if waypoint["name"] in ["grasp", "release"] or place_stopped:
                    # Hold position during grasp/release or after placement
                    config = current_config if waypoint["name"] != "release" else target_config
                else:
                    # Interpolate to target
                    linear = waypoint["name"] in ["descend", "place"]
                    config = self.interpolate_configs(current_config, target_config, alpha, linear)
                
                # Apply control
                for i, val in enumerate(config):
                    self.data.ctrl[i] = val
                self.data.ctrl[7] = waypoint["gripper"]
                
                # Compute action (for RL compatibility)
                action = self.compute_action(current_config, config, waypoint["gripper"])
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Store data at control frequency
                if frame % int(500 / self.control_frequency) == 0:  # 500Hz sim -> control freq
                    observations.append(obs)
                    actions.append(action)
                    rewards.append(0.0)  # Placeholder - compute task-specific reward
                    tactile_readings.append(obs['tactile'])
                
                # Render frame
                if self.save_video and frame % 5 == 0:  # Reduce video framerate
                    cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_cam")
                    self.renderer.update_scene(self.data, camera=cam_id)
                    frame_img = self.renderer.render()
                    
                    # Add text overlay
                    cv2.putText(frame_img, f"Demo {demo_idx}: {waypoint['description']}", 
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    frames.append(frame_img)
        
        # Compute final stacking reward
        final_reward = self.compute_stacking_reward()
        
        # Package demonstration data
        demo_data = {
            'observations': observations,
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'tactile_readings': np.array(tactile_readings),
            'success': final_reward['total_reward'] > 0.8,
            'final_reward': final_reward['total_reward'],
            'episode_length': len(actions),
            'red_initial_pos': red_pos,
            'blue_initial_pos': blue_pos
        }
        
        return demo_data, frames
    
    def compute_stacking_reward(self) -> Dict[str, float]:
        """Compute stacking quality metrics."""
        # Get block positions
        red_pos = self.data.xpos[self.red_id].copy()
        blue_pos = self.data.xpos[self.blue_id].copy()
        
        # Alignment
        xy_offset = np.linalg.norm(red_pos[:2] - blue_pos[:2])
        alignment_reward = np.exp(-10 * xy_offset)
        
        # Height
        red_bottom = red_pos[2] - self.block_half_size
        blue_top = blue_pos[2] + self.block_half_size
        height_error = abs(red_bottom - blue_top)
        height_reward = np.exp(-20 * height_error)
        
        # Total reward
        total_reward = 0.5 * alignment_reward + 0.5 * height_reward
        
        return {
            'total_reward': total_reward,
            'alignment_reward': alignment_reward,
            'height_reward': height_reward,
            'xy_offset': xy_offset,
            'height_error': height_error
        }
    
    def save_demonstrations(self, demos: List[Dict], output_path: str):
        """Save demonstrations to HDF5 file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
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
                grp.attrs['red_initial_pos'] = demo['red_initial_pos']
                grp.attrs['blue_initial_pos'] = demo['blue_initial_pos']
            
            print(f"Saved {len(demos)} demonstrations to {output_path}")
    
    def collect_demonstrations(self, num_demos: int = 50, output_dir: str = "../datasets/expert"):
        """Collect multiple expert demonstrations."""
        print(f"\nCollecting {num_demos} expert demonstrations...")
        print("="*60)
        
        demonstrations = []
        all_frames = []
        
        for demo_idx in range(num_demos):
            print(f"\nDemo {demo_idx + 1}/{num_demos}:")
            
            # Randomize block positions
            red_pos, blue_pos = self.randomize_block_positions()
            print(f"  Red block: {red_pos}")
            print(f"  Blue block: {blue_pos}")
            
            # Execute demonstration
            demo_data, frames = self.execute_demonstration(demo_idx, red_pos, blue_pos)
            
            print(f"  Success: {demo_data['success']}")
            print(f"  Final reward: {demo_data['final_reward']:.3f}")
            print(f"  Episode length: {demo_data['episode_length']} steps")
            
            # Only save successful demonstrations
            if demo_data['success']:
                demonstrations.append(demo_data)
                if frames:
                    all_frames.extend(frames)
            else:
                print("  ⚠️  Demo failed - skipping")
        
        # Save demonstrations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"expert_demos_{timestamp}.hdf5")
        self.save_demonstrations(demonstrations, output_path)
        
        # Save video compilation
        if self.save_video and all_frames:
            video_path = os.path.join(output_dir, f"expert_demos_{timestamp}.mp4")
            self.save_video_compilation(all_frames, video_path)
        
        print(f"\n✅ Collected {len(demonstrations)} successful demonstrations")
        print(f"   Average reward: {np.mean([d['final_reward'] for d in demonstrations]):.3f}")
        
        return output_path
    
    def save_video_compilation(self, frames: List[np.ndarray], output_path: str):
        """Save video compilation of all demonstrations."""
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
    
    parser = argparse.ArgumentParser(description="Collect expert demonstrations for block stacking")
    parser.add_argument("--num_demos", type=int, default=50, help="Number of demonstrations to collect")
    parser.add_argument("--output_dir", type=str, default="../datasets/expert", help="Output directory")
    parser.add_argument("--save_video", action="store_true", help="Save video of demonstrations")
    parser.add_argument("--control_freq", type=float, default=20.0, help="Control frequency in Hz")
    
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
    
    print(f"\n🎉 Done! Demonstrations saved to: {output_path}")


if __name__ == "__main__":
    main()