#!/usr/bin/env python3
"""
Create a working demonstration with the fixed environment.
"""

import numpy as np
import sys
import os
import mujoco
import imageio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments'))
from environments.panda_demo_env import PandaDemoEnv


def create_demonstration():
    """Create a demonstration with proper grasping."""
    
    # Initialize environment with close blocks
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(base_dir, "franka_emika_panda", "panda_close_blocks.xml")
    
    # Temporarily monkey-patch the environment to use our XML
    original_init = PandaDemoEnv.__init__
    def custom_init(self, camera_name="demo_cam"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        
        from tactile_sensor import TactileSensor
        self.tactile_sensor = TactileSensor(
            model=self.model,
            data=self.data,
            n_taxels_x=3, 
            n_taxels_y=4,
            left_finger_name="left_finger",
            right_finger_name="right_finger"
        )
        self._setup_model_ids()
        print("🎬 Demo Environment with Close Blocks Ready!")
        print(f"   🎯 Control dim: {self.model.nu}")
    
    PandaDemoEnv.__init__ = custom_init
    env = PandaDemoEnv()
    obs = env.reset()
    
    frames = []
    
    print("\n" + "="*60)
    print("CREATING WORKING DEMONSTRATION")
    print("="*60)
    
    # Get initial info
    ee_site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if ee_site_id >= 0:
        ee_pos = env.data.site_xpos[ee_site_id].copy()
    else:
        hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ee_pos = env.data.xpos[hand_id].copy()
    
    target_pos = obs['target_block_pos']
    distance = np.linalg.norm(ee_pos - target_pos)
    
    print(f"\nInitial state:")
    print(f"  EE position: {ee_pos}")
    print(f"  Target position: {target_pos}")
    print(f"  Distance: {distance:.3f}m")
    
    # Initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    # Phase 1: Approach (50 steps)
    print("\n📍 Phase 1: Approach")
    for step in range(50):
        # Calculate error
        if ee_site_id >= 0:
            ee_pos = env.data.site_xpos[ee_site_id].copy()
        else:
            ee_pos = env.data.xpos[hand_id].copy()
        
        error = target_pos - ee_pos
        error[2] += 0.05  # Aim above block
        
        # Simple P-controller
        action = np.zeros(8)
        action[0] = np.clip(error[0] * 2.0, -1, 1)
        action[1] = np.clip(error[1] * 2.0, -1, 1)
        action[2] = np.clip(error[2] * 1.5, -1, 1)
        action[7] = -1.0  # Open gripper
        
        obs = env.step(action)
        
        if step % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if step % 10 == 0:
            dist = np.linalg.norm(ee_pos[:2] - target_pos[:2])
            print(f"  Step {step}: XY distance = {dist:.3f}m")
    
    # Phase 2: Descend (30 steps)
    print("\n📍 Phase 2: Descend")
    for step in range(30):
        action = np.zeros(8)
        action[2] = -0.3  # Descend
        action[7] = -1.0  # Keep open
        
        obs = env.step(action)
        
        if step % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if step % 10 == 0:
            if ee_site_id >= 0:
                ee_pos = env.data.site_xpos[ee_site_id].copy()
            else:
                ee_pos = env.data.xpos[hand_id].copy()
            height_diff = ee_pos[2] - target_pos[2]
            print(f"  Step {step}: Height above block = {height_diff:.3f}m")
    
    # Phase 3: Grasp (30 steps)
    print("\n📍 Phase 3: Grasp")
    for step in range(30):
        action = np.zeros(8)
        action[7] = 1.0  # Close gripper
        
        obs = env.step(action)
        
        if step % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if step % 10 == 0:
            tactile_sum = np.sum(obs['tactile'])
            print(f"  Step {step}: Tactile sum = {tactile_sum:.1f}")
    
    # Phase 4: Lift (40 steps)
    print("\n📍 Phase 4: Lift")
    initial_block_z = obs['target_block_pos'][2]
    
    for step in range(40):
        action = np.zeros(8)
        action[2] = 0.5  # Lift up
        action[7] = 1.0  # Keep closed
        
        obs = env.step(action)
        
        if step % 2 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if step % 10 == 0:
            current_z = obs['target_block_pos'][2]
            lift_height = current_z - initial_block_z
            print(f"  Step {step}: Lift height = {lift_height*1000:.1f}mm")
    
    # Add final frames
    for _ in range(10):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    
    # Final stats
    final_block_z = obs['target_block_pos'][2]
    total_lift = final_block_z - initial_block_z
    max_tactile = np.sum(obs['tactile'])
    
    print(f"\n📊 Final Statistics:")
    print(f"  Total frames: {len(frames)}")
    print(f"  Total lift: {total_lift*1000:.1f}mm")
    print(f"  Final tactile: {max_tactile:.1f}")
    
    if total_lift > 0.005:
        print(f"\n✅ SUCCESS! Block lifted {total_lift*1000:.1f}mm")
    else:
        print(f"\n❌ FAILURE: Only lifted {total_lift*1000:.1f}mm")
    
    # Save video
    if frames:
        output_path = "../../datasets/working_demonstration.mp4"
        print(f"\nSaving video with {len(frames)} frames...")
        imageio.mimsave(output_path, frames, fps=30, quality=8, codec='libx264')
        print(f"Video saved to: {output_path}")
        
        # GIF version
        gif_path = output_path.replace('.mp4', '.gif')
        gif_frames = frames[::3]
        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
        print(f"GIF saved to: {gif_path}")
    
    return frames


if __name__ == "__main__":
    create_demonstration()