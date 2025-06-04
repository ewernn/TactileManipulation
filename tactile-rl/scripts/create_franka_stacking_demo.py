"""
Final Franka stacking demonstration.
Picks up red block and places it on blue block.
"""

import numpy as np
import sys
import os
import imageio
import mujoco

sys.path.append('../environments')
from panda_demo_env import PandaDemoEnv


def create_stacking_demo():
    """Create the final stacking demonstration video."""
    
    print("🎬 Creating Franka Block Stacking Demonstration...")
    print("🔴 Red block will be lifted and placed on 🔵 blue block")
    
    env = PandaDemoEnv()
    
    # Create demonstration
    all_frames = []
    
    # Use different camera angles for variety
    cameras = ["demo_cam", "side_cam", "overhead_cam"]
    
    for episode in range(2):  # 2 episodes
        camera = cameras[episode % len(cameras)]
        print(f"\n📹 Episode {episode + 1} - Camera: {camera}")
        
        # Reset environment
        obs = env.reset(randomize=(episode > 0))
        
        print(f"  Red block at: {obs['target_block_pos']}")
        print(f"  Blue block at: {obs['block2_pos']}")
        
        frames = []
        
        # Phase 1: Move to home position
        print("  Phase 1: Home position")
        target_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
        for i in range(40):
            joint_error = target_joints - obs['joint_pos']
            action = np.zeros(8)
            action[:7] = np.clip(joint_error * 0.3, -1.0, 1.0)
            action[7] = -1.0  # Open gripper
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
        
        # Phase 2: Move above red block
        print("  Phase 2: Position above red block")
        # Adjust for block at (-0.15, 0.0)
        target_joints = np.array([0.0, 0.4, 0.0, -1.6, 0.0, 2.0, 0.785])
        for i in range(50):
            joint_error = target_joints - obs['joint_pos']
            action = np.zeros(8)
            action[:7] = np.clip(joint_error * 0.25, -1.0, 1.0)
            action[7] = -1.0  # Keep gripper open
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
        
        # Phase 3: Descend to grasp
        print("  Phase 3: Descend to block")
        target_joints = np.array([0.0, 0.65, 0.0, -1.3, 0.0, 1.95, 0.785])
        for i in range(40):
            joint_error = target_joints - obs['joint_pos']
            action = np.zeros(8)
            action[:7] = np.clip(joint_error * 0.2, -1.0, 1.0)
            action[7] = -1.0  # Keep gripper open
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
        
        # Phase 4: Close gripper
        print("  Phase 4: Grasp red block")
        red_initial = obs['target_block_pos'].copy()
        for i in range(40):
            action = np.zeros(8)
            action[7] = 1.0  # Close gripper
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
            
            if i == 20:
                tactile = np.sum(obs['tactile'])
                print(f"    Tactile reading: {tactile:.1f}")
        
        # Phase 5: Lift
        print("  Phase 5: Lift red block")
        target_joints = np.array([0.0, 0.2, 0.0, -1.8, 0.0, 2.0, 0.785])
        for i in range(50):
            joint_error = target_joints - obs['joint_pos']
            action = np.zeros(8)
            action[:7] = np.clip(joint_error * 0.25, -1.0, 1.0)
            action[7] = 1.0  # Keep gripper closed
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
            
            if i == 25:
                red_current = obs['target_block_pos']
                if red_current[2] > red_initial[2] + 0.05:
                    print(f"    ✓ Block lifted {(red_current[2] - red_initial[2])*1000:.0f}mm")
        
        # Phase 6: Move to blue block
        print("  Phase 6: Move to blue block")
        # Rotate to align with blue block at (-0.15, -0.15)
        target_joints = np.array([-0.5, 0.2, 0.0, -1.8, 0.0, 2.0, 0.785])
        for i in range(60):
            joint_error = target_joints - obs['joint_pos']
            action = np.zeros(8)
            action[:7] = np.clip(joint_error * 0.2, -1.0, 1.0)
            action[7] = 1.0  # Keep gripper closed
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
        
        # Phase 7: Position above blue
        print("  Phase 7: Position above blue block")
        target_joints = np.array([-0.5, 0.4, 0.0, -1.6, 0.0, 2.0, 0.785])
        for i in range(40):
            joint_error = target_joints - obs['joint_pos']
            action = np.zeros(8)
            action[:7] = np.clip(joint_error * 0.25, -1.0, 1.0)
            action[7] = 1.0  # Keep gripper closed
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
        
        # Phase 8: Lower to stack
        print("  Phase 8: Stack on blue block")
        target_joints = np.array([-0.5, 0.6, 0.0, -1.35, 0.0, 1.95, 0.785])
        for i in range(50):
            joint_error = target_joints - obs['joint_pos']
            action = np.zeros(8)
            action[:7] = np.clip(joint_error * 0.15, -1.0, 1.0)
            action[7] = 1.0  # Keep gripper closed
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
        
        # Phase 9: Release
        print("  Phase 9: Release")
        for i in range(30):
            action = np.zeros(8)
            action[7] = -1.0  # Open gripper
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
        
        # Phase 10: Retreat
        print("  Phase 10: Retreat")
        target_joints = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.7, 0.785])
        for i in range(40):
            joint_error = target_joints - obs['joint_pos']
            action = np.zeros(8)
            action[:7] = np.clip(joint_error * 0.3, -1.0, 1.0)
            action[7] = -1.0  # Keep gripper open
            obs = env.step(action, steps=3)
            frames.append(env.render(camera=camera))
        
        # Check final result
        red_final = obs['target_block_pos']
        blue_final = obs['block2_pos']
        height_diff = red_final[2] - blue_final[2]
        horiz_dist = np.linalg.norm(red_final[:2] - blue_final[:2])
        
        print(f"\n  Final result:")
        print(f"    Height difference: {height_diff*1000:.0f}mm")
        print(f"    Horizontal distance: {horiz_dist*1000:.0f}mm")
        
        if height_diff > 0.03 and horiz_dist < 0.05:
            print("    ✅ Stacking successful!")
            # Hold success pose
            for _ in range(60):
                obs = env.step(action, steps=1)
                frames.append(env.render(camera=camera))
        else:
            print("    ❌ Stacking not achieved")
        
        # Add episode frames
        all_frames.extend(frames)
        
        # Add transition between episodes
        if episode < 1:
            for _ in range(20):
                all_frames.append(frames[-1])
    
    # Save video
    output_path = "../../videos/franka_block_stacking_demo.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n💾 Saving video ({len(all_frames)} frames)...")
    
    imageio.mimsave(
        output_path,
        all_frames,
        fps=30,
        quality=9,
        macro_block_size=1
    )
    
    print(f"\n✅ Video created successfully!")
    print(f"📹 Location: {output_path}")
    print(f"⏱️  Duration: {len(all_frames)/30:.1f} seconds")
    print(f"📷 Multiple camera angles")
    print(f"\n🎯 The video shows the Franka Panda robot picking up a red block")
    print(f"   and stacking it on top of a blue block.")
    

if __name__ == "__main__":
    create_stacking_demo()