"""
Create a quick visualization of the grasping attempt.
This will render frames showing the gripper approaching and attempting to grasp the cube.
"""

import sys
import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.tactile_grasping_env import TactileGraspingEnv

def create_grasp_visualization():
    """Create a visualization of the grasping attempt."""
    
    env = TactileGraspingEnv(use_tactile=True)
    
    # Storage for visualization data
    frames = []
    positions = []
    gripper_widths = []
    tactile_readings = []
    
    print("Recording grasping sequence...")
    
    obs = env.reset()
    cube_pos = obs['object_state'][:3].copy()
    
    # Record initial state
    if env.ee_site_id:
        ee_pos = env.data.site_xpos[env.ee_site_id].copy()
    else:
        ee_pos = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")].copy()
    
    positions.append({'ee': ee_pos.copy(), 'cube': cube_pos.copy()})
    gripper_widths.append(obs['proprio'][14])
    tactile_readings.append(np.sum(obs['tactile']))
    
    # Phase 1: Move to cube (30 steps)
    target = cube_pos.copy()
    target[2] += 0.03
    
    for step in range(30):
        if env.ee_site_id:
            ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        else:
            ee_pos = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")].copy()
            
        error = target - ee_pos
        
        action = np.zeros(8)
        action[0] = np.clip(error[0] * 3, -1, 1)
        action[1] = np.clip(error[1] * 3, -1, 1)
        action[2] = np.clip(error[2] * 3, -1, 1)
        action[7] = -1.0  # Open
        
        obs, _, _, _, _ = env.step(action)
        
        # Record state
        positions.append({'ee': ee_pos.copy(), 'cube': obs['object_state'][:3].copy()})
        gripper_widths.append(obs['proprio'][14])
        tactile_readings.append(np.sum(obs['tactile']))
        
        # Render frame every 5 steps
        if step % 5 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    
    # Phase 2: Descend (20 steps)
    for step in range(20):
        action = np.zeros(8)
        action[2] = -0.3
        action[7] = -1.0
        
        obs, _, _, _, _ = env.step(action)
        
        if env.ee_site_id:
            ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        else:
            ee_pos = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")].copy()
            
        positions.append({'ee': ee_pos.copy(), 'cube': obs['object_state'][:3].copy()})
        gripper_widths.append(obs['proprio'][14])
        tactile_readings.append(np.sum(obs['tactile']))
        
        if step % 4 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    
    # Phase 3: Grasp (20 steps)
    for step in range(20):
        action = np.zeros(8)
        action[7] = 1.0  # Close
        
        obs, _, _, _, _ = env.step(action)
        
        if env.ee_site_id:
            ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        else:
            ee_pos = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")].copy()
            
        positions.append({'ee': ee_pos.copy(), 'cube': obs['object_state'][:3].copy()})
        gripper_widths.append(obs['proprio'][14])
        tactile_readings.append(np.sum(obs['tactile']))
        
        if step % 4 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    
    # Phase 4: Lift (20 steps)
    for step in range(20):
        action = np.zeros(8)
        action[2] = 0.5
        action[7] = 1.0
        
        obs, _, _, _, _ = env.step(action)
        
        if env.ee_site_id:
            ee_pos = env.data.site_xpos[env.ee_site_id].copy()
        else:
            ee_pos = env.data.xpos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")].copy()
            
        positions.append({'ee': ee_pos.copy(), 'cube': obs['object_state'][:3].copy()})
        gripper_widths.append(obs['proprio'][14])
        tactile_readings.append(np.sum(obs['tactile']))
        
        if step % 4 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: X-Y trajectory
    ee_x = [p['ee'][0] for p in positions]
    ee_y = [p['ee'][1] for p in positions]
    cube_x = [p['cube'][0] for p in positions]
    cube_y = [p['cube'][1] for p in positions]
    
    ax1.plot(ee_x, ee_y, 'b-', label='End Effector', linewidth=2)
    ax1.plot(cube_x, cube_y, 'g-', label='Cube', linewidth=2)
    ax1.scatter(ee_x[0], ee_y[0], c='blue', s=100, marker='o', label='EE Start')
    ax1.scatter(ee_x[-1], ee_y[-1], c='blue', s=100, marker='x', label='EE End')
    ax1.scatter(cube_x[0], cube_y[0], c='green', s=100, marker='s', label='Cube')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('X-Y Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Z trajectory over time
    steps = list(range(len(positions)))
    ee_z = [p['ee'][2] for p in positions]
    cube_z = [p['cube'][2] for p in positions]
    
    ax2.plot(steps, ee_z, 'b-', label='End Effector Z', linewidth=2)
    ax2.plot(steps, cube_z, 'g-', label='Cube Z', linewidth=2)
    ax2.axvline(x=30, color='r', linestyle='--', alpha=0.7, label='Start Descend')
    ax2.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='Start Grasp')
    ax2.axvline(x=70, color='purple', linestyle='--', alpha=0.7, label='Start Lift')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Z Position (m)')
    ax2.set_title('Z Position Over Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Gripper width
    ax3.plot(steps, gripper_widths, 'r-', linewidth=2)
    ax3.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='Start Grasp')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Gripper Width (m)')
    ax3.set_title('Gripper Width Over Time')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Tactile readings
    ax4.plot(steps, tactile_readings, 'm-', linewidth=2)
    ax4.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='Start Grasp')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Total Tactile Reading')
    ax4.set_title('Tactile Feedback Over Time')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('../datasets/grasp_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved grasp analysis plot to ../datasets/grasp_analysis.png")
    
    # Save video if we have frames
    if frames:
        print(f"Saving video with {len(frames)} frames...")
        imageio.mimsave('../datasets/grasp_sequence.mp4', frames, fps=5, quality=8)
        print("Saved grasp sequence video to ../datasets/grasp_sequence.mp4")
    
    # Print summary
    print(f"\nGrasp Analysis Summary:")
    print(f"Initial EE position: {positions[0]['ee']}")
    print(f"Final EE position: {positions[-1]['ee']}")
    print(f"Initial cube position: {positions[0]['cube']}")
    print(f"Final cube position: {positions[-1]['cube']}")
    print(f"Cube movement: {np.linalg.norm(positions[-1]['cube'] - positions[0]['cube']):.3f}m")
    print(f"Min gripper width: {min(gripper_widths):.3f}m")
    print(f"Max tactile reading: {max(tactile_readings):.3f}")
    print(f"Final tactile reading: {tactile_readings[-1]:.3f}")
    
    return positions, gripper_widths, tactile_readings, frames

if __name__ == "__main__":
    create_grasp_visualization()