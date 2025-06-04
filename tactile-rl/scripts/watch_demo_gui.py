#!/usr/bin/env python3
"""
Interactive GUI viewer for expert demonstrations.
"""

import numpy as np
import h5py
import os
import sys
import mujoco
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'environments'))
from environments.panda_demo_env import PandaDemoEnv


class DemoViewer:
    """Interactive demonstration viewer with playback controls."""
    
    def __init__(self, dataset_path, demo_idx=0):
        self.dataset_path = dataset_path
        self.demo_idx = demo_idx
        self.current_step = 0
        self.playing = False
        
        # Load demonstration data
        self.load_demo_data()
        
        # Initialize environment
        self.env = PandaDemoEnv(camera_name="demo_cam")
        
        # Setup GUI
        self.setup_gui()
        
    def load_demo_data(self):
        """Load demonstration data from HDF5."""
        with h5py.File(self.dataset_path, 'r') as f:
            demo_key = f'demo_{self.demo_idx}'
            if demo_key not in f:
                raise ValueError(f"Demo {self.demo_idx} not found in dataset")
            
            demo_group = f[demo_key]
            self.actions = demo_group['actions'][:]
            self.joint_positions = demo_group['observations/joint_pos'][:]
            self.gripper_positions = demo_group['observations/gripper_pos'][:]
            self.tactile_readings = demo_group['tactile_readings'][:]
            self.rewards = demo_group['rewards'][:]
            self.target_positions = demo_group['observations/target_block_pos'][:]
            
        self.n_steps = len(self.actions)
        print(f"Loaded demo {self.demo_idx} with {self.n_steps} steps")
        
    def setup_gui(self):
        """Setup matplotlib GUI."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main simulation view
        self.ax_sim = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_sim.set_title('Simulation View')
        self.ax_sim.axis('off')
        
        # Tactile sensor view
        self.ax_tactile = plt.subplot2grid((3, 3), (0, 2))
        self.ax_tactile.set_title('Tactile Readings')
        
        # Joint positions
        self.ax_joints = plt.subplot2grid((3, 3), (1, 2))
        self.ax_joints.set_title('Joint Positions')
        self.ax_joints.set_ylabel('Position (rad)')
        
        # Rewards
        self.ax_reward = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        self.ax_reward.set_title('Cumulative Reward')
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Reward')
        
        # Block height
        self.ax_height = plt.subplot2grid((3, 3), (2, 2))
        self.ax_height.set_title('Block Height')
        self.ax_height.set_xlabel('Step')
        self.ax_height.set_ylabel('Height (m)')
        
        # Add playback controls
        plt.subplots_adjust(bottom=0.15)
        
        # Slider
        ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Step', 0, self.n_steps-1, 
                            valinit=0, valstep=1, valfmt='%d')
        self.slider.on_changed(self.update_step)
        
        # Play/Pause button
        ax_play = plt.axes([0.85, 0.05, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.toggle_play)
        
        # Initialize plots
        self.update_step(0)
        
    def update_step(self, step):
        """Update visualization for given step."""
        step = int(step)
        self.current_step = step
        
        # Reset environment and set state
        self.env.reset()
        
        # Set joint positions
        for i in range(7):
            joint_addr = self.env.model.jnt_qposadr[self.env.arm_joint_ids[i]]
            self.env.data.qpos[joint_addr] = self.joint_positions[step, i]
        
        # Set gripper
        gripper_ctrl = 0.04 if self.gripper_positions[step] > 0.5 else 0.0
        self.env.data.ctrl[7] = gripper_ctrl
        
        # Forward simulation
        mujoco.mj_forward(self.env.model, self.env.data)
        
        # Update simulation view
        self.ax_sim.clear()
        frame = self.env.render()
        self.ax_sim.imshow(frame)
        self.ax_sim.set_title(f'Step {step}/{self.n_steps-1}')
        self.ax_sim.axis('off')
        
        # Update tactile view
        self.ax_tactile.clear()
        tactile = self.tactile_readings[step].reshape(2, 3, 4, 3)  # 2 fingers, 3x4 taxels, 3 values
        tactile_mag = np.linalg.norm(tactile, axis=-1)  # Magnitude
        
        # Show both fingers
        combined = np.hstack([tactile_mag[0], tactile_mag[1]])
        im = self.ax_tactile.imshow(combined, cmap='hot', aspect='auto')
        self.ax_tactile.set_title(f'Tactile (max: {np.max(tactile_mag):.1f})')
        
        # Update joint positions
        self.ax_joints.clear()
        self.ax_joints.bar(range(7), self.joint_positions[step])
        self.ax_joints.set_ylim(-3.14, 3.14)
        self.ax_joints.set_xlabel('Joint')
        self.ax_joints.set_ylabel('Position (rad)')
        
        # Update reward plot
        self.ax_reward.clear()
        cumulative_rewards = np.cumsum(self.rewards[:step+1])
        self.ax_reward.plot(cumulative_rewards, 'b-')
        self.ax_reward.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        self.ax_reward.set_xlim(0, self.n_steps)
        self.ax_reward.set_xlabel('Step')
        self.ax_reward.set_ylabel('Cumulative Reward')
        self.ax_reward.grid(True)
        
        # Update block height
        self.ax_height.clear()
        block_heights = self.target_positions[:step+1, 2]
        self.ax_height.plot(block_heights, 'g-')
        self.ax_height.axvline(x=step, color='r', linestyle='--', alpha=0.5)
        self.ax_height.axhline(y=self.target_positions[0, 2], color='k', 
                               linestyle=':', alpha=0.5, label='Initial')
        self.ax_height.set_xlim(0, self.n_steps)
        self.ax_height.set_xlabel('Step')
        self.ax_height.set_ylabel('Height (m)')
        self.ax_height.legend()
        self.ax_height.grid(True)
        
        # Add info text
        info_text = f"Action: {self.actions[step]}"
        self.fig.suptitle(f'Demo {self.demo_idx} - Step {step}/{self.n_steps-1}\n{info_text}', 
                          fontsize=12)
        
        plt.draw()
        
    def toggle_play(self, event):
        """Toggle play/pause."""
        self.playing = not self.playing
        self.btn_play.label.set_text('Pause' if self.playing else 'Play')
        
        if self.playing:
            self.play_animation()
    
    def play_animation(self):
        """Play animation."""
        if not self.playing:
            return
            
        # Advance step
        next_step = (self.current_step + 1) % self.n_steps
        self.slider.set_val(next_step)
        
        # Schedule next frame
        self.fig.canvas.draw_idle()
        self.play_timer = self.fig.canvas.new_timer(interval=50)  # 20 FPS
        self.play_timer.add_callback(self.play_animation)
        self.play_timer.start()
        
    def show(self):
        """Show the GUI."""
        plt.show()


def list_demonstrations(dataset_path):
    """List available demonstrations in dataset."""
    print(f"\nChecking demonstrations in: {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as f:
        if 'data' in f:
            # Mimicgen format
            demos = [k for k in f['data'].keys() if k.startswith('demo_')]
        else:
            # Direct format
            demos = [k for k in f.keys() if k.startswith('demo_')]
        
        print(f"Found {len(demos)} demonstrations")
        
        for i, demo in enumerate(demos[:10]):  # Show first 10
            if 'data' in f:
                demo_group = f['data'][demo]
            else:
                demo_group = f[demo]
                
            if 'rewards' in demo_group:
                total_reward = np.sum(demo_group['rewards'][:])
                print(f"  {demo}: {len(demo_group['actions'])} steps, "
                      f"reward: {total_reward:.2f}")
            else:
                print(f"  {demo}: {len(demo_group['actions'])} steps")
                
        if len(demos) > 10:
            print(f"  ... and {len(demos) - 10} more")
            
    return len(demos)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive demo viewer")
    parser.add_argument("--dataset", type=str, 
                       default="../../datasets/expert_demonstrations.hdf5",
                       help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=int, default=0,
                       help="Demo index to view")
    parser.add_argument("--list", action="store_true",
                       help="List available demonstrations")
    
    args = parser.parse_args()
    
    if args.list:
        list_demonstrations(args.dataset)
    else:
        print(f"\n🎮 Interactive Demo Viewer")
        print(f"Dataset: {args.dataset}")
        print(f"Demo: {args.demo}")
        print("\nControls:")
        print("  - Drag slider to navigate through steps")
        print("  - Click Play/Pause to animate")
        print("  - Close window to exit\n")
        
        viewer = DemoViewer(args.dataset, args.demo)
        viewer.show()