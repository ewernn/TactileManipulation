#!/usr/bin/env python3
"""
Diagnostic tool to visualize and tune expert policy phases with new 110° configuration
"""

import numpy as np
import sys
import os
import mujoco
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.append('../environments')
from panda_demo_env import PandaDemoEnv

class DiagnosticExpertPolicy:
    """
    Expert policy with phase visualization and tuning capabilities
    """
    
    def __init__(self, phase_actions=None):
        self.phase = "approach"
        self.step_count = 0
        self.target_reached = False
        
        # Default actions - these need tuning for 110° wrist
        if phase_actions is None:
            self.phase_actions = {
                "approach": {
                    "action": np.array([0.15, 0.25, 0.15, -0.3, 0.1, 0.1, 0.0, -1.0]),
                    "duration": 40,
                    "description": "Rotate base and move toward target"
                },
                "pre_grasp": {
                    "action": np.array([0.05, 0.2, 0.2, -0.25, 0.0, 0.1, 0.0, -1.0]),
                    "duration": 30,
                    "description": "Position above block"
                },
                "descend": {
                    "action": np.array([0.0, 0.15, 0.2, -0.1, -0.1, 0.0, 0.0, -1.0]),
                    "duration": 25,
                    "description": "Lower to block level"
                },
                "grasp": {
                    "action": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                    "duration": 20,
                    "description": "Close gripper"
                },
                "lift": {
                    "action": np.array([-0.05, -0.15, -0.15, 0.2, 0.0, 0.0, 0.0, 1.0]),
                    "duration": 35,
                    "description": "Lift block up"
                },
                "done": {
                    "action": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                    "duration": 20,
                    "description": "Hold position"
                }
            }
        else:
            self.phase_actions = phase_actions
        
        self.phases = ["approach", "pre_grasp", "descend", "grasp", "lift", "done"]
        self.phase_history = []
        
    def reset(self):
        """Reset policy state."""
        self.phase = "approach"
        self.step_count = 0
        self.target_reached = False
        self.phase_history = []
    
    def get_action(self, observation):
        """Get action for current phase"""
        current_phase_info = self.phase_actions[self.phase]
        action = current_phase_info["action"].copy()
        
        # Record phase transition
        if self.step_count == 0:
            self.phase_history.append({
                "phase": self.phase,
                "start_step": len(self.phase_history)
            })
        
        # Check if phase should transition
        if self.step_count >= current_phase_info["duration"]:
            current_idx = self.phases.index(self.phase)
            if current_idx < len(self.phases) - 1:
                self.phase = self.phases[current_idx + 1]
                self.step_count = 0
            else:
                self.phase = "done"
        else:
            self.step_count += 1
            
        return action

def visualize_phase(env, phase_name, phase_info, save_path):
    """Visualize a single phase of the expert policy"""
    obs = env.reset(randomize=False)  # Fixed position for comparison
    
    frames = []
    positions = []
    
    # Execute phase
    for step in range(phase_info["duration"]):
        action = phase_info["action"]
        obs = env.step(action, steps=1)
        
        # Record end-effector position
        hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ee_pos = env.data.xpos[hand_id].copy()
        positions.append(ee_pos)
        
        # Capture frames at key points
        if step in [0, phase_info["duration"]//2, phase_info["duration"]-1]:
            frame = env.render()
            frames.append(frame)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Show key frames
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].set_title(f"Step {[0, phase_info['duration']//2, phase_info['duration']-1][i]}")
        axes[i].axis('off')
    
    # Plot trajectory
    positions = np.array(positions)
    axes[3].plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2)
    axes[3].scatter(positions[0, 0], positions[0, 2], c='green', s=100, label='Start')
    axes[3].scatter(positions[-1, 0], positions[-1, 2], c='red', s=100, label='End')
    axes[3].set_xlabel('X Position (forward)')
    axes[3].set_ylabel('Z Position (up)')
    axes[3].set_title('End-Effector Trajectory')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.suptitle(f"Phase: {phase_name} - {phase_info['description']}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return positions[-1]  # Return final position

def test_full_sequence(env, policy, save_video=True):
    """Test the complete expert sequence and analyze results"""
    obs = env.reset(randomize=False)
    policy.reset()
    
    frames = []
    data = {
        'positions': [],
        'gripper_states': [],
        'tactile_sum': [],
        'block_heights': [],
        'actions': []
    }
    
    print("\nExecuting full expert sequence:")
    print("=" * 50)
    
    for step in range(150):
        action = policy.get_action(obs)
        data['actions'].append(action.copy())
        
        # Execute
        obs = env.step(action, steps=3)
        
        # Record data
        # Calculate end-effector position from forward kinematics
        hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ee_pos = env.data.xpos[hand_id].copy()
        data['positions'].append(ee_pos)
        data['gripper_states'].append(obs['gripper_pos'])
        data['tactile_sum'].append(np.sum(obs['tactile']))
        data['block_heights'].append(obs['target_block_pos'][2])
        
        if save_video:
            frames.append(env.render())
        
        # Print phase transitions
        if len(policy.phase_history) > 0 and policy.phase_history[-1]['start_step'] == step:
            print(f"Step {step:3d}: Entering phase '{policy.phase}' - {policy.phase_actions[policy.phase]['description']}")
    
    # Analyze results
    initial_height = 0.44
    final_height = data['block_heights'][-1]
    max_height = max(data['block_heights'])
    success = max_height > initial_height + 0.005
    
    print("\nResults:")
    print(f"  Initial block height: {initial_height:.3f}")
    print(f"  Final block height: {final_height:.3f}")
    print(f"  Max block height: {max_height:.3f}")
    print(f"  Lift amount: {max_height - initial_height:.3f}m")
    print(f"  Success: {'YES' if success else 'NO'}")
    print(f"  Max tactile contact: {max(data['tactile_sum']):.1f}")
    
    return data, frames, success

def create_tuning_interface():
    """Create an interactive tuning interface for adjusting actions"""
    print("\n🔧 Expert Policy Tuning Interface")
    print("=" * 50)
    print("Current configuration: Joint 6 = 110° (1.920 rad)")
    print("\nPhase action format: [J1, J2, J3, J4, J5, J6, J7, Gripper]")
    print("Joint velocities range: -1.0 to 1.0")
    print("Gripper: -1.0 = open, 1.0 = close")
    print("\nPress Enter to test current values, or type 'edit' to modify")
    
    return input("\nChoice: ").strip().lower() == 'edit'

def main():
    print("🔍 Expert Policy Diagnostic Tool")
    print("=" * 50)
    
    # Create environment
    env = PandaDemoEnv()
    policy = DiagnosticExpertPolicy()
    
    # Create output directory
    output_dir = "expert_policy_diagnostics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test current policy
    print("\n1. Testing current expert policy with 110° wrist configuration...")
    data, frames, success = test_full_sequence(env, policy, save_video=True)
    
    # Save diagnostic video
    if frames:
        import imageio
        video_path = os.path.join(output_dir, "current_policy_test.mp4")
        imageio.mimsave(video_path, frames, fps=30)
        print(f"\n📹 Saved test video to: {video_path}")
    
    # Visualize each phase
    print("\n2. Analyzing individual phases...")
    for phase_name in ["approach", "pre_grasp", "descend", "grasp", "lift"]:
        if phase_name in policy.phase_actions:
            phase_info = policy.phase_actions[phase_name]
            save_path = os.path.join(output_dir, f"phase_{phase_name}.png")
            final_pos = visualize_phase(env, phase_name, phase_info, save_path)
            print(f"   - {phase_name}: Final position = [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
    
    # Plot full trajectory
    positions = np.array(data['positions'])
    block_heights = np.array(data['block_heights'])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # XZ trajectory
    ax1.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 2], c='green', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 2], c='red', s=100, label='End')
    ax1.axhline(y=0.44, color='gray', linestyle='--', label='Table height')
    ax1.set_xlabel('X Position (forward)')
    ax1.set_ylabel('Z Position (up)')
    ax1.set_title('End-Effector Trajectory (Side View)')
    ax1.legend()
    ax1.grid(True)
    
    # Block height over time
    ax2.plot(block_heights, 'g-', linewidth=2)
    ax2.axhline(y=0.44, color='gray', linestyle='--', label='Initial height')
    ax2.axhline(y=0.445, color='orange', linestyle='--', label='Success threshold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Block Height (m)')
    ax2.set_title('Block Height During Manipulation')
    ax2.legend()
    ax2.grid(True)
    
    # Tactile and gripper
    ax3.plot(data['tactile_sum'], 'r-', label='Tactile sum', alpha=0.7)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(data['gripper_states'], 'b-', label='Gripper state', alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Tactile Contact', color='r')
    ax3_twin.set_ylabel('Gripper Opening', color='b')
    ax3.set_title('Tactile Sensing and Gripper State')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "full_analysis.png"))
    plt.close()
    
    print(f"\n📊 Saved analysis plots to: {output_dir}/")
    
    # Suggest improvements
    print("\n3. Suggested adjustments for 110° wrist angle:")
    print("   - May need to adjust Joint 6 velocities in approach/pre_grasp phases")
    print("   - Descend phase might need different Joint 4/5 coordination")
    print("   - Consider slower gripper closure for better grasp stability")
    
    if success:
        print("\n✅ Current policy works with 110° configuration!")
    else:
        print("\n⚠️  Current policy needs adjustment for 110° configuration")
        print("   Run with --tune flag to interactively adjust phase actions")

if __name__ == "__main__":
    main()