#!/usr/bin/env python3
"""
Systematic tuning of expert policy for 110° configuration
Tests different action parameters and visualizes results
"""

import numpy as np
import sys
import os
import mujoco
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append('../environments')
from panda_demo_env import PandaDemoEnv

class SystematicExpertTuner:
    """Test different parameters systematically"""
    
    def __init__(self, base_actions=None):
        self.phase = "approach"
        self.step_count = 0
        
        # Base actions that we'll modify
        if base_actions is None:
            self.base_actions = {
                "approach": {
                    "action": np.array([0.3, 0.4, 0.3, -0.4, 0.2, -0.1, 0.0, -1.0]),
                    "duration": 50
                },
                "pre_grasp": {
                    "action": np.array([0.2, 0.5, 0.4, -0.35, 0.1, -0.15, 0.0, -1.0]),
                    "duration": 40
                },
                "descend": {
                    "action": np.array([0.05, 0.3, 0.35, -0.25, -0.2, -0.1, 0.0, -1.0]),
                    "duration": 35
                },
                "grasp": {
                    "action": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                    "duration": 25
                },
                "lift": {
                    "action": np.array([-0.08, -0.2, -0.2, 0.25, 0.05, 0.02, 0.0, 1.0]),
                    "duration": 40
                }
            }
        else:
            self.base_actions = base_actions
            
        self.phases = ["approach", "pre_grasp", "descend", "grasp", "lift"]
        
    def reset(self):
        self.phase = "approach"
        self.step_count = 0
        
    def get_action(self, observation):
        if self.phase not in self.base_actions:
            return np.zeros(8)
            
        action = self.base_actions[self.phase]["action"].copy()
        
        if self.step_count >= self.base_actions[self.phase]["duration"]:
            # Move to next phase
            current_idx = self.phases.index(self.phase)
            if current_idx < len(self.phases) - 1:
                self.phase = self.phases[current_idx + 1]
                self.step_count = 0
            else:
                self.phase = "done"
        else:
            self.step_count += 1
            
        return action

def test_configuration(env, tuner, max_steps=200):
    """Test a configuration and return metrics"""
    obs = env.reset(randomize=False)
    tuner.reset()
    
    metrics = {
        'positions': [],
        'block_heights': [],
        'tactile_sum': [],
        'gripper_states': [],
        'success': False,
        'max_tactile': 0,
        'max_lift': 0,
        'first_contact_step': -1,
        'grasp_quality': 0
    }
    
    initial_block_height = obs['target_block_pos'][2]
    
    for step in range(max_steps):
        action = tuner.get_action(obs)
        obs = env.step(action, steps=3)
        
        # Record data
        hand_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        hand_pos = env.data.xpos[hand_id]
        
        metrics['positions'].append(hand_pos.copy())
        metrics['block_heights'].append(obs['target_block_pos'][2])
        metrics['tactile_sum'].append(np.sum(obs['tactile']))
        metrics['gripper_states'].append(obs['gripper_pos'])
        
        # Track first contact
        if metrics['tactile_sum'][-1] > 10 and metrics['first_contact_step'] == -1:
            metrics['first_contact_step'] = step
            
        # Check if lifting
        current_lift = obs['target_block_pos'][2] - initial_block_height
        metrics['max_lift'] = max(metrics['max_lift'], current_lift)
        
        if tuner.phase == "done":
            break
    
    # Calculate metrics
    metrics['max_tactile'] = max(metrics['tactile_sum'])
    metrics['success'] = metrics['max_lift'] > 0.005
    
    # Grasp quality: combination of tactile contact during grasp and successful lift
    if metrics['max_lift'] > 0:
        grasp_tactile = np.mean(metrics['tactile_sum'][-40:])  # Average during lift
        metrics['grasp_quality'] = min(1.0, grasp_tactile / 100) * metrics['max_lift'] * 200
    
    return metrics

def visualize_test_results(metrics, title="Test Results"):
    """Create comprehensive visualization of test results"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. XZ Trajectory
    ax1 = fig.add_subplot(gs[0, :2])
    positions = np.array(metrics['positions'])
    block_heights = np.array(metrics['block_heights'])
    
    ax1.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Hand trajectory')
    ax1.scatter(positions[0, 0], positions[0, 2], c='green', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 2], c='red', s=100, marker='o', label='End')
    
    # Mark first contact
    if metrics['first_contact_step'] > 0:
        contact_pos = positions[metrics['first_contact_step']]
        ax1.scatter(contact_pos[0], contact_pos[2], c='orange', s=150, marker='*', label='First contact')
    
    ax1.axhline(y=0.44, color='gray', linestyle='--', alpha=0.5, label='Block height')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Z Position (m)')
    ax1.set_title('End-Effector Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Metrics summary
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    metrics_text = f"""Test Results:
    
Success: {'YES' if metrics['success'] else 'NO'}
Max Lift: {metrics['max_lift']*1000:.1f} mm
Max Tactile: {metrics['max_tactile']:.1f}
First Contact: Step {metrics['first_contact_step']}
Grasp Quality: {metrics['grasp_quality']:.1f}
    """
    ax2.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    
    # 3. Block height over time
    ax3 = fig.add_subplot(gs[1, :])
    steps = np.arange(len(block_heights))
    ax3.plot(steps, block_heights, 'g-', linewidth=2)
    ax3.axhline(y=0.44, color='gray', linestyle='--', label='Initial height')
    ax3.axhline(y=0.445, color='orange', linestyle='--', label='Success threshold')
    ax3.fill_between(steps, 0.44, block_heights, where=(block_heights > 0.44), 
                     alpha=0.3, color='green', label='Lifted')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Block Height (m)')
    ax3.set_title('Block Height During Manipulation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Tactile and gripper
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(metrics['tactile_sum'], 'r-', linewidth=2, label='Tactile sum')
    ax4.set_ylabel('Tactile Contact', color='r')
    ax4.tick_params(axis='y', labelcolor='r')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(metrics['gripper_states'], 'b-', linewidth=2, label='Gripper')
    ax4_twin.set_ylabel('Gripper State', color='b')
    ax4_twin.tick_params(axis='y', labelcolor='b')
    
    ax4.set_xlabel('Step')
    ax4.set_title('Tactile Sensing and Gripper State')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def test_parameter_variations(env):
    """Test variations of key parameters"""
    base_tuner = SystematicExpertTuner()
    
    # Test variations
    variations = []
    
    # 1. Original baseline
    print("\n1. Testing baseline configuration...")
    metrics = test_configuration(env, base_tuner)
    variations.append(("Baseline", metrics))
    
    # 2. Slower, more careful approach
    print("\n2. Testing slower approach...")
    slow_actions = base_tuner.base_actions.copy()
    for phase in ["approach", "pre_grasp", "descend"]:
        slow_actions[phase] = {
            "action": slow_actions[phase]["action"] * 0.7,  # Slower
            "duration": int(slow_actions[phase]["duration"] * 1.3)  # Longer
        }
    slow_tuner = SystematicExpertTuner(slow_actions)
    metrics = test_configuration(env, slow_tuner)
    variations.append(("Slower Approach", metrics))
    
    # 3. More aggressive descent
    print("\n3. Testing aggressive descent...")
    aggressive_actions = base_tuner.base_actions.copy()
    aggressive_actions["descend"]["action"] = np.array([0.02, 0.4, 0.5, -0.3, -0.3, -0.15, 0.0, -1.0])
    aggressive_tuner = SystematicExpertTuner(aggressive_actions)
    metrics = test_configuration(env, aggressive_tuner)
    variations.append(("Aggressive Descent", metrics))
    
    # 4. Fine-tuned grasp
    print("\n4. Testing fine-tuned grasp...")
    grasp_actions = base_tuner.base_actions.copy()
    # Small movements during grasp to ensure good contact
    grasp_actions["grasp"]["action"] = np.array([0.01, 0.02, 0.02, -0.02, -0.01, 0.0, 0.0, 1.0])
    grasp_actions["grasp"]["duration"] = 30
    grasp_tuner = SystematicExpertTuner(grasp_actions)
    metrics = test_configuration(env, grasp_tuner)
    variations.append(("Fine Grasp", metrics))
    
    return variations

def main():
    print("🔧 Systematic Expert Policy Tuning for 110° Configuration")
    print("=" * 60)
    
    # Create environment
    env = PandaDemoEnv()
    
    # Create output directory
    output_dir = "expert_tuning_110deg"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test parameter variations
    variations = test_parameter_variations(env)
    
    # Visualize results
    print("\n📊 Creating visualizations...")
    for i, (name, metrics) in enumerate(variations):
        fig = visualize_test_results(metrics, f"{name} Configuration")
        fig.savefig(os.path.join(output_dir, f"test_{i}_{name.lower().replace(' ', '_')}.png"))
        plt.close(fig)
    
    # Summary
    print("\n📈 Summary of Results:")
    print("-" * 60)
    print(f"{'Configuration':<20} {'Success':<10} {'Max Lift (mm)':<15} {'Max Tactile':<15} {'Quality':<10}")
    print("-" * 60)
    
    best_config = None
    best_quality = -1
    
    for name, metrics in variations:
        success_str = "YES" if metrics['success'] else "NO"
        lift_mm = metrics['max_lift'] * 1000
        print(f"{name:<20} {success_str:<10} {lift_mm:<15.1f} {metrics['max_tactile']:<15.1f} {metrics['grasp_quality']:<10.1f}")
        
        if metrics['grasp_quality'] > best_quality:
            best_quality = metrics['grasp_quality']
            best_config = name
    
    print("\n🏆 Best configuration:", best_config)
    print(f"   Grasp quality score: {best_quality:.1f}")
    
    # Generate recommended actions
    print("\n💡 Recommendations:")
    print("1. The 110° wrist angle requires strong forward motion in approach phase")
    print("2. Descent phase is critical - needs careful balance of forward/down")
    print("3. Small adjustments during grasp improve contact quality")
    print("4. Consider implementing tactile-based feedback for grasp refinement")

if __name__ == "__main__":
    main()