"""
Test the updated stacking demonstration.
"""

import numpy as np
import sys
import os
import imageio

sys.path.append('../environments')
from panda_demo_env import PandaDemoEnv
from create_expert_demos import ExpertPolicy

def test_stacking():
    """Test the stacking behavior."""
    print("🧪 Testing Block Stacking Demo...")
    
    # Create environment and expert
    env = PandaDemoEnv()
    expert = ExpertPolicy()
    
    # Reset environment
    obs = env.reset(randomize=False)
    expert.reset()
    
    print(f"Initial red block pos: {obs['target_block_pos']}")
    print(f"Initial blue block pos: {obs['block2_pos']}")
    
    frames = []
    phase_changes = []
    
    # Run demonstration
    for step in range(300):
        # Get expert action
        action = expert.get_action(obs)
        
        # Track phase changes
        if len(phase_changes) == 0 or expert.phase != phase_changes[-1][1]:
            phase_changes.append((step, expert.phase))
            print(f"Step {step}: Phase changed to '{expert.phase}'")
        
        # Execute action
        obs = env.step(action, steps=3)
        
        # Capture frame
        frame = env.render()
        frames.append(frame)
        
        # Check if stacking is successful
        red_pos = obs['target_block_pos']
        blue_pos = obs['block2_pos']
        
        if step % 50 == 0:
            print(f"Step {step}:")
            print(f"  Red block: {red_pos}")
            print(f"  Blue block: {blue_pos}")
            print(f"  Distance: {np.linalg.norm(red_pos[:2] - blue_pos[:2]):.3f}")
            print(f"  Height diff: {red_pos[2] - blue_pos[2]:.3f}")
        
        # Check success
        if (red_pos[2] > blue_pos[2] + 0.03 and 
            np.linalg.norm(red_pos[:2] - blue_pos[:2]) < 0.05):
            print(f"✅ SUCCESS! Red block stacked on blue block at step {step}")
            break
        
        # Early termination if done
        if expert.phase == "done" and expert.step_count > 20:
            break
    
    # Save test video
    print(f"💾 Saving test video...")
    output_path = "test_stacking_demo.mp4"
    imageio.mimsave(
        output_path,
        frames,
        fps=30,
        quality=8,
        macro_block_size=1
    )
    
    print(f"✅ Test complete! Video saved to: {output_path}")
    print(f"Total frames: {len(frames)}")
    print("\nPhase timeline:")
    for step, phase in phase_changes:
        print(f"  Step {step:3d}: {phase}")

if __name__ == "__main__":
    test_stacking()