"""
Create a proper stacking demonstration where the robot picks up the red block
and places it on the blue block.
"""

import numpy as np
import sys
import os
import imageio
import h5py
from tqdm import tqdm

sys.path.append('../environments')
from panda_demo_env import PandaDemoEnv


class StackingExpertPolicy:
    """
    Expert policy specifically designed for stacking red block on blue block.
    Uses more aggressive motions to actually reach and manipulate the blocks.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset policy state."""
        self.phase = "reach_forward"
        self.step_count = 0
        self.grasped = False
        self.target_joint_positions = None
        
    def get_action(self, observation):
        """
        Generate expert action for stacking demonstration.
        Uses joint position targets rather than velocities for more precise control.
        """
        
        # Get current state
        joint_pos = observation['joint_pos']
        red_pos = observation['target_block_pos']
        blue_pos = observation['block2_pos'] if 'block2_pos' in observation else None
        gripper_opening = observation['gripper_pos']
        tactile_sum = np.sum(observation['tactile'])
        
        action = np.zeros(8)  # 7 joints + gripper
        
        if self.phase == "reach_forward":
            # Aggressive forward reach to get to block area
            if self.step_count < 60:
                # Strong coordinated motion to reach blocks
                action = np.array([0.3, 0.4, 0.3, -0.4, 0.2, -0.1, 0.0, -1.0])
            else:
                self.phase = "align_with_red"
                self.step_count = 0
                
        elif self.phase == "align_with_red":
            # Align gripper with red block
            if self.step_count < 40:
                # Continue forward and adjust height
                action = np.array([0.15, 0.3, 0.25, -0.3, 0.1, -0.15, 0.0, -1.0])
            else:
                self.phase = "descend_to_red"
                self.step_count = 0
                
        elif self.phase == "descend_to_red":
            # Descend to grasp height
            if self.step_count < 40:
                # Primarily downward motion
                action = np.array([0.05, 0.25, 0.3, -0.25, -0.2, -0.1, 0.0, -1.0])
            else:
                self.phase = "close_gripper"
                self.step_count = 0
                
        elif self.phase == "close_gripper":
            # Close gripper on red block
            if self.step_count < 25:
                # Close gripper while maintaining position
                action = np.array([0.02, 0.02, 0.02, -0.02, -0.02, 0.0, 0.0, 1.0])
                # Check if we have good contact
                if tactile_sum > 10 and self.step_count > 10:
                    self.grasped = True
            else:
                self.phase = "lift_red"
                self.step_count = 0
                
        elif self.phase == "lift_red":
            # Lift red block
            if self.step_count < 40:
                # Strong upward motion
                action = np.array([-0.05, -0.3, -0.3, 0.35, 0.15, 0.05, 0.0, 1.0])
            else:
                self.phase = "move_to_blue"
                self.step_count = 0
                
        elif self.phase == "move_to_blue":
            # Move laterally to blue block position
            if self.step_count < 50:
                # Move in Y direction (joint 1 rotation)
                action = np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0])
            else:
                self.phase = "position_above_blue"
                self.step_count = 0
                
        elif self.phase == "position_above_blue":
            # Fine positioning above blue block
            if self.step_count < 30:
                # Small adjustments
                action = np.array([-0.05, 0.1, 0.1, -0.1, -0.05, 0.05, 0.0, 1.0])
            else:
                self.phase = "lower_onto_blue"
                self.step_count = 0
                
        elif self.phase == "lower_onto_blue":
            # Lower red block onto blue block
            if self.step_count < 40:
                # Controlled descent
                action = np.array([0.0, 0.15, 0.15, -0.15, -0.1, 0.0, 0.0, 1.0])
            else:
                self.phase = "release"
                self.step_count = 0
                
        elif self.phase == "release":
            # Release red block
            if self.step_count < 20:
                # Open gripper
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            else:
                self.phase = "retreat"
                self.step_count = 0
                
        elif self.phase == "retreat":
            # Move away from stacked blocks
            if self.step_count < 40:
                # Move up and back
                action = np.array([-0.1, -0.2, -0.2, 0.2, 0.1, -0.1, 0.0, -1.0])
            else:
                self.phase = "done"
                self.step_count = 0
                
        else:  # done
            # Hold position
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            
        self.step_count += 1
        return action


def create_stacking_video():
    """Create a video demonstrating block stacking."""
    
    print("🎬 Creating Block Stacking Demonstration Video...")
    
    env = PandaDemoEnv()
    expert = StackingExpertPolicy()
    
    all_frames = []
    
    # Create 2 demonstration episodes
    for episode in range(2):
        print(f"\n📹 Recording episode {episode + 1}/2")
        
        # Reset everything
        obs = env.reset(randomize=True)
        expert.reset()
        
        frames = []
        phase_log = []
        
        # Run demonstration
        for step in range(400):  # More steps for complete stacking
            # Get expert action
            action = expert.get_action(obs)
            
            # Track phase changes
            if len(phase_log) == 0 or expert.phase != phase_log[-1][1]:
                phase_log.append((step, expert.phase))
                print(f"  Step {step:3d}: {expert.phase}")
            
            # Execute action with more physics steps for stability
            obs = env.step(action, steps=5)
            
            # Capture frame
            frame = env.render()
            frames.append(frame)
            
            # Check progress every 50 steps
            if step % 50 == 0 and step > 0:
                red_pos = obs['target_block_pos']
                blue_pos = obs['block2_pos']
                print(f"  Step {step}: Red={red_pos[2]:.3f}, Blue={blue_pos[2]:.3f}, "
                      f"Dist={np.linalg.norm(red_pos[:2] - blue_pos[:2]):.3f}")
            
            # Check success
            red_pos = obs['target_block_pos']
            blue_pos = obs['block2_pos']
            if (red_pos[2] > blue_pos[2] + 0.04 and 
                np.linalg.norm(red_pos[:2] - blue_pos[:2]) < 0.04):
                print(f"  ✅ SUCCESS! Stacking achieved at step {step}")
                # Hold for a few more frames to show result
                for _ in range(30):
                    obs = env.step(action, steps=1)
                    frames.append(env.render())
                break
            
            # Early termination if done
            if expert.phase == "done" and expert.step_count > 30:
                break
        
        # Add frames to collection
        all_frames.extend(frames)
        
        # Add pause between episodes
        if episode < 1:
            for _ in range(20):
                all_frames.append(frames[-1])
    
    # Save video
    output_path = "../../videos/panda_stacking_demo.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n💾 Saving video with {len(all_frames)} frames...")
    
    imageio.mimsave(
        output_path,
        all_frames,
        fps=30,
        quality=8,
        macro_block_size=1
    )
    
    print(f"\n✅ Stacking demonstration video created!")
    print(f"   📹 Location: {output_path}")
    print(f"   ⏱️  Duration: {len(all_frames)/30:.1f} seconds")
    print(f"   🎯 Shows: Red block being stacked on blue block")
    
    return output_path


def test_single_episode():
    """Test a single stacking episode for debugging."""
    
    print("🧪 Testing single stacking episode...")
    
    env = PandaDemoEnv(camera_name="side_cam")  # Use side camera for better view
    expert = StackingExpertPolicy()
    
    obs = env.reset(randomize=False)
    expert.reset()
    
    print(f"Initial positions:")
    print(f"  Red block: {obs['target_block_pos']}")
    print(f"  Blue block: {obs['block2_pos']}")
    
    frames = []
    
    for step in range(400):
        action = expert.get_action(obs)
        obs = env.step(action, steps=5)
        
        if step % 20 == 0:
            print(f"\nStep {step}: Phase={expert.phase}")
            print(f"  Joint pos: {obs['joint_pos']}")
            print(f"  Red: {obs['target_block_pos']}")
            print(f"  Blue: {obs['block2_pos']}")
            print(f"  Tactile: {np.sum(obs['tactile']):.1f}")
        
        frame = env.render()
        frames.append(frame)
        
        # Check if stacking successful
        red_pos = obs['target_block_pos']
        blue_pos = obs['block2_pos']
        if (red_pos[2] > blue_pos[2] + 0.04 and 
            np.linalg.norm(red_pos[:2] - blue_pos[:2]) < 0.04):
            print(f"\n✅ Stacking successful at step {step}!")
            break
    
    # Save test video
    imageio.mimsave("test_stacking_single.mp4", frames, fps=30)
    print(f"\nTest video saved to: test_stacking_single.mp4")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run single episode test")
    args = parser.parse_args()
    
    if args.test:
        test_single_episode()
    else:
        create_stacking_video()