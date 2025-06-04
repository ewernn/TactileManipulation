"""
Create final stacking demonstration with blocks in reachable positions.
"""

import numpy as np
import sys
import os
import imageio

sys.path.append('../environments')
from panda_demo_env import PandaDemoEnv


class FinalStackingPolicy:
    """
    Final expert policy for stacking red block on blue block.
    Designed for blocks at reachable positions.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset policy state."""
        self.phase = "approach_red"
        self.step_count = 0
        self.red_grasped = False
        
    def get_action(self, observation):
        """
        Generate expert action for stacking demonstration.
        """
        
        # Get current state
        joint_pos = observation['joint_pos']
        red_pos = observation['target_block_pos']
        blue_pos = observation['block2_pos'] if 'block2_pos' in observation else None
        gripper_opening = observation['gripper_pos']
        tactile_sum = np.sum(observation['tactile'])
        
        action = np.zeros(8)  # 7 joints + gripper
        
        if self.phase == "approach_red":
            # Move towards red block
            if self.step_count < 50:
                # Moderate forward motion
                action = np.array([0.0, 0.25, 0.2, -0.3, 0.15, -0.1, 0.0, -1.0])
            else:
                self.phase = "descend_to_red"
                self.step_count = 0
                
        elif self.phase == "descend_to_red":
            # Lower to grasp height
            if self.step_count < 40:
                # Descend while maintaining position
                action = np.array([0.0, 0.2, 0.25, -0.25, -0.15, -0.05, 0.0, -1.0])
            else:
                self.phase = "grasp_red"
                self.step_count = 0
                
        elif self.phase == "grasp_red":
            # Close gripper on red block
            if self.step_count < 30:
                # Close gripper
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                # Check for contact
                if tactile_sum > 5 and self.step_count > 15:
                    self.red_grasped = True
            else:
                self.phase = "lift_red"
                self.step_count = 0
                
        elif self.phase == "lift_red":
            # Lift red block up
            if self.step_count < 40:
                # Lift up
                action = np.array([0.0, -0.2, -0.2, 0.25, 0.1, 0.0, 0.0, 1.0])
            else:
                self.phase = "move_over_blue"
                self.step_count = 0
                
        elif self.phase == "move_over_blue":
            # Move to position above blue block
            if self.step_count < 50:
                # Rotate base to reach blue block position
                action = np.array([-0.15, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0])
            else:
                self.phase = "position_above_blue"
                self.step_count = 0
                
        elif self.phase == "position_above_blue":
            # Fine positioning
            if self.step_count < 30:
                # Small adjustments
                action = np.array([-0.05, 0.05, 0.05, -0.05, 0.0, 0.05, 0.0, 1.0])
            else:
                self.phase = "stack_on_blue"
                self.step_count = 0
                
        elif self.phase == "stack_on_blue":
            # Lower red block onto blue block
            if self.step_count < 40:
                # Controlled descent
                action = np.array([0.0, 0.15, 0.15, -0.15, -0.08, 0.0, 0.0, 1.0])
            else:
                self.phase = "release"
                self.step_count = 0
                
        elif self.phase == "release":
            # Open gripper to release red block
            if self.step_count < 20:
                # Open gripper
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            else:
                self.phase = "retreat"
                self.step_count = 0
                
        elif self.phase == "retreat":
            # Move away from stacked blocks
            if self.step_count < 30:
                # Move up and back
                action = np.array([0.0, -0.15, -0.15, 0.15, 0.05, -0.05, 0.0, -1.0])
            else:
                self.phase = "done"
                self.step_count = 0
                
        else:  # done
            # Hold position
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            
        self.step_count += 1
        return action


def create_final_stacking_video():
    """Create the final stacking demonstration video."""
    
    print("🎬 Creating Final Block Stacking Video...")
    print("📍 Blocks positioned within robot reach")
    
    env = PandaDemoEnv()
    expert = FinalStackingPolicy()
    
    all_frames = []
    success_count = 0
    
    # Create 3 demonstration episodes with different camera angles
    cameras = ["demo_cam", "side_cam", "overhead_cam"]
    
    for episode in range(3):
        camera = cameras[episode % len(cameras)]
        print(f"\n📹 Episode {episode + 1}/3 - Camera: {camera}")
        
        # Reset everything
        obs = env.reset(randomize=(episode > 0))  # First episode without randomization
        expert.reset()
        
        print(f"  Red block: {obs['target_block_pos']}")
        print(f"  Blue block: {obs['block2_pos']}")
        
        frames = []
        phase_log = []
        
        # Run demonstration
        for step in range(350):
            # Get expert action
            action = expert.get_action(obs)
            
            # Track phase changes
            if len(phase_log) == 0 or expert.phase != phase_log[-1][1]:
                phase_log.append((step, expert.phase))
                print(f"  Step {step:3d}: → {expert.phase}")
            
            # Execute action
            obs = env.step(action, steps=4)
            
            # Capture frame
            frame = env.render(camera=camera)
            frames.append(frame)
            
            # Monitor progress
            if step % 50 == 0 and step > 0:
                red_pos = obs['target_block_pos']
                blue_pos = obs['block2_pos']
                print(f"  Step {step}: Red height={red_pos[2]:.3f}, "
                      f"Distance to blue={np.linalg.norm(red_pos[:2] - blue_pos[:2]):.3f}")
            
            # Check success
            red_pos = obs['target_block_pos']
            blue_pos = obs['block2_pos']
            height_diff = red_pos[2] - blue_pos[2]
            horizontal_dist = np.linalg.norm(red_pos[:2] - blue_pos[:2])
            
            if height_diff > 0.035 and horizontal_dist < 0.04:
                print(f"  ✅ SUCCESS! Stacking achieved at step {step}")
                print(f"     Height difference: {height_diff:.3f}m")
                print(f"     Horizontal distance: {horizontal_dist:.3f}m")
                success_count += 1
                
                # Hold success pose
                for _ in range(40):
                    obs = env.step(action, steps=1)
                    frames.append(env.render(camera=camera))
                break
            
            # Early termination
            if expert.phase == "done" and expert.step_count > 20:
                print(f"  ⏹️  Demo complete at step {step}")
                break
        
        # Add frames to collection
        all_frames.extend(frames)
        
        # Add transition between episodes
        if episode < 2:
            # Fade effect
            for i in range(15):
                all_frames.append(frames[-1])
    
    # Save video
    output_path = "../../videos/panda_block_stacking_demo.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n💾 Saving video with {len(all_frames)} frames...")
    
    imageio.mimsave(
        output_path,
        all_frames,
        fps=30,
        quality=9,
        macro_block_size=1
    )
    
    print(f"\n✅ Final stacking video created!")
    print(f"   📹 Location: {output_path}")
    print(f"   ⏱️  Duration: {len(all_frames)/30:.1f} seconds")
    print(f"   🎯 Success rate: {success_count}/3 episodes")
    print(f"   📷 Multiple camera angles included")
    
    return output_path


if __name__ == "__main__":
    create_final_stacking_video()