"""
Generate a video of the tactile grasping demonstration.
"""

import numpy as np
import mujoco
import imageio
import os
from tqdm import tqdm

def generate_simple_robot_video():
    """Generate video with the working simple robot."""
    
    # Load model
    model_path = "../franka_emika_panda/panda_tactile_grasp.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Get IDs
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    
    # Reset
    mujoco.mj_resetData(model, data)
    
    # Create renderer with supported size
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = []
    
    print("Recording grasping sequence...")
    
    # Phase 1: Initial view (0.5 seconds at 30fps)
    for _ in range(15):
        renderer.update_scene(data)
        frames.append(renderer.render())
        mujoco.mj_step(model, data)
    
    # Phase 2: Position and open gripper
    data.ctrl[0] = -0.2  # Lower arm
    data.ctrl[3] = 0.04  # Open gripper
    data.ctrl[4] = 0.04
    
    for i in tqdm(range(100), desc="Positioning"):
        mujoco.mj_step(model, data)
        if i % 3 == 0:  # Record every 3rd frame for 30fps
            renderer.update_scene(data)
            frames.append(renderer.render())
    
    # Phase 3: Lower more
    data.ctrl[0] = -0.28
    for i in tqdm(range(50), desc="Lowering"):
        mujoco.mj_step(model, data)
        if i % 3 == 0:
            renderer.update_scene(data)
            frames.append(renderer.render())
    
    # Phase 4: Close gripper
    data.ctrl[3] = 0.0
    data.ctrl[4] = 0.0
    for i in tqdm(range(50), desc="Grasping"):
        mujoco.mj_step(model, data)
        if i % 3 == 0:
            renderer.update_scene(data)
            frames.append(renderer.render())
    
    # Phase 5: Lift
    initial_cube_z = data.xpos[cube_body_id][2]
    data.ctrl[0] = 0.2
    
    for i in tqdm(range(150), desc="Lifting"):
        mujoco.mj_step(model, data)
        if i % 3 == 0:
            renderer.update_scene(data)
            
            # Add text overlay showing lift height
            frame = renderer.render()
            current_height = data.xpos[cube_body_id][2]
            lift_height = current_height - initial_cube_z
            
            # Simple text overlay (would need PIL for better text)
            frames.append(frame)
    
    # Phase 6: Hold position (1 second)
    for i in range(30):
        renderer.update_scene(data)
        frames.append(renderer.render())
    
    # Save video
    output_path = "../../datasets/tactile_grasping/tactile_grasp_demo.mp4"
    print(f"\nSaving video with {len(frames)} frames...")
    imageio.mimsave(output_path, frames, fps=30, quality=8, codec='libx264')
    print(f"Video saved to: {output_path}")
    
    # Also save as GIF for easy viewing
    gif_path = "../../datasets/tactile_grasping/tactile_grasp_demo.gif"
    # Downsample for smaller GIF
    gif_frames = frames[::3]  # Every 3rd frame
    imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
    print(f"GIF saved to: {gif_path}")
    
    return len(frames)


def attempt_panda_7dof_video():
    """Try to generate video with full 7-DOF Panda (if we can get it working)."""
    
    print("\nAttempting 7-DOF Panda video...")
    print("This would require:")
    print("1. Proper inverse kinematics for arm positioning")
    print("2. Better trajectory planning for smooth motion")
    print("3. Fixing the collision issues in the original Panda XML")
    print("\nFor now, using the simplified robot that works reliably.")
    
    # TODO: Implement 7-DOF version if time permits
    pass


if __name__ == "__main__":
    # Generate video with working robot
    num_frames = generate_simple_robot_video()
    
    print(f"\n✅ Video generation complete!")
    print(f"   Total frames: {num_frames}")
    print(f"   Duration: ~{num_frames/30:.1f} seconds")
    
    # Note about 7-DOF
    print("\n📝 Note: This uses the simplified 3-DOF robot for reliability.")
    print("   A 7-DOF Panda implementation would be more impressive but requires:")
    print("   - Inverse kinematics solver")
    print("   - Motion planning")
    print("   - Fixing the original collision/control issues")