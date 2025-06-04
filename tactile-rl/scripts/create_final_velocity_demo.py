#!/usr/bin/env python3
"""
Final velocity control demonstration.
Simple and working.
"""

import numpy as np
import mujoco
import cv2

def main():
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # Renderer
    renderer = mujoco.Renderer(model, height=720, width=1280)
    
    # Video
    frames = []
    video_fps = 30
    
    print("Final Velocity Control Demo")
    print("=" * 70)
    
    # Get IDs
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
    
    # Initial positions
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    print(f"Start - EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"Start - Red: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    print(f"Distance: {np.linalg.norm(ee_pos - red_pos):.3f}m")
    print()
    
    # Simple velocity control sequence
    dt = model.opt.timestep
    
    # Phase 1: Open gripper and wait (60 steps)
    print("Phase 1: Open gripper")
    for step in range(60):
        velocities = np.zeros(8)
        velocities[7] = 255  # Open gripper
        
        # Apply velocities
        for i in range(7):
            data.ctrl[i] = data.qpos[i]  # Hold position
        data.ctrl[7] = velocities[7]
        
        mujoco.mj_step(model, data)
        
        if step % 30 == 0:
            renderer.update_scene(data, camera="demo_cam")
            frame = renderer.render()
            cv2.putText(frame, "Phase 1: Open Gripper", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            frames.append(frame)
    
    # Phase 2: Simple joint movements (120 steps)
    print("Phase 2: Move joints with velocity control")
    for step in range(120):
        velocities = np.zeros(8)
        
        # Simple sinusoidal motion for joint 6 (wrist)
        velocities[5] = 0.3 * np.sin(step * 0.05)
        
        # Keep gripper open
        velocities[7] = 255
        
        # Apply velocities by integrating
        for i in range(7):
            current_pos = data.qpos[14 + i]  # Robot joints start at qpos[14]
            new_pos = current_pos + velocities[i] * dt
            data.ctrl[i] = new_pos
        data.ctrl[7] = velocities[7]
        
        mujoco.mj_step(model, data)
        
        if step % 20 == 0:
            ee_pos = data.xpos[ee_id]
            renderer.update_scene(data, camera="demo_cam")
            frame = renderer.render()
            cv2.putText(frame, "Phase 2: Velocity Control", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, f"Joint 6 vel: {velocities[5]:.2f} rad/s", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
            cv2.putText(frame, f"EE: [{ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f}]", 
                        (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(frame)
    
    # Phase 3: Close and open gripper (60 steps)
    print("Phase 3: Gripper control")
    for step in range(60):
        velocities = np.zeros(8)
        
        # Toggle gripper
        if step < 30:
            velocities[7] = 0  # Close
            gripper_text = "CLOSING"
        else:
            velocities[7] = 255  # Open
            gripper_text = "OPENING"
        
        # Hold arm position
        for i in range(7):
            data.ctrl[i] = data.qpos[14 + i]
        data.ctrl[7] = velocities[7]
        
        mujoco.mj_step(model, data)
        
        if step % 15 == 0:
            renderer.update_scene(data, camera="demo_cam")
            frame = renderer.render()
            cv2.putText(frame, "Phase 3: Gripper Control", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, f"Gripper: {gripper_text}", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            frames.append(frame)
    
    print("\nDemo complete!")
    
    # Final positions
    ee_pos = data.xpos[ee_id]
    red_pos = data.xpos[red_id]
    print(f"End - EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"End - Red: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
    
    # Save video
    if frames:
        output_path = "../../videos/franka_final_velocity_control.mp4"
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))  # 10 FPS for slower playback
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"\nVideo saved to {output_path}")
        print(f"Shows: Gripper control and joint velocity control")

if __name__ == "__main__":
    main()