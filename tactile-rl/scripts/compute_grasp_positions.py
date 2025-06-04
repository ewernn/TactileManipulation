#!/usr/bin/env python3
"""
Compute good grasp positions for the blocks using systematic search.
"""

import numpy as np
import mujoco
import cv2

def test_configuration(model, data, joint_vals, target_pos, gripper_reach=0.103):
    """Test a joint configuration and return metrics."""
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set joint positions
    for i, val in enumerate(joint_vals):
        data.qpos[14 + i] = val
        data.ctrl[i] = val
    data.ctrl[7] = 255  # Open gripper
    
    mujoco.mj_forward(model, data)
    
    # Get positions
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    ee_pos = data.xpos[ee_id].copy()
    
    # Get gripper direction
    hand_quat = data.xquat[ee_id].copy()
    rotmat = np.zeros(9)
    mujoco.mju_quat2Mat(rotmat, hand_quat)
    rotmat = rotmat.reshape(3, 3)
    gripper_direction = rotmat[:, 2]
    
    # Calculate approximate fingertip position
    fingertip_pos = ee_pos + gripper_reach * gripper_direction
    
    # Calculate ideal hand position (fingertips should be at block center)
    ideal_hand_pos = target_pos - gripper_reach * gripper_direction
    
    # Metrics
    hand_error = np.linalg.norm(ee_pos - ideal_hand_pos)
    fingertip_error = np.linalg.norm(fingertip_pos - target_pos)
    
    # Check alignment
    to_target = target_pos - ee_pos
    to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
    alignment = np.dot(gripper_direction, to_target_norm)
    
    return {
        'hand_error': hand_error,
        'fingertip_error': fingertip_error,
        'alignment': alignment,
        'ee_pos': ee_pos,
        'fingertip_pos': fingertip_pos,
        'gripper_direction': gripper_direction
    }

def search_grasp_position(model, data, target_pos, initial_joints, iterations=20):
    """Search for good grasp position using gradient-based optimization."""
    joint_vals = initial_joints.copy()
    best_joint_vals = joint_vals.copy()
    best_error = float('inf')
    
    # Learning rates for each joint
    learning_rates = np.array([0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05])
    
    print(f"\nSearching for grasp position at target: {target_pos}")
    print("="*60)
    
    for iter in range(iterations):
        # Test current configuration
        metrics = test_configuration(model, data, joint_vals, target_pos)
        
        # Combined error (prioritize fingertip position)
        error = metrics['fingertip_error'] + 0.1 * (1.0 - metrics['alignment'])
        
        if error < best_error:
            best_error = error
            best_joint_vals = joint_vals.copy()
        
        print(f"Iter {iter:2d}: fingertip_error={metrics['fingertip_error']:.3f}, "
              f"alignment={metrics['alignment']:.3f}, total_error={error:.3f}")
        
        # If close enough, stop
        if metrics['fingertip_error'] < 0.01:
            print("Found good solution!")
            break
        
        # Compute gradients using finite differences
        gradients = np.zeros(7)
        epsilon = 0.01
        
        for i in range(7):
            # Skip certain joints for stability
            if i == 2 or i == 4:  # Skip shoulder rotation and forearm rotation
                continue
                
            # Forward difference
            joint_vals_plus = joint_vals.copy()
            joint_vals_plus[i] += epsilon
            metrics_plus = test_configuration(model, data, joint_vals_plus, target_pos)
            error_plus = metrics_plus['fingertip_error'] + 0.1 * (1.0 - metrics_plus['alignment'])
            
            # Gradient
            gradients[i] = (error_plus - error) / epsilon
        
        # Update joints
        joint_vals -= learning_rates * gradients
        
        # Clip to joint limits
        for i in range(7):
            joint_id = i + 4  # Panda joints start at index 4
            if model.jnt_limited[joint_id]:
                limits = model.jnt_range[joint_id]
                joint_vals[i] = np.clip(joint_vals[i], limits[0], limits[1])
    
    # Return best configuration
    metrics = test_configuration(model, data, best_joint_vals, target_pos)
    print(f"\nBest configuration found:")
    print(f"  Joint values: [{', '.join([f'{v:.3f}' for v in best_joint_vals])}]")
    print(f"  Fingertip error: {metrics['fingertip_error']:.3f}m")
    print(f"  Alignment: {metrics['alignment']:.3f}")
    print(f"  EE position: [{metrics['ee_pos'][0]:.3f}, {metrics['ee_pos'][1]:.3f}, {metrics['ee_pos'][2]:.3f}]")
    print(f"  Fingertip pos: [{metrics['fingertip_pos'][0]:.3f}, {metrics['fingertip_pos'][1]:.3f}, {metrics['fingertip_pos'][2]:.3f}]")
    
    return best_joint_vals, metrics

def main():
    # Load model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Block positions
    red_block_pos = np.array([0.05, 0.0, 0.445])
    blue_block_pos = np.array([0.05, -0.15, 0.445])
    
    # Initial joint configuration
    home_joints = np.array([0, -0.1, 0, -2.0, 0, 1.5, 0])
    
    # Find positions for different phases
    configs = {}
    
    # 1. Above red block (higher for approach)
    print("\n" + "="*80)
    print("FINDING: Above red block position")
    above_red_target = red_block_pos + np.array([0, 0, 0.1])  # 10cm above
    above_red_initial = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0])
    configs['above_red'], _ = search_grasp_position(model, data, above_red_target, above_red_initial)
    
    # 2. Grasp red block
    print("\n" + "="*80)
    print("FINDING: Grasp red block position")
    grasp_red_initial = configs['above_red'].copy()
    configs['grasp_red'], _ = search_grasp_position(model, data, red_block_pos, grasp_red_initial)
    
    # 3. Above blue block
    print("\n" + "="*80)
    print("FINDING: Above blue block position")
    above_blue_target = blue_block_pos + np.array([0, 0, 0.15])  # 15cm above
    above_blue_initial = np.array([-0.4, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0])
    configs['above_blue'], _ = search_grasp_position(model, data, above_blue_target, above_blue_initial)
    
    # 4. Place on blue block
    print("\n" + "="*80)
    print("FINDING: Place on blue block position")
    place_target = blue_block_pos + np.array([0, 0, 0.05])  # Account for block height
    place_initial = configs['above_blue'].copy()
    configs['place_blue'], _ = search_grasp_position(model, data, place_target, place_initial)
    
    # Print all configurations
    print("\n" + "="*80)
    print("OPTIMIZED CONFIGURATIONS:")
    print("="*80)
    print("configs = {")
    print(f'    "home": np.array({home_joints.tolist()}),')
    for name, vals in configs.items():
        print(f'    "{name}": np.array({vals.tolist()}),')
    print("}")
    
    # Test the grasping sequence
    print("\n" + "="*80)
    print("TESTING GRASP SEQUENCE:")
    print("="*80)
    
    # Renderer for visualization
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Test sequence
    sequence = [
        ("home", home_joints, 255),
        ("above_red", configs['above_red'], 255),
        ("grasp_red", configs['grasp_red'], 255),
        ("close_gripper", configs['grasp_red'], 0),
        ("lift", configs['above_red'], 0),
        ("above_blue", configs['above_blue'], 0),
        ("place_blue", configs['place_blue'], 0),
        ("open_gripper", configs['place_blue'], 255),
    ]
    
    frames = []
    for phase, joint_vals, gripper_cmd in sequence:
        print(f"\nPhase: {phase}")
        
        # Set configuration
        for i, val in enumerate(joint_vals):
            data.qpos[14 + i] = val
            data.ctrl[i] = val
        data.ctrl[7] = gripper_cmd
        
        # Step simulation
        for _ in range(100):
            mujoco.mj_step(model, data)
        
        # Get positions
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        
        ee_pos = data.xpos[ee_id]
        red_pos = data.xpos[red_id]
        
        print(f"  EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        print(f"  Red: [{red_pos[0]:.3f}, {red_pos[1]:.3f}, {red_pos[2]:.3f}]")
        
        # Render frame
        renderer.update_scene(data, camera="demo_cam")
        frame = renderer.render()
        
        # Add text
        cv2.putText(frame, phase.upper(), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frames.append(frame)
    
    # Save test images
    for i, (phase, _, _) in enumerate(sequence):
        cv2.imwrite(f"grasp_test_{i}_{phase}.png", cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
    
    print("\n✓ Saved test images")

if __name__ == "__main__":
    main()