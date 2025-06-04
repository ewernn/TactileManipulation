#!/usr/bin/env python3
"""
Debug why expert demonstrations are failing.
"""

import numpy as np
import mujoco
import sys
sys.path.append("..")

from collect_expert_demonstrations import ExpertDemonstrationCollector

def debug_single_demo():
    """Debug a single demonstration with detailed output."""
    print("Debugging expert demonstration...")
    
    # Create collector
    collector = ExpertDemonstrationCollector(save_video=False)
    
    # Test with known good block positions
    red_pos = np.array([0.55, 0.0, 0.425])
    blue_pos = np.array([0.55, -0.1, 0.425])
    
    print(f"\nInitial block positions:")
    print(f"  Red: {red_pos}")
    print(f"  Blue: {blue_pos}")
    
    # Get waypoints
    waypoints = collector.compute_waypoints(red_pos, blue_pos)
    
    print(f"\nWaypoints:")
    for wp in waypoints:
        if wp['config']:
            print(f"  {wp['name']:12} config: {[f'{c:.3f}' for c in wp['config'][:3]]}... gripper: {wp['gripper']}")
        else:
            print(f"  {wp['name']:12} config: Hold position, gripper: {wp['gripper']}")
    
    # Reset simulation
    mujoco.mj_resetDataKeyframe(collector.model, collector.data, 0)
    
    # Set block positions
    collector.set_block_positions(red_pos, blue_pos)
    
    # Execute first few waypoints with detailed tracking
    initial_config = waypoints[0]["config"]
    for i, val in enumerate(initial_config):
        joint_addr = collector.model.jnt_qposadr[collector.arm_joint_ids[i]]
        collector.data.qpos[joint_addr] = val
        collector.data.ctrl[i] = val
    collector.data.ctrl[7] = waypoints[0]["gripper"]
    
    mujoco.mj_forward(collector.model, collector.data)
    
    print(f"\nInitial state:")
    print(f"  EE pos: {collector.data.xpos[collector.ee_id]}")
    print(f"  Red pos: {collector.data.xpos[collector.red_id]}")
    print(f"  Blue pos: {collector.data.xpos[collector.blue_id]}")
    
    # Track through waypoints
    for wp_idx, waypoint in enumerate(waypoints[:5]):  # First 5 waypoints
        print(f"\n{'='*60}")
        print(f"Waypoint {wp_idx}: {waypoint['name']}")
        
        current_config = [collector.data.ctrl[i] for i in range(7)]
        target_config = waypoint["config"] if waypoint["config"] is not None else current_config
        
        # Execute with tracking
        for frame in [0, waypoint["duration"]//2, waypoint["duration"]-1]:  # Start, middle, end
            alpha = frame / waypoint["duration"]
            
            if waypoint["name"] in ["grasp", "release"]:
                config = current_config
            else:
                config = collector.interpolate_configs(current_config, target_config, alpha)
            
            for i, val in enumerate(config):
                collector.data.ctrl[i] = val
            collector.data.ctrl[7] = waypoint["gripper"]
            
            # Step physics
            for _ in range(10):  # Multiple steps
                mujoco.mj_step(collector.model, collector.data)
            
            if frame == 0 or frame == waypoint["duration"]-1:
                print(f"\n  Frame {frame} (alpha={alpha:.2f}):")
                print(f"    EE pos: {collector.data.xpos[collector.ee_id]}")
                print(f"    Red pos: {collector.data.xpos[collector.red_id]}")
                
                # Check gripper-block distance
                ee_pos = collector.data.xpos[collector.ee_id]
                red_pos = collector.data.xpos[collector.red_id]
                distance = np.linalg.norm(ee_pos - red_pos)
                print(f"    EE-Red distance: {distance:.3f}m")
                
                # Check table clearance
                gripper_bodies = ["hand", "left_finger", "right_finger"]
                lowest_z = min([collector.data.xpos[mujoco.mj_name2id(collector.model, mujoco.mjtObj.mjOBJ_BODY, b)][2] 
                               for b in gripper_bodies if mujoco.mj_name2id(collector.model, mujoco.mjtObj.mjOBJ_BODY, b) != -1])
                table_clearance = lowest_z - collector.table_height
                print(f"    Table clearance: {table_clearance:.3f}m")
    
    # Final reward check
    final_reward = collector.compute_stacking_reward()
    print(f"\n{'='*60}")
    print(f"Final stacking metrics:")
    print(f"  Total reward: {final_reward['total_reward']:.3f}")
    print(f"  XY offset: {final_reward['xy_offset']*1000:.1f}mm")
    print(f"  Height error: {final_reward['height_error']*1000:.1f}mm")
    
    # Check if blocks moved
    print(f"\nBlock movement check:")
    print(f"  Red final: {collector.data.xpos[collector.red_id]}")
    print(f"  Blue final: {collector.data.xpos[collector.blue_id]}")

if __name__ == "__main__":
    debug_single_demo()