#!/usr/bin/env python3
"""
Test the expert demonstration collection system.
"""

import numpy as np
import sys
sys.path.append("..")

from collect_expert_demonstrations import ExpertDemonstrationCollector
import h5py

def test_single_demo():
    """Test collecting a single demonstration."""
    print("Testing single demonstration collection...")
    
    # Create collector
    collector = ExpertDemonstrationCollector(save_video=False)
    
    # Test with specific block positions
    red_pos = np.array([0.55, 0.0, 0.425])
    blue_pos = np.array([0.55, -0.1, 0.425])
    
    print(f"\nRed block position: {red_pos}")
    print(f"Blue block position: {blue_pos}")
    
    # Execute demonstration
    demo_data, _ = collector.execute_demonstration(0, red_pos, blue_pos)
    
    print(f"\nDemo results:")
    print(f"  Success: {demo_data['success']}")
    print(f"  Final reward: {demo_data['final_reward']:.3f}")
    print(f"  Episode length: {demo_data['episode_length']}")
    print(f"  Action shape: {demo_data['actions'].shape}")
    print(f"  Tactile shape: {demo_data['tactile_readings'].shape}")
    
    # Check observation keys
    print(f"\nObservation keys: {list(demo_data['observations'][0].keys())}")
    
    # Check data shapes
    for key, value in demo_data['observations'][0].items():
        print(f"  {key}: shape {np.array(value).shape}")
    
    return demo_data['success']

def test_waypoint_generation():
    """Test waypoint generation for different block positions."""
    print("\nTesting waypoint generation...")
    
    collector = ExpertDemonstrationCollector()
    
    # Test different block configurations
    test_configs = [
        # Red pos, Blue pos
        (np.array([0.5, 0.0, 0.425]), np.array([0.6, 0.0, 0.425])),  # Straight line
        (np.array([0.45, -0.1, 0.425]), np.array([0.65, 0.1, 0.425])),  # Diagonal
        (np.array([0.6, 0.1, 0.425]), np.array([0.5, -0.1, 0.425])),  # Reverse
    ]
    
    for i, (red_pos, blue_pos) in enumerate(test_configs):
        print(f"\nConfiguration {i+1}:")
        print(f"  Red: {red_pos}")
        print(f"  Blue: {blue_pos}")
        
        waypoints = collector.compute_waypoints(red_pos, blue_pos)
        
        print(f"  Generated {len(waypoints)} waypoints:")
        for wp in waypoints:
            if wp['config'] is not None:
                print(f"    {wp['name']}: joint0={wp['config'][0]:.3f}")

def test_hdf5_saving():
    """Test HDF5 file saving."""
    print("\nTesting HDF5 saving...")
    
    collector = ExpertDemonstrationCollector(save_video=False)
    
    # Collect a few demos
    demos = []
    for i in range(3):
        red_pos, blue_pos = collector.randomize_block_positions()
        demo_data, _ = collector.execute_demonstration(i, red_pos, blue_pos)
        if demo_data['success']:
            demos.append(demo_data)
            print(f"  Demo {i}: Success (reward={demo_data['final_reward']:.3f})")
        else:
            print(f"  Demo {i}: Failed")
    
    # Save to HDF5
    if demos:
        test_path = "../datasets/test/test_demos.hdf5"
        collector.save_demonstrations(demos, test_path)
        
        # Verify file
        print(f"\nVerifying HDF5 file...")
        with h5py.File(test_path, 'r') as f:
            print(f"  Number of demos: {f.attrs['num_demos']}")
            print(f"  Control frequency: {f.attrs['control_frequency']} Hz")
            
            # Check first demo
            demo0 = f['demo_0']
            print(f"\n  Demo 0 structure:")
            print(f"    Episode length: {demo0.attrs['episode_length']}")
            print(f"    Success: {demo0.attrs['success']}")
            print(f"    Final reward: {demo0.attrs['final_reward']:.3f}")
            
            # Check data shapes
            print(f"\n    Data shapes:")
            print(f"      actions: {demo0['actions'].shape}")
            print(f"      rewards: {demo0['rewards'].shape}")
            
            obs = demo0['observations']
            for key in obs.keys():
                print(f"      obs/{key}: {obs[key].shape}")

def main():
    """Run all tests."""
    print("="*60)
    print("EXPERT DEMONSTRATION COLLECTION TEST")
    print("="*60)
    
    # Test 1: Single demonstration
    success = test_single_demo()
    
    # Test 2: Waypoint generation
    test_waypoint_generation()
    
    # Test 3: HDF5 saving
    test_hdf5_saving()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()