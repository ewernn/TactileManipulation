import argparse
import h5py
import numpy as np
import os
import time
from mujoco import viewer
import mujoco
import imageio
import matplotlib.pyplot as plt
from environments.tactile_sensor import TactileSensor

def replay_with_contact(dataset_path, demo_idx=0, render=True, save_video=False, model_xml_path=None):
    """
    Replay a demonstration with contact detection.
    
    Args:
        dataset_path: Path to the HDF5 dataset
        demo_idx: Index of the demonstration to replay
        render: Whether to render the simulation
        save_video: Whether to save a video of the replay
        model_xml_path: Path to the MuJoCo XML model, if not specified in the dataset
    """
    print(f"Loading dataset from {dataset_path}")
    
    # Load the dataset
    try:
        with h5py.File(dataset_path, 'r') as f:
            # Check if the dataset has the expected structure
            if 'data' not in f:
                raise ValueError(f"Dataset does not contain 'data' group")
            
            data_group = f['data']
            demo_key = f"demo_{demo_idx}"
            
            if demo_key not in data_group:
                available_demos = [k for k in data_group.keys() if k.startswith('demo_')]
                print(f"Available demos: {sorted(available_demos)[:10]}...")
                raise ValueError(f"Demo {demo_idx} not found in dataset")
            
            demo_group = data_group[demo_key]
            print(f"Loaded demo {demo_idx}")
            
            # Extract necessary data
            if 'actions' not in demo_group:
                raise ValueError(f"Demo {demo_idx} does not contain actions")
            actions = demo_group['actions'][:]
            
            if 'states' not in demo_group:
                raise ValueError(f"Demo {demo_idx} does not contain states")
            states = demo_group['states'][:]
            
            initial_state = states[0]
            
            # Extract information about the gripper for our tactile sensor
            if 'obs' in demo_group and 'robot0_gripper_qpos' in demo_group['obs']:
                gripper_qpos = demo_group['obs']['robot0_gripper_qpos'][:]
                print(f"Gripper positions shape: {gripper_qpos.shape}")
                
            # Print some information about the demo
            print(f"Demo length: {len(actions)} steps")
            print(f"Action shape: {actions.shape}")
            print(f"Initial state shape: {initial_state.shape}")
            
            # Check if the dataset has a model file attribute
            model_file = model_xml_path
            if model_file is None:
                if 'model_file' in f.attrs:
                    model_file = f.attrs['model_file']
                    print(f"Using model file from dataset: {model_file}")
                else:
                    model_file = "assets/panda_gripper.xml"
                    print(f"No model file specified, using default: {model_file}")
                    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Check if the model file exists
    if not os.path.exists(model_file):
        # Try finding the model file in common locations
        possible_paths = [
            model_file,
            os.path.join("assets", model_file),
            os.path.join("..", "assets", model_file),
            os.path.join("assets", "panda_gripper.xml"),
            os.path.join("..", "assets", "panda_gripper.xml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_file = path
                print(f"Found model file at: {model_file}")
                break
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # Load the MuJoCo model
    try:
        model = mujoco.MjModel.from_xml_path(model_file)
        data = mujoco.MjData(model)
        print(f"Loaded MuJoCo model: {model_file}")
        
        # Initialize tactile sensor
        tactile_sensor = TactileSensor(model, data)
        print("Initialized tactile sensor")
        
        # Set initial state
        # Note: We're assuming initial_state can be directly mapped to data.qpos and data.qvel
        # This may need adjustment based on the actual state structure
        try:
            # Set initial joint positions (first part of state)
            # For Panda, typically first 7 values are joint positions
            joint_count = min(7, len(initial_state), model.nq)
            data.qpos[:joint_count] = initial_state[:joint_count]
            
            # Set initial gripper position if available
            if 'obs' in demo_group and 'robot0_gripper_qpos' in demo_group['obs']:
                gripper_start = demo_group['obs']['robot0_gripper_qpos'][0]
                # Assuming the last two values in qpos control the gripper
                data.qpos[-2:] = gripper_start
            
            # Forward kinematics to update the model
            mujoco.mj_forward(model, data)
            print("Set initial state")
        except Exception as e:
            print(f"Warning: Could not set initial state: {e}")
            print("Starting with default state")
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        raise
    
    # Set up rendering
    viewer_instance = None
    if render:
        try:
            viewer_instance = viewer.launch_passive(model, data)
            print("Launched MuJoCo viewer")
        except Exception as e:
            print(f"Warning: Could not start viewer: {e}")
            render = False
    
    # Set up video recording
    video_writer = None
    if save_video:
        try:
            video_path = f"replay_demo{demo_idx}.mp4"
            video_writer = imageio.get_writer(video_path, fps=30)
            print(f"Will save video to {video_path}")
        except Exception as e:
            print(f"Warning: Could not set up video recording: {e}")
            save_video = False
    
    # Arrays to store tactile readings for each finger
    left_readings = []
    right_readings = []
    
    # Replay the demonstration
    print("Starting replay...")
    try:
        for step, action in enumerate(actions):
            # Apply action
            # For Panda, typically:
            # - First 3 values are delta XYZ position
            # - Next 3 values are delta orientation
            # - Last 1 value is gripper command (1 for open, -1 for close)
            
            # For now, simplified approach: map the action to actuator controls
            # This may need adjustment based on the control scheme
            ctrl_count = min(len(action), model.nu)
            data.ctrl[:ctrl_count] = action[:ctrl_count]
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Get tactile readings
            left_reading, right_reading = tactile_sensor.get_readings(model, data)
            left_readings.append(left_reading.copy())
            right_readings.append(right_reading.copy())
            
            # Check for contacts
            contact_info = tactile_sensor.process_contacts(model, data)
            if contact_info:
                print(f"Step {step}: Contact detected!")
                for contact in contact_info[:5]:  # Show first 5 contacts
                    print(f"  Body: {contact['body_name']}, Pos: {contact['pos']}, Force: {contact['force_norm']:.3f}")
            
            # Render if needed
            if render and viewer_instance is not None:
                try:
                    viewer_instance.sync()
                    time.sleep(0.01)  # Slow down the playback
                except Exception as e:
                    print(f"Warning: Render failed at step {step}: {e}")
            
            # Capture frame for video
            if save_video and video_writer is not None:
                try:
                    # Render an RGB frame
                    viewport_width, viewport_height = 640, 480
                    rgb_buffer = np.zeros((viewport_height, viewport_width, 3), dtype=np.uint8)
                    
                    with viewer_instance.override_camera():
                        mujoco.mjr_render(viewport_width, viewport_height, 
                                         viewer_instance._scene, 
                                         viewer_instance._context,
                                         viewer_instance._viewport)
                        mujoco.mjr_readPixels(rgb_buffer, None, 
                                             viewer_instance._viewport,
                                             viewer_instance._context)
                    
                    # Flip the buffer vertically
                    rgb_buffer = np.flipud(rgb_buffer)
                    video_writer.append_data(rgb_buffer)
                except Exception as e:
                    print(f"Warning: Video capture failed at step {step}: {e}")
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step}/{len(actions)}")
                
    except Exception as e:
        print(f"Error during replay: {e}")
    finally:
        # Clean up resources
        if video_writer is not None:
            video_writer.close()
            print(f"Video saved to replay_demo{demo_idx}.mp4")
        
        if viewer_instance is not None:
            viewer_instance.close()
            print("Viewer closed")
    
    # Convert readings to numpy arrays for visualization
    left_readings = np.array(left_readings)  # Shape: (n_steps, n_taxels)
    right_readings = np.array(right_readings)  # Shape: (n_steps, n_taxels)
    
    # Save the tactile readings
    np.save(f"tactile_readings_demo{demo_idx}.npy", {
        'left': left_readings,
        'right': right_readings
    })
    print(f"Saved tactile readings to tactile_readings_demo{demo_idx}.npy")
    
    # Visualize some tactile readings
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Sample timesteps for visualization (beginning, middle, end)
        sample_steps = [0, len(actions) // 2, len(actions) - 1]
        
        for i, step in enumerate(sample_steps):
            # Visualize left finger
            tactile_sensor.visualize_reading(left_readings[step], ax=axes[0, i], title=f"Left Finger (Step {step})")
            
            # Visualize right finger
            tactile_sensor.visualize_reading(right_readings[step], ax=axes[1, i], title=f"Right Finger (Step {step})")
        
        plt.tight_layout()
        plt.savefig(f"tactile_visualization_demo{demo_idx}.png")
        print(f"Saved tactile visualization to tactile_visualization_demo{demo_idx}.png")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not visualize tactile readings: {e}")
    
    print("Replay completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=int, default=0, help="Index of the demonstration to replay")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--save-video", action="store_true", help="Save a video of the replay")
    parser.add_argument("--model", type=str, help="Path to MuJoCo XML model")
    
    args = parser.parse_args()
    
    replay_with_contact(
        dataset_path=args.dataset,
        demo_idx=args.demo,
        render=not args.no_render,
        save_video=args.save_video,
        model_xml_path=args.model
    )
