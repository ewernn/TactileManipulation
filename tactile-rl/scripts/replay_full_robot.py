import argparse
import h5py
import numpy as np
import os
import sys
import time
import mujoco
from mujoco import viewer
import imageio
import matplotlib
# Force matplotlib to use a non-interactive backend that doesn't require a GUI window
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))  # scripts/
parent_dir = os.path.dirname(script_dir)  # tactile-rl/
sys.path.insert(0, parent_dir)
from environments.tactile_sensor import TactileSensor

def parse_object_data(object_obs):
    """
    Parse the 23-value object observation into meaningful components.
    
    The structure is:
    - cubeA position (3 values)
    - cubeA quaternion (4 values)
    - cubeB position (3 values)
    - cubeB quaternion (4 values)
    - cubeA to cubeB vector (3 values)
    - gripper to cubeA vector (3 values)
    - gripper to cubeB vector (3 values)
    
    Args:
        object_obs: Array of 23 values containing object information
        
    Returns:
        Tuple of (cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat)
    """
    cubeA_pos = object_obs[0:3]  # First 3 values are cubeA position
    cubeA_quat = object_obs[3:7]  # Next 4 values are cubeA quaternion
    
    cubeB_pos = object_obs[7:10]  # Next 3 values are cubeB position
    cubeB_quat = object_obs[10:14]  # Next 4 values are cubeB quaternion
    
    # The rest are the vectors (not needed for updating positions)
    # cubeA_to_cubeB = object_obs[14:17]
    # gripper_to_cubeA = object_obs[17:20]
    # gripper_to_cubeB = object_obs[20:23]
    
    return cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat

def replay_full_robot(dataset_path, demo_idx=0, render=True, save_video=False, model_xml_path=None, output_dir=None, camera_id=None, playback_speed=0.01):
    """
    Replay a demonstration with the full robot and contact detection.
    
    Args:
        dataset_path: Path to the HDF5 dataset
        demo_idx: Index of the demonstration to replay
        render: Whether to render the simulation
        save_video: Whether to save a video of the replay
        model_xml_path: Path to the MuJoCo XML model, if not specified in the dataset
        output_dir: Directory to save outputs (videos, visualizations, etc.)
        camera_id: ID of the camera to use for video capture
        playback_speed: Time in seconds to wait between frames (higher = slower playback)
    """
    print(f"Loading dataset from {dataset_path}")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Outputs will be saved to: {output_dir}")
    
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
            
            # Extract information about the arm and gripper
            if 'obs' in demo_group:
                if 'robot0_joint_pos' in demo_group['obs']:
                    joint_pos = demo_group['obs']['robot0_joint_pos'][:]
                    print(f"Arm joint positions shape: {joint_pos.shape}")
                
                if 'robot0_gripper_qpos' in demo_group['obs']:
                    gripper_qpos = demo_group['obs']['robot0_gripper_qpos'][:]
                    print(f"Gripper positions shape: {gripper_qpos.shape}")
                
                # Check for object observations
                if 'object' in demo_group['obs']:
                    object_obs = demo_group['obs']['object'][:]
                    print(f"Object observations shape: {object_obs.shape}")
                    if object_obs.shape[1] != 23:
                        print(f"Warning: Expected 23 values in object observations, but got {object_obs.shape[1]}")
                else:
                    print("Warning: No object observations found in dataset")
                    object_obs = None
            
            # Print some information about the demo
            print(f"Demo length: {len(actions)} steps")
            print(f"Action shape: {actions.shape}")
            print(f"Initial state shape: {initial_state.shape}")
            
            # Check if the dataset has a model file attribute
            model_file = model_xml_path
            if model_file is None:
                # Use our custom two-cube model instead of the default
                model_file = "franka_emika_panda/mjx_two_cubes.xml"
                print(f"Using two-cube model: {model_file}")
                    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Check if the model file exists
    if not os.path.exists(model_file):
        # Try finding the model file in common locations
        possible_paths = [
            model_file,
            os.path.join("franka_emika_panda", "mjx_two_cubes.xml"),
            os.path.join("..", "franka_emika_panda", "mjx_two_cubes.xml"),
            os.path.join("franka_emika_panda", model_file),
            os.path.join("..", "franka_emika_panda", model_file)
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
        
        # Get body IDs for cubes
        try:
            cubeA_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cubeA")
            cubeB_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cubeB")
            print(f"Found cube bodies: cubeA (ID: {cubeA_body_id}), cubeB (ID: {cubeB_body_id})")
            has_cubes = True
        except Exception as e:
            print(f"Warning: Could not find cube bodies: {e}")
            has_cubes = False
        
        # Initialize tactile sensor
        tactile_sensor = TactileSensor(
            model=model, 
            data=data,
            left_finger_name="left_finger",
            right_finger_name="right_finger",
            collision_geom_prefixes=["fingertip_pad_collision"]
        )
        print("Initialized tactile sensor")
        
        # Set initial state
        try:
            # Set initial arm joint positions (7 joints)
            if 'obs' in demo_group and 'robot0_joint_pos' in demo_group['obs']:
                arm_joint_pos = demo_group['obs']['robot0_joint_pos'][0]
                data.qpos[:7] = arm_joint_pos
                print("Set initial arm joint positions")
            
            # Set initial gripper position (2 joints)
            if 'obs' in demo_group and 'robot0_gripper_qpos' in demo_group['obs']:
                gripper_pos = demo_group['obs']['robot0_gripper_qpos'][0]
                data.qpos[7:9] = gripper_pos
                print("Set initial gripper positions")
            
            # Set initial cube positions if object observations are available
            if has_cubes and object_obs is not None:
                cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat = parse_object_data(object_obs[0])
                
                # Set initial positions and orientations for cubes
                # For cubeA - set position and orientation in the free joint's qpos
                cube_A_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cubeA_joint0")
                if cube_A_joint_idx != -1:
                    cube_A_qpos_start = model.jnt_qposadr[cube_A_joint_idx]
                    data.qpos[cube_A_qpos_start:cube_A_qpos_start+3] = cubeA_pos
                    data.qpos[cube_A_qpos_start+3:cube_A_qpos_start+7] = cubeA_quat
                
                # For cubeB - set position and orientation in the free joint's qpos
                cube_B_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cubeB_joint0")
                if cube_B_joint_idx != -1:
                    cube_B_qpos_start = model.jnt_qposadr[cube_B_joint_idx]
                    data.qpos[cube_B_qpos_start:cube_B_qpos_start+3] = cubeB_pos
                    data.qpos[cube_B_qpos_start+3:cube_B_qpos_start+7] = cubeB_quat
                
                print("Set initial cube positions and orientations")
            
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
            viewer_instance = viewer.launch(model, data)
            
            # IMPORTANT: Add this line to make viewer non-blocking
            viewer_instance.sync()  # Initial sync
            
            # Set camera view if needed
            if camera_id is None:
                viewer_instance.cam.azimuth = 90
                viewer_instance.cam.elevation = -15
                viewer_instance.cam.distance = 1.5
                viewer_instance.cam.lookat = [0.0, 0.0, 0.5]  # Focus on robot's workspace
                print(f"Using default full robot camera view")
            elif isinstance(camera_id, int) and 0 <= camera_id < model.ncam:
                viewer_instance.cam.azimuth = model.cam_azimuth[camera_id]
                viewer_instance.cam.elevation = model.cam_elevation[camera_id]
                viewer_instance.cam.distance = model.cam_distance[camera_id]
                print(f"Using camera {camera_id}")
            else:
                print(f"Invalid camera ID: {camera_id}")
            
            print("Launched MuJoCo viewer")
        except Exception as e:
            print(f"Warning: Could not start viewer: {e}")
            render = False
    
    # Initialize renderer for video capture if needed
    renderer = None
    video_writer = None
    if save_video:
        try:
            # Fixed dimensions - width and height were swapped before
            viewport_width, viewport_height = 640, 480
            
            # Create the renderer with correct dimensions
            renderer = mujoco.Renderer(model, viewport_height, viewport_width)
            
            video_path = os.path.join(output_dir, f"replay_demo{demo_idx}.mp4") if output_dir else f"replay_demo{demo_idx}.mp4"
            video_writer = imageio.get_writer(video_path, fps=30)
            print(f"Will save video to {video_path}")
        except Exception as e:
            print(f"Warning: Could not set up video recording: {e}")
            print("If you're seeing framebuffer errors, you might need to add this to your XML model:")
            print("<visual>\n  <global offheight=\"640\" offwidth=\"480\"/>\n</visual>")
            save_video = False
            if renderer is not None:
                try:
                    renderer.close()
                except:
                    pass
                renderer = None
    
    # Arrays to store tactile readings for each finger
    left_readings = []
    right_readings = []
    
    # Replay the demonstration
    print("Starting replay...")
    try:
        for step, action in enumerate(actions):
            # Apply action to the full robot
            # First 7 dimensions control the arm joints
            data.ctrl[:7] = action[:7]
            
            # Last dimension controls the gripper
            if len(action) > 6:
                # Map from [-1,1] to [0.0, 0.04], where -1 = closed (0.0) and 1 = open (0.04)
                finger_pos = (action[6] + 1) * 0.02  # Maps [-1,1] to [0,0.04]
                data.ctrl[7] = finger_pos
                
                # Enhanced gripper debugging
                if step % 10 == 0 or step == len(actions) - 1:  # Print at regular intervals
                    gripper_positions = data.qpos[7:9]  # Current position of gripper joints
                    print(f"Step {step}: Gripper action: {action[6]:.4f}, Mapped to: {finger_pos:.4f}")
                    print(f"        Gripper positions: {gripper_positions}")
                    print(f"        Gripper velocity: {data.qvel[7:9]}")
            
            # Update cube positions and orientations before stepping the simulation
            if has_cubes and object_obs is not None and step < len(object_obs):
                try:
                    cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat = parse_object_data(object_obs[step])
                    
                    # Update cubeA position and orientation using joint qpos
                    cube_A_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cubeA_joint0")
                    if cube_A_joint_idx != -1:
                        cube_A_qpos_start = model.jnt_qposadr[cube_A_joint_idx]
                        data.qpos[cube_A_qpos_start:cube_A_qpos_start+3] = cubeA_pos
                        data.qpos[cube_A_qpos_start+3:cube_A_qpos_start+7] = cubeA_quat
                    
                    # Update cubeB position and orientation using joint qpos
                    cube_B_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cubeB_joint0")
                    if cube_B_joint_idx != -1:
                        cube_B_qpos_start = model.jnt_qposadr[cube_B_joint_idx]
                        data.qpos[cube_B_qpos_start:cube_B_qpos_start+3] = cubeB_pos
                        data.qpos[cube_B_qpos_start+3:cube_B_qpos_start+7] = cubeB_quat
                    
                    # Forward position/velocity to update the model state
                    mujoco.mj_forward(model, data)
                    
                    if step % 50 == 0:  # Log less frequently
                        print(f"Step {step}: Updated cube positions")
                except Exception as e:
                    print(f"Warning: Could not update cube positions at step {step}: {e}")
            
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
                    time.sleep(playback_speed)  # Use configurable playback speed
                except Exception as e:
                    print(f"Warning: Render failed at step {step}: {e}")
            
            # Capture frame for video
            if save_video and renderer is not None and video_writer is not None:
                try:
                    # Update the scene with the specified camera
                    if camera_id is not None and isinstance(camera_id, int) and 0 <= camera_id < model.ncam:
                        renderer.update_scene(data, camera=camera_id)
                    else:
                        renderer.update_scene(data)
                    
                    # Render and add to video
                    pixels = renderer.render()
                    video_writer.append_data(pixels)
                except Exception as e:
                    print(f"Warning: Video capture failed at step {step}: {e}")
                    # Continue without video capture if it fails
                    if step == 0:
                        print("Disabling video capture due to errors")
                        save_video = False
                        if renderer is not None:
                            try:
                                renderer.close()
                            except:
                                pass
                            renderer = None
            
            # Print progress
            if step % 10 == 0 or step == len(actions) - 1:
                print(f"Step {step}/{len(actions)}")
                
    except Exception as e:
        print(f"Error during replay: {e}")
    finally:
        # Clean up resources
        if save_video and 'video_writer' in locals() and video_writer is not None:
            video_writer.close()
            video_path = os.path.join(output_dir, f"replay_demo{demo_idx}.mp4") if output_dir else f"replay_demo{demo_idx}.mp4"
            print(f"Video saved to {video_path}")
        
        if viewer_instance is not None:
            viewer_instance.close()
            print("Viewer closed")
    
    # Convert readings to numpy arrays for visualization
    left_readings = np.array(left_readings)  # Shape: (n_steps, n_taxels_x, n_taxels_y, 3)
    right_readings = np.array(right_readings)  # Shape: (n_steps, n_taxels_x, n_taxels_y, 3)
    
    # Save the tactile readings
    readings_path = os.path.join(output_dir, f"tactile_readings_demo{demo_idx}.npy") if output_dir else f"tactile_readings_demo{demo_idx}.npy"
    np.save(readings_path, {
        'left': left_readings,
        'right': right_readings
    })
    print(f"Saved tactile readings to {readings_path}")
    
    # Visualize some tactile readings (using non-interactive backend)
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
        viz_path = os.path.join(output_dir, f"tactile_visualization_demo{demo_idx}.png") if output_dir else f"tactile_visualization_demo{demo_idx}.png"
        plt.savefig(viz_path)
        print(f"Saved tactile visualization to {viz_path}")
        plt.close(fig)  # Explicitly close the figure
    except Exception as e:
        print(f"Warning: Could not visualize tactile readings: {e}")
    
    print("Replay completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Set default paths for dataset and model
    parser.add_argument("--dataset", type=str, 
                        default="../datasets/mimicgen/core/stack_d0.hdf5", 
                        help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=int, default=0, 
                        help="Index of the demonstration to replay")
    parser.add_argument("--render", action="store_true", 
                        help="Enable interactive visualization")
    parser.add_argument("--save-video", action="store_true", 
                        help="Save a video of the replay")
    parser.add_argument("--model", type=str, 
                        default="franka_emika_panda/stack_d0_compatible.xml",
                        help="Path to MuJoCo XML model")
    parser.add_argument("--output-dir", type=str, 
                        default="../exps",
                        help="Directory to save outputs (videos, visualizations, etc.)")
    parser.add_argument("--camera", type=int, 
                        help="Camera ID to use for viewing (0-4 for different camera views)")
    parser.add_argument("--playback-speed", type=float, default=0.01, 
                        help="Time in seconds to wait between frames (higher = slower playback)")
    
    args = parser.parse_args()
    
    replay_full_robot(
        dataset_path=args.dataset,
        demo_idx=args.demo,
        render=args.render,
        save_video=args.save_video,
        model_xml_path=args.model,
        output_dir=args.output_dir,
        camera_id=args.camera,
        playback_speed=args.playback_speed
    )