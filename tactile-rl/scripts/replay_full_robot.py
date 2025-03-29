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
import logging

# Set up logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def setup_environment(model, data, joint_names):
    """Configure environment and extract IDs for joints, actuators, bodies etc."""
    # Detect environment type by looking for specific naming patterns
    is_original_environment = any("robot0_joint" in name for name in joint_names)
    
    # Also check for specific gripper finger bodies
    left_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_leftfinger")
    right_finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_rightfinger")
    has_original_fingers = (left_finger_body_id >= 0 and right_finger_body_id >= 0)

    print(f"Using original environment model: {is_original_environment}")
    print(f"Has original fingers: {has_original_fingers}")
    
    # Set appropriate names based on detailed model analysis
    if is_original_environment or has_original_fingers:
        joint_prefix = "robot0_joint"
        left_finger_name = "gripper0_right_leftfinger"
        right_finger_name = "gripper0_right_rightfinger"
        left_finger_joint = "gripper0_right_finger_joint1"
        right_finger_joint = "gripper0_right_finger_joint2"
    else:
        # Fallback naming for non-original environments
        joint_prefix = "joint"
        left_finger_name = "left_finger"
        right_finger_name = "right_finger"
        left_finger_joint = "finger_joint1"
        right_finger_joint = "finger_joint2"
    
    # Find specific gripper joint IDs and their control indices
    left_finger_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, left_finger_joint)
    right_finger_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, right_finger_joint)
    
    # Find actuator IDs for gripper
    gripper_actuator_ids = []
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name and "gripper" in actuator_name:
            gripper_actuator_ids.append(i)
            print(f"Found gripper actuator: {actuator_name} at index {i}")
    
    # Map robot joint names to their actuator indices
    robot_joint_map = {}
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if actuator_name and "torq_j" in actuator_name:
            # Extract joint number from actuator name (e.g., robot0_torq_j1 -> 1)
            joint_num = int(actuator_name.split('j')[-1])
            robot_joint_map[joint_num] = i
            print(f"Mapped joint {joint_num} to actuator {actuator_name} at index {i}")
    
    # Get body IDs for cubes
    has_cubes = False
    cubeA_joint_name = None
    cubeB_joint_name = None
    
    try:
        # First try with specific names from the model files
        cubeA_body_id = -1
        cubeB_body_id = -1
        
        # Try different naming variations
        for cube_name in ["cubeA", "cubeA_main"]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
            if body_id >= 0:
                cubeA_body_id = body_id
                print(f"Found cubeA as '{cube_name}' (ID: {cubeA_body_id})")
                break
                
        for cube_name in ["cubeB", "cubeB_main"]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
            if body_id >= 0:
                cubeB_body_id = body_id
                print(f"Found cubeB as '{cube_name}' (ID: {cubeB_body_id})")
                break
        
        has_cubes = (cubeA_body_id >= 0 and cubeB_body_id >= 0)
        if has_cubes:
            print(f"Found cube bodies: cubeA (ID: {cubeA_body_id}), cubeB (ID: {cubeB_body_id})")
        
        # Also find the joint names for the cubes (they might differ)
        for joint_name in joint_names:
            if "cubeA" in joint_name:
                cubeA_joint_name = joint_name
            elif "cubeB" in joint_name:
                cubeB_joint_name = joint_name
        
        print(f"Cube joint names: cubeA={cubeA_joint_name}, cubeB={cubeB_joint_name}")
    except Exception as e:
        print(f"Warning: Could not find cube bodies: {e}")
        has_cubes = False
    
    # Create and return environment info dictionary
    env_info = {
        "joint_prefix": joint_prefix,
        "left_finger_name": left_finger_name,
        "right_finger_name": right_finger_name,
        "left_finger_joint": left_finger_joint,
        "right_finger_joint": right_finger_joint,
        "left_finger_joint_id": left_finger_joint_id,
        "right_finger_joint_id": right_finger_joint_id,
        "gripper_actuator_ids": gripper_actuator_ids,
        "robot_joint_map": robot_joint_map,
        "has_cubes": has_cubes,
        "cubeA_joint_name": cubeA_joint_name,
        "cubeB_joint_name": cubeB_joint_name,
    }
    
    return env_info

def apply_joint_action(action_val, joint_idx, control_mode, action_scale, current_positions, model, data, joint_map, joint_prefix):
    """Apply a single joint action based on the control mode."""
    if joint_idx not in joint_map:
        return None, current_positions
        
    actuator_idx = joint_map[joint_idx]
    
    # Get current joint position
    joint_name = f"{joint_prefix}{joint_idx}"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return None, current_positions
        
    joint_adr = model.jnt_qposadr[joint_id]
    current_pos = data.qpos[joint_adr]
    
    # Handle action differently based on control mode
    if control_mode == "velocity":
        # Use action as a desired velocity
        scaled_action = action_val * action_scale
        # We'll apply this directly to qvel later
    elif control_mode == "incremental":
        # Use action as increment to current position
        current_positions[joint_idx] += action_val * 0.1  # Small increment
        scaled_action = current_positions[joint_idx]
    else:  # "position" mode (default)
        # Original - scale action as before
        scaled_action = action_val * action_scale
    
    # Make sure we don't exceed control limits
    ctrl_range = model.actuator_ctrlrange[actuator_idx]
    scaled_action = max(ctrl_range[0], min(scaled_action, ctrl_range[1]))
    
    # Apply the action
    if control_mode == "velocity":
        # Set velocity directly
        joint_vel_adr = model.jnt_dofadr[joint_id]
        data.qvel[joint_vel_adr] = scaled_action
    else:
        # Set actuator control
        data.ctrl[actuator_idx] = scaled_action
        
    return scaled_action, current_positions

def update_cube_positions(step, object_obs, model, data, env_info):
    """Update cube positions from the dataset."""
    if not env_info["has_cubes"] or object_obs is None or step >= len(object_obs):
        return
        
    try:
        cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat = parse_object_data(object_obs[step])
        
        if env_info["cubeA_joint_name"]:
            cube_A_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, env_info["cubeA_joint_name"])
            if cube_A_joint_idx != -1:
                cube_A_qpos_start = model.jnt_qposadr[cube_A_joint_idx]
                data.qpos[cube_A_qpos_start:cube_A_qpos_start+3] = cubeA_pos
                data.qpos[cube_A_qpos_start+3:cube_A_qpos_start+7] = cubeA_quat
        
        if env_info["cubeB_joint_name"]:
            cube_B_joint_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, env_info["cubeB_joint_name"])
            if cube_B_joint_idx != -1:
                cube_B_qpos_start = model.jnt_qposadr[cube_B_joint_idx]
                data.qpos[cube_B_qpos_start:cube_B_qpos_start+3] = cubeB_pos
                data.qpos[cube_B_qpos_start+3:cube_B_qpos_start+7] = cubeB_quat
        
        mujoco.mj_forward(model, data)
    except Exception as e:
        print(f"Warning: Failed to update cube positions at step {step}: {e}")

def apply_gripper_action(gripper_cmd, gripper_actuator_ids, data, debug=False, physics_step=0):
    """Apply gripper action based on command value."""
    if len(gripper_actuator_ids) < 2:
        return
        
    if gripper_cmd < 0:  # Open gripper
        data.ctrl[gripper_actuator_ids[0]] = 0.04
        data.ctrl[gripper_actuator_ids[1]] = -0.04
        if debug and physics_step == 0:
            print("  Command: OPEN GRIPPER")
    else:  # Close gripper
        data.ctrl[gripper_actuator_ids[0]] = 0.0
        data.ctrl[gripper_actuator_ids[1]] = 0.0
        if debug and physics_step == 0:
            print("  Command: CLOSE GRIPPER")

def visualize_tactile_readings(left_readings, right_readings, output_path, tactile_sensor):
    """Create visualization of tactile readings and save to file."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Sample timesteps for visualization (beginning, middle, end)
        sample_steps = [0, len(left_readings) // 2, len(left_readings) - 1]
        
        for i, step in enumerate(sample_steps):
            # Visualize left finger
            tactile_sensor.visualize_reading(left_readings[step], ax=axes[0, i], title=f"Left Finger (Step {step})")
            
            # Visualize right finger
            tactile_sensor.visualize_reading(right_readings[step], ax=axes[1, i], title=f"Right Finger (Step {step})")
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Saved tactile visualization to {output_path}")
        plt.close(fig)  # Explicitly close the figure
    except Exception as e:
        print(f"Warning: Could not visualize tactile readings: {e}")

def load_dataset(dataset_path, demo_idx):
    """Load demonstration data from an HDF5 file."""
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
            
            # Extract data (actions, states, observations)
            result = {
                'actions': demo_group['actions'][:] if 'actions' in demo_group else None,
                'states': demo_group['states'][:] if 'states' in demo_group else None,
                'object_obs': None
            }
            
            if 'obs' in demo_group:
                if 'object' in demo_group['obs']:
                    result['object_obs'] = demo_group['obs']['object'][:]
            
            return result
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def setup_rendering(model, data, render, save_video, camera_id, output_dir, demo_idx):
    """Set up rendering and video capture."""
    viewer_instance = None
    renderer = None
    video_writer = None
    
    if render:
        try:
            viewer_instance = viewer.launch(model, data)
            viewer_instance.sync()  # Initial sync
            
            # Set camera view if needed
            if camera_id is None:
                viewer_instance.cam.azimuth = 90
                viewer_instance.cam.elevation = -15
                viewer_instance.cam.distance = 1.5
                viewer_instance.cam.lookat = [0.0, 0.0, 0.5]
                print(f"Using default full robot camera view")
            elif isinstance(camera_id, int) and 0 <= camera_id < model.ncam:
                viewer_instance.cam.azimuth = model.cam_azimuth[camera_id]
                viewer_instance.cam.elevation = model.cam_elevation[camera_id]
                viewer_instance.cam.distance = model.cam_distance[camera_id]
                print(f"Using camera {camera_id}")
            else:
                print(f"Invalid camera ID: {camera_id}")
        except Exception as e:
            print(f"Warning: Could not start viewer: {e}")
            render = False
    
    if save_video:
        try:
            viewport_width, viewport_height = 640, 480
            renderer = mujoco.Renderer(model, viewport_height, viewport_width)
            
            video_path = os.path.join(output_dir, f"replay_demo{demo_idx}.mp4") if output_dir else f"replay_demo{demo_idx}.mp4"
            
            # Add explicit check for directory permissions
            output_dir_path = os.path.dirname(video_path)
            if not os.access(output_dir_path, os.W_OK):
                print(f"WARNING: No write permission for output directory: {output_dir_path}")
                
            # Add more verbose messaging about video path
            print(f"Will save video to {os.path.abspath(video_path)}")
            
            # More explicit writer setup with higher quality
            video_writer = imageio.get_writer(video_path, fps=30, quality=8, macro_block_size=None)
                
        except Exception as e:
            print(f"Warning: Could not set up video recording: {e}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()
            save_video = False
            if renderer is not None:
                try:
                    renderer.close()
                except:
                    pass
                renderer = None
    
    return {
        'viewer_instance': viewer_instance,
        'renderer': renderer,
        'video_writer': video_writer,
        'render': render,
        'save_video': save_video
    }

class Robot:
    """Class to encapsulate robot joint/actuator operations."""
    
    def __init__(self, model, data, env_info):
        """Initialize the robot with model and environment information."""
        self.model = model
        self.data = data
        self.env_info = env_info
        self.joint_prefix = env_info['joint_prefix']
        self.robot_joint_map = env_info['robot_joint_map']
        self.gripper_actuator_ids = env_info['gripper_actuator_ids']
        
        # Cache joint IDs and addresses
        self.joint_ids = {}
        for i in range(7):
            joint_name = f"{self.joint_prefix}{i+1}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                self.joint_ids[i+1] = {
                    'id': joint_id,
                    'qpos_adr': model.jnt_qposadr[joint_id],
                    'qvel_adr': model.jnt_dofadr[joint_id]
                }
    
    def apply_joint_action(self, joint_idx, action_val, control_mode, action_scale):
        """Apply action to a specific joint."""
        if joint_idx not in self.joint_ids or joint_idx not in self.robot_joint_map:
            return None
            
        actuator_idx = self.robot_joint_map[joint_idx]
        joint_data = self.joint_ids[joint_idx]
        
        # Handle action based on control mode
        if control_mode == "velocity":
            # Direct velocity control
            scaled_action = action_val * action_scale
            self.data.qvel[joint_data['qvel_adr']] = scaled_action
        elif control_mode == "incremental":
            # Incremental position change
            current_pos = self.data.qpos[joint_data['qpos_adr']]
            new_pos = current_pos + action_val * 0.1  # Small increment
            
            # Apply limits
            ctrl_range = self.model.actuator_ctrlrange[actuator_idx]
            scaled_action = max(ctrl_range[0], min(new_pos, ctrl_range[1]))
            self.data.ctrl[actuator_idx] = scaled_action
        else:  # "position" mode
            # Direct position control
            scaled_action = action_val * action_scale
            
            # Apply limits
            ctrl_range = self.model.actuator_ctrlrange[actuator_idx]
            scaled_action = max(ctrl_range[0], min(scaled_action, ctrl_range[1]))
            self.data.ctrl[actuator_idx] = scaled_action
        
        return scaled_action
    
    def apply_gripper_action(self, gripper_cmd):
        """Apply action to the gripper based on command value."""
        if len(self.gripper_actuator_ids) < 2:
            return
            
        if gripper_cmd < 0:  # Open gripper
            self.data.ctrl[self.gripper_actuator_ids[0]] = 0.04
            self.data.ctrl[self.gripper_actuator_ids[1]] = -0.04
        else:  # Close gripper
            self.data.ctrl[self.gripper_actuator_ids[0]] = 0.0
            self.data.ctrl[self.gripper_actuator_ids[1]] = 0.0

def replay_full_robot(dataset_path, demo_idx=0, render=True, save_video=False, model_xml_path=None, output_dir=None, camera_id=None, playback_speed=0.01, action_scale=20.0, control_mode="velocity"):
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
        action_scale: Scaling factor to apply to arm joint actions (default: 20.0)
        control_mode: How to interpret actions: "position" (default), "velocity", or "incremental"
    """
    print(f"Loading dataset from {dataset_path}")
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Outputs will be saved to: {output_dir}")
    
    # Load the dataset
    demo_data = load_dataset(dataset_path, demo_idx)
    actions = demo_data['actions']
    object_obs = demo_data['object_obs']
    
    # Check if the dataset has a model file attribute
    model_file = model_xml_path
    if model_file is None:
        # Use our custom two-cube model instead of the default
        model_file = "franka_emika_panda/mjx_two_cubes.xml"
        print(f"Using two-cube model: {model_file}")
                    
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
        logger.info(f"Loaded MuJoCo model: {model_file}")
        
        # Extract joint names for environment setup
        joint_names = []
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                joint_names.append(joint_name)
        
        # Set up environment and get all IDs in one call
        env_info = setup_environment(model, data, joint_names)
        
        # Add print statement to check timestep (kept for debugging)
        print(f"MuJoCo timestep: {model.opt.timestep} seconds")
        
        # Use environment info to initialize tactile sensor
        tactile_sensor = TactileSensor(
            model=model, 
            data=data,
            left_finger_name=env_info["left_finger_name"],
            right_finger_name=env_info["right_finger_name"],
            collision_geom_prefixes=["fingertip_pad_collision"]
        )
        
        # Setup rendering
        render_info = setup_rendering(model, data, render, save_video, camera_id, output_dir, demo_idx)
        viewer_instance = render_info['viewer_instance']
        renderer = render_info['renderer']
        video_writer = render_info['video_writer']
        render = render_info['render']
        save_video = render_info['save_video']
        
        # Arrays to store tactile readings for each finger
        left_readings = []
        right_readings = []
        
        # Precompute all joint and actuator IDs once
        joint_ids = {}
        for i in range(7):
            joint_name = f"{env_info['joint_prefix']}{i+1}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                joint_ids[i+1] = {
                    'id': joint_id,
                    'qpos_adr': model.jnt_qposadr[joint_id],
                    'qvel_adr': model.jnt_dofadr[joint_id]
                }
        
        # After loading the dataset and before starting the replay loop
        # Initialize robot pose from the first state in the dataset
        if demo_data['states'] is not None and len(demo_data['states']) > 0:
            initial_state = demo_data['states'][0]
            
            # Set initial joint positions for the robot
            for i in range(min(7, len(initial_state))):
                joint_idx = i+1
                if joint_idx in joint_ids and joint_idx in env_info['robot_joint_map']:
                    joint_data = joint_ids[joint_idx]
                    # Set the initial joint position
                    data.qpos[joint_data['qpos_adr']] = initial_state[i]
            
            # Also set initial gripper position if available
            if len(initial_state) > 7:
                # Apply initial gripper position
                gripper_pos = initial_state[7]
                if len(env_info["gripper_actuator_ids"]) >= 2:
                    data.ctrl[env_info["gripper_actuator_ids"][0]] = gripper_pos
                    data.ctrl[env_info["gripper_actuator_ids"][1]] = -gripper_pos
            
            # Update the physics with the new position
            mujoco.mj_forward(model, data)
            print("Initialized robot to demonstration starting pose")
        
        # Define steps to analyze in detail
        analyze_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Analyze first 10 steps
        
        # Number of physics steps per frame
        steps_per_frame = max(1, int(0.01 / model.opt.timestep))  # Aim for ~100Hz control rate
        print(f"Running {steps_per_frame} physics steps per frame")
        
        # Replay the demonstration with much faster physics
        print("Starting replay...")
        
        try:
            # Use the cached IDs in the simulation loop
            for step, action in enumerate(actions):
                # Add detailed analysis for steps we want to examine
                if step in analyze_steps:
                    print(f"\n=== DETAILED ANALYSIS FOR STEP {step} ===")
                    print(f"  Raw action vector: {action}")
                    
                    # Print joint positions, velocities and the applied actions
                    for i in range(min(7, len(action))):
                        joint_idx = i+1
                        if joint_idx in joint_ids and joint_idx in env_info['robot_joint_map']:
                            joint_data = joint_ids[joint_idx]
                            pos = data.qpos[joint_data['qpos_adr']]
                            vel = data.qvel[joint_data['qvel_adr']]
                            scaled = action[i] * action_scale
                            print(f"  Joint {joint_idx}: pos={pos:.4f}, vel={vel:.4f}, raw_action={action[i]:.4f}, scaled={scaled:.4f}")
                    
                    # Print gripper command
                    gripper_cmd = action[6] if len(action) > 6 else 0
                    print(f"  Gripper command: {gripper_cmd:.4f} ({'OPEN' if gripper_cmd < 0 else 'CLOSE'})")
                    
                    # Print cube positions if available
                    if object_obs is not None and step < len(object_obs):
                        cubeA_pos, cubeA_quat, cubeB_pos, cubeB_quat = parse_object_data(object_obs[step])
                        print(f"  CubeA position: {cubeA_pos}")
                        print(f"  CubeA quaternion: {cubeA_quat}")
                        print(f"  CubeB position: {cubeB_pos}")
                        print(f"  CubeB quaternion: {cubeB_quat}")
                
                # Apply joint actions
                for i in range(min(7, len(action))):
                    joint_idx = i+1
                    if joint_idx in joint_ids and joint_idx in env_info['robot_joint_map']:
                        actuator_idx = env_info['robot_joint_map'][joint_idx]
                        joint_data = joint_ids[joint_idx]
                        
                        # Apply action based on control mode
                        if control_mode == "velocity":
                            scaled_action = action[i] * action_scale
                            data.qvel[joint_data['qvel_adr']] = scaled_action
                        else:
                            # Position-based control
                            scaled_action = action[i] * action_scale
                            data.ctrl[actuator_idx] = scaled_action
                        
                        # Additional debugging code can go here...
                
                # Extract gripper command
                gripper_cmd = action[6] if len(action) > 6 else 0
                
                # Run physics steps
                for physics_step in range(steps_per_frame):
                    # Apply gripper action
                    apply_gripper_action(
                        gripper_cmd=gripper_cmd,
                        gripper_actuator_ids=env_info["gripper_actuator_ids"],
                        data=data,
                        debug=(step in analyze_steps),
                        physics_step=physics_step
                    )
                    
                    # Update cube positions
                    update_cube_positions(
                        step=step,
                        object_obs=object_obs,
                        model=model,
                        data=data,
                        env_info=env_info
                    )
                    
                    # Step the simulation
                    mujoco.mj_step(model, data)
                
                # Get tactile readings
                left_reading, right_reading = tactile_sensor.get_readings(model, data)
                left_readings.append(left_reading.copy())
                right_readings.append(right_reading.copy())
                
                # Print less frequent progress updates
                if step % 20 == 0 or step == len(actions) - 1:
                    print(f"Step {step}/{len(actions)}")
                
                # Render if needed (moved outside if/else to run for every step)
                if render and viewer_instance is not None:
                    viewer_instance.sync()
                    time.sleep(playback_speed)  # Still use the configurable playback speed
                
                # Capture frame for video (moved outside if/else to capture every step)
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
        except Exception as e:
            print(f"Error during replay: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up resources
            if save_video and 'video_writer' in locals() and video_writer is not None:
                try:
                    video_writer.close()
                    video_path = os.path.join(output_dir, f"replay_demo{demo_idx}.mp4") if output_dir else f"replay_demo{demo_idx}.mp4"
                    
                    # Check if video file exists and has meaningful content
                    if os.path.exists(video_path) and os.path.getsize(video_path) > 10000:  # At least 10KB
                        print(f"Video successfully saved to {video_path} ({os.path.getsize(video_path)/1024:.1f} KB)")
                    else:
                        print(f"WARNING: Video file may be empty or invalid: {video_path}")
                        if os.path.exists(video_path):
                            print(f"  File size: {os.path.getsize(video_path)} bytes (should be >10KB)")
                        else:
                            print(f"  File doesn't exist!")
                except Exception as e:
                    print(f"Error while finalizing video: {e}")
            
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
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        raise

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
    parser.add_argument("--action-scale", type=float, default=20.0,
                        help="Scale factor for arm joint actions (default: 20.0)")
    parser.add_argument("--control-mode", type=str, choices=["position", "velocity", "incremental"],
                        default="velocity", 
                        help="How to interpret actions (default: velocity)")
    
    # Add note about the playback speed here
    print("NOTE: Default playback_speed is 0.01 sec between frames.")
    print("      To make the robot move 5× faster, use --playback-speed 0.002")
    
    args = parser.parse_args()
    
    # Add a clear message about the speed issue
    if args.playback_speed == 0.01:
        print("\nNOTE: Running at 0.01 sec/step which is 5x slower than real-time.")
        print("To run at real-time speed, use: --playback-speed 0.002")
        print("To run at 2x real-time speed, use: --playback-speed 0.001\n")
        
    replay_full_robot(args.dataset, args.demo, args.render, args.save_video, args.model, args.output_dir, args.camera, args.playback_speed, args.action_scale, args.control_mode)
        