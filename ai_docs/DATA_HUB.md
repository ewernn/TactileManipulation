# 📊 Data Collection & Management Hub

*Central documentation for data collection, storage, and dataset management*

## 🚀 Quick Reference

### **Key Files**
- Main collection script: `tactile-rl/scripts/collect_direct.py`
- Dataset location: `datasets/tactile_grasping/`
- Analysis tools: `tactile-rl/scripts/create_summary.py`
- HDF5 explorer: `tactile-rl/scripts/explore_hdf5_summary.py`

### **Common Operations**
- Collect demos → `python collect_direct.py --n_demos 50`
- Explore dataset → `python explore_hdf5_summary.py datasets/demos.hdf5`
- Generate summary → `python create_summary.py`
- Replay demos → `python replay_demo.py --file demos.hdf5 --demo 0`

## 🏗️ Dataset Structure

### **HDF5 Format**
```
demos.hdf5
├── data/
│   ├── demo_0/
│   │   ├── observations/
│   │   │   ├── robot_joint_pos     (300, 4)
│   │   │   ├── robot_joint_vel     (300, 4)
│   │   │   ├── gripper_pos         (300, 2)
│   │   │   ├── object_pos          (300, 3)
│   │   │   ├── object_quat         (300, 4)
│   │   │   ├── tactile_left        (300, 3, 4, 3)
│   │   │   └── tactile_right       (300, 3, 4, 3)
│   │   ├── actions                  (300, 4)
│   │   └── rewards                  (300,)
│   ├── demo_1/
│   └── ...
└── metadata/
    ├── n_demos                      50
    ├── env_name                     "SimpleTactileEnv"
    └── collection_time              "2025-05-27"
```

### **Observation Fields**
| Field | Shape | Description |
|-------|-------|-------------|
| robot_joint_pos | (n_joints,) | Joint positions (rad) |
| robot_joint_vel | (n_joints,) | Joint velocities (rad/s) |
| gripper_pos | (2,) | Finger positions (m) |
| object_pos | (3,) | Object xyz position |
| object_quat | (4,) | Object orientation |
| tactile_left | (3, 4, 3) | Left finger tactile |
| tactile_right | (3, 4, 3) | Right finger tactile |

## 🔧 Data Collection Implementation

### **Expert Policy Collection**
```python
# collect_direct.py
def collect_episode(env):
    """Collect single demonstration with 4-phase policy"""
    obs, _ = env.reset()
    observations = []
    actions = []
    
    # Phase 1: Approach (100 steps)
    for t in range(100):
        action = np.array([0.0, -0.2, 0.0, -1.0])  # Move down, gripper open
        obs, reward, done, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
    
    # Phase 2: Descend (50 steps)
    for t in range(50):
        action = np.array([0.0, -0.5, 0.0, -1.0])  # Faster descent
        obs, reward, done, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
    
    # Phase 3: Grasp (50 steps)
    for t in range(50):
        action = np.array([0.0, -0.5, 0.0, 1.0])  # Close gripper
        obs, reward, done, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
    
    # Phase 4: Lift (100 steps)
    for t in range(100):
        action = np.array([0.0, 0.8, 0.0, 1.0])  # Lift up
        obs, reward, done, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
    
    # Check success
    final_height = env.data.body('object').xpos[2]
    success = final_height > 0.1
    
    return {
        'observations': observations,
        'actions': actions,
        'success': success,
        'final_height': final_height
    }
```

### **HDF5 Storage**
```python
def save_demonstrations(demos, filename):
    """Save demonstrations to HDF5 format"""
    with h5py.File(filename, 'w') as f:
        # Create groups
        data_grp = f.create_group('data')
        meta_grp = f.create_group('metadata')
        
        # Save metadata
        meta_grp.attrs['n_demos'] = len(demos)
        meta_grp.attrs['env_name'] = 'SimpleTactileEnv'
        meta_grp.attrs['collection_time'] = str(datetime.now())
        
        # Save each demo
        for i, demo in enumerate(demos):
            demo_grp = data_grp.create_group(f'demo_{i}')
            
            # Save observations
            obs_grp = demo_grp.create_group('observations')
            first_obs = demo['observations'][0]
            
            for key in first_obs:
                data = np.array([o[key] for o in demo['observations']])
                obs_grp.create_dataset(key, data=data, compression='gzip')
            
            # Save actions
            actions = np.array(demo['actions'])
            demo_grp.create_dataset('actions', data=actions, compression='gzip')
            
            # Save metadata
            demo_grp.attrs['success'] = demo['success']
            demo_grp.attrs['final_height'] = demo['final_height']
            demo_grp.attrs['length'] = len(demo['observations'])
```

## 🎯 Data Collection Strategies

### **1. Scripted Expert**
```python
# Fixed sequence of actions
def scripted_policy(obs, phase):
    if phase == 'approach':
        return np.array([0, -0.2, 0, -1])  # Move down, open
    elif phase == 'grasp':
        return np.array([0, -0.5, 0, 1])   # Continue down, close
    elif phase == 'lift':
        return np.array([0, 0.8, 0, 1])    # Move up, stay closed
```

### **2. Teleoperation**
```python
# Keyboard or joystick control
def teleop_policy(keyboard_input):
    action = np.zeros(4)
    if keyboard_input == 'w':
        action[1] = 0.5  # Move up
    elif keyboard_input == 's':
        action[1] = -0.5  # Move down
    # ... more mappings
    return action
```

### **3. Learned Policy**
```python
# Use trained model
def learned_policy(obs, model):
    obs_tensor = torch.FloatTensor(flatten_obs(obs))
    with torch.no_grad():
        action = model(obs_tensor).numpy()
    return action
```

## 📊 Data Analysis Tools

### **Dataset Summary**
```python
# create_summary.py
def analyze_dataset(hdf5_path):
    """Generate dataset statistics and plots"""
    with h5py.File(hdf5_path, 'r') as f:
        n_demos = f['metadata'].attrs['n_demos']
        
        successes = []
        heights = []
        tactile_forces = []
        
        for i in range(n_demos):
            demo = f[f'data/demo_{i}']
            successes.append(demo.attrs['success'])
            heights.append(demo.attrs['final_height'])
            
            # Analyze tactile
            tactile_left = demo['observations/tactile_left'][-1]
            tactile_right = demo['observations/tactile_right'][-1]
            total_force = np.sum(tactile_left[:, :, 0]) + np.sum(tactile_right[:, :, 0])
            tactile_forces.append(total_force)
    
    # Generate plots
    plot_success_rate(successes)
    plot_height_distribution(heights)
    plot_tactile_analysis(tactile_forces)
```

### **Visualization Tools**
```python
def visualize_trajectory(demo_data):
    """Plot robot trajectory and tactile evolution"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Joint positions over time
    joint_pos = demo_data['observations/robot_joint_pos']
    axes[0].plot(joint_pos)
    axes[0].set_title('Joint Positions')
    
    # Object height
    obj_pos = demo_data['observations/object_pos']
    axes[1].plot(obj_pos[:, 2])
    axes[1].set_title('Object Height')
    
    # Tactile force magnitude
    tactile_left = demo_data['observations/tactile_left']
    force_magnitude = np.sum(tactile_left[:, :, :, 0], axis=(1, 2))
    axes[2].plot(force_magnitude)
    axes[2].set_title('Total Tactile Force')
```

## 🐛 Common Issues

### **Memory Issues with Large Datasets**
```python
# Use chunked reading
def process_large_dataset(hdf5_path, chunk_size=10):
    with h5py.File(hdf5_path, 'r') as f:
        n_demos = f['metadata'].attrs['n_demos']
        
        for start in range(0, n_demos, chunk_size):
            end = min(start + chunk_size, n_demos)
            chunk_data = load_demos_range(f, start, end)
            process_chunk(chunk_data)
```

### **Corrupted Demonstrations**
```python
# Validation during collection
def validate_demo(demo):
    """Check demo integrity"""
    # Check lengths match
    n_obs = len(demo['observations'])
    n_act = len(demo['actions'])
    assert n_obs == n_act + 1, "Observation/action mismatch"
    
    # Check for NaN
    for obs in demo['observations']:
        for key, value in obs.items():
            assert not np.any(np.isnan(value)), f"NaN in {key}"
    
    return True
```

### **Dataset Compatibility**
```python
# Convert between formats
def convert_to_robomimic_format(our_dataset, output_path):
    """Convert to RoboMimic HDF5 format"""
    # Implementation depends on specific requirements
    pass
```

## 📈 Performance Metrics

### **Collection Statistics**
- Collection rate: ~130 demos/second
- Average demo length: 300 steps
- Success rate: 100% (with scripted expert)
- Storage: ~1 MB per demo (compressed)

### **Key Metrics Tracked**
| Metric | Description |
|--------|-------------|
| Success Rate | % of demos achieving goal |
| Lift Height | Final object height |
| Grasp Quality | Tactile-based metric |
| Episode Length | Steps to completion |
| Contact Events | Number of contacts |

## 🔗 Related Documentation
- Environment setup → `/ai_docs/ENVIRONMENT_HUB.md`
- Tactile processing → `/ai_docs/TACTILE_HUB.md`
- Training pipelines → `/ai_docs/TRAINING_HUB.md`
- Data patterns → `/ai_docs/PATTERNS/DATA_PATTERNS.md`