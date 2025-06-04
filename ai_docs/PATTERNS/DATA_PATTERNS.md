# 📊 Data Handling Patterns

*Common patterns for data collection, storage, and processing in robotics*

## 🎯 HDF5 Data Storage Patterns

### **Standard Dataset Structure**
```python
import h5py
import numpy as np
from datetime import datetime

def create_dataset_structure(filename):
    """Create standard HDF5 dataset structure"""
    with h5py.File(filename, 'w') as f:
        # Metadata group
        meta = f.create_group('metadata')
        meta.attrs['created'] = str(datetime.now())
        meta.attrs['version'] = '1.0'
        meta.attrs['env_name'] = 'TactileGrasping'
        
        # Data group
        data = f.create_group('data')
        
        # Optional groups
        f.create_group('statistics')  # Computed stats
        f.create_group('models')      # Trained models
        
    return filename
```

### **Efficient Demo Storage**
```python
def save_demonstration(hdf5_file, demo_id, trajectory):
    """Save single demonstration efficiently"""
    with h5py.File(hdf5_file, 'a') as f:
        # Create demo group
        demo_grp = f['data'].create_group(f'demo_{demo_id}')
        
        # Save observations (nested structure)
        obs_grp = demo_grp.create_group('observations')
        first_obs = trajectory['observations'][0]
        
        for key, value in first_obs.items():
            # Stack all timesteps
            if isinstance(value, dict):
                # Nested observations
                sub_grp = obs_grp.create_group(key)
                for sub_key, sub_value in value.items():
                    data = np.array([o[key][sub_key] for o in trajectory['observations']])
                    sub_grp.create_dataset(sub_key, data=data, compression='gzip')
            else:
                # Direct observations
                data = np.array([o[key] for o in trajectory['observations']])
                obs_grp.create_dataset(key, data=data, compression='gzip')
        
        # Save actions
        actions = np.array(trajectory['actions'])
        demo_grp.create_dataset('actions', data=actions, compression='gzip')
        
        # Save rewards if available
        if 'rewards' in trajectory:
            rewards = np.array(trajectory['rewards'])
            demo_grp.create_dataset('rewards', data=rewards, compression='gzip')
        
        # Demo metadata
        demo_grp.attrs['success'] = trajectory.get('success', False)
        demo_grp.attrs['length'] = len(trajectory['observations'])
        demo_grp.attrs['total_reward'] = sum(trajectory.get('rewards', [0]))
```

### **Lazy Loading Pattern**
```python
class LazyDataset:
    """Load data on-demand to save memory"""
    def __init__(self, hdf5_path):
        self.path = hdf5_path
        self._file = None
        self._demo_keys = None
    
    def __enter__(self):
        self._file = h5py.File(self.path, 'r')
        self._demo_keys = list(self._file['data'].keys())
        return self
    
    def __exit__(self, *args):
        if self._file:
            self._file.close()
    
    def __len__(self):
        return len(self._demo_keys)
    
    def __getitem__(self, idx):
        """Load single demonstration"""
        demo = self._file['data'][self._demo_keys[idx]]
        return {
            'observations': {k: v[:] for k, v in demo['observations'].items()},
            'actions': demo['actions'][:],
            'success': demo.attrs.get('success', False)
        }
```

## 🔧 Data Collection Patterns

### **Robust Collection with Recovery**
```python
def collect_demonstrations_robust(env, n_demos, save_path):
    """Collect demos with error recovery"""
    collected = []
    failures = []
    
    with h5py.File(save_path, 'w') as f:
        # Initialize structure
        create_dataset_structure(save_path)
        
        pbar = tqdm(total=n_demos, desc="Collecting demos")
        demo_id = 0
        
        while len(collected) < n_demos:
            try:
                # Collect single demo
                trajectory = collect_single_demo(env)
                
                # Validate before saving
                if validate_trajectory(trajectory):
                    # Save immediately to avoid memory issues
                    save_demonstration(f, demo_id, trajectory)
                    collected.append(demo_id)
                    demo_id += 1
                    pbar.update(1)
                else:
                    failures.append({
                        'reason': 'validation_failed',
                        'demo_id': demo_id
                    })
                    
            except Exception as e:
                failures.append({
                    'reason': str(e),
                    'demo_id': demo_id
                })
                # Reset environment after failure
                env.reset()
        
        # Save collection statistics
        f['metadata'].attrs['n_demos'] = len(collected)
        f['metadata'].attrs['n_failures'] = len(failures)
        
    return collected, failures
```

### **Parallel Data Collection**
```python
from multiprocessing import Pool, Queue
import pickle

def collect_worker(args):
    """Worker process for parallel collection"""
    worker_id, env_config, n_demos, queue = args
    
    # Create environment in worker
    env = create_env(env_config)
    demos = []
    
    for i in range(n_demos):
        try:
            demo = collect_single_demo(env)
            demos.append(demo)
            queue.put(('progress', worker_id, i))
        except Exception as e:
            queue.put(('error', worker_id, str(e)))
    
    return demos

def collect_parallel(env_config, total_demos, n_workers=4):
    """Collect demonstrations in parallel"""
    demos_per_worker = total_demos // n_workers
    manager = mp.Manager()
    queue = manager.Queue()
    
    # Create worker arguments
    worker_args = [
        (i, env_config, demos_per_worker, queue)
        for i in range(n_workers)
    ]
    
    # Start collection
    with Pool(n_workers) as pool:
        # Start async collection
        results = pool.map_async(collect_worker, worker_args)
        
        # Monitor progress
        pbar = tqdm(total=total_demos)
        while not results.ready():
            try:
                msg_type, worker_id, data = queue.get(timeout=0.1)
                if msg_type == 'progress':
                    pbar.update(1)
                elif msg_type == 'error':
                    print(f"Worker {worker_id} error: {data}")
            except:
                pass
        
        # Collect results
        all_demos = []
        for worker_demos in results.get():
            all_demos.extend(worker_demos)
    
    return all_demos
```

## 🎯 Data Preprocessing Patterns

### **Normalization Pipeline**
```python
class DataNormalizer:
    """Normalize observations and actions"""
    def __init__(self):
        self.obs_mean = {}
        self.obs_std = {}
        self.action_mean = None
        self.action_std = None
    
    def fit(self, dataset):
        """Compute normalization statistics"""
        # Collect all observations
        all_obs = defaultdict(list)
        all_actions = []
        
        for demo in dataset:
            for obs in demo['observations']:
                for key, value in obs.items():
                    all_obs[key].append(value)
            all_actions.extend(demo['actions'])
        
        # Compute statistics
        for key, values in all_obs.items():
            data = np.array(values)
            self.obs_mean[key] = data.mean(axis=0)
            self.obs_std[key] = data.std(axis=0) + 1e-6
        
        actions = np.array(all_actions)
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0) + 1e-6
    
    def normalize_obs(self, obs):
        """Normalize observation dict"""
        normalized = {}
        for key, value in obs.items():
            normalized[key] = (value - self.obs_mean[key]) / self.obs_std[key]
        return normalized
    
    def normalize_action(self, action):
        """Normalize action"""
        return (action - self.action_mean) / self.action_std
    
    def denormalize_action(self, normalized_action):
        """Convert back to original scale"""
        return normalized_action * self.action_std + self.action_mean
```

### **Data Augmentation**
```python
def augment_demonstration(demo, augmentation_config):
    """Apply data augmentation to demonstration"""
    augmented = []
    
    # Original demo
    augmented.append(demo)
    
    # Noise augmentation
    if augmentation_config.get('add_noise', False):
        noisy_demo = copy.deepcopy(demo)
        for i, obs in enumerate(noisy_demo['observations']):
            # Add observation noise
            for key in ['robot_joint_pos', 'object_pos']:
                if key in obs:
                    noise = np.random.normal(0, 0.01, obs[key].shape)
                    obs[key] = obs[key] + noise
            
            # Add action noise
            if i < len(noisy_demo['actions']):
                action_noise = np.random.normal(0, 0.05, noisy_demo['actions'][i].shape)
                noisy_demo['actions'][i] = np.clip(
                    noisy_demo['actions'][i] + action_noise, -1, 1
                )
        augmented.append(noisy_demo)
    
    # Time reversal (for reversible tasks)
    if augmentation_config.get('time_reversal', False):
        reversed_demo = {
            'observations': demo['observations'][::-1],
            'actions': demo['actions'][::-1] * -1,  # Reverse actions
            'success': demo['success']
        }
        augmented.append(reversed_demo)
    
    return augmented
```

## 📊 Data Analysis Patterns

### **Dataset Statistics**
```python
def compute_dataset_statistics(hdf5_path):
    """Comprehensive dataset analysis"""
    stats = {
        'n_demos': 0,
        'n_success': 0,
        'total_steps': 0,
        'avg_episode_length': 0,
        'action_stats': {},
        'observation_stats': {},
        'tactile_stats': {}
    }
    
    with h5py.File(hdf5_path, 'r') as f:
        demos = f['data']
        stats['n_demos'] = len(demos)
        
        episode_lengths = []
        all_actions = []
        tactile_forces = []
        
        for demo_name in demos:
            demo = demos[demo_name]
            
            # Basic stats
            if demo.attrs.get('success', False):
                stats['n_success'] += 1
            
            length = demo['actions'].shape[0]
            episode_lengths.append(length)
            stats['total_steps'] += length
            
            # Action statistics
            all_actions.append(demo['actions'][:])
            
            # Tactile statistics
            if 'observations/tactile_left' in demo:
                left_tactile = demo['observations/tactile_left'][:]
                right_tactile = demo['observations/tactile_right'][:]
                
                # Total force per timestep
                total_force = (
                    np.sum(left_tactile[:, :, :, 0], axis=(1, 2)) +
                    np.sum(right_tactile[:, :, :, 0], axis=(1, 2))
                )
                tactile_forces.append(total_force)
        
        # Compute aggregated stats
        stats['avg_episode_length'] = np.mean(episode_lengths)
        stats['success_rate'] = stats['n_success'] / stats['n_demos']
        
        # Action statistics
        all_actions = np.concatenate(all_actions, axis=0)
        stats['action_stats'] = {
            'mean': all_actions.mean(axis=0).tolist(),
            'std': all_actions.std(axis=0).tolist(),
            'min': all_actions.min(axis=0).tolist(),
            'max': all_actions.max(axis=0).tolist()
        }
        
        # Tactile statistics
        if tactile_forces:
            all_forces = np.concatenate(tactile_forces)
            stats['tactile_stats'] = {
                'mean_force': float(all_forces.mean()),
                'max_force': float(all_forces.max()),
                'force_std': float(all_forces.std())
            }
    
    return stats
```

### **Data Validation**
```python
def validate_dataset(hdf5_path):
    """Validate dataset integrity"""
    errors = []
    warnings = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Check structure
        if 'data' not in f:
            errors.append("Missing 'data' group")
            return errors, warnings
        
        if 'metadata' not in f:
            warnings.append("Missing 'metadata' group")
        
        # Check each demonstration
        for demo_name in f['data']:
            demo = f['data'][demo_name]
            
            # Required fields
            if 'observations' not in demo:
                errors.append(f"{demo_name}: Missing observations")
            if 'actions' not in demo:
                errors.append(f"{demo_name}: Missing actions")
            
            # Check dimensions
            if 'observations' in demo and 'actions' in demo:
                n_obs = len(list(demo['observations'].values())[0])
                n_act = len(demo['actions'])
                
                if n_obs != n_act + 1:
                    warnings.append(
                        f"{demo_name}: Observation/action mismatch "
                        f"({n_obs} obs, {n_act} actions)"
                    )
            
            # Check for NaN/Inf
            for dataset in demo.values():
                if isinstance(dataset, h5py.Dataset):
                    data = dataset[:]
                    if np.any(np.isnan(data)):
                        errors.append(f"{demo_name}/{dataset.name}: Contains NaN")
                    if np.any(np.isinf(data)):
                        errors.append(f"{demo_name}/{dataset.name}: Contains Inf")
    
    return errors, warnings
```

## 🔗 Related Patterns
- Environment data → `/ai_docs/PATTERNS/ENVIRONMENT_PATTERNS.md`
- Sensor data → `/ai_docs/PATTERNS/SENSOR_PATTERNS.md`
- Training data → `/ai_docs/PATTERNS/TRAINING_PATTERNS.md`