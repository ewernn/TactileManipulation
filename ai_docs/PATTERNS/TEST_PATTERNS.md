# 🧪 Testing Patterns

*Common patterns for testing robotic systems and validating implementations*

## 🎯 Environment Testing Patterns

### **Basic Environment Test**
```python
import pytest
import numpy as np
from environments.simple_tactile_env import SimpleTactileEnv

class TestEnvironmentBasics:
    """Test basic environment functionality"""
    
    def test_environment_creation(self):
        """Test environment can be created"""
        env = SimpleTactileEnv()
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None
        
    def test_reset(self):
        """Test environment reset"""
        env = SimpleTactileEnv()
        obs, info = env.reset()
        
        # Check observation structure
        assert isinstance(obs, dict)
        assert 'robot_joint_pos' in obs
        assert 'tactile_left' in obs
        assert 'tactile_right' in obs
        
        # Check dimensions
        assert obs['tactile_left'].shape == (3, 4, 3)
        assert obs['tactile_right'].shape == (3, 4, 3)
        
    def test_step(self):
        """Test environment step"""
        env = SimpleTactileEnv()
        obs, _ = env.reset()
        
        # Random action
        action = env.action_space.sample()
        
        # Step
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Validate outputs
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
```

### **Physics Validation Tests**
```python
class TestPhysicsValidation:
    """Validate physics behavior"""
    
    def test_gravity(self):
        """Test that objects fall under gravity"""
        env = SimpleTactileEnv()
        env.reset()
        
        # Get initial object height
        initial_height = env.data.body('object').xpos[2]
        
        # Step with gripper open
        for _ in range(100):
            action = np.array([0, 0, 0, -1])  # Gripper open
            env.step(action)
            
        # Object should have fallen
        final_height = env.data.body('object').xpos[2]
        assert final_height < initial_height
        
    def test_collision(self):
        """Test collision detection"""
        env = SimpleTactileEnv()
        env.reset()
        
        # Move gripper to object
        for _ in range(50):
            action = np.array([0, -0.5, 0, -1])  # Move down
            env.step(action)
            
        # Close gripper
        for _ in range(50):
            action = np.array([0, 0, 0, 1])  # Close
            obs, _, _, _, _ = env.step(action)
            
        # Should have tactile readings
        left_force = np.sum(obs['tactile_left'][:, :, 0])
        right_force = np.sum(obs['tactile_right'][:, :, 0])
        
        assert left_force > 0 or right_force > 0
```

## 🔧 Sensor Testing Patterns

### **Tactile Sensor Tests**
```python
class TestTactileSensor:
    """Test tactile sensor functionality"""
    
    def test_sensor_initialization(self):
        """Test sensor creation"""
        from environments.tactile_sensor import TactileSensor
        
        sensor = TactileSensor(n_taxels_x=3, n_taxels_y=4)
        assert sensor.n_taxels_x == 3
        assert sensor.n_taxels_y == 4
        
    def test_no_contact_readings(self):
        """Test sensor with no contacts"""
        env = SimpleTactileEnv()
        env.reset()
        
        # Ensure no contacts
        env.data.qpos[env.gripper_joint_ids] = 0.04  # Full open
        mujoco.mj_forward(env.model, env.data)
        
        left, right = env.tactile_sensor.get_readings(env.model, env.data)
        
        # Should be all zeros
        assert np.allclose(left, 0)
        assert np.allclose(right, 0)
        
    def test_contact_detection(self):
        """Test sensor detects contacts"""
        env = SimpleTactileEnv()
        
        # Create controlled contact scenario
        env.reset()
        
        # Position gripper around object
        env.data.qpos[:3] = [0, -0.1, 0]  # Positioned to grasp
        env.data.qpos[env.gripper_joint_ids] = 0.01  # Partially closed
        mujoco.mj_forward(env.model, env.data)
        
        # Force simulation step to generate contacts
        mujoco.mj_step(env.model, env.data)
        
        left, right = env.tactile_sensor.get_readings(env.model, env.data)
        
        # Should have some readings
        total_force = np.sum(left[:, :, 0]) + np.sum(right[:, :, 0])
        assert total_force > 0
```

### **Noise and Calibration Tests**
```python
def test_sensor_noise():
    """Test sensor noise characteristics"""
    env = SimpleTactileEnv()
    env.reset()
    
    # Collect multiple readings in same state
    readings = []
    for _ in range(100):
        left, right = env.tactile_sensor.get_readings(env.model, env.data)
        readings.append(np.concatenate([left.flatten(), right.flatten()]))
        
    readings = np.array(readings)
    
    # Check noise statistics
    mean = np.mean(readings, axis=0)
    std = np.std(readings, axis=0)
    
    # Noise should be present but small
    assert np.any(std > 0)  # Has noise
    assert np.all(std < 0.1)  # But not too much
```

## 🎯 Policy Testing Patterns

### **Policy Evaluation Test**
```python
class TestPolicyEvaluation:
    """Test trained policies"""
    
    def test_scripted_policy(self):
        """Test scripted expert policy"""
        from scripts.collect_direct import scripted_expert_policy
        
        env = SimpleTactileEnv()
        obs, _ = env.reset()
        
        success_count = 0
        
        for episode in range(10):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action = scripted_expert_policy(obs, env.step_count)
                obs, reward, done, truncated, _ = env.step(action)
                
            # Check if successful
            final_height = env.data.body('object').xpos[2]
            if final_height > 0.1:
                success_count += 1
                
        # Should have high success rate
        assert success_count >= 8  # 80% success
        
    def test_learned_policy(self):
        """Test learned policy performance"""
        import joblib
        
        # Load policy
        policy = joblib.load('models/policy.pkl')
        env = SimpleTactileEnv()
        
        # Evaluate
        rewards = []
        for episode in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                obs_vector = preprocess_observation(obs)
                action = policy.predict([obs_vector])[0]
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                
            rewards.append(episode_reward)
            
        # Check performance
        mean_reward = np.mean(rewards)
        assert mean_reward > 0  # Should get positive rewards
```

### **Robustness Tests**
```python
def test_policy_robustness():
    """Test policy under various conditions"""
    env = SimpleTactileEnv()
    policy = load_policy()
    
    test_conditions = [
        {'object_mass': 0.05},  # Light object
        {'object_mass': 0.2},   # Heavy object
        {'friction': 0.5},      # Low friction
        {'friction': 2.0},      # High friction
        {'noise_std': 0.02},    # More sensor noise
    ]
    
    results = {}
    
    for condition_name, params in test_conditions:
        # Modify environment
        if 'object_mass' in params:
            env.model.body_mass[env.object_body_id] = params['object_mass']
        elif 'friction' in params:
            env.model.geom_friction[:] = params['friction']
        elif 'noise_std' in params:
            env.tactile_sensor.noise_std = params['noise_std']
            
        # Test policy
        success_rate = evaluate_policy(env, policy, n_episodes=20)
        results[condition_name] = success_rate
        
        # Reset environment
        env = SimpleTactileEnv()  # Fresh environment
        
    # Policy should be reasonably robust
    for condition, success_rate in results.items():
        assert success_rate > 0.5, f"Policy failed under {condition}"
```

## 🐛 Integration Testing Patterns

### **End-to-End Test**
```python
def test_full_pipeline():
    """Test complete data collection to training pipeline"""
    
    # 1. Environment creation
    env = SimpleTactileEnv()
    assert env is not None
    
    # 2. Data collection
    from scripts.collect_direct import collect_demonstrations
    demos = collect_demonstrations(env, n_demos=5)
    assert len(demos) == 5
    
    # 3. Save data
    save_path = 'test_demos.hdf5'
    save_demonstrations(demos, save_path)
    assert os.path.exists(save_path)
    
    # 4. Load and validate data
    with h5py.File(save_path, 'r') as f:
        assert 'data' in f
        assert len(f['data']) == 5
        
    # 5. Train simple policy
    from scripts.train_simple import train_policy
    model = train_policy(save_path)
    assert model is not None
    
    # 6. Evaluate policy
    success_rate = evaluate_policy(env, model, n_episodes=5)
    assert success_rate > 0
    
    # Cleanup
    os.remove(save_path)
```

### **Performance Benchmarks**
```python
import time

def benchmark_environment():
    """Benchmark environment performance"""
    env = SimpleTactileEnv()
    
    # Reset benchmark
    reset_times = []
    for _ in range(100):
        start = time.time()
        env.reset()
        reset_times.append(time.time() - start)
        
    # Step benchmark
    env.reset()
    step_times = []
    for _ in range(1000):
        action = env.action_space.sample()
        start = time.time()
        env.step(action)
        step_times.append(time.time() - start)
        
    # Report results
    print(f"Reset: {np.mean(reset_times)*1000:.2f}ms ± {np.std(reset_times)*1000:.2f}ms")
    print(f"Step: {np.mean(step_times)*1000:.2f}ms ± {np.std(step_times)*1000:.2f}ms")
    print(f"FPS: {1.0/np.mean(step_times):.1f}")
    
    # Performance requirements
    assert np.mean(reset_times) < 0.1  # < 100ms
    assert np.mean(step_times) < 0.01  # < 10ms (>100 FPS)
```

## 📊 Test Utilities

### **Visualization for Debugging**
```python
def visualize_test_trajectory(env, policy, save_path='test_trajectory.mp4'):
    """Create video of test episode for debugging"""
    frames = []
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        # Render
        frame = env.render()
        
        # Add debug info
        frame = add_debug_overlay(frame, obs, env)
        frames.append(frame)
        
        # Step
        action = policy(obs)
        obs, reward, done, truncated, _ = env.step(action)
        
    # Save video
    save_video(frames, save_path)
    return save_path

def add_debug_overlay(frame, obs, env):
    """Add debug information to frame"""
    import cv2
    
    # Add text overlay
    text_lines = [
        f"Step: {env.step_count}",
        f"Gripper: {obs['gripper_pos'][0]:.3f}",
        f"Object Z: {obs['object_pos'][2]:.3f}",
        f"Left Force: {np.sum(obs['tactile_left'][:,:,0]):.2f}",
        f"Right Force: {np.sum(obs['tactile_right'][:,:,0]):.2f}"
    ]
    
    y = 30
    for line in text_lines:
        cv2.putText(frame, line, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        
    return frame
```

### **Test Data Fixtures**
```python
@pytest.fixture
def sample_trajectory():
    """Provide sample trajectory for testing"""
    return {
        'observations': [
            {
                'robot_joint_pos': np.random.randn(4),
                'object_pos': np.array([0, 0, 0.05]),
                'tactile_left': np.zeros((3, 4, 3)),
                'tactile_right': np.zeros((3, 4, 3))
            }
            for _ in range(100)
        ],
        'actions': [np.random.randn(4) for _ in range(100)],
        'rewards': [0.0] * 100,
        'success': True
    }

@pytest.fixture
def test_environment():
    """Provide configured test environment"""
    env = SimpleTactileEnv()
    env.reset(seed=42)  # Deterministic
    return env
```

## 🔗 Related Patterns
- Environment testing → `/ai_docs/ENVIRONMENT_HUB.md`
- Data validation → `/ai_docs/PATTERNS/DATA_PATTERNS.md`
- Performance metrics → `/ai_docs/VISUALIZATION_HUB.md`