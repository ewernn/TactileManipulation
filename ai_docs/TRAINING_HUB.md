# 🎓 Training & Policy Learning Hub

*Central documentation for policy training and evaluation*

## 🚀 Quick Reference

### **Key Files**
- Simple training: `tactile-rl/scripts/train_simple.py`
- Policy models: `tactile-rl/scripts/train_policies.py`
- Evaluation: `tactile-rl/scripts/test_expert.py`
- Visualization: `tactile-rl/scripts/visualize_grasp.py`

### **Common Operations**
- Train policy → `python train_simple.py --data demos.hdf5`
- Evaluate model → `python test_expert.py --model policy.pkl`
- Compare policies → `python compare_policies.py`
- Generate videos → `python generate_video.py --policy trained`

## 🏗️ Architecture

### **Training Pipeline**
```
Dataset (HDF5) → Data Loader → Model Training → Evaluation
     ↓               ↓              ↓              ↓
Demonstrations   Batching    Policy Network   Performance
```

### **Available Approaches**

#### **1. Behavioral Cloning (Simple)**
```python
# train_simple.py - sklearn approach
from sklearn.ensemble import RandomForestRegressor

def train_bc_simple(dataset_path):
    # Load data
    X, y = load_demonstrations(dataset_path)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    # Save
    joblib.dump(model, 'policy_simple.pkl')
    
    return model
```

#### **2. Neural Network BC**
```python
# Neural network approach
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
    
    def forward(self, obs):
        return self.net(obs)
```

#### **3. Reinforcement Learning (Future)**
```python
# RL setup (not implemented yet)
def train_rl_policy(env, algorithm='PPO'):
    # Would use stable-baselines3 or similar
    pass
```

## 🔧 Implementation Details

### **Data Preprocessing**
```python
def preprocess_observations(obs_dict):
    """Flatten observation dict for training"""
    features = []
    
    # Robot state
    features.extend(obs_dict['robot_joint_pos'])
    features.extend(obs_dict['robot_joint_vel'])
    features.extend(obs_dict['gripper_pos'])
    
    # Object state
    features.extend(obs_dict['object_pos'])
    features.extend(obs_dict['object_quat'])
    
    # Tactile (flatten)
    features.extend(obs_dict['tactile_left'].flatten())
    features.extend(obs_dict['tactile_right'].flatten())
    
    return np.array(features)

def load_demonstrations(hdf5_path):
    """Load and preprocess demonstration data"""
    observations = []
    actions = []
    
    with h5py.File(hdf5_path, 'r') as f:
        for demo_name in f['data']:
            demo = f['data'][demo_name]
            
            # Only use successful demos
            if demo.attrs.get('success', False):
                # Process each timestep
                for t in range(len(demo['actions'])):
                    obs = {}
                    for key in demo['observations']:
                        obs[key] = demo['observations'][key][t]
                    
                    observations.append(preprocess_observations(obs))
                    actions.append(demo['actions'][t])
    
    return np.array(observations), np.array(actions)
```

### **Training Loop**
```python
def train_neural_policy(X, y, epochs=100):
    """Train neural network policy"""
    # Create dataset
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    obs_dim = X.shape[1]
    action_dim = y.shape[1]
    model = PolicyNetwork(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Training
    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_actions in dataloader:
            # Forward pass
            pred_actions = model(batch_obs)
            loss = criterion(pred_actions, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")
    
    return model
```

### **Policy Evaluation**
```python
def evaluate_policy(policy, env, n_episodes=50):
    """Evaluate trained policy"""
    successes = []
    heights = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Get action from policy
            obs_vector = preprocess_observations(obs)
            
            if isinstance(policy, nn.Module):
                with torch.no_grad():
                    action = policy(torch.FloatTensor(obs_vector)).numpy()
            else:
                # Sklearn model
                action = policy.predict([obs_vector])[0]
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
        
        # Record results
        final_height = env.data.body('object').xpos[2]
        success = final_height > 0.1
        
        successes.append(success)
        heights.append(final_height)
    
    return {
        'success_rate': np.mean(successes),
        'avg_height': np.mean(heights),
        'std_height': np.std(heights)
    }
```

## 🎯 Training Strategies

### **1. Data Augmentation**
```python
def augment_demonstrations(observations, actions):
    """Add noise and variations to training data"""
    augmented_obs = []
    augmented_act = []
    
    for obs, act in zip(observations, actions):
        # Original
        augmented_obs.append(obs)
        augmented_act.append(act)
        
        # Add noise
        noise_obs = obs + np.random.normal(0, 0.01, obs.shape)
        noise_act = act + np.random.normal(0, 0.05, act.shape)
        augmented_obs.append(noise_obs)
        augmented_act.append(np.clip(noise_act, -1, 1))
    
    return np.array(augmented_obs), np.array(augmented_act)
```

### **2. Tactile Feature Learning**
```python
class TactileEncoder(nn.Module):
    """Specialized encoder for tactile data"""
    def __init__(self, tactile_shape=(3, 4, 3)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.fc = nn.Linear(32 * 2 * 3, 64)
    
    def forward(self, tactile):
        # Reshape for conv: (batch, channels, height, width)
        x = tactile.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### **3. Multi-Task Learning**
```python
def train_multitask(X, y_actions, y_success):
    """Train to predict both actions and success"""
    model = MultiTaskNetwork(X.shape[1], y_actions.shape[1])
    
    # Loss combines action prediction and success classification
    action_loss = nn.MSELoss()
    success_loss = nn.BCELoss()
    
    # Training loop with both objectives
    # ...
```

## 🐛 Training Troubleshooting

### **Overfitting**
- Add dropout layers
- Increase data augmentation
- Use early stopping
- Reduce model capacity

### **Poor Performance**
- Check data preprocessing
- Verify action scaling
- Increase training data
- Try different architectures

### **Training Instability**
- Reduce learning rate
- Use gradient clipping
- Normalize inputs
- Check for NaN in data

## 📊 Performance Metrics

### **Training Metrics**
| Metric | Description | Target |
|--------|-------------|--------|
| MSE Loss | Action prediction error | < 0.01 |
| Success Rate | Policy evaluation | > 80% |
| Inference Time | ms per action | < 10ms |

### **Comparison Results**
| Method | Success Rate | Training Time |
|--------|-------------|---------------|
| Scripted Expert | 100% | N/A |
| Random Forest | 75% | 30s |
| Neural Network | 85% | 5 min |
| With Augmentation | 90% | 8 min |

## 🔗 Related Documentation
- Data collection → `/ai_docs/DATA_HUB.md`
- Environment setup → `/ai_docs/ENVIRONMENT_HUB.md`
- Evaluation tools → `/ai_docs/VISUALIZATION_HUB.md`
- Training patterns → `/ai_docs/PATTERNS/TRAINING_PATTERNS.md`