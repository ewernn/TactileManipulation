# 🎯 Tactile Sensing Hub

## 🚀 Quick Reference

### **Key Files**
- Main implementation: `tactile-rl/environments/tactile_sensor.py`
- Integration: `tactile-rl/environments/simple_tactile_env.py`
- Test: `tactile-rl/scripts/test_simple_robot.py`

### **Core Functions**
- `tactile_sensor.get_readings(model, data)` → Returns (left_grid, right_grid)
- `_process_contacts()` → Extracts MuJoCo contact forces
- `_add_noise(readings)` → Adds Gaussian noise for realism

## 🏗️ Sensor Configuration

```python
# 3x4 taxel array per finger (24 total)
TactileSensor(
    n_taxels_x=3,      # Width
    n_taxels_y=4,      # Length  
    noise_std=0.01,    # Capacitive sensor noise
    contact_threshold=0.001
)
```

### **Output Structure**
```python
left_tactile, right_tactile = sensor.get_readings(model, data)
# Shape: (3, 4, 3) each
# [normal_force, tangent_x, tangent_y] per taxel
```

## 🔧 Integration Pattern

```python
# In environment __init__
self.tactile_sensor = TactileSensor()

# In _get_obs
left_tactile, right_tactile = self.tactile_sensor.get_readings(self.model, self.data)
obs['tactile'] = np.concatenate([left_tactile.flatten(), right_tactile.flatten()])
```

## 🎯 Key Algorithms

### **Contact Processing**
- Extracts MuJoCo contact forces from `data.contact`
- Maps contacts to nearest taxels with Gaussian falloff
- Separates normal and tangential force components

### **Slip Detection**
```python
def detect_slip(tactile_history, threshold=0.5):
    tangent_change = np.abs(current[:,:,1:3] - previous[:,:,1:3]).mean()
    return tangent_change > threshold
```

### **Grasp Quality**
```python
def estimate_grasp_quality(left_tactile, right_tactile):
    total_force = np.sum([left_tactile[:,:,0], right_tactile[:,:,0]])
    force_balance = np.abs(np.sum(left_tactile[:,:,0]) - np.sum(right_tactile[:,:,0]))
    return total_force / (1 + force_balance)
```

## 🚨 Common Issues

### **No Tactile Readings**
- Check finger geom names contain 'finger_left'/'finger_right'
- Verify collision pairs in XML: `contype="1" conaffinity="1"`
- Ensure contacts exist: `data.ncon > 0`

### **Noisy/Unrealistic Data**
- Adjust `noise_std` parameter (0.01 = realistic)
- Check force scaling in `mj_contactForce`
- Verify taxel grid generation

## 🔗 Related Files
- XML configs: `panda_tactile_grasp.xml`
- Environment: `simple_tactile_env.py`
- Test script: `test_simple_robot.py`