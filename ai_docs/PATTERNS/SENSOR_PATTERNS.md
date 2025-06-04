# 🎯 Tactile Sensor Patterns

*Common patterns for tactile sensor implementation and integration*

## 🎯 Sensor Implementation Patterns

### **Basic Tactile Sensor**
```python
class BasicTactileSensor:
    """Simple tactile sensor with grid layout"""
    def __init__(self, n_rows=3, n_cols=4, n_components=3):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_components = n_components  # [normal, tangent_x, tangent_y]
        
        # Initialize sensor state
        self.reset()
        
    def reset(self):
        """Reset sensor readings"""
        self.left_readings = np.zeros((self.n_rows, self.n_cols, self.n_components))
        self.right_readings = np.zeros((self.n_rows, self.n_cols, self.n_components))
        
    def get_readings(self, model, data):
        """Get tactile readings from MuJoCo contacts"""
        # Reset readings
        self.reset()
        
        # Process all contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            
            # Get contact geometry names
            geom1_name = model.geom(contact.geom1).name
            geom2_name = model.geom(contact.geom2).name
            
            # Extract contact force
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            
            # Distribute to appropriate sensor
            if 'finger_left' in geom1_name or 'finger_left' in geom2_name:
                self._add_contact_to_grid(contact, force[:3], self.left_readings)
            elif 'finger_right' in geom1_name or 'finger_right' in geom2_name:
                self._add_contact_to_grid(contact, force[:3], self.right_readings)
                
        return self.left_readings.copy(), self.right_readings.copy()
```

### **Advanced Tactile with Spatial Distribution**
```python
class SpatialTactileSensor:
    """Tactile sensor with spatial force distribution"""
    def __init__(self, finger_dims=(0.02, 0.04), 
                 n_taxels_x=3, n_taxels_y=4,
                 distribution_sigma=0.005):
        self.finger_dims = finger_dims
        self.n_taxels_x = n_taxels_x
        self.n_taxels_y = n_taxels_y
        self.sigma = distribution_sigma
        
        # Generate taxel positions
        self.taxel_positions = self._generate_taxel_positions()
        
    def _generate_taxel_positions(self):
        """Create grid of taxel positions"""
        x_spacing = self.finger_dims[0] / (self.n_taxels_x - 1)
        y_spacing = self.finger_dims[1] / (self.n_taxels_y - 1)
        
        positions = []
        for i in range(self.n_taxels_x):
            for j in range(self.n_taxels_y):
                x = i * x_spacing - self.finger_dims[0] / 2
                y = j * y_spacing - self.finger_dims[1] / 2
                positions.append([x, y])
                
        return np.array(positions)
    
    def _distribute_force(self, contact_point, force, readings):
        """Distribute force across nearby taxels with Gaussian falloff"""
        # Project contact point to 2D sensor surface
        contact_2d = contact_point[:2]  # Assuming Z is normal
        
        for i in range(self.n_taxels_x):
            for j in range(self.n_taxels_y):
                taxel_idx = i * self.n_taxels_y + j
                taxel_pos = self.taxel_positions[taxel_idx]
                
                # Distance from contact to taxel
                dist = np.linalg.norm(contact_2d - taxel_pos)
                
                # Gaussian weight
                weight = np.exp(-dist**2 / (2 * self.sigma**2))
                
                # Apply weighted force
                readings[i, j] += force * weight
```

### **Noise and Calibration Patterns**
```python
class CalibratedTactileSensor:
    """Tactile sensor with realistic noise and calibration"""
    def __init__(self, base_sensor, 
                 noise_std=0.01,
                 bias_range=(-0.05, 0.05),
                 scale_range=(0.95, 1.05)):
        self.base_sensor = base_sensor
        self.noise_std = noise_std
        
        # Generate per-taxel calibration parameters
        n_taxels = base_sensor.n_taxels_x * base_sensor.n_taxels_y
        self.bias_left = np.random.uniform(*bias_range, (n_taxels, 3))
        self.bias_right = np.random.uniform(*bias_range, (n_taxels, 3))
        self.scale_left = np.random.uniform(*scale_range, (n_taxels, 3))
        self.scale_right = np.random.uniform(*scale_range, (n_taxels, 3))
        
    def get_readings(self, model, data):
        """Get readings with noise and calibration effects"""
        # Get base readings
        left_raw, right_raw = self.base_sensor.get_readings(model, data)
        
        # Apply calibration
        left_cal = self._apply_calibration(left_raw, self.scale_left, self.bias_left)
        right_cal = self._apply_calibration(right_raw, self.scale_right, self.bias_right)
        
        # Add noise
        if self.noise_std > 0:
            left_cal += np.random.normal(0, self.noise_std, left_cal.shape)
            right_cal += np.random.normal(0, self.noise_std, right_cal.shape)
            
        # Ensure non-negative forces
        left_cal = np.maximum(left_cal, 0)
        right_cal = np.maximum(right_cal, 0)
        
        return left_cal, right_cal
    
    def _apply_calibration(self, raw_readings, scale, bias):
        """Apply per-taxel calibration"""
        flat_readings = raw_readings.reshape(-1, 3)
        calibrated = flat_readings * scale + bias
        return calibrated.reshape(raw_readings.shape)
```

## 🔧 Sensor Processing Patterns

### **Tactile Feature Extraction**
```python
class TactileFeatureExtractor:
    """Extract high-level features from raw tactile data"""
    
    @staticmethod
    def compute_contact_features(tactile_left, tactile_right):
        """Compute contact-based features"""
        features = {}
        
        # Total force
        features['total_force_left'] = np.sum(tactile_left[:, :, 0])
        features['total_force_right'] = np.sum(tactile_right[:, :, 0])
        features['total_force'] = features['total_force_left'] + features['total_force_right']
        
        # Force balance
        if features['total_force'] > 0:
            features['force_balance'] = (
                features['total_force_left'] / features['total_force']
            )
        else:
            features['force_balance'] = 0.5
        
        # Contact area (number of active taxels)
        threshold = 0.1  # Force threshold
        features['contact_area_left'] = np.sum(tactile_left[:, :, 0] > threshold)
        features['contact_area_right'] = np.sum(tactile_right[:, :, 0] > threshold)
        
        # Center of pressure
        features['cop_left'] = TactileFeatureExtractor._compute_cop(tactile_left[:, :, 0])
        features['cop_right'] = TactileFeatureExtractor._compute_cop(tactile_right[:, :, 0])
        
        # Shear ratio
        normal_force = tactile_left[:, :, 0] + tactile_right[:, :, 0]
        shear_force = np.sqrt(
            (tactile_left[:, :, 1] + tactile_right[:, :, 1])**2 +
            (tactile_left[:, :, 2] + tactile_right[:, :, 2])**2
        )
        features['mean_shear_ratio'] = np.mean(
            shear_force / (normal_force + 1e-6)
        )
        
        return features
    
    @staticmethod
    def _compute_cop(force_grid):
        """Compute center of pressure"""
        total_force = np.sum(force_grid)
        if total_force < 1e-6:
            return np.array([0.5, 0.5])  # Center
            
        # Compute weighted position
        rows, cols = force_grid.shape
        row_indices, col_indices = np.meshgrid(
            np.arange(rows), np.arange(cols), indexing='ij'
        )
        
        cop_row = np.sum(row_indices * force_grid) / total_force
        cop_col = np.sum(col_indices * force_grid) / total_force
        
        # Normalize to [0, 1]
        return np.array([cop_row / (rows - 1), cop_col / (cols - 1)])
```

### **Slip Detection Pattern**
```python
class SlipDetector:
    """Detect slip from tactile time series"""
    def __init__(self, window_size=5, slip_threshold=0.5):
        self.window_size = window_size
        self.threshold = slip_threshold
        self.history = []
        
    def update(self, tactile_left, tactile_right):
        """Update with new tactile readings"""
        # Store recent history
        self.history.append({
            'left': tactile_left.copy(),
            'right': tactile_right.copy(),
            'timestamp': time.time()
        })
        
        # Keep fixed window size
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
    def detect_slip(self):
        """Detect if slip is occurring"""
        if len(self.history) < 2:
            return False, 0.0
            
        # Compute shear force changes
        shear_changes = []
        
        for i in range(1, len(self.history)):
            prev = self.history[i-1]
            curr = self.history[i]
            
            # Shear force magnitude
            prev_shear_left = np.sqrt(
                prev['left'][:, :, 1]**2 + prev['left'][:, :, 2]**2
            )
            curr_shear_left = np.sqrt(
                curr['left'][:, :, 1]**2 + curr['left'][:, :, 2]**2
            )
            
            # Rate of change
            dt = curr['timestamp'] - prev['timestamp']
            shear_rate = np.mean(np.abs(curr_shear_left - prev_shear_left)) / dt
            shear_changes.append(shear_rate)
            
        # Check if slip detected
        mean_shear_rate = np.mean(shear_changes)
        is_slipping = mean_shear_rate > self.threshold
        
        return is_slipping, mean_shear_rate
```

### **Grasp Quality Estimation**
```python
class GraspQualityEstimator:
    """Estimate grasp quality from tactile data"""
    
    def __init__(self, min_force=0.5, optimal_force=2.0, max_force=5.0):
        self.min_force = min_force
        self.optimal_force = optimal_force
        self.max_force = max_force
        
    def estimate_quality(self, tactile_left, tactile_right):
        """Compute grasp quality score [0, 1]"""
        scores = {}
        
        # 1. Force magnitude score
        total_force = (
            np.sum(tactile_left[:, :, 0]) + 
            np.sum(tactile_right[:, :, 0])
        )
        
        if total_force < self.min_force:
            scores['force'] = total_force / self.min_force
        elif total_force <= self.optimal_force:
            scores['force'] = 1.0
        elif total_force <= self.max_force:
            scores['force'] = 1.0 - (total_force - self.optimal_force) / (self.max_force - self.optimal_force)
        else:
            scores['force'] = 0.0
            
        # 2. Force distribution score
        left_std = np.std(tactile_left[:, :, 0])
        right_std = np.std(tactile_right[:, :, 0])
        mean_std = (left_std + right_std) / 2
        scores['distribution'] = 1.0 / (1.0 + mean_std)
        
        # 3. Symmetry score
        left_total = np.sum(tactile_left[:, :, 0])
        right_total = np.sum(tactile_right[:, :, 0])
        if left_total + right_total > 0:
            balance = min(left_total, right_total) / max(left_total, right_total)
        else:
            balance = 0
        scores['symmetry'] = balance
        
        # 4. Contact area score
        contact_threshold = 0.1
        left_contact = np.sum(tactile_left[:, :, 0] > contact_threshold)
        right_contact = np.sum(tactile_right[:, :, 0] > contact_threshold)
        total_taxels = tactile_left.shape[0] * tactile_left.shape[1] * 2
        scores['contact_area'] = (left_contact + right_contact) / total_taxels
        
        # Combined score
        weights = {
            'force': 0.4,
            'distribution': 0.2,
            'symmetry': 0.2,
            'contact_area': 0.2
        }
        
        overall_score = sum(
            scores[key] * weights[key] 
            for key in scores
        )
        
        return overall_score, scores
```

## 🎯 Integration Patterns

### **Sensor Manager**
```python
class TactileSensorManager:
    """Manage multiple tactile sensors and processing"""
    def __init__(self):
        self.sensors = {}
        self.processors = {}
        self.data_history = []
        
    def add_sensor(self, name, sensor):
        """Register a sensor"""
        self.sensors[name] = sensor
        
    def add_processor(self, name, processor):
        """Register a processor"""
        self.processors[name] = processor
        
    def update(self, model, data):
        """Update all sensors and processors"""
        current_data = {}
        
        # Collect sensor data
        for name, sensor in self.sensors.items():
            current_data[name] = sensor.get_readings(model, data)
            
        # Process data
        processed_data = {}
        for name, processor in self.processors.items():
            if hasattr(processor, 'process'):
                processed_data[name] = processor.process(current_data)
                
        # Store history
        self.data_history.append({
            'raw': current_data,
            'processed': processed_data,
            'timestamp': time.time()
        })
        
        # Limit history size
        if len(self.data_history) > 1000:
            self.data_history.pop(0)
            
        return current_data, processed_data
```

### **Tactile-Based Controller**
```python
class TactileGraspController:
    """Controller that uses tactile feedback"""
    def __init__(self, target_force=2.0, slip_prevention=True):
        self.target_force = target_force
        self.slip_prevention = slip_prevention
        self.slip_detector = SlipDetector()
        self.quality_estimator = GraspQualityEstimator()
        
    def compute_gripper_action(self, tactile_left, tactile_right):
        """Compute gripper action from tactile feedback"""
        # Update slip detector
        self.slip_detector.update(tactile_left, tactile_right)
        
        # Check for slip
        is_slipping, slip_rate = self.slip_detector.detect_slip()
        
        # Get current force
        current_force = (
            np.sum(tactile_left[:, :, 0]) + 
            np.sum(tactile_right[:, :, 0])
        )
        
        # Compute action
        if is_slipping and self.slip_prevention:
            # Emergency: close more
            action = 1.0
        elif current_force < self.target_force * 0.8:
            # Too little force: close
            action = 0.5
        elif current_force > self.target_force * 1.2:
            # Too much force: open slightly
            action = -0.2
        else:
            # Good force: maintain
            action = 0.0
            
        return action
```

## 🔗 Related Patterns
- Environment integration → `/ai_docs/ENVIRONMENT_HUB.md`
- Control integration → `/ai_docs/PATTERNS/CONTROL_PATTERNS.md`
- Data processing → `/ai_docs/PATTERNS/DATA_PATTERNS.md`