# 🎨 Visualization & Analysis Hub

*Central documentation for visualization, plotting, and video generation*

## 🚀 Quick Reference

### **Key Files**
- Summary plots: `tactile-rl/scripts/create_summary.py`
- Video generation: `tactile-rl/scripts/generate_video.py`
- Tactile visualization: `tactile-rl/scripts/visualize_grasp.py`
- Replay tools: `tactile-rl/scripts/replay_demo.py`

### **Common Operations**
- Generate summary → `python create_summary.py`
- Create demo video → `python generate_video.py --demo 0`
- Visualize tactile → `python visualize_grasp.py --live`
- Export frames → `python replay_demo.py --save_frames`

## 🏗️ Visualization Components

### **1. Performance Summary Plots**
```python
# create_summary.py
def create_project_summary(dataset_path):
    """Generate comprehensive performance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Success rate comparison
    plot_success_comparison(axes[0, 0])
    
    # Lift height distribution
    plot_height_distribution(axes[0, 1], dataset_path)
    
    # Tactile force analysis
    plot_tactile_analysis(axes[1, 0], dataset_path)
    
    # Grasp stability metrics
    plot_grasp_metrics(axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('datasets/tactile_grasping/project_summary.png', dpi=300)
```

### **2. Video Generation**
```python
# generate_video.py
def create_demo_video(env, policy=None, output_path='demo.mp4'):
    """Generate video of task execution"""
    frames = []
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        # Render frame
        frame = env.render()
        
        # Add tactile overlay
        frame_with_tactile = add_tactile_visualization(
            frame, obs['tactile_left'], obs['tactile_right']
        )
        frames.append(frame_with_tactile)
        
        # Get action
        if policy:
            action = policy(obs)
        else:
            action = scripted_expert(obs)
        
        obs, _, done, _, _ = env.step(action)
    
    # Save video
    save_video(frames, output_path, fps=30)
    
    # Also create GIF
    save_gif(frames[::3], output_path.replace('.mp4', '.gif'), fps=10)
```

### **3. Tactile Heatmaps**
```python
def visualize_tactile_sensors(left_tactile, right_tactile):
    """Create tactile force heatmaps"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left finger
    left_normal = left_tactile[:, :, 0]  # Normal force component
    im1 = ax1.imshow(left_normal, cmap='hot', interpolation='bilinear')
    ax1.set_title('Left Finger Tactile')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Length')
    plt.colorbar(im1, ax=ax1, label='Force (N)')
    
    # Right finger
    right_normal = right_tactile[:, :, 0]
    im2 = ax2.imshow(right_normal, cmap='hot', interpolation='bilinear')
    ax2.set_title('Right Finger Tactile')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Length')
    plt.colorbar(im2, ax=ax2, label='Force (N)')
    
    return fig
```

## 🔧 Advanced Visualizations

### **Real-time Tactile Display**
```python
class TactileVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.im1 = self.ax1.imshow(np.zeros((3, 4)), cmap='hot', vmin=0, vmax=5)
        self.im2 = self.ax2.imshow(np.zeros((3, 4)), cmap='hot', vmin=0, vmax=5)
        
    def update(self, left_tactile, right_tactile):
        """Update display with new tactile readings"""
        self.im1.set_data(left_tactile[:, :, 0])
        self.im2.set_data(right_tactile[:, :, 0])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
```

### **Trajectory Visualization**
```python
def plot_3d_trajectory(demo_data):
    """3D visualization of end-effector trajectory"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    ee_positions = demo_data['ee_positions']
    obj_positions = demo_data['object_positions']
    
    # Plot paths
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
            'b-', label='End-effector', linewidth=2)
    ax.plot(obj_positions[:, 0], obj_positions[:, 1], obj_positions[:, 2], 
            'r-', label='Object', linewidth=2)
    
    # Mark key points
    ax.scatter(*ee_positions[0], color='green', s=100, label='Start')
    ax.scatter(*ee_positions[-1], color='red', s=100, label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
```

### **Force Analysis Plots**
```python
def analyze_grasp_forces(demo_data):
    """Detailed force analysis visualization"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Total force over time
    left_forces = demo_data['tactile_left']
    right_forces = demo_data['tactile_right']
    
    total_left = np.sum(left_forces[:, :, :, 0], axis=(1, 2))
    total_right = np.sum(right_forces[:, :, :, 0], axis=(1, 2))
    
    axes[0].plot(total_left, label='Left Finger')
    axes[0].plot(total_right, label='Right Finger')
    axes[0].set_ylabel('Total Normal Force (N)')
    axes[0].legend()
    
    # Force distribution
    axes[1].plot(total_left + total_right, 'k-', label='Combined Force')
    axes[1].axhline(y=2.0, color='r', linestyle='--', label='Min Grasp Force')
    axes[1].set_ylabel('Combined Force (N)')
    axes[1].legend()
    
    # Balance metric
    balance = np.abs(total_left - total_right) / (total_left + total_right + 1e-6)
    axes[2].plot(1 - balance, 'g-')
    axes[2].set_ylabel('Force Balance Score')
    axes[2].set_xlabel('Time Steps')
    axes[2].set_ylim([0, 1])
```

## 🎯 Visualization Patterns

### **Side-by-Side Comparison**
```python
def compare_policies_visual(env, policies, labels):
    """Compare multiple policies visually"""
    n_policies = len(policies)
    fig, axes = plt.subplots(n_policies, 4, figsize=(16, 4*n_policies))
    
    for i, (policy, label) in enumerate(zip(policies, labels)):
        # Run episode
        frames, tactile_data = run_episode_with_recording(env, policy)
        
        # Show key frames
        key_frames = [0, len(frames)//3, 2*len(frames)//3, -1]
        for j, frame_idx in enumerate(key_frames):
            axes[i, j].imshow(frames[frame_idx])
            axes[i, j].set_title(f'{label} - Step {frame_idx}')
            axes[i, j].axis('off')
```

### **Performance Metrics Dashboard**
```python
def create_dashboard(results_dict):
    """Create comprehensive performance dashboard"""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig)
    
    # Success rate
    ax1 = fig.add_subplot(gs[0, :2])
    plot_success_rates(ax1, results_dict)
    
    # Height distribution
    ax2 = fig.add_subplot(gs[0, 2:])
    plot_height_distributions(ax2, results_dict)
    
    # Tactile patterns
    ax3 = fig.add_subplot(gs[1, :2])
    plot_tactile_patterns(ax3, results_dict)
    
    # Timing analysis
    ax4 = fig.add_subplot(gs[1, 2:])
    plot_timing_analysis(ax4, results_dict)
    
    # Example frames
    for i in range(4):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(results_dict['example_frames'][i])
        ax.axis('off')
```

## 🐛 Visualization Troubleshooting

### **Video Generation Issues**
```python
# Check ffmpeg installation
import subprocess
try:
    subprocess.run(['ffmpeg', '-version'], check=True)
except:
    print("Install ffmpeg: brew install ffmpeg")

# Alternative: use imageio
import imageio
imageio.mimsave('output.gif', frames, fps=10)
```

### **Memory Issues with Large Videos**
```python
# Stream frames instead of storing all
def create_video_streaming(env, policy, output_path):
    writer = imageio.get_writer(output_path, fps=30)
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        frame = env.render()
        writer.append_data(frame)
        
        action = policy(obs)
        obs, _, done, _, _ = env.step(action)
    
    writer.close()
```

### **Plot Quality Issues**
```python
# High-quality plot settings
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'lines.linewidth': 2
})
```

## 📊 Output Examples

### **Generated Files**
| File | Description | Location |
|------|-------------|----------|
| project_summary.png | Performance metrics | datasets/tactile_grasping/ |
| tactile_grasp_demo.mp4 | Full demo video | datasets/tactile_grasping/ |
| tactile_grasp_demo.gif | Animated GIF | datasets/tactile_grasping/ |
| force_analysis.png | Force patterns | datasets/tactile_grasping/ |

### **Video Specifications**
- Resolution: 640x480 (default)
- FPS: 30 (MP4), 10 (GIF)
- Codec: H.264 (MP4)
- Duration: 5-10 seconds typical

## 🔗 Related Documentation
- Data analysis → `/ai_docs/DATA_HUB.md`
- Environment rendering → `/ai_docs/ENVIRONMENT_HUB.md`
- Tactile visualization → `/ai_docs/TACTILE_HUB.md`
- Video patterns → `/ai_docs/PATTERNS/VISUALIZATION_PATTERNS.md`