#!/usr/bin/env python3
"""
Google Colab setup script for training BC and RL policies.
Copy this to a Colab notebook and run the cells.
"""

# ============================================================================
# CELL 1: Initial Setup
# ============================================================================
"""
# Check GPU and install dependencies
!nvidia-smi
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Install required packages
!pip install mujoco h5py tensorboard matplotlib

# For video rendering (optional)
!apt-get update && apt-get install -y xvfb ffmpeg
!pip install imageio imageio-ffmpeg
"""

# ============================================================================
# CELL 2: Clone Repository and Setup
# ============================================================================
"""
# Clone your repository
!git clone https://github.com/YOUR_USERNAME/TactileManipulation.git
%cd TactileManipulation/tactile-rl

# Or if private repo:
# !git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/TactileManipulation.git

# Verify structure
!ls -la
!ls scripts/
!ls ../datasets/expert/
"""

# ============================================================================
# CELL 3: Mount Google Drive (for saving checkpoints)
# ============================================================================
"""
from google.colab import drive
drive.mount('/content/drive')

# Create checkpoint directory on Drive
import os
checkpoint_dir = '/content/drive/MyDrive/tactile_manipulation_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Checkpoints will be saved to: {checkpoint_dir}")
"""

# ============================================================================
# CELL 4: Copy Expert Demonstrations
# ============================================================================
"""
# If demos are in your repo
demo_path = '../datasets/expert/expert_demos_20250603_184458.hdf5'

# Or upload from local
# from google.colab import files
# uploaded = files.upload()
# demo_path = list(uploaded.keys())[0]

# Verify demos
import h5py
with h5py.File(demo_path, 'r') as f:
    print(f"Loaded {f.attrs['num_demos']} demonstrations")
    print(f"Environment: {f.attrs['env_name']}")
    print(f"Control frequency: {f.attrs['control_frequency']} Hz")
"""

# ============================================================================
# CELL 5: Train BC Policy
# ============================================================================
"""
# Train BC policy with optimal settings for T4
!python scripts/train_bc_policy.py \
    --demos {demo_path} \
    --epochs 150 \
    --batch_size 256 \
    --lr 1e-3 \
    --hidden_dims 256 256 \
    --dropout 0.1 \
    --save_dir {checkpoint_dir}/bc_run_$(date +%Y%m%d_%H%M%S) \
    --device cuda \
    --log_interval 10
"""

# ============================================================================
# CELL 6: Monitor Training (in another cell)
# ============================================================================
"""
# Plot training curves
import json
import matplotlib.pyplot as plt

# Load training history
with open(f'{checkpoint_dir}/bc_run_*/training_history.json', 'r') as f:
    history = json.load(f)

epochs = [h['epoch'] for h in history]
train_losses = [h['train_loss'] for h in history]
val_losses = [h['val_loss'] for h in history]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train')
plt.plot(epochs, val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('BC Training Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, [h['lr'] for h in history])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.tight_layout()
plt.show()
"""

# ============================================================================
# CELL 7: Evaluate BC Policy
# ============================================================================
"""
# Create evaluation script
eval_code = '''
import torch
import numpy as np
import mujoco
from train_bc_policy import BCPolicy

# Load trained model
checkpoint = torch.load(f'{checkpoint_dir}/bc_run_*/best_model.pt')
model = BCPolicy().cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load normalization stats
norm_stats = checkpoint['norm_stats']

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation loss: {checkpoint['val_loss']:.4f}")

# Test on a few random inputs
test_obs = torch.randn(5, 52).cuda()
with torch.no_grad():
    actions = model(test_obs)
    print(f"\\nSample actions:")
    print(actions.cpu().numpy())
'''

exec(eval_code)
"""

# ============================================================================
# CELL 8: Save Model for Local Use
# ============================================================================
"""
# Download the best model
from google.colab import files

# Find best model path
import glob
model_paths = glob.glob(f'{checkpoint_dir}/bc_run_*/best_model.pt')
if model_paths:
    best_model_path = model_paths[0]
    print(f"Downloading: {best_model_path}")
    files.download(best_model_path)
    
    # Also download training history
    history_path = best_model_path.replace('best_model.pt', 'training_history.json')
    files.download(history_path)
"""

# ============================================================================
# CELL 9: Prevent Colab Timeout
# ============================================================================
"""
# Run this JavaScript in console to prevent disconnection
# Press Ctrl+Shift+I to open console, paste this:

function ClickConnect(){
    console.log("Keeping connection alive...");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
"""

# ============================================================================
# CELL 10: Training Tips for Colab
# ============================================================================
"""
Training Tips:

1. **Save checkpoints frequently to Drive**
   - Colab can disconnect, so save every 10-20 epochs
   - Use Google Drive for persistence

2. **Use Colab Pro for longer sessions**
   - Free tier: ~12 hours max, can disconnect
   - Pro: 24 hours, more reliable
   
3. **Batch your experiments**
   - Train multiple seeds in one session
   - Use wandb or tensorboard for tracking

4. **Monitor GPU memory**
   !nvidia-smi

5. **For RL training later**
   - Start with shorter runs (100k steps)
   - Use checkpoint resuming
   - Consider Colab Pro for A100 access

6. **Quick performance test**
   - T4 should process ~1000 batches/second for BC
   - If slower, check for CPU bottlenecks
"""