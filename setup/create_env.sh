# Create a new conda environment
# conda create -n tactile-rl python=3.8
conda activate tactile-rl

# Install MuJoCo (optimized for M1 Pro)
pip install mujoco

# Install PyTorch (Apple Silicon version)
pip install torch torchvision torchaudio

# Install robomimic and dependencies
pip install robomimic

# Install other dependencies
pip install numpy gymnasium matplotlib pandas
