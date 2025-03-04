# Tactile-Enhanced Robotic Manipulation

A reinforcement learning project demonstrating how simulated tactile feedback enhances robotic manipulation for humanoid robotics applications. This project augments the RoboMimic dataset with simulated tactile sensors to improve sample efficiency and manipulation performance.

## Project Overview

This project showcases how tactile sensing can significantly improve manipulation tasks relevant to humanoid robotics. Using a 2-finger gripper simulation (with potential extension to more complex hands), we:

1. Augment existing demonstrations with simulated tactile feedback
2. Train both tactile-enhanced and standard models
3. Compare performance across various manipulation challenges
4. Visualize the impact of tactile sensing on decision-making

## Key Features

- **Tactile Sensing Simulation**: 3×4 taxel grid on each gripper finger with simulated capacitive response
- **Multi-Modal Learning**: Integration of visual, proprioceptive, and tactile information
- **Sample Efficient Learning**: Offline reinforcement learning using demonstration data
- **Comparative Analysis**: Performance metrics with and without tactile feedback

## Installation

### Prerequisites

- macOS with M1/M2 chip
- Python 3.8+
- 32GB RAM recommended

### Environment Setup

```bash
# Create a new conda environment
conda create -n tactile-rl python=3.8
conda activate tactile-rl

# Install MuJoCo
pip install mujoco

# Install PyTorch (Apple Silicon version)
pip install torch torchvision torchaudio

# Install robomimic and dependencies
pip install robomimic

# Install other dependencies
pip install numpy gymnasium matplotlib pandas
```

### Dataset Setup

```bash
# Download RoboMimic dataset (choose one or more tasks)
python -m robomimic.scripts.download_datasets --tasks can_picking
python -m robomimic.scripts.download_datasets --tasks lift
python -m robomimic.scripts.download_datasets --tasks square_insertion
```

## Project Structure

```
tactile-rl/
├── data/                        # Datasets and processed data
├── environments/                # Custom MuJoCo environments
│   └── tactile_gripper.py       # Gripper with tactile sensors
├── models/                      # Neural network architectures
│   ├── policy_network.py        # Policy networks with tactile processing
│   └── tactile_encoder.py       # Encoder for tactile information
├── scripts/                     # Utility scripts
│   ├── augment_dataset.py       # Add tactile data to demonstrations
│   ├── visualize_tactile.py     # Visualize tactile readings
│   └── extend_shadow_hand.py    # Optional extension to 5-finger hand
├── training/                    # Training algorithms
│   ├── bc_training.py           # Behavior Cloning implementation  
│   └── cql_training.py          # Conservative Q-Learning implementation
├── visualization/               # Visualization tools
├── main.py                      # Main training script
└── evaluate.py                  # Evaluation script
```

## Usage

### Data Augmentation

First, augment the dataset with tactile information:

```bash
python scripts/augment_dataset.py --dataset can_picking --output augmented_can_picking
```

This replays the demonstrations in simulation and records tactile sensor readings at each timestep.

### Training

Train models with and without tactile information:

```bash
# Train with tactile
python main.py --dataset augmented_can_picking --use_tactile True --algorithm cql

# Train without tactile (baseline)
python main.py --dataset augmented_can_picking --use_tactile False --algorithm cql
```

### Evaluation

Compare performance between models:

```bash
python evaluate.py --tactile_model models/tactile_cql.pt --baseline_model models/baseline_cql.pt
```

## Tactile Sensor Implementation

The tactile sensors are implemented as a 3×4 grid on each finger of the gripper. Each taxel reads:

- Normal force (pressure)
- Shear forces (x, y directions)
- Contact binary state

The sensor simulation converts MuJoCo contact data into realistic tactile readings by:

1. Finding contact points within each sensor pad area
2. Applying a Gaussian spatial model to distribute forces across taxels
3. Adding realistic noise based on capacitive sensor characteristics
4. Processing readings into tactile "images" for the learning algorithm

## Learning Architecture

The model uses a multi-modal architecture:

- **Visual Encoder**: ResNet-based feature extractor for camera input
- **Tactile Encoder**: CNN-based encoder for tactile "images"
- **Proprioceptive Encoder**: MLP for joint positions/velocities
- **Fusion Module**: Cross-attention mechanism for feature integration
- **Policy Network**: Actor-critic architecture for control

## Recommended Tasks

Three RoboMimic tasks are particularly well-suited for demonstrating tactile benefits:

1. **Can Picking**: Grasping cylindrical objects with curved surfaces
   - Tactile benefit: Surface curvature detection, slip prevention

2. **Lift**: Simple grasping and lifting of objects
   - Tactile benefit: Grasp stability, force regulation

3. **Square Insertion**: Placing square pegs into matching holes
   - Tactile benefit: Contact detection for alignment, force feedback for insertion

## Results

Key performance metrics include:

- Success rate comparison across tasks
- Sample efficiency (learning curves)
- Robustness to variations in object properties
- Visualization of attention on tactile features during critical manipulation phases

## Tesla Optimus Relevance

This project is directly relevant to the Tesla Optimus program in several ways:

1. **Manipulation Enhancement**: Addresses the challenge of precise object manipulation
2. **Sensor Fusion**: Demonstrates effective integration of multiple sensor modalities
3. **Sample Efficiency**: Shows improved learning with fewer demonstrations
4. **Safety**: Tactile feedback prevents excessive force application
5. **Adaptability**: Improves handling of diverse objects and conditions

## Future Work

Potential extensions to this project:

- **Extension to 5-finger Shadow Hand**: Implementing the same approach on a more dexterous hand model
- Implementation on a real robot with tactile sensors
- Integration with vision-language models for instruction following
- Exploration of active tactile exploration strategies
- Curriculum learning for complex manipulation sequences

## Acknowledgments

- RoboMimic framework and datasets
- MuJoCo simulation environment