#!/usr/bin/env python3
"""
Test different block positions for grasping.
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt

# Load the model
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Create renderer
renderer = mujoco.Renderer(model, height=600, width=800)

# Test configurations
configs = [
    {
        "name": "Original",
        "red": [0.4, 0.0, 0.47],
        "blue": [0.4, -0.1, 0.47]
    },
    {
        "name": "Closer to Robot",
        "red": [0.35, 0.0, 0.47],
        "blue": [0.35, -0.1, 0.47]
    },
    {
        "name": "Wider Spacing",
        "red": [0.4, 0.05, 0.47],
        "blue": [0.4, -0.15, 0.47]
    },
    {
        "name": "Stacked",
        "red": [0.4, 0.0, 0.47],
        "blue": [0.4, 0.0, 0.52]
    }
]

# Test each configuration
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, config in enumerate(configs):
    # Reset
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Set positions
    data.qpos[0:3] = config["red"]
    data.qpos[7:10] = config["blue"]
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Let physics settle
    for _ in range(50):
        mujoco.mj_step(model, data)
    
    # Render
    renderer.update_scene(data, camera="demo_cam")
    pixels = renderer.render()
    
    axes[idx].imshow(pixels)
    axes[idx].set_title(config["name"])
    axes[idx].axis('off')
    
    # Print settled positions
    print(f"\n{config['name']}:")
    print(f"  Red: {data.qpos[0:3]}")
    print(f"  Blue: {data.qpos[7:10]}")

plt.tight_layout()
plt.savefig('block_configurations.png', dpi=150, bbox_inches='tight')
print("\nSaved block_configurations.png")
plt.close()