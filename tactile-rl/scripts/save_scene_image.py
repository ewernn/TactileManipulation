#!/usr/bin/env python3
"""
Save a static image of the scene.
"""

import numpy as np
import mujoco
import imageio

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Reset and step forward
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Create renderer
renderer = mujoco.Renderer(model, height=720, width=1280)

# Render from different angles
angles = [
    ("front", [1.5, 0, 0.8], [0.4, 0, 0.4]),
    ("side", [0.4, -1.5, 0.8], [0.4, 0, 0.4]),
    ("top", [0.4, 0, 2.0], [0.4, 0, 0.4]),
    ("robot_view", [-0.5, -1.0, 0.8], [0.4, 0, 0.4])
]

for name, pos, lookat in angles:
    # Set camera
    renderer.update_scene(data, camera="demo_cam")
    
    # Render
    pixels = renderer.render()
    
    # Save image
    filename = f"scene_{name}.png"
    imageio.imwrite(filename, pixels)
    print(f"Saved {filename}")

print("\nCheck the current directory for scene images!")
print("If blocks are in wrong positions, we'll need to debug further.")