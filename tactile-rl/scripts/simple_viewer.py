#!/usr/bin/env python3
"""
Simple viewer using MuJoCo's built-in viewer.
"""

import mujoco
import mujoco.viewer

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"

print("Loading model...")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("Launching viewer...")
print("\nControls:")
print("  - Left mouse: Rotate camera")
print("  - Right mouse: Pan camera") 
print("  - Scroll: Zoom")
print("  - Ctrl+Left click on joints: Drag to move")
print("  - Space: Pause/unpause")
print("  - Esc: Exit")

# Launch the basic viewer
mujoco.viewer.launch(model, data)