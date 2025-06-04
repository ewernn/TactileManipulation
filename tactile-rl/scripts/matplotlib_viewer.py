#!/usr/bin/env python3
"""
View scene using matplotlib (no OpenGL required).
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt

# Load the scene
xml_path = "../franka_emika_panda/panda_demo_scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Reset and forward
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# Run a few steps to let things settle
for _ in range(10):
    mujoco.mj_step(model, data)

# Create renderer
renderer = mujoco.Renderer(model, height=480, width=640)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

views = [
    ("Front View", "demo_cam"),
    ("Side View", "side_cam"), 
    ("Top View", "overhead_cam"),
    ("Wrist View", "wrist_cam")
]

for ax, (title, cam) in zip(axes, views):
    try:
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        ax.imshow(pixels)
        ax.set_title(title)
        ax.axis('off')
    except:
        ax.text(0.5, 0.5, f"Camera '{cam}' not found", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

plt.tight_layout()
plt.savefig('scene_views.png', dpi=150, bbox_inches='tight')
print("Saved scene_views.png")

# Also print positions
print("\nObject positions:")
table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
red_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block2")

print(f"Table: {data.xpos[table_id]}")
print(f"Red block: {data.xpos[red_id]}")
print(f"Blue block: {data.xpos[blue_id]}")

plt.show()