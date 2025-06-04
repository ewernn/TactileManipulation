#!/usr/bin/env python3
"""
Interactive script to position blocks and save configuration.
"""

import numpy as np
import mujoco
import time
import sys
import tty
import termios

# Constants for qpos addressing
RED_BLOCK_QPOS = 0   # Red block starts at qpos[0]
BLUE_BLOCK_QPOS = 7   # Blue block starts at qpos[7]
ROBOT_QPOS = 14       # Robot starts at qpos[14]

def get_single_keypress():
    """Get a single keypress without Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    # Load the model
    xml_path = "../franka_emika_panda/panda_demo_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Reset to keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    print("=" * 60)
    print("POSITION BLOCKS - Terminal Version")
    print("=" * 60)
    print("\nControls:")
    print("  Space: Toggle physics")
    print("  Red Block:   Q/A (Y), W/S (X), E/D (Z)")
    print("  Blue Block:  J/L (Y), I/K (X), U/H (Z)")
    print("  P: Print positions")
    print("  X: Save and exit")
    print("  ESC: Exit without saving")
    print("=" * 60)
    
    paused = True
    move_speed = 0.01
    
    while True:
        # Get key
        key = get_single_keypress()
        
        # Handle exit
        if key == '\x1b':  # ESC
            print("\nExiting without saving...")
            break
        elif key.lower() == 'x':
            print("\nSaving configuration...")
            # Print final positions
            red_pos = data.qpos[RED_BLOCK_QPOS:RED_BLOCK_QPOS+3]
            blue_pos = data.qpos[BLUE_BLOCK_QPOS:BLUE_BLOCK_QPOS+3]
            print(f"\nFinal positions:")
            print(f"Red block:  {red_pos}")
            print(f"Blue block: {blue_pos}")
            
            # Save to file
            with open("block_positions.txt", "w") as f:
                f.write(f"Red block position: {red_pos[0]:.3f} {red_pos[1]:.3f} {red_pos[2]:.3f}\n")
                f.write(f"Blue block position: {blue_pos[0]:.3f} {blue_pos[1]:.3f} {blue_pos[2]:.3f}\n")
            print("Saved to block_positions.txt")
            break
        
        # Toggle physics
        elif key == ' ':
            paused = not paused
            print(f"\nPhysics: {'PAUSED' if paused else 'RUNNING'}")
        
        # Print positions
        elif key.lower() == 'p':
            red_pos = data.qpos[RED_BLOCK_QPOS:RED_BLOCK_QPOS+3]
            blue_pos = data.qpos[BLUE_BLOCK_QPOS:BLUE_BLOCK_QPOS+3]
            print(f"\nRed block:  {red_pos}")
            print(f"Blue block: {blue_pos}")
        
        # Red block controls
        elif key.lower() == 'q':
            data.qpos[RED_BLOCK_QPOS+1] += move_speed
        elif key.lower() == 'a':
            data.qpos[RED_BLOCK_QPOS+1] -= move_speed
        elif key.lower() == 'w':
            data.qpos[RED_BLOCK_QPOS] += move_speed
        elif key.lower() == 's':
            data.qpos[RED_BLOCK_QPOS] -= move_speed
        elif key.lower() == 'e':
            data.qpos[RED_BLOCK_QPOS+2] += move_speed
        elif key.lower() == 'd':
            data.qpos[RED_BLOCK_QPOS+2] -= move_speed
        
        # Blue block controls
        elif key.lower() == 'j':
            data.qpos[BLUE_BLOCK_QPOS+1] -= move_speed
        elif key.lower() == 'l':
            data.qpos[BLUE_BLOCK_QPOS+1] += move_speed
        elif key.lower() == 'i':
            data.qpos[BLUE_BLOCK_QPOS] += move_speed
        elif key.lower() == 'k':
            data.qpos[BLUE_BLOCK_QPOS] -= move_speed
        elif key.lower() == 'u':
            data.qpos[BLUE_BLOCK_QPOS+2] += move_speed
        elif key.lower() == 'h':
            data.qpos[BLUE_BLOCK_QPOS+2] -= move_speed
        
        # Update physics
        if paused:
            mujoco.mj_forward(model, data)
        else:
            mujoco.mj_step(model, data)
        
        # Brief pause
        time.sleep(0.01)

if __name__ == "__main__":
    main()