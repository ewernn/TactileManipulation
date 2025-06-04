#!/usr/bin/env python3
"""
Find the correct joint positions to reach and grasp the blocks.
Interactive script to tune positions.
"""

import numpy as np
import mujoco
import mujoco.viewer

class GraspPositionFinder:
    def __init__(self):
        # Load model
        self.xml_path = "../franka_emika_panda/panda_demo_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Block positions
        self.red_block_pos = np.array([0.05, 0.0, 0.445])
        self.blue_block_pos = np.array([0.05, -0.15, 0.445])
        
        # Gripper parameters
        self.gripper_reach = 0.103  # Distance from hand to fingertips
        
        # Current target
        self.current_target = "red"
        
        # Joint limits for reference
        self.joint_limits = []
        for i in range(7):
            joint_id = i + 4  # Panda joints start at index 4
            if self.model.jnt_limited[joint_id]:
                limits = self.model.jnt_range[joint_id]
                self.joint_limits.append((limits[0], limits[1]))
            else:
                self.joint_limits.append((-np.pi, np.pi))
        
        # Initial position
        self.reset_position()
        
    def reset_position(self):
        """Reset to home position."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Set initial joint positions
        self.joint_vals = np.array([0, -0.1, 0, -2.0, 0, 1.5, 0])
        for i, val in enumerate(self.joint_vals):
            self.data.qpos[14 + i] = val
            self.data.ctrl[i] = val
        self.data.ctrl[7] = 255  # Open gripper
        
        mujoco.mj_forward(self.model, self.data)
    
    def get_info(self):
        """Get current position info."""
        # Get IDs
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        red_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        blue_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block2")
        
        # Get positions
        ee_pos = self.data.xpos[ee_id].copy()
        red_pos = self.data.xpos[red_id].copy()
        blue_pos = self.data.xpos[blue_id].copy()
        
        # Calculate fingertip position (approximately)
        hand_quat = self.data.xquat[ee_id].copy()
        rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(rotmat, hand_quat)
        rotmat = rotmat.reshape(3, 3)
        
        # Gripper points along local Z axis
        gripper_direction = rotmat[:, 2]
        fingertip_pos = ee_pos + self.gripper_reach * gripper_direction
        
        # Target position
        if self.current_target == "red":
            target_pos = red_pos
            ideal_hand_x = red_pos[0] - self.gripper_reach - 0.01  # Slightly before block
        else:
            target_pos = blue_pos
            ideal_hand_x = blue_pos[0] - self.gripper_reach - 0.01
        
        info = {
            "ee_pos": ee_pos,
            "fingertip_pos": fingertip_pos,
            "target_pos": target_pos,
            "distance_to_target": np.linalg.norm(fingertip_pos - target_pos),
            "x_error": ideal_hand_x - ee_pos[0],
            "y_error": target_pos[1] - ee_pos[1],
            "z_error": target_pos[2] - ee_pos[2],
            "gripper_direction": gripper_direction
        }
        
        return info
    
    def print_status(self):
        """Print current status."""
        info = self.get_info()
        
        print("\n" + "="*60)
        print(f"Target: {self.current_target} block")
        print(f"Joint positions: [{', '.join([f'{v:.3f}' for v in self.joint_vals])}]")
        print(f"Hand position:     [{info['ee_pos'][0]:.3f}, {info['ee_pos'][1]:.3f}, {info['ee_pos'][2]:.3f}]")
        print(f"Fingertip approx:  [{info['fingertip_pos'][0]:.3f}, {info['fingertip_pos'][1]:.3f}, {info['fingertip_pos'][2]:.3f}]")
        print(f"Target position:   [{info['target_pos'][0]:.3f}, {info['target_pos'][1]:.3f}, {info['target_pos'][2]:.3f}]")
        print(f"Distance to target: {info['distance_to_target']:.3f}m")
        print(f"Position errors:   X={info['x_error']:.3f}, Y={info['y_error']:.3f}, Z={info['z_error']:.3f}")
        print(f"Gripper pointing:  [{info['gripper_direction'][0]:.3f}, {info['gripper_direction'][1]:.3f}, {info['gripper_direction'][2]:.3f}]")
        
        # Check if gripper is pointing toward target
        to_target = info['target_pos'] - info['ee_pos']
        to_target_norm = to_target / np.linalg.norm(to_target)
        alignment = np.dot(info['gripper_direction'], to_target_norm)
        print(f"Alignment with target: {alignment:.3f} (1.0 = perfect)")
    
    def adjust_joint(self, joint_idx, delta):
        """Adjust a joint position."""
        if 0 <= joint_idx < 7:
            new_val = self.joint_vals[joint_idx] + delta
            # Clip to limits
            new_val = np.clip(new_val, self.joint_limits[joint_idx][0], self.joint_limits[joint_idx][1])
            self.joint_vals[joint_idx] = new_val
            
            # Apply to simulation
            self.data.qpos[14 + joint_idx] = new_val
            self.data.ctrl[joint_idx] = new_val
            mujoco.mj_forward(self.model, self.data)
    
    def run_interactive(self):
        """Run interactive tuning."""
        print("\nGRASP POSITION FINDER")
        print("=" * 60)
        print("Commands:")
        print("  1-7: Select joint to adjust")
        print("  +/-: Increase/decrease selected joint by 0.05 rad")
        print("  ++/--: Increase/decrease by 0.2 rad")
        print("  r: Reset to home position")
        print("  t: Toggle target (red/blue)")
        print("  g: Test grasp (close gripper)")
        print("  o: Open gripper")
        print("  s: Save current position")
        print("  p: Print current status")
        print("  v: Launch viewer")
        print("  q: Quit")
        
        selected_joint = 0
        
        self.print_status()
        
        while True:
            cmd = input(f"\nJoint {selected_joint+1} selected > ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd in '1234567':
                selected_joint = int(cmd) - 1
                print(f"Selected joint {selected_joint+1}")
                limits = self.joint_limits[selected_joint]
                print(f"  Current: {self.joint_vals[selected_joint]:.3f}")
                print(f"  Limits: [{limits[0]:.3f}, {limits[1]:.3f}]")
            elif cmd == '+':
                self.adjust_joint(selected_joint, 0.05)
                self.print_status()
            elif cmd == '-':
                self.adjust_joint(selected_joint, -0.05)
                self.print_status()
            elif cmd == '++':
                self.adjust_joint(selected_joint, 0.2)
                self.print_status()
            elif cmd == '--':
                self.adjust_joint(selected_joint, -0.2)
                self.print_status()
            elif cmd == 'r':
                self.reset_position()
                print("Reset to home position")
                self.print_status()
            elif cmd == 't':
                self.current_target = "blue" if self.current_target == "red" else "red"
                print(f"Switched to {self.current_target} block")
                self.print_status()
            elif cmd == 'g':
                self.data.ctrl[7] = 0  # Close gripper
                for _ in range(50):
                    mujoco.mj_step(self.model, self.data)
                print("Gripper closed")
                self.print_status()
            elif cmd == 'o':
                self.data.ctrl[7] = 255  # Open gripper
                for _ in range(50):
                    mujoco.mj_step(self.model, self.data)
                print("Gripper opened")
                self.print_status()
            elif cmd == 's':
                print(f"\nSaved position for {self.current_target} block:")
                print(f'"{self.current_target}_grasp": np.array({list(self.joint_vals)}),')
            elif cmd == 'p':
                self.print_status()
            elif cmd == 'v':
                print("Launching viewer... (close viewer window to return)")
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    while viewer.is_running():
                        # Update control
                        for i in range(7):
                            self.data.ctrl[i] = self.joint_vals[i]
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                print("Viewer closed")
            else:
                print("Unknown command")

if __name__ == "__main__":
    finder = GraspPositionFinder()
    finder.run_interactive()