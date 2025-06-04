#!/usr/bin/env python3
"""
Comprehensive diagnostic for robot movement issues.
"""

import numpy as np
import mujoco
import matplotlib.pyplot as plt

class MovementDiagnostic:
    def __init__(self):
        # Load model
        self.xml_path = "../franka_emika_panda/panda_demo_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Get joint info
        self.joint_names = []
        self.joint_ids = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and 'panda_joint' in name:
                self.joint_names.append(name)
                self.joint_ids.append(i)
        
        print(f"Found {len(self.joint_ids)} panda joints: {self.joint_names}")
        
        # Control indices
        self.ctrl_indices = list(range(7))  # 7 joints
        
        # Data storage
        self.time_data = []
        self.action_data = []
        self.ctrl_data = []
        self.qpos_data = []
        self.qvel_data = []
        self.ee_pos_data = []
        self.expected_pos_data = []
        
    def reset_robot(self):
        """Reset to initial configuration."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Set initial joint positions
        joint_vals = [0, -0.1, 0, -2.0, 0, 1.5, 0]
        for i, val in enumerate(joint_vals):
            self.data.qpos[14 + i] = val
            self.data.ctrl[i] = val
        self.data.ctrl[7] = 255  # Gripper open
        
        mujoco.mj_forward(self.model, self.data)
        
    def test_movement(self, test_name, action_sequence, duration=2.0):
        """Test a specific movement pattern."""
        print(f"\n{'='*80}")
        print(f"Testing: {test_name}")
        print(f"{'='*80}")
        
        self.reset_robot()
        
        # Clear data
        self.time_data.clear()
        self.action_data.clear()
        self.ctrl_data.clear()
        self.qpos_data.clear()
        self.qvel_data.clear()
        self.ee_pos_data.clear()
        self.expected_pos_data.clear()
        
        # Initial state
        initial_qpos = [self.data.qpos[14 + i] for i in range(7)]
        expected_pos = initial_qpos.copy()
        
        dt = self.model.opt.timestep
        n_steps = int(duration / dt)
        
        # Get end-effector ID
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        
        print(f"Timestep: {dt}, Total steps: {n_steps}")
        print(f"Initial joint positions: {[f'{p:.3f}' for p in initial_qpos]}")
        
        for step in range(n_steps):
            t = step * dt
            
            # Get action for this step
            if callable(action_sequence):
                action = action_sequence(t)
            else:
                action = action_sequence
            
            # Store action
            self.action_data.append(action.copy())
            self.time_data.append(t)
            
            # Apply control - VELOCITY CONTROL
            for i in range(7):
                current_pos = self.data.qpos[14 + i]
                # Integrate velocity to get target position
                target_pos = current_pos + action[i] * dt
                self.data.ctrl[i] = target_pos
                
                # Update expected position
                expected_pos[i] += action[i] * dt
            
            # Store control values
            self.ctrl_data.append([self.data.ctrl[i] for i in range(7)])
            
            # Step physics
            mujoco.mj_step(self.model, self.data)
            
            # Store actual positions and velocities
            actual_qpos = [self.data.qpos[14 + i] for i in range(7)]
            actual_qvel = [self.data.qvel[14 + i] for i in range(7)]
            ee_pos = self.data.xpos[ee_id].copy()
            
            self.qpos_data.append(actual_qpos)
            self.qvel_data.append(actual_qvel)
            self.ee_pos_data.append(ee_pos)
            self.expected_pos_data.append(expected_pos.copy())
            
            # Print detailed info every 0.2 seconds
            if step % int(0.2 / dt) == 0:
                print(f"\nTime: {t:.2f}s")
                print(f"Actions:      {[f'{a:6.3f}' for a in action[:7]]}")
                print(f"Ctrl values:  {[f'{self.data.ctrl[i]:6.3f}' for i in range(7)]}")
                print(f"Actual pos:   {[f'{p:6.3f}' for p in actual_qpos]}")
                print(f"Expected pos: {[f'{p:6.3f}' for p in expected_pos]}")
                print(f"Position err: {[f'{actual_qpos[i]-expected_pos[i]:6.3f}' for i in range(7)]}")
                print(f"Velocities:   {[f'{v:6.3f}' for v in actual_qvel]}")
                print(f"EE position:  [{ee_pos[0]:6.3f}, {ee_pos[1]:6.3f}, {ee_pos[2]:6.3f}]")
                
                # Check if control is being applied
                for i in range(7):
                    if abs(action[i]) > 0.01 and abs(actual_qvel[i]) < 0.001:
                        print(f"  ⚠️ Joint {i} commanded {action[i]:.3f} but velocity is {actual_qvel[i]:.3f}")
        
        return {
            'time': np.array(self.time_data),
            'actions': np.array(self.action_data),
            'ctrl': np.array(self.ctrl_data),
            'qpos': np.array(self.qpos_data),
            'qvel': np.array(self.qvel_data),
            'ee_pos': np.array(self.ee_pos_data),
            'expected_pos': np.array(self.expected_pos_data)
        }
    
    def plot_results(self, results, test_name):
        """Plot diagnostic results."""
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle(f"Movement Diagnostic: {test_name}", fontsize=16)
        
        time = results['time']
        
        # Plot actions
        ax = axes[0, 0]
        for i in range(7):
            ax.plot(time, results['actions'][:, i], label=f'Joint {i}')
        ax.set_ylabel('Commanded Velocity')
        ax.set_title('Actions (Velocity Commands)')
        ax.grid(True)
        ax.legend()
        
        # Plot control values
        ax = axes[0, 1]
        for i in range(7):
            ax.plot(time, results['ctrl'][:, i], label=f'Joint {i}')
        ax.set_ylabel('Control Value')
        ax.set_title('Control Values (Target Positions)')
        ax.grid(True)
        
        # Plot actual positions
        ax = axes[1, 0]
        for i in range(7):
            ax.plot(time, results['qpos'][:, i], label=f'Joint {i}')
        ax.set_ylabel('Position (rad)')
        ax.set_title('Actual Joint Positions')
        ax.grid(True)
        
        # Plot position errors
        ax = axes[1, 1]
        pos_errors = results['qpos'] - results['expected_pos']
        for i in range(7):
            ax.plot(time, pos_errors[:, i], label=f'Joint {i}')
        ax.set_ylabel('Position Error (rad)')
        ax.set_title('Position Error (Actual - Expected)')
        ax.grid(True)
        
        # Plot velocities
        ax = axes[2, 0]
        for i in range(7):
            ax.plot(time, results['qvel'][:, i], label=f'Joint {i}')
        ax.set_ylabel('Velocity (rad/s)')
        ax.set_title('Actual Joint Velocities')
        ax.grid(True)
        
        # Plot end-effector position
        ax = axes[2, 1]
        ax.plot(time, results['ee_pos'][:, 0], 'r-', label='X')
        ax.plot(time, results['ee_pos'][:, 1], 'g-', label='Y')
        ax.plot(time, results['ee_pos'][:, 2], 'b-', label='Z')
        ax.set_ylabel('Position (m)')
        ax.set_title('End-Effector Position')
        ax.legend()
        ax.grid(True)
        
        # Plot control effectiveness
        ax = axes[3, 0]
        for i in range(3):  # Just first 3 joints
            commanded = results['actions'][:, i]
            achieved = results['qvel'][:, i]
            effectiveness = np.where(np.abs(commanded) > 0.01, 
                                   achieved / (commanded + 1e-6), 0)
            ax.plot(time, effectiveness, label=f'Joint {i}')
        ax.set_ylabel('Effectiveness Ratio')
        ax.set_title('Control Effectiveness (Achieved/Commanded)')
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        ax.legend()
        
        # Joint limits check
        ax = axes[3, 1]
        ax.text(0.1, 0.9, "Joint Limits Check:", transform=ax.transAxes, fontsize=12, weight='bold')
        y_pos = 0.8
        for i in range(7):
            min_pos = np.min(results['qpos'][:, i])
            max_pos = np.max(results['qpos'][:, i])
            limit_low = self.model.jnt_range[self.joint_ids[i], 0] if self.model.jnt_limited[self.joint_ids[i]] else -np.inf
            limit_high = self.model.jnt_range[self.joint_ids[i], 1] if self.model.jnt_limited[self.joint_ids[i]] else np.inf
            
            status = "OK"
            if min_pos <= limit_low + 0.01 or max_pos >= limit_high - 0.01:
                status = "AT LIMIT!"
            
            ax.text(0.1, y_pos, f"Joint {i}: [{min_pos:.2f}, {max_pos:.2f}] / [{limit_low:.2f}, {limit_high:.2f}] {status}",
                    transform=ax.transAxes, fontsize=10,
                    color='red' if status != "OK" else 'black')
            y_pos -= 0.1
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'movement_diagnostic_{test_name.replace(" ", "_")}.png')
        print(f"\nSaved plot to: movement_diagnostic_{test_name.replace(' ', '_')}.png")
        
def main():
    diag = MovementDiagnostic()
    
    # Test 1: Constant velocity on Joint 1
    print("\n" + "="*80)
    print("TEST 1: Constant velocity on Joint 1 (Shoulder)")
    print("Expected: Joint 1 should move continuously")
    action1 = np.zeros(8)
    action1[1] = 0.5  # 0.5 rad/s on shoulder
    results1 = diag.test_movement("Joint 1 Constant Velocity", action1, duration=2.0)
    diag.plot_results(results1, "joint1_constant")
    
    # Test 2: Multiple joints
    print("\n" + "="*80)
    print("TEST 2: Multiple joints moving")
    print("Expected: Joints 1, 3 should move")
    action2 = np.zeros(8)
    action2[1] = 0.3   # Shoulder
    action2[3] = -0.2  # Elbow
    results2 = diag.test_movement("Multiple Joints", action2, duration=2.0)
    diag.plot_results(results2, "multiple_joints")
    
    # Test 3: Time-varying action
    print("\n" + "="*80)
    print("TEST 3: Time-varying action (sine wave)")
    print("Expected: Joint 1 should oscillate")
    def sine_action(t):
        action = np.zeros(8)
        action[1] = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine
        return action
    results3 = diag.test_movement("Sine Wave", sine_action, duration=4.0)
    diag.plot_results(results3, "sine_wave")
    
    # Test 4: Check if it's a control mode issue
    print("\n" + "="*80)
    print("TEST 4: Direct position control test")
    diag.reset_robot()
    
    print("\nDirect control test - setting ctrl values directly:")
    initial_pos = [diag.data.qpos[14 + i] for i in range(7)]
    print(f"Initial positions: {[f'{p:.3f}' for p in initial_pos]}")
    
    # Set control to move joint 1
    target_pos = initial_pos.copy()
    target_pos[1] += 0.5  # Move joint 1 by 0.5 rad
    
    for i in range(7):
        diag.data.ctrl[i] = target_pos[i]
    
    print(f"Target positions:  {[f'{p:.3f}' for p in target_pos]}")
    print(f"Control values:    {[f'{diag.data.ctrl[i]:.3f}' for i in range(7)]}")
    
    # Step multiple times
    for _ in range(100):
        mujoco.mj_step(diag.model, diag.data)
    
    final_pos = [diag.data.qpos[14 + i] for i in range(7)]
    print(f"Final positions:   {[f'{p:.3f}' for p in final_pos]}")
    print(f"Position changes:  {[f'{final_pos[i]-initial_pos[i]:.3f}' for i in range(7)]}")
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    # Check control mode
    print(f"\nControl Configuration:")
    print(f"Number of controls (nu): {diag.model.nu}")
    print(f"Number of joints (njnt): {diag.model.njnt}")
    print(f"Number of DoFs (nv): {diag.model.nv}")
    
    # Check actuator configuration
    print(f"\nActuator Configuration:")
    for i in range(diag.model.nu):
        actuator_name = mujoco.mj_id2name(diag.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  Actuator {i}: {actuator_name}")
        if i < 7:  # Joint actuators
            joint_id = diag.model.actuator_trnid[i, 0]
            joint_name = mujoco.mj_id2name(diag.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            print(f"    Controls joint: {joint_name}")
            print(f"    Control range: [{diag.model.actuator_ctrlrange[i, 0]:.2f}, {diag.model.actuator_ctrlrange[i, 1]:.2f}]")

if __name__ == "__main__":
    main()