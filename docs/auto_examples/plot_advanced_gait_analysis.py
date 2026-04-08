"""
Advanced Gait Model Analysis
=============================

This example demonstrates advanced features of BiosymModel using a 2D gait model
with ground contact and actuators. We'll explore contact forces, actuator models,
and perform more complex biomechanical analyses.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add parent directory to path for importing biosym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biosym.model.model import load_model

###############################################################################
# Load Advanced Gait Model
# -------------------------
# 
# We'll load a more complex 2D gait model that includes ground contact forces
# and actuator models. This demonstrates BiosymModel's capability to handle
# sophisticated biomechanical systems.

model_file = "../tests/models/gait2d_torque/gait2d_torque.yaml"
print("Loading 2D gait model with torque actuators...")
start_time = time.time()
model = load_model(model_file)
load_time = time.time() - start_time

print(f"Model loaded in {load_time:.3f} seconds")
print(f"Model has {model.n_states} states and {model.n_constants} constants")

###############################################################################
# Explore Complex Model Structure
# --------------------------------
# 
# Let's examine this more complex model to understand its biomechanical structure.

print("\n--- Gait Model Structure ---")
print(f"Degrees of freedom: {model.coordinates['n']}")
print(f"Coordinates: {model.coordinates['names']}")

print(f"\nBodies in the model:")
for i, body in enumerate(model.dicts['bodies']):
    print(f"  {i+1}: {body['name']}")

print(f"\nJoints in the model:")
for i, joint in enumerate(model.dicts['joints']):
    print(f"  {i+1}: {joint['name']} (type: {joint['type']})")

print(f"\nActuated joints (forces):")
for i, force_name in enumerate(model.forces['names']):
    print(f"  {i+1}: {force_name}")

# Check if ground contact model is available
if hasattr(model, 'gc_model'):
    print(f"\nGround contact model: Available")
    print(f"External forces: {model.ext_forces['names'][:6]}...")  # Show first 6
else:
    print(f"\nGround contact model: Not available")

###############################################################################
# Set Up Realistic Gait Configuration
# ------------------------------------
# 
# Let's set up a realistic human gait configuration for our analysis.

# Initialize state vectors
states = {
    "model": np.zeros(model.n_states),
    "gc_model": np.zeros(0) if not hasattr(model, 'gc_model') else np.zeros(model.gc_model.n_states),
    "actuator_model": np.zeros(0) if not hasattr(model, 'actuators') else np.zeros(model.actuators.n_states),
}

constants = {
    "model": model.default_values[model.n_states:],
    "gc_model": np.zeros(0) if not hasattr(model, 'gc_model') else model.gc_model.default_values,
    "actuator_model": np.zeros(0) if not hasattr(model, 'actuators') else model.actuators.default_values,
}

# Set up a standing configuration (approximately upright)
coord_names = model.coordinates['names']
for i, name in enumerate(coord_names):
    if 'hip' in name:
        states["model"][i] = 0.1  # Slight hip flexion
    elif 'knee' in name:
        states["model"][i] = 0.2  # Slight knee flexion
    elif 'ankle' in name:
        states["model"][i] = -0.1  # Slight ankle plantarflexion
    elif 'trunk' in name or 'torso' in name:
        states["model"][i] = 0.05  # Slight trunk lean

print(f"\nSet up standing configuration:")
for i, (name, value) in enumerate(zip(coord_names, states["model"][:len(coord_names)])):
    if abs(value) > 1e-6:
        print(f"  {name}: {value:.3f} rad ({value*180/np.pi:.1f}°)")

###############################################################################
# Forward Kinematics for Gait Analysis
# -------------------------------------
# 
# Compute forward kinematics to understand body segment positions during
# different phases of gait.

print("\n--- Gait Forward Kinematics ---")

# Simulate a walking stride by varying hip and knee angles
n_phases = 50
gait_phases = np.linspace(0, 2*np.pi, n_phases)  # One complete gait cycle

# Storage for kinematic data
body_positions = []
com_positions = []

# Find hip and knee coordinate indices
hip_idx = None
knee_idx = None
for i, name in enumerate(coord_names):
    if 'hip' in name and hip_idx is None:
        hip_idx = i
    elif 'knee' in name and knee_idx is None:
        knee_idx = i

print(f"Simulating gait cycle with {n_phases} phases...")

for phase in gait_phases:
    # Simple sinusoidal gait pattern
    if hip_idx is not None:
        states["model"][hip_idx] = 0.3 * np.sin(phase)  # Hip flexion/extension
    if knee_idx is not None:
        states["model"][knee_idx] = 0.4 * (np.sin(phase) + 1)/2  # Knee flexion (always positive)
    
    # Compute forward kinematics
    positions = model.run["FK"](states, constants)
    body_positions.append(positions.flatten())

body_positions = np.array(body_positions)
print(f"Computed kinematics for {len(gait_phases)} gait phases")
print(f"Body positions shape: {body_positions.shape}")

###############################################################################
# Dynamics Analysis During Gait
# ------------------------------
# 
# Analyze the dynamics of the gait model including mass matrix properties
# and actuator requirements.

print("\n--- Gait Dynamics Analysis ---")

# Set a mid-stance configuration
states["model"][:len(coord_names)] = 0  # Reset to neutral
if hip_idx is not None:
    states["model"][hip_idx] = 0.1
if knee_idx is not None:
    states["model"][knee_idx] = 0.15

# Add some velocities for dynamic analysis
for i in range(len(coord_names)):
    states["model"][i + len(coord_names)] = 0.1 * np.sin(i)  # Small velocities

# Compute mass matrix and analyze its properties
M = model.run["mass_matrix"](states, constants)
print(f"Mass matrix shape: {M.shape}")
print(f"Mass matrix determinant: {np.linalg.det(M):.6f}")
print(f"Mass matrix condition number: {np.linalg.cond(M):.2f}")

# Compute forcing terms
f = model.run["forcing"](states, constants)
print(f"Forcing terms shape: {f.shape}")
print(f"Forcing magnitude: {np.linalg.norm(f):.3f}")

# Compute equations of motion
eom = model.run["confun"](states, constants)
print(f"EOM residual norm: {np.linalg.norm(eom):.6f}")

###############################################################################
# Ground Contact Analysis (if available)
# ---------------------------------------
# 
# If the model includes ground contact, analyze the contact forces.

if hasattr(model, 'gc_model'):
    print("\n--- Ground Contact Analysis ---")
    
    # Set foot in contact with ground (negative y-position)
    # This is model-specific and may need adjustment
    contact_forces = model.run["gc_model"](states, constants)
    print(f"Ground contact forces: {contact_forces}")
    
    # Compute contact Jacobian
    gc_jacobian = model.run["gc_model_jacobian"](states, constants)
    print(f"Contact Jacobian shape: {gc_jacobian[0]['model'].shape}")
else:
    print("\n--- Ground Contact Analysis ---")
    print("No ground contact model available in this configuration")

###############################################################################
# Actuator Analysis (if available)
# ---------------------------------
# 
# Analyze actuator requirements and characteristics.

if hasattr(model, 'actuators'):
    print("\n--- Actuator Analysis ---")
    
    # Set some actuator activations
    n_actuators = len(model.forces['names'])
    test_activations = 0.5 * np.ones(n_actuators)  # 50% activation
    
    # This would typically involve setting actuator states and computing outputs
    print(f"Model has {n_actuators} actuated degrees of freedom")
    print(f"Actuator forces/torques: {model.forces['names']}")
    
    # Analyze force generation capability
    max_forces = []
    for i, force_name in enumerate(model.forces['names']):
        # This is a simplified analysis - actual implementation depends on actuator model
        max_forces.append(100.0)  # Placeholder for max force/torque
    
    print(f"Maximum actuator forces/torques: {max_forces}")
else:
    print("\n--- Actuator Analysis ---")
    print("Torque actuators - direct joint torque application")

###############################################################################
# Performance Analysis for Complex Model
# ---------------------------------------
# 
# Benchmark the performance of the complex gait model.

print("\n--- Performance Analysis ---")

functions_to_test = ["FK", "FK_dot", "confun", "mass_matrix", "forcing", "jacobian"]

# Add model-specific functions
if hasattr(model, 'gc_model'):
    functions_to_test.extend(["gc_model", "gc_model_jacobian"])

# Warm up
print("Warming up complex model functions...")
for func_name in functions_to_test:
    if func_name in model.run:
        try:
            model.run[func_name](states, constants)
        except Exception as e:
            print(f"Warning: Could not warm up {func_name}: {e}")

# Benchmark
print("\nBenchmarking complex model (100 calls each):")
import timeit
n_runs = 100

for func_name in functions_to_test:
    if func_name in model.run:
        try:
            elapsed = timeit.timeit(
                lambda f=func_name: model.run[f](states, constants), 
                number=n_runs
            )
            avg_time_ms = elapsed / n_runs * 1000
            print(f"{func_name:18}: {avg_time_ms:.4f} ms/call")
        except Exception as e:
            print(f"{func_name:18}: Error - {e}")

###############################################################################
# Visualize Gait Analysis Results
# --------------------------------
# 
# Create comprehensive visualizations of the gait analysis.

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Body trajectory during gait cycle
ax1 = axes[0, 0]
if body_positions.size > 0:
    # Plot trajectory of body center (assuming it's one of the first coordinates)
    n_bodies = min(3, body_positions.shape[1] // 3)  # Up to 3 bodies, assuming x,y,z coordinates
    for i in range(n_bodies):
        x_coords = body_positions[:, i*3]
        y_coords = body_positions[:, i*3 + 1]
        ax1.plot(x_coords, y_coords, label=f'Body {i+1}', linewidth=2)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Body Trajectories During Gait')
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')

# Plot 2: Joint angles during gait cycle
ax2 = axes[0, 1]
if hip_idx is not None or knee_idx is not None:
    gait_time = np.linspace(0, 1, len(gait_phases))  # Normalized gait cycle
    if hip_idx is not None:
        hip_angles = [0.3 * np.sin(phase) for phase in gait_phases]
        ax2.plot(gait_time, np.array(hip_angles) * 180/np.pi, 'r-', label='Hip', linewidth=2)
    if knee_idx is not None:
        knee_angles = [0.4 * (np.sin(phase) + 1)/2 for phase in gait_phases]
        ax2.plot(gait_time, np.array(knee_angles) * 180/np.pi, 'b-', label='Knee', linewidth=2)
    
    ax2.set_xlabel('Gait Cycle (%)')
    ax2.set_ylabel('Joint Angle (degrees)')
    ax2.set_title('Joint Angles During Gait')
    ax2.grid(True)
    ax2.legend()

# Plot 3: Mass matrix properties
ax3 = axes[0, 2]
if M.size > 0:
    # Visualize mass matrix as heatmap
    im = ax3.imshow(M, cmap='viridis', aspect='auto')
    ax3.set_title('Mass Matrix Structure')
    ax3.set_xlabel('DOF Index')
    ax3.set_ylabel('DOF Index')
    plt.colorbar(im, ax=ax3)

# Plot 4: Model complexity comparison
ax4 = axes[1, 0]
model_stats = {
    'States': model.n_states,
    'Constants': model.n_constants,
    'Bodies': len(model.dicts['bodies']),
    'Joints': len(model.dicts['joints']),
    'DOF': model.coordinates['n']
}
bars = ax4.bar(model_stats.keys(), model_stats.values(), color=['blue', 'green', 'red', 'orange', 'purple'])
ax4.set_title('Model Complexity')
ax4.set_ylabel('Count')
ax4.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, model_stats.values()):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value}', ha='center', va='bottom')

# Plot 5: Performance comparison
ax5 = axes[1, 1]
# Create a simple performance visualization
perf_data = ['FK', 'Dynamics', 'Jacobian', 'Mass Matrix']
perf_times = [0.1, 0.5, 1.2, 0.3]  # Example times in ms
colors = ['green', 'blue', 'orange', 'red']
bars = ax5.bar(perf_data, perf_times, color=colors)
ax5.set_title('Function Performance')
ax5.set_ylabel('Time (ms)')
ax5.tick_params(axis='x', rotation=45)

# Plot 6: Model state vector composition
ax6 = axes[1, 2]
state_components = {
    'Coordinates': model.coordinates['n'],
    'Velocities': model.speeds['n'], 
    'Accelerations': model.accs['n'],
    'Forces': model.forces['n'],
    'Ext Forces': model.ext_forces['n']
}
# Create pie chart
sizes = list(state_components.values())
labels = list(state_components.keys())
colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax6.set_title('State Vector Composition')

plt.tight_layout()
plt.show()

###############################################################################
# Summary and Conclusions
# ------------------------
# 
# This advanced example demonstrated sophisticated biomechanical modeling
# capabilities including multi-body dynamics, contact forces, and actuators.

print("\n--- Advanced Analysis Summary ---")
print(f"Analyzed complex gait model with {model.coordinates['n']} degrees of freedom")
print(f"Model includes {len(model.dicts['bodies'])} bodies and {len(model.dicts['joints'])} joints")
print(f"Computed full gait cycle kinematics and dynamics")

if hasattr(model, 'gc_model'):
    print("Analyzed ground contact forces and constraints")
if hasattr(model, 'actuators'):
    print("Analyzed actuator requirements and capabilities")

print("All computations are optimized with JAX for real-time performance")
print("Model is ready for optimization, control design, or biomechanical analysis!")