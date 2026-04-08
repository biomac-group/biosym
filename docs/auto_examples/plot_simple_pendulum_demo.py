"""
Simple BiosymModel Example
==========================

This example demonstrates the core functionality of BiosymModel with
a simple pendulum model. It covers loading, structure inspection,
and basic computations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for importing biosym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biosym.model.model import load_model

###############################################################################
# Load and Inspect the Model
# ---------------------------
# 
# First, let's load a simple pendulum model and understand its structure.

print("Loading pendulum model...")
model_file = "../tests/models/pendulum.xml"
model = load_model(model_file)

print(f"✓ Model loaded successfully!")
print(f"  States: {model.n_states}")
print(f"  Constants: {model.n_constants}")
print(f"  Coordinates: {model.coordinates['names']}")
print(f"  Bodies: {[body['name'] for body in model.dicts['bodies']]}")

###############################################################################
# Understanding the State Vector
# -------------------------------
# 
# The model uses a structured state vector containing positions, velocities,
# accelerations, and forces. Let's explore its organization.

print(f"\n--- State Vector Structure ---")
print(f"Total states: {model.n_states}")
print(f"Coordinates ({model.coordinates['n']}): indices {model.coordinates['idx']} to {model.coordinates['idx'] + model.coordinates['n'] - 1}")
print(f"Speeds ({model.speeds['n']}): indices {model.speeds['idx']} to {model.speeds['idx'] + model.speeds['n'] - 1}")
print(f"Accelerations ({model.accs['n']}): indices {model.accs['idx']} to {model.accs['idx'] + model.accs['n'] - 1}")
print(f"Forces ({model.forces['n']}): indices {model.forces['idx']} to {model.forces['idx'] + model.forces['n'] - 1}")

###############################################################################
# Basic Model Evaluation
# -----------------------
# 
# Let's evaluate some basic model functions using the simple interface
# from the main script in model.py.

print(f"\n--- Basic Model Evaluation ---")

# Create state and constant vectors as shown in the model.py main script
states = {
    "model": np.zeros(model.n_states),
    "gc_model": np.zeros(0),
    "actuator_model": np.zeros(0),
}

constants = {
    "model": np.zeros(model.n_constants),  # Using zeros like in the main script
    "gc_model": np.zeros(0),
    "actuator_model": np.zeros(0),
}

print("Available model functions:")
for func_name in sorted(model.run.keys()):
    print(f"  • {func_name}")

# Test forward kinematics
print(f"\n--- Forward Kinematics Test ---")
try:
    # Set a small pendulum angle
    states["model"][0] = 0.3  # ~17 degrees
    positions = model.run["FK"](states, constants)
    print(f"✓ Forward kinematics successful")
    print(f"  Output shape: {positions.shape}")
    print(f"  End-effector position: ({positions[0, 0]:.3f}, {positions[0, 1]:.3f})")
except Exception as e:
    print(f"✗ Forward kinematics failed: {e}")

# Test equations of motion
print(f"\n--- Dynamics Test ---")
try:
    eom = model.run["confun"](states, constants)
    print(f"✓ Equations of motion successful")
    print(f"  EOM residual shape: {eom.shape}")
    print(f"  Residual norm: {np.linalg.norm(eom):.6f}")
except Exception as e:
    print(f"✗ Equations of motion failed: {e}")

# Test mass matrix
try:
    M = model.run["mass_matrix"](states, constants)
    print(f"✓ Mass matrix computation successful")
    print(f"  Mass matrix shape: {M.shape}")
    print(f"  Determinant: {np.linalg.det(M):.6f}")
except Exception as e:
    print(f"✗ Mass matrix failed: {e}")

###############################################################################
# Performance Demonstration
# --------------------------
# 
# Show the performance benefits of JAX compilation.

print(f"\n--- Performance Demonstration ---")

import timeit

# Warm up functions (JIT compilation)
print("Warming up functions...")
for func_name in ["FK", "confun", "mass_matrix"]:
    if func_name in model.run:
        try:
            model.run[func_name](states, constants)
        except:
            pass

# Benchmark
n_runs = 1000
print(f"\nBenchmarking ({n_runs} runs each):")

for func_name in ["FK", "confun", "mass_matrix"]:
    if func_name in model.run:
        try:
            elapsed = timeit.timeit(
                lambda f=func_name: model.run[f](states, constants),
                number=n_runs
            )
            avg_time = elapsed / n_runs * 1000  # milliseconds
            print(f"  {func_name:12}: {avg_time:.4f} ms/call")
        except Exception as e:
            print(f"  {func_name:12}: Error - {e}")

###############################################################################
# Trajectory Computation
# -----------------------
# 
# Compute forward kinematics for different pendulum angles.

print(f"\n--- Trajectory Computation ---")

angles = np.linspace(-np.pi/2, np.pi/2, 20)
positions = []

print(f"Computing trajectory for {len(angles)} angles...")

for angle in angles:
    try:
        states["model"][0] = angle
        pos = model.run["FK"](states, constants)
        positions.append([pos[0, 0], pos[0, 1]])  # x, y coordinates
    except Exception as e:
        print(f"Error at angle {angle:.2f}: {e}")
        positions.append([np.nan, np.nan])

positions = np.array(positions)
print(f"✓ Trajectory computed successfully")

###############################################################################
# Visualization
# -------------
# 
# Create a simple plot showing the pendulum trajectory.

plt.figure(figsize=(10, 6))

# Plot 1: Pendulum trajectory
plt.subplot(1, 2, 1)
valid_idx = ~np.isnan(positions[:, 0])
if np.any(valid_idx):
    plt.plot(positions[valid_idx, 0], positions[valid_idx, 1], 'b-o', linewidth=2, markersize=4)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Pendulum End-Effector Trajectory')
    plt.grid(True)
    plt.axis('equal')

# Plot 2: Position vs angle
plt.subplot(1, 2, 2)
if np.any(valid_idx):
    plt.plot(angles[valid_idx] * 180/np.pi, positions[valid_idx, 0], 'r-', label='X position')
    plt.plot(angles[valid_idx] * 180/np.pi, positions[valid_idx, 1], 'b-', label='Y position')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Position (m)')
    plt.title('Position vs Pendulum Angle')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

###############################################################################
# Summary
# -------
# 
# This example demonstrated the essential features of BiosymModel.

print(f"\n" + "="*60)
print("EXAMPLE SUMMARY")
print("="*60)
print(f"✓ Successfully loaded {model.coordinates['n']}-DOF pendulum model")
print(f"✓ Demonstrated forward kinematics computation")
print(f"✓ Showed dynamics evaluation (EOM, mass matrix)")
print(f"✓ Illustrated JAX compilation performance benefits")
print(f"✓ Computed and visualized pendulum trajectory")
print(f"\nThe model is ready for:")
print(f"  • Optimization and control applications")
print(f"  • Dynamic simulation studies")  
print(f"  • Biomechanical analysis")
print("="*60)