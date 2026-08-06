"""
Basic Model Loading and Usage
==============================

This example demonstrates how to load a BiosymModel and perform basic operations
including forward kinematics, dynamics computations, and performance analysis.

We'll use a simple pendulum model to illustrate the core functionality of the
BiosymModel class.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import timeit
import os

from biosym.model.model import load_model
from biosym.utils import states
from biosym.utils.paths import find_repo_root

# sphinx_gallery_start_ignore
# biosym is importable regardless of CWD, but example data/model paths below
# are given relative to the repo root, so chdir there for reproducibility.
current_dir = find_repo_root()
os.chdir(current_dir)
# sphinx_gallery_end_ignore

###############################################################################
# Load the Model
# --------------
# 
# First, we load a simple pendulum model from an XML file. The load_model
# function handles caching automatically, so subsequent loads will be faster.
# We toggle force_rebuild to True to ensure we load from the XML file directly and not from cache.

model_file = os.path.join(current_dir, "tests", "models", "pendulum.xml")
print("Loading pendulum model...")
start_time = time.time()
model = load_model(model_file, force_rebuild=True)
load_time = time.time() - start_time

print(f"Model loaded in {load_time:.3f} seconds")
print(f"Model has {model.n_states} states and {model.n_constants} constants")

###############################################################################
# Explore Model Structure
# ------------------------
#
# Let's examine the structure of our loaded model to understand its components.
# Model symbols are exposed as namedtuples, so they are accessed via dot notation.

print("\n--- Model Structure ---")
print(f"Coordinates: {model.coordinates.names}")
print(f"Speeds: {model.speeds.names}")
print(f"Forces: {model.tau.names}")

print(f"\nBodies in the model:")
for i, body in enumerate(model.dicts['bodies']):
    mass = body['mass'][0] if isinstance(body['mass'], list) else body['mass']
    com = body['com'] if 'com' in body else np.zeros(3)
    inertia = body['inertia'] if 'inertia' in body else np.zeros((3, 3))
    print(f"  {i}: {body['name']} (mass: {mass:.3f} kg, com: {com}, inertia: {inertia})")

print(f"\nJoints in the model:")
for i, joint in enumerate(model.dicts['joints']):
    print(f"  {i}: {joint['name']} (type: {joint['type']})")

###############################################################################
# Set Up Initial Conditions
# --------------------------
#
# Before running any computations, we need to set up the state and constant
# vectors. The model provides default ``States``/``Constants`` instances that
# we can use directly (or modify via ``.replace(...)``).

states_obj = model.default_states
constants_obj = model.default_constants

print(f"\nInitialized states vector with {states_obj.size()} elements")
print(f"Initialized constants vector with {constants_obj.flatten().shape[0]} elements")

###############################################################################
# Forward Kinematics Analysis
# ----------------------------
# 
# Now let's compute the forward kinematics for different pendulum angles
# to understand how the end-effector moves through space.

print("\n--- Forward Kinematics Analysis ---")

# Define a range of pendulum angles
angles = np.linspace(-np.pi/2, np.pi/2, 50)
angles2 = np.linspace(-2*np.pi, 2*np.pi, 50)
positions = []
velocities = []

# Set a small angular velocity on the first hinge for velocity computations
states_obj = states_obj.replace(qd=jnp.array(states_obj.qd).at[0].set(0.5))  # angular velocity in rad/s

print("Computing forward kinematics for 50 different angles...")

for angle, angle2 in zip(angles, angles2):
    q = jnp.array(states_obj.q).at[0].set(angle)   # Set angle of the first hinge
    q = q.at[1].set(angle2)                        # Set angle of the second hinge
    states_obj = states_obj.replace(q=q)

    # Compute forward kinematics (positions)
    pos = model.run["FK_vis"](states_obj, constants_obj)[-1, :2]
    positions.append(pos.flatten())

    # Compute velocity kinematics
    vel = model.run["FK_dot"](states_obj, constants_obj)[-1, :2]
    velocities.append(vel.flatten())

positions = np.array(positions)
velocities = np.array(velocities)

print(f"Forward kinematics computed for {len(angles)} configurations")
print(f"Position output shape: {positions.shape}")

plt.plot(positions[:, 0], positions[:, 1], 'b-')
plt.title('Pendulum End-Effector Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid()
plt.axis('equal')
plt.show()

###############################################################################
# Dynamics Computations
# ----------------------
# 
# Let's compute the equations of motion and examine the mass matrix and
# forcing terms for our pendulum model.

print("\n--- Dynamics Analysis ---")

# Set initial conditions: 45 degrees with some angular velocity
states_obj = states_obj.replace(q=jnp.array(states_obj.q).at[0].set(np.pi / 4))   # 45 degrees
states_obj = states_obj.replace(qd=jnp.array(states_obj.qd).at[0].set(1.0))       # 1 rad/s angular velocity

# Compute equations of motion residual (Kane's method)
eom_residual = model.run["kane"](states_obj, constants_obj)
print(f"EOM residual: {eom_residual}")

# Compute mass matrix
mass_matrix = model.run["mass_matrix"](states_obj, constants_obj)
print(f"Mass matrix shape: {mass_matrix.shape}")
print(f"Mass matrix:\n{mass_matrix}")

# Compute forcing terms (Coriolis, centrifugal, gravity)
forcing = model.run["forcing"](states_obj, constants_obj)
print(f"Forcing terms: {forcing}")

# Compute Jacobian of the equations of motion for sensitivity analysis.
# The result is a States pytree: one Jacobian block per state field
# (each block has shape (n_eom_residuals, *field_shape)).
jacobian = model.run["kane_jacobian"](states_obj, constants_obj)
print(f"Jacobian: {jacobian}")

###############################################################################