"""
Batching
=============================

Aka how can I use biosym for deep learning applications?
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from biosym.model.model import load_model
from biosym.utils import states
from biosym.utils.paths import find_repo_root
import jax

# sphinx_gallery_start_ignore
# biosym is importable regardless of CWD, but example data/model paths below
# are given relative to the repo root, so chdir there for reproducibility.
current_dir = find_repo_root()
os.chdir(current_dir)
# sphinx_gallery_end_ignore

###############################################################################
# Load 2D Gait Model
# -------------------------
# 
# We'll load a more complex 2D gait model that includes ground contact forces
# and actuator models. This demonstrates BiosymModel's capability to handle
# sophisticated biomechanical systems.

model_file = os.path.join(current_dir, "tests", "models", "gait2d_torque", "gait2d_torque.yaml")
print("Loading 2D gait model with torque actuators...")
start_time = time.time()
model = load_model(model_file, force_rebuild=True)
load_time = time.time() - start_time

print(f"Model loaded in {load_time:.3f} seconds")
print(f"Model has {model.n_states} states and {model.n_constants} constants")


###############################################################################
# Create batches of movement data

# Initialize state vector (positions, velocities, accelerations, forces, etc.)
states_dict_0 = model.default_states
print(states_dict_0)

# Create a batch of 1000 identical state vectors
batch_size = 1000
states_ = states.concatenate([states_dict_0] * batch_size)
print(states_)

# For any function in the model, you can now pass in the batched states using jax.vmap
# e.g. here compute the output of the dynamics (constraint) function
# The input axes are defined as (0, None) meaning the first argument (states) is batched
# while the second argument (constants) is not batched
dynamics_fn = jax.vmap(model.run["kane"], in_axes=(0, None))
dynamics_output = dynamics_fn(states_, model.default_constants)
print("Dynamics output shape with batching:", dynamics_output.shape)

###############################################################################
# Performance of batching (optional)
# -------------------------
# Check if jax finds a GPU

print("Available devices:", jax.devices())
start_time = time.time()
dynamics_output = dynamics_fn(states_, model.default_constants)
end_time = time.time()
print(f"Computed dynamics for batch of size {batch_size} in {end_time - start_time:.4f} seconds")

start_time = time.time()
for i in range(batch_size):
    dynamics_output_single = model.run["kane"](states_[i], model.default_constants)
end_time = time.time()
print(f"Computed dynamics for batch of size {batch_size} without batching in {end_time - start_time:.4f} seconds")
