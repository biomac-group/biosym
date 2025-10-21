"""
Batching
=============================

Aka how can I use biosym for deep learning applications?
"""

from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
import time
# sphinx_gallery_start_ignore
import sys
import os

# Add parent directory to path for importing biosym
import os
import sys
# For documentation builds, handle __file__ not being defined


# Find the .git root directory, then set that as current dir
def _find_git_root(start_path):
    p = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(p, ".git")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent

try:
    start_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ may not be defined (e.g., in some doc builds); fall back to cwd
    start_path = os.path.abspath(os.getcwd())

_git_root = _find_git_root(start_path)
if _git_root:
    current_dir = _git_root
else:
    current_dir = start_path

# Set working directory and ensure repository root is on sys.path for imports
os.chdir(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# sphinx_gallery_end_ignore

from biosym.model.model import load_model
from biosym.utils import states
import jax

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
states_dict_0 = model.default_inputs
print(states_dict_0)

# Create a batch of 1000 identical state vectors
batch_size = 1000
states_ = states.stack_dataclasses([states_dict_0] * batch_size)
print(states_)

# For any function in the model, you can now pass in the batched states using jax.vmap
# e.g. here compute the output of the dynamics (constraint) function
# The input axes are defined as (0, None) meaning the first argument (states) is batched
# while the second argument (constants) is not batched
dynamics_fn = jax.vmap(model.run["confun"], in_axes=(0, None))
dynamics_output = dynamics_fn(states_.states, states_.constants)
print("Dynamics output shape with batching:", dynamics_output.shape)

###############################################################################
# Performance of batching (optional)
# -------------------------
# Check if jax finds a GPU

print("Available devices:", jax.devices())
start_time = time.time()
dynamics_output = dynamics_fn(states_.states, states_.constants)
end_time = time.time()
print(f"Computed dynamics for batch of size {batch_size} in {end_time - start_time:.4f} seconds")

start_time = time.time()
for i in range(batch_size):
    dynamics_output_single = model.run["confun"](states_.states[i], states_.constants)
end_time = time.time()
print(f"Computed dynamics for batch of size {batch_size} without batching in {end_time - start_time:.4f} seconds")
