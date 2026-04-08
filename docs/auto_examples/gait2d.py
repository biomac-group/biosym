"""
Predictive Gait Simulation in 2D
=============================

This is a recreation of "gait2d", or 2D gait simulations in general
"""

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
# Example YAML Configuration
# ---------------------------
# 
# Here's an example of what a standing2d.yaml for defining a 2D standing optimal control problem might look like:
# Key elements: 
# collocation: (The collocation method and settings)
# settings: (Model file, number of nodes, discretization method, output file, ipopt settings),
# objectives: (Objective functions to minimize, e.g., effort or tracking terms),
# constraints: (Dynamics, ground contact, actuator constraints, periodicity etc.),
# initial_guess: (Type of initial guess, e.g., random, from model, from file etc.),
# optimization bounds: (Bounds on states, controls, durations etc. (usually a property of the model).

standing_yaml_config = """
collocation:
  name: script2d_torque_driven
  description: Setup file for 2D gait model with torque-driven actuators
  settings:
    model: tests/models/gait2d_torque/gait2d_torque.yaml
    #model: tests/models/pendulum.xml
    nnodes: 1
    discretization:
      type: euler
      mode: backward
      weight: 10
    output:
      file: "~/.biosym/standing2d.pkl"
    tol: 5e-4

  objectives:
    - name: torque_term
      weight: 1
      args:
        exponent: 11 # Somehow this works best

  constraints:
    - name: dynamics
      weight: "1/BW"
    - name: ground_contact
      weight: "1/BW"
    - name: actuators
      weight: "1/BW"

  #initial_guess:
  #  type: random
  #  seed: 42
  bounds:
    from_model: true
    start_at_origin: true
"""

###############################################################################
# Run optimization problem
# ------------------------
# 
# Now that we have our model and configuration set up, we can use the yaml file 
# to run an optimization problem to find a standing posture.
# Usually, you would see IPOPT's log as well, but it gets redirected when building the documentation.

# sphinx_gallery_start_ignore
os.chdir(current_dir)
# sphinx_gallery_end_ignore
from biosym.ocp import collocation

ocp = collocation.Collocation(current_dir+"/examples/standing2d.yaml")
start_ = time.time()
solution = ocp.solve()
end_ = time.time()
print(f"Optimization completed in {end_ - start_:.2f} seconds")

###############################################################################
# 2D Walking Configuration example
# ------------------------------
# 
# Similarly, you can set up a YAML configuration for a 2D walking gait analysis.
# This would involve defining appropriate objectives, constraints, and settings
# for the walking task.

walking_yaml_config = """
collocation:
  name: script2d_torque_driven
  description: Setup file for 2D gait model with torque-driven actuators
  settings:
    model: tests/models/gait2d_torque/gait2d_torque.yaml
    nnodes: 50
    discretization:
      type: euler
      args:
        mode: backward # default: backward
        vars: q # FK or q, default: q
        weight: 10 # Standard weight for the discretization
        adaptive_h: false
    output:
      file: "~/.biosym/walking2d.pkl"
    tol: 1e-5
    acceptable_dual_inf_tol: 1e-2
    constr_viol_tol: 1e-4
    max_iter: 10000

  objectives:
    - name: torque_term
      weight: 1e-2
      args:
        exponent: 3
    - name: jerk_term
      weight: 1e-2

  constraints:
    - name: dynamics
      weight: "1/BW" <-- Constraints are scaled so that they are in the same magnitude
    - name: ground_contact
      weight: "1/BW" <-- 1/BW means that the constraint violation is scaled by body weight
    - name: actuators
      weight: "1/BW"
    - name: periodicity
      weight: 1
      args:
        symmetry: true
        exclude: [0] # Forward movement: exclude the first dimension --> pelvis_tx in 2D model

  initial_guess: # <-- We use the previous standing solution as initial guess
    type: from_file
    file: "~/.biosym/standing2d.pkl"

  bounds:
    from_model: true
    start_at_origin: true # <-- When not tracking data, starting at origin is reasonable
    dur: [0.4,0.7] # <-- Write upper / lower bounds in brackets
    speed: 1.3 # <-- If it is a set value, write it directly
"""
  
###############################################################################
# Solve standing problem
# ------------------------
# 
# You can use the above YAML configuration to set up and solve a walking gait
# optimization problem similarly to the standing example.

# sphinx_gallery_start_ignore
os.chdir(current_dir)
# sphinx_gallery_end_ignore
ocp_walking = collocation.Collocation(current_dir+"/examples/walking2d.yaml")
# The solve method returns x (the optimal solution) and info (ipopt information)
start = time.time()
x, info = ocp_walking.solve()
end = time.time()
print(f"Optimization completed in {end - start:.2f} seconds")
