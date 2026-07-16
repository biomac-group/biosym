"""
Presents several BaseActuator instances as one self.actuators object, so a model
with several actuator types (coordinate + torque + muscle) still exposes the
single interface model.py expects. Not a BaseActuator subclass -- it just fans
calls out to its members and combines their results.

Under the joint-torque model every member's forward() returns the
same thing: a full joint-sized (n_dof) torque array, with that member's torques
at the joints it drives and zero elsewhere. So combining members is simply
summing those arrays elementwise -- no load-type routing, no special cases.
"""

import jax.numpy as jnp


class MultiActuator:
    def __init__(self, models):
        if not models:
            raise ValueError("MultiActuator requires at least one actuator model.")
        self.models = list(models)

    # --- counts (sums over members) ---
    def get_n_actuators(self):
        return sum(m.get_n_actuators() for m in self.models)

    def get_n_states(self):
        return sum(m.get_n_states() for m in self.models)

    def get_n_constants(self):
        return sum(m.get_n_constants() for m in self.models)

    def get_n_constraints(self, *args, **kwargs):
        return sum(m.get_n_constraints(*args, **kwargs) for m in self.models)

    def get_nnz(self, *args, **kwargs):
        return sum(m.get_nnz(*args, **kwargs) for m in self.models)

    # --- joints driven: union across members ---
    def get_actuated_joints(self):
        joints = []
        for m in self.models:
            joints.extend(m.get_actuated_joints())
        return joints

    def get_actuators(self):
        merged = {}
        for m in self.models:
            if hasattr(m, "get_actuators"):
                merged.update(m.get_actuators())
        return merged

    # --- lifecycle: fan out to members ---
    def reset(self):
        for m in self.models:
            m.reset()

    def process_eom(self, model):
        for m in self.models:
            m.process_eom(model)

    # --- forward: sum members' joint-torque arrays ---
    def forward(self, states, constants, model):
        """Sum every member's joint-sized torque array. Each member returns an
        (n_dof,) or (n_samples, n_dof) array with its torques at its joints and
        zeros elsewhere, so a plain elementwise sum gives the combined per-joint
        torque.

        NOTE: every member receives the full states/constants. Each member reads
        its own actuator states from states.actuator_model. This is correct as
        long as members read only their own slice; if several members ever share
        the actuator_model vector, this must slice it per member first -- flagged,
        not yet needed (the current models don't mix multiple state-bearing
        actuators)."""
        total = None
        for m in self.models:
            f = m.forward(states, constants, model)
            total = f if total is None else total + f
        return total