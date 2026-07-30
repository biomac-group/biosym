
"""
A container that presents several BaseContact instances through the single-
object interface model.py expects of `gc_model`. Built by contact_parser when
a model has more than one contact force law (e.g. some geometries governed by
Hunt-Crossley, others by a spring-damper). Deliberately not a BaseContact
subclass: it has no pairs/geometries of its own, it just fans calls out to its
members and merges their per-body outputs.
"""

import jax.numpy as jnp
import numpy as np


class MultiContact:
    """Presents a list of BaseContact instances as one contact model."""

    def __init__(self, models):
        if not models:
            raise ValueError("MultiContact requires at least one contact model.")
        self.models = list(models)

        # Unified body ordering: union of all members' bodies, first-seen.
        self.bodies = []
        for m in self.models:
            for b in m.get_bodies():
                if b not in self.bodies:
                    self.bodies.append(b)

        # For each member, where each of its bodies lands in the unified order.
        # Static integer arrays -> safe inside the jitted forward.
        self._scatter = [
            np.array([self.bodies.index(b) for b in m.get_bodies()])
            for m in self.models
        ]

    # --- structural queries (unions / sums over members) ---
    def get_bodies(self):
        return list(self.bodies)

    def get_n_bodies(self):
        return len(self.bodies)

    def get_n_constraints(self, *args, **kwargs):
        return sum(m.get_n_constraints() for m in self.models)

    def get_n_states(self):
        return sum(m.get_n_states() for m in self.models)

    def get_n_constants(self):
        return sum(m.get_n_constants() for m in self.models)

    def get_states(self):
        names = []
        for m in self.models:
            names.extend(m.get_states())
        return names

    def get_constants(self):
        names = []
        for m in self.models:
            names.extend(m.get_constants())
        return names

    # --- lifecycle: fan out to members ---
    def process_eom(self, model, **kwargs):
        for m in self.models:
            m.process_eom(model, **kwargs)

    def reset(self):
        for m in self.models:
            m.reset()

    def forward(self, states, constants, model):
        """Sum each member's per-body forces/moments into the unified body
        ordering. NOTE: members are currently passed the full states/constants
        unchanged. That is correct as long as every member has 0 gc_model
        states/constants (the case for the penalty laws here, which hardcode
        their parameters). If a member ever exposes gc_model states/constants,
        this must slice states.gc_model / constants.gc_model into per-member
        chunks first -- flagged, not yet needed."""
        n = self.get_n_bodies()
        total_forces = jnp.zeros((n, 3))
        total_moments = jnp.zeros((n, 3))
        for m, scatter in zip(self.models, self._scatter):
            f, mo = m.forward(states, constants, model)
            total_forces = total_forces.at[scatter].add(f)
            total_moments = total_moments.at[scatter].add(mo)
        return total_forces, total_moments

    def plot(self, states, model, mode, ax, **kwargs):
        return [m.plot(states, model, mode, ax, **kwargs) for m in self.models]