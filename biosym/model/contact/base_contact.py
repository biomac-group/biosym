"""
Abstract base class for contact force laws.

A "contact" (a force law such as Hunt-Crossley) takes a set of geometry
pairs and turns each pair's penetration -- reported by the geometries via
BaseGeometry.penetration() -- into a contact force. It owns the physics of
response (stiffness, damping, friction) but none of the kinematics or overlap
math, which live on the geometries.

See biosym.model.contact.base_geometry.BaseGeometry for the geometry side.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np


class BaseContact(ABC):
    """
    Abstract base class for contact force laws.

    Parameters
    ----------
    pairs : list of (BaseGeometry, BaseGeometry)
        The contact pairs this force law acts on. Each pair is
        (feature, other): `feature` is the moving contact geometry whose body
        receives the force (a sphere/point on a limb); `other` is what it
        contacts (a ground half-space, or another sphere for self-collision).
        All pairs share this instance's force parameters -- pairs needing
        different parameters belong in separate BaseContact instances, which
        is the contact_parser's call.

    Attributes
    ----------
    pairs : list of (BaseGeometry, BaseGeometry)
        As passed in.
    geometries : list of BaseGeometry
        Unique geometries across all pairs, deduplicated by identity in
        first-seen order. build_kinematics is called once per entry, so a
        ground half-space shared by twenty spheres is built only once.
    bodies : list of str
        Unique real parent bodies that receive forces, first-seen order,
        excluding the "ground_frame" (the ground has no body origin /
        no force slot). This ordering is the contract: get_bodies() reports it
        and forward() must return per-body forces/moments in the same order.

    Notes
    -----
    The structural bookkeeping here (which bodies are involved, building
    geometry kinematics, summing forces onto bodies) is shared and concrete.
    Subclasses implement only the force-law-specific methods.

    See Also
    --------
    biosym.model.contact.contact_models.contact_springdamper.SpringDamper
    biosym.model.contact.contact_models.contact_huntcrossley.HuntCrossley
    """

    def __init__(self, pairs, **kwargs):
        if not pairs:
            raise ValueError(f"{type(self).__name__} requires at least one contact pair.")
        self.pairs = [tuple(p) for p in pairs]

        seen = set()
        self.geometries = []
        for feature, other in self.pairs:
            for g in (feature, other):
                if id(g) not in seen:
                    seen.add(id(g))
                    self.geometries.append(g)

        self.bodies = []
        for g in self.geometries:
            if g.parent_body != "ground_frame" and g.parent_body not in self.bodies:
                self.bodies.append(g.parent_body)

    # ------------------------------------------------------------------
    # Structural queries -- concrete, derivable from the pairs.
    # ------------------------------------------------------------------
    def get_bodies(self):
        """
        Bodies that receive contact forces, in slot order. Called by model.py
        to size the external-force slots, so its ordering must
        match forward()'s output ordering.
        """
        return list(self.bodies)

    def get_n_bodies(self):
        """Number of bodies that receive contact forces."""
        return len(self.bodies)

    def get_n_constraints(self, *args, **kwargs):
        """
        Number of explicit NLP constraints this force law adds. Defaults to 0:
        smooth penalty laws (Hunt-Crossley, spring-damper) impose none.
        """
        return 0

    def _body_index(self, body_name):
        """Slot index of `body_name` within self.bodies, for building the
        body_indices arrays _aggregate_to_bodies consumes."""
        return self.bodies.index(body_name)

    def _build_geometries(self, model):
        """
        Build every unique geometry's symbolic kinematics exactly once.
        Concrete process_eom calls this first, before reading any pos_expr /
        normal_expr or calling penetration() (which needs the other geometry's
        normal_expr already populated). Runs at process_eom time -- the
        lambdify cost is paid once here, not inside the jitted forward.
        """
        for geometry in self.geometries:
            geometry.build_kinematics(model)

    def _aggregate_to_bodies(self, contributions, body_indices):
        """
        Sum per-contribution 3-vectors (forces or moments) onto the bodies,
        given which body each contribution acts on. Pure JAX (bincount under
        vmap) and differentiable in `contributions`, so it is safe inside the
        jitted/differentiated forward.

        Parameters
        ----------
        contributions : jax.Array, shape (n, 3)
        body_indices : array-like of int, shape (n,)
            Slot index (into self.bodies) for each contribution. Passed in,
            not stored, so a subclass can list a force and its equal-and-
            opposite reaction as two contributions aimed at two different
            bodies -- needed for sphere-sphere, where both bodies move.

        Returns
        -------
        jax.Array, shape (n_bodies, 3), in self.bodies order.
        """
        n_bodies = self.get_n_bodies()
        mapping = jnp.tile(jnp.asarray(body_indices), (3, 1))

        def _bincount(idx, weights):
            return jnp.bincount(idx, weights=weights, length=n_bodies)

        return jax.vmap(_bincount)(mapping, contributions.T).T

    # ------------------------------------------------------------------
    # Force-law-specific interface -- abstract.
    # ------------------------------------------------------------------
    @abstractmethod
    def get_n_states(self):
        """
        Number of extra states this force law adds to its OWN sub-vector
        (states.gc_model), separate from the main model state vector.
        _register_contact_model uses this to size default_inputs via
        np.zeros(get_n_states()). 0 for a memoryless penalty law.
        """

    @abstractmethod
    def get_n_constants(self):
        """
        Number of extra constants in the OWN sub-vector (constants.gc_model) --
        e.g. stiffness/damping/friction, if exposed as optimization constants
        rather than hardcoded into the symbolic force expression. 0 if
        hardcoded (the simplest first implementation).
        """

    @abstractmethod
    def get_states(self):
        """Names of the extra gc_model states (length == get_n_states())."""

    @abstractmethod
    def get_constants(self):
        """Names of the extra gc_model constants (length == get_n_constants())."""

    @abstractmethod
    def process_eom(self, model, **kwargs):
        """
        Build this force law's symbolic contribution. Called by model.py as
        gc_model.process_eom(self, body_weight=...), after the sympy model
        exists.

        Expected algorithm:
          1. self._build_geometries(model)               -- kinematics, once each
          2. for each (feature, other) in self.pairs:
                 pen = feature.penetration(other)         -- the standard dict
                 turn pen["depth"], pen["normal_velocity"],
                 pen["tangential_velocity"], pen["normal"] into a 3-vector
                 force on `feature` (and its reaction on `other`, where that
                 body moves)
          3. stack the per-pair force expressions and lambdify once (over
             model._symbols, plus the gc_model constant symbols if any are exposed),
             for fast evaluation in forward(). Geometry kinematics are
             functions of the main model vector; force parameters, if exposed,
             come from the gc_model constants.

        **kwargs carries options; body_weight (sum of body masses) is passed
        for parameter scaling, as in the old ContactPoints.
        """

    @abstractmethod
    def forward(self, states, constants, model):
        """
        Evaluate contact forces for the current state, aggregated per body.

        IMPORTANT: model.py registers this as
            jax.jit(jax.jacobian(partial(forward, model=self)))
        so forward is differentiated w.r.t. `states` and JIT-compiled. It must
        be pure JAX: build from jnp / lambdified-JAX expressions only, no
        Python branching on traced values, no in-place numpy on traced arrays.
        Static, structure-derived integer index arrays are fine.

        Reads: geometry kinematics via the lambdified expressions over the main
        model vector (s_flat / c_flat); any exposed force
        parameters via the contact sub-vector (constants.gc_model).

        Expected algorithm: evaluate the lambdified per-pair forces; form
        moment arms (geometry position minus its body origin, from
        model.run["FK"]) and cross with the force; sum to bodies via
        self._aggregate_to_bodies, dropping ground contributions.

        Returns
        -------
        (jax.Array, jax.Array)
            Forces and moments per body, global frame, in get_bodies() order.
            These fill the model's ext_forces / ext_torques slots -- the blank
            symbolic loads _create_eom baked into the EOM. The tuple is
            differentiated as a pytree by jax.jacobian.
        """

    @abstractmethod
    def reset(self):
        """Reset any internal/hidden state (no-op for memoryless laws)."""

    @abstractmethod
    def plot(self, states, model, mode, ax, **kwargs):
        """Visualize the contact geometries and resulting forces."""
