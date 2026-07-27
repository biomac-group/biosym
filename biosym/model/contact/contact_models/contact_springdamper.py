
import jax.numpy as jnp
import numpy as np
from sympy import Matrix, lambdify, sqrt

from biosym.model.contact.base_contact import BaseContact


class SpringDamper(BaseContact):
    """
    Linear (Kelvin-Voigt) spring-damper contact with smoothed Coulomb
    friction -- the law the legacy yaml-based ContactPoints used.

    Normal force (linear in depth):
        f_n = k_eff * x_+ * (1 + b * xdot) + grad_bias * x
    Friction (single-coefficient smoothed Coulomb):
        F_f = -mu * f_n * vt / sqrt(|vt|^2 + v_reg^2)
    The grad_bias * x term is a small linear-in-raw-depth bias (legacy
    "point towards ground" term) that keeps a nonzero force gradient even
    where the smoothed x_+ has flattened out away from contact.

    Stiffness is body-weight scaled (k_eff = k * body_weight * 9.81) at
    process_eom time, matching the legacy code -- the yaml k values are
    dimensionless multipliers, not physical stiffnesses. (Unlike HuntCrossley,
    whose OpenSim stiffness is already physical and is used as-is.)

    Parameters are hardcoded into the symbolic expression, so this law adds no
    states and no constants to the optimization.
    """

    def __init__(self, pairs, k, b, mu, eps_depth=1e-3, v_reg=1e-2, grad_bias=1e-1, **kwargs):
        super().__init__(pairs, **kwargs)
        self.k = float(k)
        self.b = float(b)
        self.mu = float(mu)
        self.eps_depth = float(eps_depth)
        self.v_reg = float(v_reg)
        self.grad_bias = float(grad_bias)
        self.k_eff = self.k        # set for real in process_eom (body-weight scaled)

    # ------------------------------------------------------------------
    # States / constants: all parameters hardcoded.
    # ------------------------------------------------------------------
    def get_n_states(self):
        return 0

    def get_n_constants(self):
        return 0

    def get_states(self):
        return []

    def get_constants(self):
        return []

    # ------------------------------------------------------------------
    # The spring-damper per-pair force formula.
    # ------------------------------------------------------------------
    def _pair_force(self, pen):
        """Symbolic (3, 1) contact force on the feature, from one penetration
        dict. The reaction on `other` is added by process_eom."""
        x = pen["depth"]
        xdot = pen["normal_velocity"]
        normal = pen["normal"]
        vt = pen["tangential_velocity"]

        # Smooth positive depth: ~x when penetrating, ~0 when clear.
        x_pos = 0.5 * (sqrt(x ** 2 + self.eps_depth ** 2) + x)

        # Linear elastic term with linear (velocity-proportional) damping,
        # plus a small bias linear in raw (unsmoothed) depth so a nonzero
        # gradient toward contact survives even where x_pos has flattened
        # out to ~0 (legacy "point towards ground" term).
        f_n = self.k_eff * x_pos * (1 + self.b * xdot) + self.grad_bias * x
        force_normal = f_n * normal

        # Smoothed single-coefficient Coulomb friction.
        v_s_sq = vt.dot(vt)
        force_friction = -self.mu * f_n * vt / sqrt(v_s_sq + self.v_reg ** 2)

        return force_normal + force_friction

    # ------------------------------------------------------------------
    # Symbolic assembly.
    # ------------------------------------------------------------------
    def process_eom(self, model, **kwargs):
        # Body-weight scaling, as in the legacy ContactPoints. Done here (not
        # in __init__) because body_weight is only known at process_eom time.
        # Recompute from self.k each call so re-running can't double-scale.
        body_weight = kwargs.get("body_weight", 1.0)
        self.k_eff = self.k * body_weight * 9.81

        self._build_geometries(model)

        body_keys = list(model.rigid_bodies.keys())
        force_exprs, pos_exprs = [], []
        contrib_body_slot, contrib_fk_idx = [], []

        def _add_contribution(force_expr, geometry):
            force_exprs.append(force_expr)
            pos_exprs.append(geometry.pos_expr)
            contrib_body_slot.append(self._body_index(geometry.parent_body))
            contrib_fk_idx.append(body_keys.index(geometry.parent_body))

        for feature, other in self.pairs:
            pen = feature.penetration(other)
            force_on_feature = self._pair_force(pen)

            _add_contribution(force_on_feature, feature)
            if other.parent_body != "ground_frame":
                _add_contribution(-force_on_feature, other)

        force_matrix = Matrix.vstack(*[f.T for f in force_exprs])
        pos_matrix = Matrix.vstack(*[p.T for p in pos_exprs])
        self.force_fn = lambdify(model._symbols, force_matrix, modules="jax", cse=True, docstring_limit=2)
        self.pos_fn = lambdify(model._symbols, pos_matrix, modules="jax", cse=True, docstring_limit=2)

        self._contrib_body_slot = np.array(contrib_body_slot)
        self._contrib_fk_idx = np.array(contrib_fk_idx)

    # ------------------------------------------------------------------
    # Numeric evaluation (pure JAX -- differentiated & jitted upstream).
    # ------------------------------------------------------------------
    def forward(self, states, constants, model):
        s_flat = states.filter('model').flatten()
        c_flat = constants.filter('model').flatten()
        forces = self.force_fn(*s_flat, *c_flat)
        positions = self.pos_fn(*s_flat, *c_flat)

        body_positions = model.run["FK"](states, constants)[self._contrib_fk_idx]
        moment_arms = positions - body_positions
        moments = jnp.cross(moment_arms, forces)

        body_forces = self._aggregate_to_bodies(forces, self._contrib_body_slot)
        body_moments = self._aggregate_to_bodies(moments, self._contrib_body_slot)
        return body_forces, body_moments

    def reset(self):
        pass

    # ------------------------------------------------------------------
    # Basic visualization: each contribution's point + force arrow.
    # ------------------------------------------------------------------
    def plot(self, states, model, mode, ax, **kwargs):
        factor = kwargs.get("force_scale", 1e-3)
        case = kwargs.get("case", "3D")
        non_zero_axes = kwargs.get("non_zero_axes", [0, 1])
        frame = kwargs.get("frame", 0)

        if mode == "init":
            # Store the full sequence + constants so update can slice by frame
            # (update is called with states=False).
            s = states.states if hasattr(states, "states") else states
            c = states.constants if hasattr(states, "constants") else model.default_constants
            self._plot_states = s
            self._plot_constants = c

        # Pick one frame from the stored sequence.
        s = self._plot_states
        c = self._plot_constants
        s_frame = s[frame] if np.asarray(s.q).ndim > 1 else s

        s_flat = s_frame.filter('model').flatten()
        c_flat = c.filter('model').flatten()
        positions = np.asarray(self.pos_fn(*s_flat, *c_flat))
        forces = np.asarray(self.force_fn(*s_flat, *c_flat))

        if mode == "init":
            self._plot_markers, self._plot_force_lines = [], []
            for i in range(positions.shape[0]):
                tip = positions[i] + factor * forces[i]
                if case == "2D":
                    a0, a1 = non_zero_axes
                    (m,) = ax.plot([positions[i, a0]], [positions[i, a1]], c="k", marker="o")
                    (fl,) = ax.plot([positions[i, a0], tip[a0]],
                                    [positions[i, a1], tip[a1]], c="darkgreen")
                else:
                    (m,) = ax.plot([positions[i, 0]], [positions[i, 1]], [positions[i, 2]],
                                   c="k", marker="o")
                    (fl,) = ax.plot([positions[i, 0], tip[0]],
                                    [positions[i, 1], tip[1]],
                                    [positions[i, 2], tip[2]], c="darkgreen")
                self._plot_markers.append(m)
                self._plot_force_lines.append(fl)
            return self._plot_markers, self._plot_force_lines

        if mode == "update":
            for i, (m, fl) in enumerate(zip(self._plot_markers, self._plot_force_lines)):
                tip = positions[i] + factor * forces[i]
                if case == "2D":
                    a0, a1 = non_zero_axes
                    m.set_data([positions[i, a0]], [positions[i, a1]])
                    fl.set_data([positions[i, a0], tip[a0]], [positions[i, a1], tip[a1]])
                else:
                    m.set_data([positions[i, 0]], [positions[i, 1]])
                    m.set_3d_properties(positions[i, 2])
                    fl.set_data([positions[i, 0], tip[0]], [positions[i, 1], tip[1]])
                    fl.set_3d_properties([positions[i, 2], tip[2]])
            return

        raise ValueError("Invalid mode. Must be 'init' or 'update'.")