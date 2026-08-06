"""
A Custom Ground-Contact Model: Sliding Contact Point
=====================================================

This tutorial shows how to build a custom ground-contact (GC) model from
scratch and plug it into biosym, instead of using the built-in
``huntcrossley``/``springdamper`` force laws and contact geometry.

The model (from SPINNpose, ACM TIST 2026): instead of two fixed contact
points per foot (heel and toe), use a *single* contact point per foot that
slides continuously between the heel and toe reference points as a smooth
(tanh) function of the foot's own orientation. Physically, this behaves like
the heel point while the foot is flat/dorsiflexed and smoothly hands off to
the toe point as the foot rotates into push-off, without a hard switch.

Only the contact *force* is a decision-variable state here; the contact
point's position and velocity are explicitly calculated from the model states, 
and a constraint ties the force state to the Hunt-Crossley force law evaluated 
at that algebraic position/velocity. The force could also be computed directly
from the Hunt-Crossley law, but this showcases how stateful contact forces can 
be implemented in a custom GC model. Additionally, having stateful GC models 
might improve jacobian conditioning. 
"""
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from biosym.utils.paths import find_repo_root

# sphinx_gallery_start_ignore
# biosym is importable regardless of CWD, but example data/model paths below
# are given relative to the repo root, so chdir there for reproducibility.
current_dir = find_repo_root()
os.chdir(current_dir)
# sphinx_gallery_end_ignore

###############################################################################
# Writing the model
# ------------------
#
# A GC model is any subclass of ``BaseContact`` (see
# ``biosym/model/contact/base_contact.py`` for the full interface docs). We
# don't use the force+geometry pipeline the built-in force laws share (it
# assumes purely kinematic contact geometry); instead we parse a 
# small XML block and build all the kinematics with sympy ourselves.

from sympy import Matrix, atan2, lambdify, sqrt, tanh

from biosym.model.contact.base_contact import BaseContact
from biosym.model.contact import contact_parser


class SlidingHuntCrossley(BaseContact):
    """
    Hunt-Crossley ground contact with one sliding contact point per foot.

    States (decision variables): ``fx``, ``fy`` per foot -- the contact
    force. Contact-point position and velocity are algebraic functions of
    q/qdot. For each foot CP positions are computed twice, once as if the
    contact point were rigidly at the heel and once as if it were rigidly at
    the toe (each via the standard rigid-body point-velocity formula,
    v = v_origin + omega x r), then blended with a tanh(steepness * phi)
    weight, where phi is the foot's own orientation relative to the ground.
    Two constraints per foot tie fx/fy to the Hunt-Crossley normal/
    friction law evaluated at that algebraic position/velocity, similar to
    ``biosym/model/contact/contact_models/contact_huntcrossley.py``.
    """

    def __init__(self, xml_root, body_weight=1.0):
        # The __init__'s job is mainly to parse the xml:
        # e.g. get positions and material properties for the contact points
        self.body_weight = body_weight
        
        points = {}
        for el in xml_root.findall("slide_point"):
            name = el.get("name")
            points[name] = {
                "body": el.get("body"),
                "heel": np.array([float(x) for x in el.get("heel").split()]),
                "toe": np.array([float(x) for x in el.get("toe").split()]),
                "steepness": float(el.get("steepness", 7.0)),
            }
        self.points = points
        self.point_names = list(points.keys())

        # We need to store the bodies that the GC model acts on to plug it correctly into a biosym model
        self._bodies = []
        for p in points.values():
            if p["body"] not in self._bodies:
                self._bodies.append(p["body"])

        self.stiffness = float(xml_root.get("stiffness"))
        self.dissipation = float(xml_root.get("dissipation"))
        self.static_friction = float(xml_root.get("static_friction"))
        self.dynamic_friction = float(xml_root.get("dynamic_friction"))
        self.viscous_friction = float(xml_root.get("viscous_friction"))
        self.transition_velocity = float(xml_root.get("transition_velocity", 0.01))
        self.eps_depth = 1e-4
        self.eps_diss = 5e-2
        self.grad_bias = 1e-1

        # ! State names must contain "_r"/"_l" for periodicity.get_symmetry_indices !
        # to auto-pair left/right for the periodicity constraint used in the
        # gait OCPs below.
        self.state_vector = []
        for name in self.point_names:
            self.state_vector.extend([f"{name}_fx", f"{name}_fy"])

    # ------------------------------------------------------------------
    # Structural bookkeeping (BaseContact's own helpers use these).
    # ------------------------------------------------------------------
    def get_bodies(self):
        return list(self._bodies)

    def get_n_bodies(self):
        return len(self._bodies)

    def get_n_states(self):
        # Each CP has fx, fy as states
        return 2 * len(self.point_names)

    def get_n_constants(self):
        """
            In principle, this GC model has a lot of constants:
            - CP positions & smoothness
            - Stiffness & damping, dissipation
            However, we parse them from the .xml and compile them in the model for simplicity.
            You could also add them here, and then backpropagate to these, and optimize them 
            e.g. with optimal control.
        """
        return 0

    def get_states(self):
        return list(self.state_vector)

    def get_constants(self):
        # See above.
        return []

    def get_n_constraints(self, model, settings):
        # Each state has one constraint per node: F = g(q,qdot)
        return self.get_n_states() * settings.get("nnodes")

    def get_nnz(self, model, settings):
        # Dense per-node Jacobian block, we just return jacobian towards the full state here.
        # In OCPs, the zeros will be masked & eventually removed by the jit compiler.
        # If you define the precise jacobian indices, compilation might be faster.
        nvpn = model.opt_states.size()
        return self.get_n_constraints(model, settings) * nvpn

    def reset(self):
        pass

    def process_eom(self, model, **kwargs):
        """
            We use the model's sympy definitions to build the ground contact points. 
            This hook will get called after the model is built and creates any callable 
            function (or equation of motion (eqm), so you could give this mass properties 
            as well. In this case, we only build the forward kinematics and their derivatives.
        """
        self.body_weight = kwargs.get("body_weight", self.body_weight)
        bw_n = float(self.body_weight)
        if bw_n < 200.0:  # looks like a mass in kg, not a weight in Newtons
            bw_n *= 9.81
        self.body_weight = bw_n

        body_keys = list(model.rigid_bodies.keys())
        targets = []  # flat: [f_n, f_t, x_nom, y_nom] per foot, in point order
        body_slots, fk_idx = [], []

        for name in self.point_names:
            p = self.points[name]
            body = p["body"]
            ref_frame = model.reference_frames[body]
            origin = model.body_origins[body]

            # Express the CP position (and speed) in global coordinates
            parent_pos = origin.pos_from(model.origin)
            p_x = parent_pos.dot(model.ground_frame.x)
            p_y = parent_pos.dot(model.ground_frame.y)

            parent_vel = origin.vel(model.ground_frame)
            dp_x = parent_vel.dot(model.ground_frame.x)
            dp_y = parent_vel.dot(model.ground_frame.y)

            omega_z = ref_frame.ang_vel_in(model.ground_frame).dot(model.ground_frame.z)

            # 2D rotation from body frame to ground frame.
            R11 = ref_frame.x.dot(model.ground_frame.x)
            R12 = ref_frame.y.dot(model.ground_frame.x)
            R21 = ref_frame.x.dot(model.ground_frame.y)
            R22 = ref_frame.y.dot(model.ground_frame.y)

            def _rigid_point_kinematics(local_x, local_y):
                """World position/velocity of a point rigidly fixed at
                (local_x, local_y) in the body frame -- standard
                v = v_origin + omega x r formula."""
                wx = p_x + R11 * local_x + R12 * local_y
                wy = p_y + R21 * local_x + R22 * local_y
                wxdot = dp_x + omega_z * (R12 * local_x - R11 * local_y)
                wydot = dp_y + omega_z * (R22 * local_x - R21 * local_y)
                return wx, wy, wxdot, wydot

            heel_x, heel_y, heel_xdot, heel_ydot = _rigid_point_kinematics(*p["heel"][:2])
            toe_x, toe_y, toe_xdot, toe_ydot = _rigid_point_kinematics(*p["toe"][:2])

            # Foot orientation vs. ground, and the tanh blend weight (as in
            # SPINNpose): position and velocity are blended with the *same*
            # weight, i.e. the weight is treated as locally frozen for the
            # velocity blend (no product-rule term for its own time
            # derivative) -- the same simplification the reference
            # implementation makes.
            phi = atan2(R21, R11)
            w = (tanh(p["steepness"] * phi) + 1) / 2

            x_nom = w * heel_x + (1 - w) * toe_x
            y_nom = w * heel_y + (1 - w) * toe_y
            xdot_nom = w * heel_xdot + (1 - w) * toe_xdot
            ydot_nom = w * heel_ydot + (1 - w) * toe_ydot

            # Hunt-Crossley normal + friction law (same formulas as
            # contact_huntcrossley.py::HuntCrossley._pair_force), evaluated
            # at the algebraic sliding-point depth/velocity instead of a
            # BaseGeometry.penetration() result.
            depth = -y_nom
            depth_rate = -ydot_nom
            vt = xdot_nom

            x_pos = 0.5 * (sqrt(depth ** 2 + self.eps_depth ** 2) + depth)
            diss = 1 + 1.5 * self.dissipation * depth_rate
            diss_pos = 0.5 * (sqrt(diss ** 2 + self.eps_diss ** 2) + diss)
            f_n = self.stiffness * x_pos ** 1.5 * diss_pos + self.grad_bias * depth

            v_s_sq = vt ** 2
            blend = self.dynamic_friction + 2 * (self.static_friction - self.dynamic_friction) / (
                1 + v_s_sq / self.transition_velocity ** 2
            )
            coulomb = blend * vt / sqrt(v_s_sq + self.transition_velocity ** 2)
            viscous = self.viscous_friction * vt
            f_t = -f_n * (coulomb + viscous)

            targets.extend([f_n, f_t, x_nom, y_nom])
            body_slots.append(self._bodies.index(body))
            fk_idx.append(body_keys.index(body))

        targets = model._replace_dyn(Matrix(targets))
        self._target_fn = lambdify(model._symbols, targets, modules="jax", cse=True)
        self._body_slots = np.array(body_slots)
        self._fk_idx = np.array(fk_idx)


    def _make_residual_fn(self, model):
        """Build the per-node residual closure. `model` is only used for
        structural/static bookkeeping (_materialize's shapes, the already-
        built self._target_fn) -- captured via closure rather than passed as
        a jax.jit argument, since it isn't itself an array."""
        # _materialize is a compabitily layer between states and model.symbols (which was used above)
        from biosym.constraints.dynamics import _materialize

        n_points = len(self.point_names)
        body_weight = self.body_weight

        def _residual(s, c):
            s = _materialize(s, model)
            s_flat = s.filter("model").flatten()
            c_flat = c.filter("model").flatten()
            targets = jnp.asarray(self._target_fn(*s_flat, *c_flat)).flatten()

            res = []
            # for-looks are slow to compile in jax, so this is more for readability
            for i in range(n_points):
                f_n_target, f_t_target = targets[4 * i], targets[4 * i + 1]
                # fx, fy states are normalized by body weight (fractions of
                # BW, O(1) in magnitude), same convention as Gait2dcContact --
                # keeps both the state values and this residual well-
                # conditioned instead of O(body-weight-in-newtons).
                fx_bw = s.gc_model[2 * i + 0]
                fy_bw = s.gc_model[2 * i + 1]
                res.append(fy_bw - f_n_target / body_weight)
                res.append(fx_bw - f_t_target / body_weight)
            return jnp.stack(res)

        return _residual

    def constraints(self, states, constants, model, settings):
        # Function to define 0 = F - g(q, qdot)
        # e.g. forces in the states should be the same as Hunt-Crossley calculations
        states_dict, globals_dict = states
        nnodes = settings.get("nnodes")

        residual_fn = self._make_residual_fn(model)
        c_fun = jax.jit(jax.vmap(residual_fn, in_axes=(0, None)))
        res = c_fun(states_dict[:nnodes], constants)
        return res.reshape(-1)

    def jacobian(self, states, constants, model, settings):
        states_dict, globals_dict = states
        nnodes = settings.get("nnodes")
        n_points = len(self.point_names)
        ncons = 2 * n_points
        nvpn = states_dict.get_n_states()

        # Building the jacobian - the easy part, we can just call jax.jacobian on the gc model.
        residual_fn = self._make_residual_fn(model)
        jac_fun = jax.jit(jax.vmap(jax.jacobian(residual_fn, argnums=0), in_axes=(0, None)))
        jac = jac_fun(states_dict[:nnodes], constants)

        # Building the jacobian - the harder part: build the non-zero dictionary
        # rows and columns are set that the jacobian has block-diagonal structure
        node_indices = jnp.arange(nnodes)
        row_blocks = node_indices[:, None] * ncons + jnp.arange(ncons)[None, :]
        col_blocks = node_indices[:, None] * nvpn + jnp.arange(nvpn)[None, :]

        rows = jnp.repeat(row_blocks, nvpn, axis=1).flatten()
        cols = jnp.tile(col_blocks, (1, ncons)).flatten()
        data = jac.to_array().reshape(nnodes, -1).flatten()
        # flat COO format vectors are returned.
        return rows, cols, data

    # ------------------------------------------------------------------
    # forward(): fx, fy are already states -- no recomputation needed, just
    # locate them at the algebraic sliding-point position for the moment arm.
    # ------------------------------------------------------------------
    def forward(self, states, constants, model):
        from biosym.constraints.dynamics import _materialize

        states = _materialize(states, model)
        s_flat = states.filter("model").flatten()
        c_flat = constants.filter("model").flatten()
        targets = jnp.asarray(self._target_fn(*s_flat, *c_flat)).flatten()

        n_points = len(self.point_names)
        # gc_model states are BW-normalized (see _residual) -- scale back to
        # Newtons here.
        forces = jnp.stack([
            jnp.array([
                states.gc_model[2 * i + 0] * self.body_weight,
                states.gc_model[2 * i + 1] * self.body_weight,
                0.0,
            ])
            for i in range(n_points)
        ])
        positions = jnp.stack([
            jnp.array([targets[4 * i + 2], targets[4 * i + 3], 0.0])
            for i in range(n_points)
        ])

        body_positions = model.run["FK"](states, constants)[self._fk_idx]
        moment_arms = positions - body_positions
        moments = jnp.cross(moment_arms, forces)

        # From our base class - make sure that the forces and moments appear at the correct bodies
        body_forces = self._aggregate_to_bodies(forces, self._body_slots)
        body_moments = self._aggregate_to_bodies(moments, self._body_slots)
        return body_forces, body_moments

    # ------------------------------------------------------------------
    # Visualization: one marker + force arrow per foot, at the algebraic
    # sliding contact point.
    # ------------------------------------------------------------------
    def plot(self, states, model, mode, ax, **kwargs):
        from biosym.constraints.dynamics import _materialize

        factor = kwargs.get("force_scale", 1e-3)
        case = kwargs.get("case", "3D")
        non_zero_axes = kwargs.get("non_zero_axes", [0, 1])
        frame = kwargs.get("frame", 0)

        if mode == "init":
            s = states.states if hasattr(states, "states") else states
            c = states.constants if hasattr(states, "constants") else model.default_constants
            self._plot_states = s
            self._plot_constants = c

        s = self._plot_states
        c = self._plot_constants
        s_frame = s[frame] if np.asarray(s.q).ndim > 1 else s
        s_frame_full = _materialize(s_frame, model)

        s_flat = s_frame_full.filter("model").flatten()
        c_flat = c.filter("model").flatten()
        targets = np.asarray(self._target_fn(*s_flat, *c_flat)).flatten()
        n_points = len(self.point_names)
        positions = np.array([[targets[4 * i + 2], targets[4 * i + 3], 0.0] for i in range(n_points)])
        # gc_model states are BW-normalized (see _residual) -- scale back to
        # Newtons for the force-arrow display.
        forces = np.array([
            [s_frame.gc_model[2 * i + 0] * self.body_weight, s_frame.gc_model[2 * i + 1] * self.body_weight, 0.0]
            for i in range(n_points)
        ])
        body_positions = np.asarray(model.run["FK"](s_frame_full, c))[self._fk_idx]

        if mode == "init":
            self._plot_markers, self._plot_force_lines, self._plot_body_lines = [], [], []
            for i in range(n_points):
                tip = positions[i] + factor * forces[i]
                if case == "2D":
                    a0, a1 = non_zero_axes
                    (m,) = ax.plot([positions[i, a0]], [positions[i, a1]], c="k", marker="o")
                    (fl,) = ax.plot([positions[i, a0], tip[a0]], [positions[i, a1], tip[a1]], c="darkgreen")
                    (bl,) = ax.plot([body_positions[i, a0], positions[i, a0]],
                                    [body_positions[i, a1], positions[i, a1]], c="k")
                else:
                    (m,) = ax.plot([positions[i, 0]], [positions[i, 1]], [positions[i, 2]], c="k", marker="o")
                    (fl,) = ax.plot([positions[i, 0], tip[0]], [positions[i, 1], tip[1]],
                                    [positions[i, 2], tip[2]], c="darkgreen")
                    (bl,) = ax.plot([body_positions[i, 0], positions[i, 0]],
                                    [body_positions[i, 1], positions[i, 1]],
                                    [body_positions[i, 2], positions[i, 2]], c="k")
                self._plot_markers.append(m)
                self._plot_force_lines.append(fl)
                self._plot_body_lines.append(bl)

            if case == "2D":
                ax.fill_between([-5, 5], -10, 0, color="grey", alpha=0.3)
            else:
                xg, yg = np.linspace(-5, 5, 10), np.linspace(-5, 5, 10)
                Xg, Yg = np.meshgrid(xg, yg)
                ax.plot_surface(Xg, Yg, np.zeros(Xg.shape), color="grey", alpha=0.3)

            # stickfigure.py's animation loop only calls plot(..., mode="update")
            # if this init call returned a 4-tuple (it checks len(...) == 4
            # before wiring up per-frame updates) -- the 4th slot is unused
            # here (no same-body contact-point-to-contact-point lines to draw,
            # since there's just one sliding point per foot).
            return self._plot_markers, self._plot_force_lines, self._plot_body_lines, []

        if mode == "update":
            for i, (m, fl, bl) in enumerate(zip(self._plot_markers, self._plot_force_lines, self._plot_body_lines)):
                tip = positions[i] + factor * forces[i]
                if case == "2D":
                    a0, a1 = non_zero_axes
                    m.set_data([positions[i, a0]], [positions[i, a1]])
                    fl.set_data([positions[i, a0], tip[a0]], [positions[i, a1], tip[a1]])
                    bl.set_data([body_positions[i, a0], positions[i, a0]],
                                [body_positions[i, a1], positions[i, a1]])
                else:
                    m.set_data([positions[i, 0]], [positions[i, 1]])
                    m.set_3d_properties(positions[i, 2])
                    fl.set_data([positions[i, 0], tip[0]], [positions[i, 1], tip[1]])
                    fl.set_3d_properties([positions[i, 2], tip[2]])
                    bl.set_data([body_positions[i, 0], positions[i, 0]], [body_positions[i, 1], positions[i, 1]])
                    bl.set_3d_properties([body_positions[i, 2], positions[i, 2]])
            return

        raise ValueError("Invalid mode. Must be 'init' or 'update'.")


###############################################################################
# Registering the model
# ----------------------
#
# ``register_contact_model`` is the public extension point for GC models
# that (like this one) don't fit the pairs/geometry force-law pipeline
# ``huntcrossley``/``springdamper`` use. It just records a name -> constructor
# mapping; nothing is built yet. It must run *before* ``load_model`` parses a
# YAML/XML referencing this type as the GC model's state count gets baked
# into the symbolic model and JAX-compiled functions during model
# construction, so it can't be swapped in on an already-built model.

contact_parser.register_contact_model("huntcrossley_sliding", SlidingHuntCrossley)

###############################################################################
# Load the model
# --------------
#
# ``gait2d_sliding.yaml`` reuses the ordinary shared gait2d skeleton (bodies,
# joints, actuators) and only swaps in our sliding-contact ground-contact
# model, defined in ``gait2d_ground_contact_sliding.xml`` right next to this
# script. Both files in full, so this is reproducible standalone:
#
# ``examples/advanced/gait2d_sliding.yaml``:
#
# .. code-block:: yaml
#
#     model:
#       name: gait2d.xml # copy of tests/models/gait2d/gait2d.xml -- nothing custom about the bodies/joints
#       additional_parameters:
#         joints:
#           jointK2: 5000
#           jointD: 2
#         ground_contact:
#           replace_existing: true
#           file: "gait2d_ground_contact_sliding.xml" # the interesting part -- see below
#         actuators:
#           replace_existing: true
#           file: "gait2d_actuators.xml" # copy of tests/models/gait2d/gait2d_actuators.xml
#
# ``examples/advanced/gait2d_ground_contact_sliding.xml``:
#
# .. code-block:: xml
#
#     <ground_contact_model type="huntcrossley_sliding"
#       stiffness="1e6" dissipation="2.0"
#       static_friction="0.8" dynamic_friction="0.8" viscous_friction="0.5"
#       transition_velocity="0.2">
#       <slide_point name="slide_r" body="foot_r" heel="-0.06 -0.0702 0" toe="0.1636 -0.0702 0" steepness="7.0"></slide_point>
#       <slide_point name="slide_l" body="foot_l" heel="-0.06 -0.0702 0" toe="0.1636 -0.0702 0" steepness="7.0"></slide_point>
#     </ground_contact_model>

from biosym.model.model import load_model

model_file = os.path.join(current_dir, "examples", "advanced", "gait2d_sliding.yaml")
print("Loading 2D gait model with a sliding-contact-point ground contact model...")
start_time = time.time()
model = load_model(model_file, force_rebuild=True)
print(f"Model loaded in {time.time() - start_time:.3f} seconds")
print(f"Model has {model.n_states} states and {model.n_constants} constants")
print(f"Ground contact model added {model.gc_model.get_n_states()} states: {model.gc_model.get_states()}")

###############################################################################
# Standing equilibrium
# ---------------------
#
# As always, first solve for a standing equilibrium.
# This tests that the new contact model's states/constraints are internally
# consistent (force states should settle at sane values, roughly half of
# body weight per foot while standing on both feet).
#
# ``examples/advanced/standing2d_sliding.yaml``:
#
# .. code-block:: yaml
#
#     collocation:
#       name: script2d_sliding_contact
#       description: Standing equilibrium for the 2D gait model with a sliding-contact-point ground contact model
#       settings:
#         model: examples/advanced/gait2d_sliding.yaml
#         nnodes: 1
#         discretization:
#           type: euler
#           mode: backward
#           weight: 1
#         output:
#           file: "~/.biosym/standing2d_sliding.pkl"
#         tol: 5e-4
#
#       objectives:
#         - name: effort_term
#           weight: 100
#           args:
#             exponent: 5
#
#       constraints:
#         - name: dynamics
#
#       bounds:
#         from_model: true
#         start_at_origin: true

from biosym.ocp import collocation

# sphinx_gallery_start_ignore
# sphinx-gallery resets the CWD to this script's own directory before each
# code block below, so re-establish the repo root before any relative model
# paths inside the YAML configs get resolved.
os.chdir(current_dir)
# sphinx_gallery_end_ignore
ocp_standing = collocation.Collocation(
    os.path.join(current_dir, "examples", "advanced", "standing2d_sliding.yaml"),
    force_rebuild=True,
)
start_ = time.time()
solution_standing = ocp_standing.solve(visualize=False)
print(f"Standing optimization completed in {time.time() - start_:.2f} seconds")

###############################################################################
# Running at 3.5 m/s, effort-only
# ---------------------------------
#
# To test the model further, we run a predictive simulation 3.5 m/s, using
# effort term as the sole objective. Standing is used as initial guess.
#
# ``examples/advanced/running2d_sliding.yaml``:
#
# .. code-block:: yaml
#
#     collocation:
#       name: script2d_sliding_contact_running
#       description: >
#         Template-free running at 3.5 m/s (past the walk-run transition speed --
#         the optimizer is free to find a running/bouncing gait, stance and flight
#         phases are not hard-coded, they emerge from the contact forces) with the
#         sliding-contact-point ground contact model -- effort minimization only,
#         no tracking data.
#       settings:
#         model: examples/advanced/gait2d_sliding.yaml
#         nnodes: 50
#         discretization:
#           type: euler
#           args:
#             mode: backward
#             vars: q
#             weight: 1
#             adaptive_h: false
#         output:
#           file: "~/.biosym/running2d_sliding.pkl"
#         tol: 5e-4
#         acceptable_dual_inf_tol: 1e-3
#         acceptable_tol: 1e-3
#         constr_viol_tol: 1e-4
#         max_iter: 6000
#
#       objectives:
#         - name: effort_term
#           weight: 225
#           args:
#             exponent: 5
#
#       constraints:
#         - name: dynamics
#         - name: periodicity
#           args:
#             symmetry: true
#             exclude: [0]
#
#       initial_guess:
#         type: from_file
#         file: "~/.biosym/standing2d_sliding.pkl"
#
#       bounds:
#         from_model: true
#         start_at_origin: true
#         dur: [0.1, 0.4]
#         speed: 3.5

# sphinx_gallery_start_ignore
os.chdir(current_dir)
# sphinx_gallery_end_ignore
ocp_running = collocation.Collocation(
    os.path.join(current_dir, "examples", "advanced", "running2d_sliding.yaml"),
    force_rebuild=True,
)
start = time.time()
solution_running = ocp_running.solve(visualize=False)
print(f"Running optimization completed in {time.time() - start:.2f} seconds")

###############################################################################
# Animate the result
# -------------------
#
# The results is slowed down for visualization. The visualization will look 
# uncanny, as the contact model only plots the position of the CoP
# and disregards the actual foot shape.


def _upsample_states(states, factor):
    """Linearly interpolate a per-node States sequence to `factor` times as
    many frames (visualization only -- not used anywhere in the OCP itself).
    Interpolates every field batched per-node (leading dim == node count);
    a solved trajectory can also carry single-instance placeholder fields
    (e.g. qdd/tau/ext_forces/ext_torques with no leading node dim at all) --
    those are left untouched, only the truly time-varying fields are resampled."""
    n_nodes = np.asarray(states.q).shape[0]

    def _interp(field):
        arr = np.asarray(field)
        n = arr.shape[0]
        n_new = (n - 1) * factor + 1
        x_old = np.arange(n)
        x_new = np.linspace(0, n - 1, n_new)
        flat = arr.reshape(n, -1)
        out = np.empty((n_new, flat.shape[1]), dtype=flat.dtype)
        for j in range(flat.shape[1]):
            out[:, j] = np.interp(x_new, x_old, flat[:, j])
        # States.__getitem__ only recognizes jnp.ndarray leaves (see
        # _pad_state_for_fk's comment in stickfigure.py) -- a plain numpy
        # array here silently breaks per-frame slicing downstream.
        return jnp.asarray(out.reshape((n_new,) + arr.shape[1:]))

    updates = {
        name: _interp(getattr(states, name))
        for name in states.names
        if np.asarray(getattr(states, name)).shape[0] == n_nodes
    }
    return states.replace(**updates)


from biosym.visualization import stickfigure

upsampled_states = _upsample_states(solution_running.states, factor=3)
running_animation = stickfigure.plot_stick_figure(
    ocp_running.model, (upsampled_states, solution_running.globals), notebook=False,
    playback_speed=0.1,  # 10x slow motion, ~30 fps after upsampling
)
