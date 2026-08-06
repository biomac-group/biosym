"""
Presents several BaseActuator instances as one self.actuators object, so a model
with several actuator types (coordinate + torque + muscle) still exposes the
single interface model.py expects. Not a BaseActuator subclass -- it just fans
calls out to its members and combines their results.

Under the joint-torque model every member's forward() returns the
same thing: a full joint-sized (n_dof) torque array, with that member's torques
at the joints it drives and zero elsewhere. So combining members is simply
summing those arrays elementwise -- no load-type routing, no special cases.

Members' own actuator-state/constant vectors (`states.actuator_model`,
`constants.actuator_model`) are concatenated in `self.models` order (the only
order anything -- `get_n_states()`, `bounds`, this class -- agrees on), so
`forward()`/`reset()` slice each member's own segment out before delegating.

`constraints()` forwards to each constraint-bearing member (e.g. Hill2d/
Millard2012's force-equilibrium + activation-dynamics constraints) with the
same per-member slicing as `forward()`, and concatenates their results.
`jacobian()` deliberately does NOT hand-derive its own sparse (rows, cols,
data) structure the way Hill2d does (per-node blocks, predecessor-node
columns, dur/h columns for periodic/backward-difference discretizations --
see hill2d.py's `jacobian()`) -- reimplementing that generically for
arbitrary member combinations would be a large, error-prone undertaking for
comparatively little payoff at this actuator-constraint block's scale.
Instead it computes a plain `jax.jacobian` of `MultiActuator`'s own
`constraints()` and declares every structurally-possible entry present (a
fully dense block per constraint row, matching the same trade dynamics.py's
own nnodes==1 dynamics block already makes: `nnz = ncons_model * nvpn_opt`).
This is deliberately not as sparse as Hill2d's own hand-derived Jacobian
would be for a single-actuator-type model, in exchange for correctness by
construction and much less code -- `get_nnz()` is overridden to match.
"""

import jax
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

    def get_nnz(self, model, settings):
        """Overrides the naive "sum of member get_nnz()" (each member's own
        *sparse* nnz count) -- jacobian() below declares a fully dense block
        instead (see module docstring), so this must match that exactly:
        cyipopt requires jacobian()'s returned data length to equal get_nnz()
        exactly, not merely bound it (same contract Hill2d's own get_nnz()
        documents)."""
        n_constraints = self.get_n_constraints(model, settings)
        if n_constraints == 0:
            return 0
        nnodes = settings.get("nnodes")
        nvpn = model.opt_states.size()
        # +1 dur column only when nnodes > 1 (no time dimension/dur variable
        # exists for a single-node problem -- must match jacobian()'s own
        # nnodes==1 branch, which never adds one either).
        return n_constraints * (nnodes * nvpn + (1 if nnodes > 1 else 0))

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

    @property
    def bounds(self):
        """Concatenation of each member's state bounds, in `self.models` order
        (matching how their actuator-state vectors are concatenated)."""
        mins = [m.bounds["states"]["min"] for m in self.models]
        maxs = [m.bounds["states"]["max"] for m in self.models]
        return {"states": {"min": jnp.concatenate(mins), "max": jnp.concatenate(maxs)}}

    # --- lifecycle: fan out to members ---
    def reset(self):
        for m in self.models:
            m.reset()

    def process_eom(self, model):
        for m in self.models:
            m.process_eom(model)

    def _member_slices(self):
        """(member, state_start, n_states, const_start, n_consts) tuples, in
        `self.models` order. States and constants are independently-sized
        vectors, so each gets its own running offset."""
        state_offset = 0
        const_offset = 0
        out = []
        for m in self.models:
            n_states = m.get_n_states()
            n_consts = m.get_n_constants()
            out.append((m, state_offset, n_states, const_offset, n_consts))
            state_offset += n_states
            const_offset += n_consts
        return out

    # --- forward: sum members' joint-torque arrays ---
    def forward(self, states, constants, model, states_prev=None, h=None):
        """Sum every member's joint-sized torque array. Each member returns an
        (n_dof,) or (n_samples, n_dof) array with its torques at its joints and
        zeros elsewhere, so a plain elementwise sum gives the combined per-joint
        torque.

        Each member is given its own slice of states.actuator_model /
        constants.actuator_model (by consecutive offset in `self.models`
        order) so members reading `states.actuator_model` directly (as
        CoordinateActuator, Hill2d, and Millard2012 all do) see only their own
        states, not the full combined vector. states_prev (if given) is
        sliced the same way, so a Hill2d member can derive its own fiber
        velocity from its own previous-node states."""
        total = None
        for m, s_start, n_states, c_start, n_consts in self._member_slices():
            member_states = states.replace(actuator_model=states.actuator_model[..., s_start : s_start + n_states])
            member_constants = constants.replace(
                actuator_model=constants.actuator_model[..., c_start : c_start + n_consts]
            )
            member_states_prev = (
                states_prev.replace(actuator_model=states_prev.actuator_model[..., s_start : s_start + n_states])
                if states_prev is not None
                else None
            )
            f = m.forward(member_states, member_constants, model, states_prev=member_states_prev, h=h)
            total = f if total is None else total + f
        return total

    # --- constraints: forward to constraint-bearing members, concatenate ---
    def constraints(self, states, constants, model, settings):
        """`states` is `(states_list, globals_dict)`, matching Hill2d's own
        `constraints()` signature and how `dynamics.py` calls
        `model.actuator_model.constraints(...)`. Members with no constraints
        of their own (e.g. CoordinateActuator, which doesn't define
        constraints()/jacobian() at all) are skipped. Each remaining member
        gets its own slice of `states_list.actuator_model`/
        `constants.actuator_model` (same `_member_slices()` offsets as
        `forward()`), and results are concatenated in `self.models` order --
        the same order `get_n_constraints()` already sums in, so row
        semantics line up automatically for callers."""
        states_list, globals_dict = states
        pieces = []
        for m, s_start, n_states, c_start, n_consts in self._member_slices():
            if m.get_n_constraints(model, settings) == 0:
                continue
            member_states_list = states_list.replace(
                actuator_model=states_list.actuator_model[..., s_start : s_start + n_states]
            )
            member_constants = constants.replace(
                actuator_model=constants.actuator_model[..., c_start : c_start + n_consts]
            )
            pieces.append(m.constraints((member_states_list, globals_dict), member_constants, model, settings))
        return jnp.concatenate(pieces) if pieces else jnp.zeros((0,))

    # --- jacobian: autodiff on constraints() itself, declared fully dense ---
    def jacobian(self, states, constants, model, settings):
        """Computes `jax.jacobian` of this class's own `constraints()`
        directly, rather than hand-deriving a sparse (rows, cols, data)
        structure the way Hill2d does -- see the module docstring for why.
        Declares every structurally-possible entry present (dense per
        constraint row), so `get_nnz()` above is overridden to match this
        exactly (cyipopt requires a fixed-length `data` array matching
        whatever `get_nnz()`/`jacobianstructure()` first declared).

        `adaptive_h` discretization isn't supported (same restriction Hill2d
        itself has for its own jacobian()).
        """
        if settings.get("discretization", {}).get("args", {}).get("adaptive_h", False):
            raise NotImplementedError(
                "MultiActuator.jacobian() does not support adaptive_h discretization "
                "(matches Hill2d's own restriction for the same reason: dur/h column "
                "bookkeeping would need to become per-interval)."
            )

        nnodes = settings.get("nnodes")
        nnodes_dur = settings.get("nnodes_dur", nnodes)
        nvpn = model.opt_states.size()

        # nnodes == 1 has no time dimension at all -- no dur/h decision
        # variable exists for it (matches Hill2d's own jacobian(), which never
        # touches globals_dict in that case either, see hill2d.py's nnodes<=1
        # branch). globals_dict may genuinely be None then (no Globals object
        # was ever built), so differentiating w.r.t. it must be skipped
        # entirely rather than merely guarded against a None *value*.
        if nnodes > 1:
            jac_states_list, jac_globals = jax.jacobian(lambda s: self.constraints(s, constants, model, settings))(
                states
            )
        else:
            states_list, _ = states
            jac_states_list = jax.jacobian(
                lambda sl: self.constraints((sl, None), constants, model, settings)
            )(states_list)
            jac_globals = None

        # States block: concatenate q/qd/gc_model/actuator_model per node,
        # same field order dynamics.py's own _flatten_model_jac uses, giving
        # (n_constraints, nnodes, nvpn); reshape's C-order flattening then
        # matches dynamics.py's "node*nvpn + local_idx" column convention
        # exactly (nnodes is the slower-varying axis, nvpn the faster one).
        dense = jnp.concatenate(
            (jac_states_list.q, jac_states_list.qd, jac_states_list.gc_model, jac_states_list.actuator_model),
            axis=-1,
        )
        n_constraints = dense.shape[0]
        dense = dense.reshape(n_constraints, nnodes * nvpn)

        if jac_globals is not None:
            # dur column: fixed index nnodes_dur * nvpn, same convention as
            # dynamics.py's _h_column for the non-adaptive case. globals_dict.dur
            # has shape (1,), so its jacobian is (n_constraints, 1).
            dur_col = nnodes_dur * nvpn
            dur_grad = jac_globals.dur.reshape(n_constraints, 1)
            dense = jnp.concatenate([dense, dur_grad], axis=-1)
            col_idx = jnp.concatenate([jnp.arange(nnodes * nvpn), jnp.array([dur_col])])
        else:
            col_idx = jnp.arange(nnodes * nvpn)

        total_cols = col_idx.shape[0]
        rows = jnp.repeat(jnp.arange(n_constraints), total_cols)
        cols = jnp.tile(
            col_idx,
            n_constraints,
        )
        data = dense.reshape(-1)
        return rows, cols, data