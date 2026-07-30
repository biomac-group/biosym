import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from biosym.constraints.base_constraint import BaseConstraint


# any constraint needs to be named Constraint, otherwise it will not be found by the OCP class
class Constraint(BaseConstraint):
    """
    Base class for dynamics constraints in the biosym package.

    This class provides a template for implementing specific dynamics constraints.
    It includes methods for evaluating the constraint function, computing the Jacobian,
    and retrieving information about the constraint.
    """

    def __init__(self, model, settings, args):
        self.model = model
        self.settings = settings.copy()
        self.settings["nvpn"] = model.opt_states.size()
        self.nvar = settings.get("nvar")
        self.ncons_model = len(self.model.fr)
        self.nq = self.model.coordinates.n
        self.nvpn_opt = model.opt_states.size()
        self.qd_mirror = _periodic_qd_mirror(model, settings)
        self.bodymass = model.variables[(model.variables['name'].str.startswith('m_')) & (model.variables['type'] == "constant")]['x0'].sum()

    def _get_info(self):
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Base dynamics constraint class for biosym constraints.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "nnz": self.get_nnz(),
            "nnz_model": self.ncons_model * self.settings.get("nvpn") * self.settings.get("nnodes"),
            "ncons": self.get_n_constraints(),
            "ncons_pernode": self.ncons_model,
            "n_dyn_nodes": self._n_dyn_nodes(),
            "bodymass": self.bodymass,
            "nq": self.nq,
        }

    def get_confun(self):
        modelfn_bwd = partial(conf_bwd, self.model)
        modelfn_zero = partial(conf_zero, self.model)
        modelfn_periodic = (
            partial(conf_bwd, self.model, qd_prev_mirror=self.qd_mirror)
            if self.qd_mirror is not None else None
        )
        return jax.jit(partial(
            confun, modelfn_bwd, modelfn_zero, modelfn_periodic,
            settings=self.settings, info=self._get_info(), model=self.model,
        ))

    def get_jacobian(self):
        # argnums includes 2 (h): qdd = (qd - qd_prev)/h depends on h, and h in
        # turn depends on the global `dur` (or, if adaptive, on its own h[i]
        # decision variable) -- unlike the pre-refactor system, where qdd was a
        # free variable with zero dependence on dur, this dependency is now real
        # and must be included or IPOPT gets a wrong/missing "dur" gradient.
        modelfn_bwd = jax.jacobian(partial(conf_bwd, self.model), argnums=(0, 1, 2))
        modelfn_zero = jax.jacobian(partial(conf_zero, self.model))
        modelfn_periodic = (
            jax.jacobian(partial(conf_bwd, self.model, qd_prev_mirror=self.qd_mirror), argnums=(0, 1, 2))
            if self.qd_mirror is not None else None
        )
        return jax.jit(partial(
            jacobian, modelfn_bwd, modelfn_zero, modelfn_periodic,
            settings=self.settings, info=self._get_info(), model=self.model,
        ))

    def _n_dyn_nodes(self):
        # Node 0 has no predecessor to finite-difference qdd against -- *unless*
        # the problem is periodic, in which case node 0's natural predecessor is
        # the mirrored last node of the cycle (see _periodic_qd_mirror): that's
        # exactly what the pre-refactor system enforced implicitly, by tying the
        # extra periodic node's qdd (via discretization) to the mirror of qdd_0
        # (via periodicity). With a periodicity constraint, every node 0..nnodes-1
        # is therefore constrained. Without one, node 0's dynamics equation only
        # ever *defined* its own (free) qdd -- it constrained nothing else -- so
        # it's dropped (nodes 1..nnodes-1 remain fully constrained via their
        # predecessor). For nnodes == 1 there is no predecessor at all: qdd is
        # pinned to zero (equilibrium), which *is* a real constraint.
        nnodes = self.settings.get("nnodes")
        if nnodes == 1:
            return 1
        return nnodes if self.qd_mirror is not None else nnodes - 1

    def get_n_constraints(self):
        return self.ncons_model * self._n_dyn_nodes() + self.model.actuator_model.get_n_constraints(self.model, self.settings) + self.model.gc_model.get_n_constraints(self.model, self.settings)

    def get_nnz(self):
        nnz = 0
        if self.model.actuator_model.get_n_constraints(self.model, self.settings) > 0:
            nnz += (self.model.actuator_model.get_nnz(self.model, self.settings))
        if self.model.gc_model.get_n_constraints(self.model, self.settings) > 0:
            nnz += (self.model.gc_model.get_nnz(self.model, self.settings))
        nnodes = self.settings.get("nnodes")
        if nnodes > 1:
            # Each remaining node's finite-difference qdd also depends on its
            # predecessor's qd (one extra qd-sized block per node) and on the
            # step size h, i.e. on the global dur (or, if adaptive, on its own
            # h[i]): one extra scalar column per node.
            nnz += self.ncons_model * (self.nvpn_opt + self.nq + 1) * self._n_dyn_nodes()
        else:
            nnz += self.ncons_model * self.nvpn_opt
        return nnz


def _materialize(states, model):
    """
    Fill placeholder zeros for any missing derived fields (qdd, tau, ext_forces,
    ext_torques). Some model.run[...] functions (e.g. contact model forward
    passes) unpack `states.filter('model').flatten()` positionally against a
    fixed symbol list, so they need all 6 fields present as real arrays even
    though the values of these particular ones are functionally unused there.
    """
    fills = {}
    if states.qdd is None:
        fills["qdd"] = jnp.zeros(model.accs.n)
    if states.tau is None:
        fills["tau"] = jnp.zeros(model.tau.n)
    if states.ext_forces is None:
        fills["ext_forces"] = jnp.zeros(model.ext_forces.n)
    if states.ext_torques is None:
        fills["ext_torques"] = jnp.zeros(model.ext_torques.n)
    return states.replace(**fills) if fills else states


def calc_forces(states, model, constants=None):
    """
    First stage of the dynamics evaluation: run the ground-contact and actuator
    models' forward passes and inject the resulting forces/torques into states.
    """
    if constants is None:
        constants = model.default_constants
    states = _materialize(states, model)
    external_forces, external_torques = model.run["gc_model"](states, constants)
    internal_forces = model.run["actuator_model"](states, constants)
    return states.replace(
        tau=internal_forces.flatten(),
        ext_forces=external_forces.flatten(),
        ext_torques=external_torques.flatten(),
    )


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def confun_mm_tau(states, constants, model):
    mm = model.run['mass_matrix'](states, constants)
    forcing = model.run['forcing'](states, constants)
    qdd = states.qdd

    # R = M(q)*qdd - forcing
    inertial_force = jnp.matmul(mm, qdd[..., None])[..., 0]
    residuals = inertial_force - forcing
    return residuals


@confun_mm_tau.defjvp
def jvpfun(model, primals, tangents):
    states, constants = primals
    dstates, dconstants = tangents

    mm = model.run['mass_matrix'](states, constants)
    forcing = model.run['forcing'](states, constants)
    qdd = states.qdd
    dqdd = dstates.qdd

    inertial_force = jnp.matmul(mm, qdd[..., None])[..., 0]
    residuals = inertial_force - forcing

    # Compute d(M(q)*qdd) and dforcing using JVP of vector-valued functions.
    # This avoids computing the tangent of the full mass matrix (O(n^2))
    # and instead computes the tangent of the contracted vector (O(n)).
    def mm_qdd_and_forcing(s, c):
        M = model.run['mass_matrix'](s, c)
        F = model.run['forcing'](s, c)
        M_qdd = jnp.matmul(M, qdd[..., None])[..., 0]
        return M_qdd, F

    _, (dmm_qdd, dforcing) = jax.jvp(
        lambda s, c: mm_qdd_and_forcing(s, c),
        (states, constants),
        (dstates, dconstants)
    )

    mm_dqdd = jnp.matmul(mm, dqdd[..., None])[..., 0]
    dresiduals = dmm_qdd + mm_dqdd - dforcing

    return residuals, dresiduals


def _conf_core(model, states, constants):
    states_filled = calc_forces(states, model, constants)
    return confun_mm_tau(states_filled, constants, model)


def _periodic_qd_mirror(model, settings):
    """
    If a periodicity constraint is configured, return the qd permutation that
    maps the end of the gait cycle onto node 0 (used to close the loop for
    node 0's finite-difference qdd, since node 0 otherwise has no predecessor).
    Returns None if there's no periodicity constraint at all.
    """
    for c in settings.get("constraints", []) or []:
        if c.get("name") == "periodicity":
            args = c.get("args") or {}
            nq = model.coordinates.n
            if args.get("symmetry", False):
                from biosym.constraints.periodicity import get_symmetry_indices
                id_symmetry = get_symmetry_indices(model)
                return jnp.array([id_symmetry[nq + i] - nq for i in range(nq)])
            return jnp.arange(nq)
    return None


def conf_bwd(model, states, states_prev, h, constants, qd_prev_mirror=None):
    """
    qdd is derived from a backward-Euler finite difference of qd,
    qdd = (qd - qd_prev) / h. If qd_prev_mirror is given (periodic wrap at
    node 0, tying it to the mirrored last node of the cycle), it's applied
    to states_prev.qd before differencing.

    The residual is returned multiplied by h: M(q)*qdd - forcing, scaled by
    h, is M(q)*(qd - qd_prev) - h*forcing -- the same zero set (h > 0 always),
    but without the 1/h and 1/h^2 terms that a bare "divide by h" formulation
    puts into the Jacobian's qd/qd_prev and h/dur columns. Those blow up as h
    shrinks (e.g. once dur drifts toward its lower bound with many nodes),
    degrading the constraint's conditioning; this form stays well-scaled.
    Autodiff resolves the multiply-by-h/divide-by-h cancellation exactly, so
    this is not merely an algebraic simplification done by hand -- it changes
    what's actually returned (and hence exposed to the NLP solver).
    """
    qd_prev = states_prev.qd if qd_prev_mirror is None else states_prev.qd[qd_prev_mirror]
    states = states.replace(qdd=(states.qd - qd_prev) / h)
    return h * _conf_core(model, states, constants)


def conf_zero(model, states, constants):
    """Single-node problems: no neighbor exists at all, so qdd == 0."""
    states = states.replace(qdd=jnp.zeros_like(states.qd))
    return _conf_core(model, states, constants)


def _node_h(globals_dict, settings, n_intervals):
    if settings.get("discretization", {}).get("args", {}).get("adaptive_h", False):
        return globals_dict.h[:n_intervals]
    nnodes_dur = settings.get("nnodes_dur")
    return jnp.ones(n_intervals) * globals_dict.dur / (nnodes_dur - 1)


def compute_qdd_fd(states_list, globals_dict, settings):
    """
    Backward-Euler finite-difference qdd from qd, for every node in states_list.
    Node 0 reuses the node-0/1 interval's value, since there is no earlier
    neighbor to difference against.
    """
    n_total = len(states_list)
    if n_total == 1:
        return jnp.zeros_like(states_list.qd)
    h = _node_h(globals_dict, settings, n_total - 1)
    qdd_bwd = (states_list.qd[1:] - states_list.qd[:-1]) / h[:, None]
    return jnp.concatenate([qdd_bwd[0:1], qdd_bwd], axis=0)


def confun(modelfn_bwd, modelfn_zero, modelfn_periodic, states_list, globals_dict, settings, info, model):
    nnodes = settings.get("nnodes")
    constants = model.default_constants

    if nnodes == 1:
        vals = modelfn_zero(states_list[0], constants)[None, ...]
    elif modelfn_periodic is not None:
        # h[k] is the step from node k to node k+1; h[-1] is the wrap step
        # (last real node -> the extra periodic node / mirrored node 0).
        h = _node_h(globals_dict, settings, nnodes)
        val0 = modelfn_periodic(states_list[0], states_list[nnodes - 1], h[-1], constants)[None, ...]
        vals_rest = jax.vmap(modelfn_bwd, in_axes=(0, 0, 0, None))(
            states_list[1:nnodes], states_list[0 : nnodes - 1], h[: nnodes - 1], constants
        )
        vals = jnp.concatenate([val0, vals_rest], axis=0)
    else:
        h = _node_h(globals_dict, settings, nnodes - 1)
        vals = jax.vmap(modelfn_bwd, in_axes=(0, 0, 0, None))(
            states_list[1:nnodes], states_list[0 : nnodes - 1], h, constants
        )
    data_model = (1 / info['bodymass'] * vals.squeeze()).reshape(-1)

    if model.actuator_model.get_n_constraints(model, settings) > 0:
        c_act = model.actuator_model.constraints((states_list, globals_dict), constants, model, settings)
        data_out = jnp.concatenate([data_model, c_act.flatten().squeeze()])
    else:
        data_out = data_model

    if model.gc_model.get_n_constraints(model, settings) > 0:
        raise NotImplementedError("Ground contact constraints in unified dynamics not yet implemented.")
    return data_out


def _flatten_model_jac(jac):
    return jnp.concatenate((jac.q, jac.qd, jac.gc_model, jac.actuator_model), axis=-1)


def _h_column(nnodes, nnodes_dur, nvpn, adaptive_h, interval_idx):
    """
    Column index (and d(h)/d(decision variable) scale factor) for the global
    variable that a given interval's step size h derives from: dur itself when
    h is a fixed fraction of it, or that interval's own h[i] decision variable
    when adaptive. Mirrors discretization.py's duration/h column convention.
    """
    if adaptive_h:
        return nnodes * nvpn + interval_idx, 1.0
    return nnodes_dur * nvpn, 1.0 / (nnodes_dur - 1)


def jacobian(modelfn_bwd, modelfn_zero, modelfn_periodic, states_list, globals_dict, settings, info, model):
    nvpn = states_list[0].size()
    nq = info["nq"]
    nnodes = settings.get("nnodes")
    nnodes_dur = settings.get("nnodes_dur")
    ncons_sympy = info["ncons_pernode"]
    constants = model.default_constants
    adaptive_h = settings.get("discretization", {}).get("args", {}).get("adaptive_h", False)

    # Sparsity structure (rows/cols) below is built with plain NumPy instead of
    # jnp, and outside of vmap/lax control flow. It depends only on Python ints
    # (nnodes, nvpn, nq, ncons_sympy, ...) known at trace time, never on traced
    # values, so tracing it with jnp only bloats the HLO graph XLA has to
    # optimize (arange/tile/repeat/concatenate per node) without changing the
    # result -- costly at high nnodes since it scales with node count. Building
    # it eagerly in Python/NumPy keeps only the real per-node AD values (data)
    # flowing through jax/vmap; the indices are embedded as constants.
    if nnodes == 1:
        jac_0 = modelfn_zero(states_list[0], constants)
        row_block0 = np.arange(ncons_sympy)
        col_block0 = np.arange(nvpn)
        rows_out = np.repeat(row_block0, nvpn)
        cols_out = np.tile(col_block0, ncons_sympy)
        data_out = _flatten_model_jac(jac_0).flatten()
    else:
        periodic = modelfn_periodic is not None
        h = _node_h(globals_dict, settings, nnodes if periodic else nnodes - 1)

        # Nodes 1..nnodes-1: dense current-node block + a "previous node" qd
        # neighbor block + a scalar h/dur column. If not periodic, node 0 is
        # not constrained here (see _n_dyn_nodes) and rows start at node 1's
        # block instead of node 0's.
        jac_curr, jac_prev, jac_h = jax.vmap(modelfn_bwd, in_axes=(0, 0, 0, None))(
            states_list[1:nnodes], states_list[0 : nnodes - 1], h[: nnodes - 1] if periodic else h, constants
        )

        row_offset = 0 if periodic else -1

        def _node_block_indices(n):
            row_block = (n + row_offset) * ncons_sympy + np.arange(ncons_sympy)
            col_block = nvpn * n + np.arange(nvpn)
            rows_curr = np.repeat(row_block, nvpn)
            cols_curr = np.tile(col_block, ncons_sympy)

            # qd sits right after q in the node's flattened state vector.
            col_block_prev = (n - 1) * nvpn + nq + np.arange(nq)
            rows_prev = np.repeat(row_block, nq)
            cols_prev = np.tile(col_block_prev, ncons_sympy)

            # This node's own dynamics equation uses h[n - 1] (interval n-1 -> n).
            h_col, h_scale = _h_column(nnodes, nnodes_dur, nvpn, adaptive_h, n - 1)
            rows_h = row_block
            cols_h = np.full((ncons_sympy,), h_col, dtype=int)

            rows = np.concatenate((rows_curr, rows_prev, rows_h))
            cols = np.concatenate((cols_curr, cols_prev, cols_h))
            return rows, cols, h_scale

        _indices_and_scale = [_node_block_indices(n) for n in range(1, nnodes)]
        rows_rest = np.concatenate([r for r, _, _ in _indices_and_scale])
        cols_rest = np.concatenate([c for _, c, _ in _indices_and_scale])
        h_scale_rest = jnp.asarray([s for _, _, s in _indices_and_scale])

        def _node_block_data(jac_c, jac_p, jac_h_n, h_scale_n):
            data_curr = _flatten_model_jac(jac_c).flatten()
            data_prev = jac_p.qd.flatten()
            data_h = jac_h_n * h_scale_n
            return jnp.concatenate((data_curr, data_prev, data_h))

        data_rest = jax.vmap(_node_block_data)(jac_curr, jac_prev, jac_h, h_scale_rest)
        data_rest = data_rest.reshape(-1)

        if periodic:
            # Node 0: dense current-node block + a mirrored "wrap" qd neighbor
            # block + the wrap interval's h/dur column, pointing at the last
            # real dynamics node (nnodes - 1) / interval (nnodes - 1).
            jac_curr0, jac_wrap0, jac_h0 = modelfn_periodic(states_list[0], states_list[nnodes - 1], h[-1], constants)
            row_block0 = np.arange(ncons_sympy)
            col_block0 = np.arange(nvpn)
            rows_curr0 = np.repeat(row_block0, nvpn)
            cols_curr0 = np.tile(col_block0, ncons_sympy)
            data_curr0 = _flatten_model_jac(jac_curr0).flatten()

            col_block_wrap0 = (nnodes - 1) * nvpn + nq + np.arange(nq)
            rows_wrap0 = np.repeat(row_block0, nq)
            cols_wrap0 = np.tile(col_block_wrap0, ncons_sympy)
            data_wrap0 = jac_wrap0.qd.flatten()

            h_col0, h_scale0 = _h_column(nnodes, nnodes_dur, nvpn, adaptive_h, nnodes - 1)
            rows_h0 = row_block0
            cols_h0 = np.full((ncons_sympy,), h_col0, dtype=int)
            data_h0 = jac_h0 * h_scale0

            rows_out = np.concatenate((rows_curr0, rows_wrap0, rows_h0, rows_rest))
            cols_out = np.concatenate((cols_curr0, cols_wrap0, cols_h0, cols_rest))
            data_out = jnp.concatenate((data_curr0, data_wrap0, data_h0, data_rest))
        else:
            rows_out, cols_out, data_out = rows_rest, cols_rest, data_rest

    data_out = (1 / info['bodymass'] * data_out).reshape(-1)

    if model.actuator_model.get_n_constraints(model, settings) > 0:
        rows_act_con, cols_act_con, data_act_con = model.actuator_model.jacobian(
            (states_list, globals_dict), constants, model, settings
        )
        rows_act_con = rows_act_con + (info.get('ncons_pernode') * info.get('n_dyn_nodes'))  # Shift row indices to avoid overlap
        rows_out = jnp.concatenate([rows_out, rows_act_con], axis=0)
        cols_out = jnp.concatenate([cols_out, cols_act_con], axis=0)
        data_out = jnp.concatenate([data_out, data_act_con], axis=0)
    return rows_out, cols_out, data_out
