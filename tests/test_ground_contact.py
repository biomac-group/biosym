from types import SimpleNamespace

import jax.numpy as jnp

from biosym.constraints.ground_contact import Constraint, jacobian
from biosym.utils.states import Constants, States


class _DebugGcModel:
    def get_n_states(self):
        return 2

    def get_n_constraints(self):
        return 0

    def forward(self, states, constants, model):
        q = states.q.flatten()
        qd = states.qd.flatten()
        tau = states.tau.flatten()
        gc = states.gc_model.flatten()
        act = states.actuator_model.flatten()

        J_q_f = (0.0 + jnp.arange(12, dtype=float)).reshape(6, 2)
        J_qd_f = (0.0 + jnp.arange(6, dtype=float)).reshape(6, 1)
        J_tau_f = (0.0 + jnp.arange(6, dtype=float)).reshape(6, 1)
        J_gc_f = (100.0 + jnp.arange(12, dtype=float)).reshape(6, 2)
        J_act_f = (200.0 + jnp.arange(18, dtype=float)).reshape(6, 3)

        J_q_m = (300.0 + jnp.arange(12, dtype=float)).reshape(6, 2)
        J_qd_m = (300.0 + jnp.arange(6, dtype=float)).reshape(6, 1)
        J_tau_m = (300.0 + jnp.arange(6, dtype=float)).reshape(6, 1)
        J_gc_m = (400.0 + jnp.arange(12, dtype=float)).reshape(6, 2)
        J_act_m = (500.0 + jnp.arange(18, dtype=float)).reshape(6, 3)

        forces = (J_q_f @ q + J_qd_f @ qd + J_tau_f @ tau + J_gc_f @ gc + J_act_f @ act).reshape(2, 3)
        moments = (J_q_m @ q + J_qd_m @ qd + J_tau_m @ tau + J_gc_m @ gc + J_act_m @ act).reshape(2, 3)
        return forces, moments


class _DebugActuators:
    def get_n_states(self):
        return 3


class _DebugModel:
    def __init__(self):
        # lean model: 2 q-coords, 1 qd speed, 1 tau force  → nvpn_model = 4
        self.coordinates = {"n": 2, "idx": 0}
        self.speeds = {"n": 1, "idx": 2}
        self.forces = {"n": 1, "idx": 3}
        self.ext_forces = SimpleNamespace(idx=0, n=6)
        self.ext_torques = SimpleNamespace(idx=6, n=6)
        self.fr = [0, 1]
        self.state_vector = [""] * 16
        self.gc_model = _DebugGcModel()
        self.contact_model = self.gc_model
        self.default_constants = Constants()
        self.actuators = _DebugActuators()
        self.run = {"gc_model_jacobian": self.gc_model_jacobian}

    def gc_model_jacobian(self, states, constants):
        # Return lean field attributes: q, qd, tau (no monolithic .model)
        # Shapes: (n_bodies, n_forces_per_body, n_vars) = (2, 3, *)
        force = SimpleNamespace(
            q=jnp.arange(12, dtype=float).reshape(2, 3, 2),
            qd=jnp.arange(6, dtype=float).reshape(2, 3, 1),
            qdd=None,
            tau=jnp.arange(6, dtype=float).reshape(2, 3, 1),
            ext_forces=600.0 + jnp.arange(36, dtype=float).reshape(2, 3, 6),
            ext_torques=700.0 + jnp.arange(36, dtype=float).reshape(2, 3, 6),
            gc_model=100.0 + jnp.arange(12, dtype=float).reshape(2, 3, 2),
            actuator_model=200.0 + jnp.arange(18, dtype=float).reshape(2, 3, 3),
        )
        moment = SimpleNamespace(
            q=300.0 + jnp.arange(12, dtype=float).reshape(2, 3, 2),
            qd=300.0 + jnp.arange(6, dtype=float).reshape(2, 3, 1),
            qdd=None,
            tau=300.0 + jnp.arange(6, dtype=float).reshape(2, 3, 1),
            ext_forces=800.0 + jnp.arange(36, dtype=float).reshape(2, 3, 6),
            ext_torques=900.0 + jnp.arange(36, dtype=float).reshape(2, 3, 6),
            gc_model=400.0 + jnp.arange(12, dtype=float).reshape(2, 3, 2),
            actuator_model=500.0 + jnp.arange(18, dtype=float).reshape(2, 3, 3),
        )
        return force, moment


def test_ground_contact_jacobian_includes_gc_model_states_in_nnz():
    model = _DebugModel()
    settings = {"nnodes": 2, "nvar": 0, "is_lean_aba": True, "nvpn": 21}
    constraint = Constraint(model, settings, args=None)
    info = constraint._get_info()

    # Lean state: q(2) + qd(1) + tau(1) + ext_forces(6) + ext_torques(6) = 16 model vars; gc_model(2); actuator_model(3) → nvpn=21
    state = States(
        q=jnp.zeros((2, 2)),
        qd=jnp.zeros((2, 1)),
        tau=jnp.zeros((2, 1)),
        ext_forces=jnp.zeros((2, 6)),
        ext_torques=jnp.zeros((2, 6)),
        gc_model=jnp.zeros((2, 2)),
        actuator_model=jnp.zeros((2, 3)),
    )
    states_list = state

    rows, cols, data = jacobian(model, states_list, None, constraint.settings, info)

    ncons_per_node = info["ncons_pernode"]
    nvpn = constraint.settings["nvpn"]
    node_width = states_list[0].size()

    assert constraint.settings["nvpn_model"] == 16
    assert constraint.settings["nvpn_gc_model"] == 2
    assert constraint.settings["nvpn_actuator_model"] == 3
    assert nvpn == 21
    assert info["nnz"] == settings["nnodes"] * ncons_per_node * nvpn
    assert rows.shape == cols.shape == data.shape == (info["nnz"],)

    first_node = slice(0, ncons_per_node * nvpn)
    second_node = slice(ncons_per_node * nvpn, 2 * ncons_per_node * nvpn)

    assert int(cols[first_node].min()) == 0
    assert int(cols[first_node].max()) == nvpn - 1
    assert int(cols[second_node].min()) == node_width
    assert int(cols[second_node].max()) == node_width + nvpn - 1

    first_node_rows = rows[first_node]
    first_node_cols = cols[first_node]
    first_node_data = data[first_node]

    expected_gc = jnp.vstack((model.gc_model_jacobian(None, None)[0].gc_model, model.gc_model_jacobian(None, None)[1].gc_model)).reshape(ncons_per_node, 2)
    expected_actuator = jnp.vstack(
        (model.gc_model_jacobian(None, None)[0].actuator_model, model.gc_model_jacobian(None, None)[1].actuator_model)
    ).reshape(ncons_per_node, 3)

    # The sparse COO output is assembled as sequential blocks per variable group
    # (model, then gc_model, then actuator_model), each block ordered row-major
    # over (ncons_per_node, block_width) -- not interleaved per-row with nvpn stride.
    nvpn_model = constraint.settings["nvpn_model"]
    nvpn_gc_model = constraint.settings["nvpn_gc_model"]
    nvpn_actuator_model = constraint.settings["nvpn_actuator_model"]
    gc_block_start = nvpn_model * ncons_per_node
    act_block_start = gc_block_start + nvpn_gc_model * ncons_per_node

    for r in range(ncons_per_node):
        # gc_model columns are 16 and 17
        gc_indices = gc_block_start + r * nvpn_gc_model + jnp.arange(nvpn_gc_model)
        assert jnp.all(first_node_cols[gc_indices] == jnp.array([16, 17]))
        assert jnp.all(first_node_rows[gc_indices] == r)
        assert jnp.allclose(first_node_data[gc_indices], expected_gc[r])

        # actuator_model columns are 18, 19, 20
        act_indices = act_block_start + r * nvpn_actuator_model + jnp.arange(nvpn_actuator_model)
        assert jnp.all(first_node_cols[act_indices] == jnp.array([18, 19, 20]))
        assert jnp.all(first_node_rows[act_indices] == r)
        assert jnp.allclose(first_node_data[act_indices], expected_actuator[r])
