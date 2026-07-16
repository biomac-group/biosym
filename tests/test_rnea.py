import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from biosym.model import model as model_module
from biosym.utils.aba import get_aba_jax_model, get_aba_biosym
from biosym.utils.rnea import get_rnea_jax_model, get_rnea_biosym


def test_rnea_aba_equivalence():
    """Verify that RNEA is the mathematical inverse of ABA.

    Specifically: ABA(q, qd, tau, ext, c) -> qdd, then RNEA(q, qd, qdd, ext, c) -> tau_rnea.
    tau_rnea at actuated joints must match the input tau, and must be 0 at unactuated joints.
    """
    m = model_module.load_model("tests/models/gait2d_torque/gait2d_torque.yaml", force_rebuild=False)
    assert m is not None

    states = m.default_states
    constants = m.default_constants

    q = states.q if states.q is not None else jnp.zeros(m.coordinates.n)
    qd = states.qd if states.qd is not None else jnp.zeros(m.speeds.n)
    tau = states.tau if states.tau is not None else jnp.zeros(m.tau.n)
    ext = jnp.zeros(m.ext_forces.n + m.ext_torques.n)
    consts_model = constants.filter("model").to_array()

    # Get the JAX kernels
    aba_fn = get_aba_jax_model(m)
    rnea_fn = get_rnea_jax_model(m)

    # 1. Forward dynamics to get a physically consistent acceleration qdd
    qdd = aba_fn(q, qd, tau, ext, consts_model)

    # 2. Inverse dynamics with RNEA
    tau_rnea = rnea_fn(q, qd, qdd, ext, consts_model)

    # 3. Build the expected full torque mapping (actuated joints match tau, unactuated match 0)
    # Exposing the internal body spec joint coordinate map
    from biosym.utils.aba import _build_parent_index_map, _state_index_map, _force_index_map
    
    bodies, parent_indices = _build_parent_index_map(list(m.dicts["bodies"]))
    qd_index_map = _state_index_map(m.speeds.names, "qd_")
    tau_index_map = _force_index_map(m)

    tau_expected = np.zeros(m.speeds.n)
    for body in bodies:
        for joint in body.get("joints", []):
            joint_name = joint["name"]
            qd_idx = qd_index_map[joint_name]
            tau_idx = tau_index_map.get(joint_name, -1)
            if tau_idx >= 0:
                tau_expected[qd_idx] = tau[tau_idx]
            else:
                tau_expected[qd_idx] = 0.0

    tau_expected = jnp.asarray(tau_expected)

    # Compare tau_rnea to the expected torque mapping
    assert jnp.allclose(tau_rnea, tau_expected, rtol=1e-6, atol=1e-6)


def test_rnea_jacobian():
    """Verify that the Jacobian of RNEA can be computed cleanly and is fast to compile/run."""
    m = model_module.load_model("tests/models/gait2d_torque/gait2d_torque.yaml", force_rebuild=False)
    assert m is not None

    states = m.default_states
    constants = m.default_constants

    q = states.q if states.q is not None else jnp.zeros(m.coordinates.n)
    qd = states.qd if states.qd is not None else jnp.zeros(m.speeds.n)
    qdd = jnp.zeros(m.speeds.n)
    ext = jnp.zeros(m.ext_forces.n + m.ext_torques.n)
    consts_model = constants.filter("model").to_array()

    rnea_fn = get_rnea_jax_model(m)

    # Differentiate RNEA with respect to q, qd, and qdd
    jac_q_fn = jax.jit(jax.jacobian(rnea_fn, argnums=0))
    jac_qd_fn = jax.jit(jax.jacobian(rnea_fn, argnums=1))
    jac_qdd_fn = jax.jit(jax.jacobian(rnea_fn, argnums=2))

    # Compile the Jacobians
    jac_q = jac_q_fn(q, qd, qdd, ext, consts_model)
    jac_qd = jac_qd_fn(q, qd, qdd, ext, consts_model)
    jac_qdd = jac_qdd_fn(q, qd, qdd, ext, consts_model)

    assert jac_q.shape == (m.speeds.n, m.coordinates.n)
    assert jac_qd.shape == (m.speeds.n, m.speeds.n)
    assert jac_qdd.shape == (m.speeds.n, m.speeds.n)


def test_rnea_biosym_compatibility():
    """Verify that the biosym-compatible RNEA wrapper works perfectly with StatesDict."""
    m = model_module.load_model("tests/models/gait2d_torque/gait2d_torque.yaml", force_rebuild=False)
    assert m is not None

    states = m.default_states
    constants = m.default_constants

    # Use public biosym wrappers
    aba_biosym = get_aba_biosym(m)
    rnea_biosym = get_rnea_biosym(m)

    # 1. Forward dynamics to get qdd
    qdd_sim = aba_biosym(states, constants)

    # 2. Store qdd_sim in the states object to simulate actual optimization constraints usage
    updated_states = states.replace(qdd=qdd_sim)

    # 3. Call RNEA with updated states
    tau_rnea = rnea_biosym(updated_states, constants)

    # Verify matching dimensions
    assert tau_rnea.shape == (m.speeds.n,)

def test_rnea_mass_matrix_equivalence():
    """Verify that the mass matrix computed via RNEA matches the one from model.run['mass_matrix']."""
    m = model_module.load_model("tests/models/gait2d_torque/gait2d_torque.yaml", force_rebuild=False)
    assert m is not None

    states = m.default_states
    constants = m.default_constants

    q = states.q if states.q is not None else jnp.zeros(m.coordinates.n)
    qd = states.qd if states.qd is not None else jnp.zeros(m.speeds.n)
    ext = jnp.zeros(m.ext_forces.n + m.ext_torques.n)
    consts_model = constants.filter("model").to_array()

    # Compute mass matrix via RNEA
    rnea_fn = get_rnea_jax_model(m)

    def compute_mass_matrix(q_val, qd_val, ext_val, consts_val):
        # rnea_fn returns the full inverse-dynamics torque M(q)@qdd + C(q,qd) + g(q) - J^T@ext,
        # so the Coriolis/gravity/external bias (its value at qdd=0) must be subtracted off
        # before each unit-qdd evaluation isolates a column of M(q).
        n = m.speeds.n
        bias = rnea_fn(q_val, qd_val, jnp.zeros(n), ext_val, consts_val)
        mass_matrix = jnp.zeros((n, n))
        for i in range(n):
            qdd_unit = jnp.zeros(n).at[i].set(1.0)
            tau_rnea = rnea_fn(q_val, qd_val, qdd_unit, ext_val, consts_val)
            mass_matrix = mass_matrix.at[:, i].set(tau_rnea - bias)
        return mass_matrix

    mass_matrix_rnea = compute_mass_matrix(q, qd, ext, consts_model)

    # Compare with model.run['mass_matrix']
    mass_matrix_biosym = m.run["mass_matrix"](states, constants)

    assert jnp.allclose(mass_matrix_rnea, mass_matrix_biosym, rtol=1e-6, atol=1e-6)
    