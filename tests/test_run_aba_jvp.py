import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from biosym.model import model as model_module
from biosym.utils.aba import get_aba_jax_model

def test_run_aba_custom_jvp_equivalence():
    """Verify that the custom JVP of run_aba returns identical JVP derivatives compared to standard AD."""
    m = model_module.load_model("tests/models/gait2d_torque/gait2d_torque.yaml", force_rebuild=False)
    assert m is not None

    states = m.default_states
    constants = m.default_constants

    # Inputs
    q = states.q if states.q is not None else jnp.zeros(m.coordinates.n)
    qd = states.qd if states.qd is not None else jnp.zeros(m.speeds.n)
    tau = states.tau if states.tau is not None else jnp.zeros(m.tau.n)
    ext = jnp.zeros(m.ext_forces.n + m.ext_torques.n)
    consts_model = constants.filter("model").to_array()

    aba_fn = get_aba_jax_model(m)

    # Tangents for JVP comparison
    key = jax.random.PRNGKey(123)
    dq = jax.random.normal(key, q.shape) * 0.1
    dqd = jax.random.normal(key, qd.shape) * 0.1
    dtau = jax.random.normal(key, tau.shape) * 0.1
    dext = jax.random.normal(key, ext.shape) * 0.1
    dconsts = jax.random.normal(key, consts_model.shape) * 0.1

    # Standard AD vs Custom JVP (now active on get_aba_jax_model)
    # We can retrieve the un-decorated primal implementation or define it to compare
    # Wait, our custom JVP is decorated on the returned run_aba function.
    # To get standard AD, we can just differentiate standard _run_aba_impl or we can compare with solving directly:
    M_obj = m.run["mass_matrix"]
    F_obj = m.run["forcing"]

    # We want to check that run_aba with custom JVP produces the correct JVP derivative compared to explicit solve
    def custom_aba_solve(q_val, qd_val, tau_val, ext_val, consts_val):
        return aba_fn(q_val, qd_val, tau_val, ext_val, consts_val)

    def explicit_solve(q_val, qd_val, tau_val, ext_val, consts_val):
        model_sizes = [np.prod(getattr(constants, name).shape) for name in ("g", "mass", "inertia", "com", "offset")]
        splits = np.cumsum(model_sizes)[:-1]
        g_v, mass_v, inertia_v, com_v, offset_v = jnp.split(consts_val, splits)
        consts_obj = constants.replace(
            g=g_v.reshape(constants.g.shape),
            mass=mass_v.reshape(constants.mass.shape),
            inertia=inertia_v.reshape(constants.inertia.shape),
            com=com_v.reshape(constants.com.shape),
            offset=offset_v.reshape(constants.offset.shape),
        )
        ef_n = m.ext_forces.n
        ef = ext_val[:ef_n]
        et = ext_val[ef_n:]
        # Primal acceleration doesn't matter for mass matrix/forcing, but we pass zero to be consistent
        s_obj = states.replace(
            q=q_val,
            qd=qd_val,
            qdd=jnp.zeros_like(q_val),
            tau=tau_val,
            ext_forces=ef,
            ext_torques=et,
        )
        M = M_obj(s_obj, consts_obj)
        F = F_obj(s_obj, consts_obj)
        return jnp.linalg.solve(M, F).flatten()

    y_cust, dy_cust = jax.jvp(custom_aba_solve, (q, qd, tau, ext, consts_model), (dq, dqd, dtau, dext, dconsts))
    y_exp, dy_exp = jax.jvp(explicit_solve, (q, qd, tau, ext, consts_model), (dq, dqd, dtau, dext, dconsts))

    assert jnp.allclose(y_cust, y_exp, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(dy_cust, dy_exp, rtol=1e-5, atol=1e-5)
