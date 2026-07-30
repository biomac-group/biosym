import jax
import jax.numpy as jnp

from biosym.constraints.discretization import confun_q, jacobian_q
from biosym.utils.states import States, Globals


def test_discretization_jacobian_matches_autodiff_for_mixed_dependents():
    info = {
        "nvar": 1,
        "mode": "backward",
        "adaptive_h": True,
        "sections": {
            "contact_model": {"states": jnp.array([0]), "derivatives": jnp.array([1])},
            "actuator_model": {"states": jnp.array([0]), "derivatives": jnp.array([1])},
        },
        "ncons_per_node": 3,
        "ncons": 3,
        "nnz": 12,
    }
    settings = {"nnodes_dur": 2}
    globals_dict = Globals(dur=jnp.array([0.5]), speed=jnp.array([0.0]), h=jnp.array([0.5]))

    # nvar=1: q, qd, qdd each of length 1 per node, plus gc_model/actuator_model of length 2
    states_list = States(
        q=jnp.array([[1.0], [1.2]], dtype=float),
        qd=jnp.array([[2.0], [2.2]], dtype=float),
        qdd=jnp.array([[3.0], [3.2]], dtype=float),
        gc_model=jnp.array([[4.0, 5.0], [4.3, 5.3]], dtype=float),
        actuator_model=jnp.array([[6.0, 7.0], [6.4, 7.4]], dtype=float),
    )
    state_size = states_list[0].size()

    # flat vector contains: [node0 (size 7), node1 (size 7), h (size 1)]
    def residual(flat_vector):
        state_block = flat_vector[:-1]
        h_val = flat_vector[-1]

        node0 = state_block[:state_size]
        node1 = state_block[state_size: 2 * state_size]

        q = jnp.stack((node0[0:1], node1[0:1]), axis=0)
        qd = jnp.stack((node0[1:2], node1[1:2]), axis=0)
        qdd = jnp.stack((node0[2:3], node1[2:3]), axis=0)
        gc_model = jnp.stack((node0[3:5], node1[3:5]), axis=0)
        actuator_model = jnp.stack((node0[5:7], node1[5:7]), axis=0)

        candidate_states = States(
            q=q,
            qd=qd,
            qdd=qdd,
            gc_model=gc_model,
            actuator_model=actuator_model,
        )
        return confun_q(
            candidate_states,
            Globals(dur=jnp.array([0.5]), speed=jnp.array([0.0]), h=jnp.atleast_1d(h_val)),
            settings,
            info,
        )

    flat_vector = jnp.concatenate(
        (
            states_list[0].flatten(),
            states_list[1].flatten(),
            jnp.array([0.5], dtype=float),
        )
    )
    dense_jacobian = jax.jacrev(residual)(flat_vector)

    rows, cols, data = jacobian_q(states_list, globals_dict, settings, info)
    sparse_jacobian = jnp.zeros_like(dense_jacobian).at[rows, cols].set(data)

    assert jnp.unique(rows).shape[0] == info["ncons_per_node"]
    assert jnp.allclose(sparse_jacobian, dense_jacobian)


def test_discretization_jacobian_q_matches_autodiff_for_constant_duration():
    info = {
        "nvar": 1,
        "mode": "backward",
        "adaptive_h": False,
        "sections": {
            "contact_model": {"states": jnp.array([0]), "derivatives": jnp.array([1])},
            "actuator_model": {"states": jnp.array([0]), "derivatives": jnp.array([1])},
        },
        "ncons_per_node": 3,
        "ncons": 3,
        "nnz": 12,
    }
    settings = {"nnodes_dur": 2}
    globals_dict = Globals(dur=jnp.array([0.5]), speed=jnp.array([0.0]), h=jnp.zeros((0,)))

    # nvar=1: q, qd, qdd each of length 1 per node, plus gc_model/actuator_model of length 2
    states_list = States(
        q=jnp.array([[1.0], [1.2]], dtype=float),
        qd=jnp.array([[2.0], [2.2]], dtype=float),
        qdd=jnp.array([[3.0], [3.2]], dtype=float),
        gc_model=jnp.array([[4.0, 5.0], [4.3, 5.3]], dtype=float),
        actuator_model=jnp.array([[6.0, 7.0], [6.4, 7.4]], dtype=float),
    )
    state_size = states_list[0].size()

    # flat vector contains: [node0 (size 7), node1 (size 7), dur (size 1)]
    def residual(flat_vector):
        state_block = flat_vector[:-1]
        dur = flat_vector[-1]

        node0 = state_block[:state_size]
        node1 = state_block[state_size: 2 * state_size]

        q = jnp.stack((node0[0:1], node1[0:1]), axis=0)
        qd = jnp.stack((node0[1:2], node1[1:2]), axis=0)
        qdd = jnp.stack((node0[2:3], node1[2:3]), axis=0)
        gc_model = jnp.stack((node0[3:5], node1[3:5]), axis=0)
        actuator_model = jnp.stack((node0[5:7], node1[5:7]), axis=0)

        candidate_states = States(
            q=q,
            qd=qd,
            qdd=qdd,
            gc_model=gc_model,
            actuator_model=actuator_model,
        )
        return confun_q(candidate_states, Globals(dur=jnp.atleast_1d(dur), speed=jnp.array([0.0]), h=jnp.zeros((0,))), settings, info)

    flat_vector = jnp.concatenate(
        (
            states_list[0].flatten(),
            states_list[1].flatten(),
            jnp.array([0.5], dtype=float),
        )
    )
    dense_jacobian = jax.jacrev(residual)(flat_vector)

    rows, cols, data = jacobian_q(states_list, globals_dict, settings, info)
    sparse_jacobian = jnp.zeros_like(dense_jacobian).at[rows, cols].set(data)

    assert state_size == 7
    assert int(cols.max()) == flat_vector.size - 1
    assert jnp.allclose(sparse_jacobian, dense_jacobian)
