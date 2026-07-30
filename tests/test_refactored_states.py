import warnings
import pytest
import jax
import jax.numpy as jnp

from biosym.utils.states import States, Constants, concatenate, resample


def test_states_optional_fields_default_to_none():
    """Verify that q, qd, qdd, tau, ext_forces, ext_torques default to None in States."""
    states = States()
    assert states.q is None
    assert states.qd is None
    assert states.qdd is None
    assert states.tau is None
    assert states.ext_forces is None
    assert states.ext_torques is None


def test_jax_pytree_names_and_metadata():
    """Verify that names and metadata are correctly serialized and deserialized in PyTrees.

    `names` is derived automatically from which physical fields are populated
    (it drives `to_array`/`filter`'s iteration order), so it is not settable
    by the caller. Arbitrary descriptive data (e.g. per-DOF names) belongs in
    `metadata`, which round-trips as provided.
    """
    metadata = {
        "joint_limits": {
            "hip_flexion_r": [-0.5, 2.0]
        },
        "model_type": "gait2d",
        "coordinate_names": ["pelvis_tx", "pelvis_ty", "hip_flexion_r"],
    }

    states = States(
        q=jnp.array([1.0, 2.0, 3.0]),
        metadata=metadata
    )

    # Check leaves and structure
    leaves, treedef = jax.tree_util.tree_flatten(states)

    # Reconstruct from PyTree
    states_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

    assert states_reconstructed.names == ["q"]
    assert states_reconstructed.metadata == metadata


def test_backward_compatibility_with_legacy_unpickled_states():
    """Verify that unpickling / __setstate__ handles old pickled objects missing new fields or using dq."""
    legacy_state = {
        "q": jnp.array([1.0]),
        "dq": jnp.array([2.0]),
        "ddq": jnp.array([3.0])
    }

    states = States.__new__(States)
    states.__setstate__(legacy_state)

    assert jnp.allclose(states.q, jnp.array([1.0]))
    assert jnp.allclose(states.qd, jnp.array([2.0]))
    assert jnp.allclose(states.qdd, jnp.array([3.0]))
    assert states.names is None
    assert states.metadata is None


def test_concatenate():
    """Verify that concatenate appends time-series of states to each other."""
    s1 = States(q=jnp.array([1.0, 2.0]))
    s2 = States(q=jnp.array([3.0, 4.0]))
    
    # Concatenate list of States
    result = concatenate([s1, s2])
    
    # Check shape/value. Since the individual q arrays were 1D, they are unsqueezed to (1, 2)
    # and concatenated along axis 0 to shape (2, 2)
    assert result.q.shape == (2, 2)
    assert jnp.allclose(result.q, jnp.array([[1.0, 2.0], [3.0, 4.0]]))


def test_resample():
    """Verify that resample changes the length of the time series."""
    # Time series of length 2
    s = States(
        q=jnp.array([[1.0, 10.0], [3.0, 30.0]])
    )
    
    # Resample to length 3
    resampled = resample(s, 3)
    
    assert resampled.q.shape == (3, 2)
    assert jnp.allclose(resampled.q, jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    
    # If no length is there (1D array), it should just expand/tile
    s_single = States(q=jnp.array([1.0, 10.0]))
    resampled_single = resample(s_single, 4)
    assert resampled_single.q.shape == (4, 2)
    assert jnp.allclose(resampled_single.q, jnp.array([[1.0, 10.0]] * 4))
