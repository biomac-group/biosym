import jax
import jax.numpy as jnp
from collections import namedtuple
from flax.struct import dataclass
from dataclasses import fields, is_dataclass
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map, tree_leaves
from jax import vmap
from biosym.utils.states import StatesDict

@jax.jit
def x_to_states_dict(x, states_dict, globals_dict=None):
    """
    Convert a flat array x into a StatesDict based on the structure of states_dict.
    :param x: Flat array containing the state values.
    :param states_dict: StatesDict defining the structure of the states.
    :return: A new StatesDict with the states filled from x.
    """
    # 1) Build an “example” single state by slicing index 0 out of each array in the batch
    example_state = jax.tree_util.tree_map(
        lambda arr: arr[0] if isinstance(arr, jnp.ndarray) else arr,
        states_dict.states
    )

    # 2) Flatten that example to get size 'd' and an unravel_fn
    flat_ex, unravel_state = ravel_pytree(example_state)
    d = flat_ex.shape[0]

    # 3) Batch size N
    N = states_dict.states.model.shape[0]

    # 4) Split x: first N*d entries for states, rest for globals
    flat_states = x[: N * d].reshape((N, d))
    rest       = x[N * d :]

    if globals_dict is not None:
        flat_glob, unravel_glob = ravel_pytree(globals_dict)
        new_globals = unravel_glob(rest)
    else:
        new_globals = None

    # 5) Rebuild the batched States by vmap‑ing the unravel
    new_states = vmap(unravel_state)(flat_states)

    # 7) Return a new StatesDict (keep constants unchanged)
    return StatesDict(
        states=new_states,
        constants=states_dict.constants,
    ), new_globals

@jax.jit
def states_dict_to_x(states_dict, globals_dict=None):
    # Flatten each batched state
    batch_flat = vmap(lambda s: ravel_pytree(s)[0])(states_dict.states)

    # Flatten globals if present
    if globals_dict is not None:
        globals_flat, _ = ravel_pytree(globals_dict)
        full_flat = jnp.concatenate([batch_flat.flatten(), globals_flat])
    else:
        full_flat = batch_flat.flatten()

    return full_flat

def get_state_row(obj, idx):
    """
    Recursively extract the idx-th row from all JAX arrays in a flax.struct.dataclass.
    Preserves the dataclass structure.

    :param obj: A flax.struct.dataclass or nested structure.
    :param idx: Index to extract (along axis 0).
    :return: A new structure with only the idx-th row from each array field.
    """
    if is_dataclass(obj):
        return obj.__class__(**{
            f.name: get_state_row(getattr(obj, f.name), idx)
            for f in fields(obj)
        })
    elif isinstance(obj, jnp.ndarray) and obj.ndim > 0:
        return obj[idx]
    else:
        return obj  # Pass through scalars or non-array fields
