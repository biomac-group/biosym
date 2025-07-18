import jax
import jax.numpy as jnp
from collections import namedtuple
from flax.struct import dataclass
from dataclasses import fields, is_dataclass

@jax.jit
def x_to_states_dict(x, idx):
    """
    Convert a vector of states to a dictionary format.
    :return: A function that converts a vector of states to a dictionary.
    """
    pass

def states_list_to_dict(states_list):
    """
    Convert a list of states to a dictionary format.
    :param states_list: List of states.
    :return: Dictionary with states as keys and their values as lists.
    """
    return {key: {k: jnp.vstack([state[key][k] for state in states_list]) for k, v in value.items()} for key, value in states_list[0].items() if key != 'input_names'}

@jax.jit
def get_single_state(states_dict, n):
    """
    Get a single state from the states dictionary.
    :param states_dict: The states dictionary.
    :param n: The index of the state to retrieve.
    :return: The state at index n.
    """
    # Loop over the nested dictionary (depth=2) and return the state at index n
    return {key: {k: v[n] for k, v in value.items()} for key, value in states_dict.items() if key != 'input_names'}

@jax.jit
def sum_states_dicts(states_dicts, weights=None):
    """
    Sum a list of states dictionaries.
    :param states_dicts: List of states dictionaries.
    :return: A single states dictionary with summed values.
    """
    if weights is None:
        weights = [1] * len(states_dicts)
    
    return {key: {k: jnp.sum(jnp.array([state[key][k] * weight for state, weight in zip(states_dicts, weights)]), axis=0)
                 for k, v in value.items()} for key, value in states_dicts[0].items() if key != 'input_names'}



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
