"""
Utility functions for optimal control problem setup and data conversion.

This module provides helper functions for converting between different data
representations used in optimization, particularly for handling state vectors
and parameter structures in the biosym framework.
"""

from dataclasses import fields, is_dataclass

import jax
import jax.numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree

from biosym.utils.states import Globals, StatesDict


@jax.jit
def x_to_states_dict(x, states_dict, globals_dict=None):
    """
    Convert a flat optimization vector into structured state and global dictionaries.
    
    This function is essential for interfacing with optimization algorithms that
    work with flat parameter vectors, converting them back to the structured
    data formats used throughout the biosym framework.
    
    Parameters
    ----------
    x : jnp.ndarray
        Flat array containing state values and potentially global parameters.
        Structure: [N*d state values, global parameters...]
    states_dict : StatesDict
        Template StatesDict defining the structure and dimensions of states.
        Used to determine how to reshape the flat array.
    globals_dict : dict, optional
        Template for global variables structure. If provided, extracts global
        parameters from the end of x.
        
    Returns
    -------
    tuple
        Tuple containing:
        - new_states_dict: StatesDict with states filled from x
        - new_globals: Globals object with parameters from x (or None)
        
    Notes
    -----
    - Expects x to contain N*d state values followed by global parameters
    - Uses JAX tree operations for efficient array reshaping
    - Preserves constants from the input states_dict unchanged
    - JIT-compiled for performance during optimization
    """
    # 1) Build an “example” single state by slicing index 0 out of each array in the batch
    example_state = jax.tree_util.tree_map(
        lambda arr: arr[0] if isinstance(arr, jnp.ndarray) else arr, states_dict.states
    )

    # 2) Flatten that example to get size 'd' and an unravel_fn
    flat_ex, unravel_state = ravel_pytree(example_state)
    d = flat_ex.shape[0]

    # 3) Batch size N
    N = states_dict.states.model.shape[0]

    # 4) Split x: first N*d entries for states, rest for globals
    flat_states = x[: N * d].reshape((N, d))
    rest = x[N * d :]
    if globals_dict is not None:
        new_globals = Globals(dur=x[-2], speed=x[-1]) if len(rest) == 2 else 71101
        if len(rest) != 2:
            raise ValueError(f"Expected 2 global parameters (dur, speed), got {len(rest)}")
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
    """
    Convert structured state and global dictionaries into a flat optimization vector.
    
    This is the inverse operation of x_to_states_dict, converting from the structured
    data formats used in biosym back to the flat parameter vectors required by
    optimization algorithms.
    
    Parameters
    ----------
    states_dict : StatesDict
        Structured state dictionary containing batched state variables.
    globals_dict : dict, optional
        Global variables dictionary to append to the flat vector.
        
    Returns
    -------
    jnp.ndarray
        Flat array containing all state values followed by global parameters.
        Structure: [N*d state values, global parameters...]
        
    Notes
    -----
    - Uses JAX tree flattening for efficient conversion
    - Concatenates state and global parameters into single vector
    - JIT-compiled for performance during optimization
    - Essential for interfacing with gradient-based optimizers
    """
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
    Extract a specific time point from batched state data structures.
    
    This function recursively extracts the idx-th row from all JAX arrays
    in a dataclass structure, preserving the original dataclass organization.
    Useful for extracting individual time points from optimization trajectories.
    
    Parameters
    ----------
    obj : dataclass or nested structure
        A dataclass (typically from flax.struct) or nested structure containing
        JAX arrays representing batched states over time.
    idx : int
        Index of the time point to extract (along axis 0 of arrays).
        
    Returns
    -------
    dataclass or structure
        New structure with the same type as input, containing only the
        idx-th time point from each array field. Non-array fields are
        passed through unchanged.
        
    Notes
    -----
    - Recursively processes nested dataclass structures
    - Preserves dataclass types and field organization
    - Handles scalar and non-array fields by passing them through
    - Useful for analyzing specific time points in optimization trajectories
    """
    if is_dataclass(obj):
        return obj.__class__(**{f.name: get_state_row(getattr(obj, f.name), idx) for f in fields(obj)})
    if isinstance(obj, jnp.ndarray) and obj.ndim > 0:
        return obj[idx]
    return obj  # Pass through scalars or non-array fields
