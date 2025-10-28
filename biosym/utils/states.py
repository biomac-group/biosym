"""
Core data structures for biomechanical modeling and optimization in biosym.

This module defines the fundamental data structures used throughout the biosym
framework for representing states, constants, and global parameters in
biomechanical simulations and optimal control problems. These structures are
built on JAX and Flax for efficient computation and automatic differentiation.

The module provides hierarchical data organization:
- States: Time-varying model variables (positions, velocities, activations)
- Constants: Time-invariant model parameters (masses, lengths, gains)
- Globals: Problem-level parameters (duration, speed)
- StatesDict: Container combining states and constants for complete model representation

Key Features:
- JAX-compatible data structures for efficient computation
- Automatic differentiation support through Flax dataclasses
- Vectorized operations for batch processing
- Indexing and slicing operations for trajectory analysis
- Utility functions for stacking, reducing, and converting data structures
"""

from dataclasses import field, replace
from typing import Literal

import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class States:
    """
    Time-varying state variables for biomechanical models.
    
    This dataclass represents the dynamic state of a biomechanical system
    at one or more time points. It contains arrays for different model
    components that change over time during simulation or optimization.
    
    Attributes
    ----------
    model : jnp.ndarray
        Primary model state variables (positions, velocities, accelerations).
        Shape typically (n_timesteps, n_model_dofs).
    gc_model : jnp.ndarray
        Ground contact model state variables (contact forces, positions).
        Shape typically (n_timesteps, n_gc_dofs).
    actuator_model : jnp.ndarray
        Actuator/muscle model state variables (activations, forces).
        Shape typically (n_timesteps, n_actuator_dofs).
    h : jnp.ndarray
        Time step sizes for adaptive time stepping in optimization.
        Shape typically (n_timesteps,) or (n_timesteps, 1).
        
    Notes
    -----
    - Uses Flax dataclass for JAX compatibility and automatic differentiation
    - All fields default to empty arrays if not provided
    - Supports vectorized operations across time steps
    - Essential for representing trajectories in optimal control problems
    """
    model: field(default_factory=lambda: jnp.zeros((0,)))  # Default to empty array if not provided
    gc_model: field(default_factory=lambda: jnp.zeros((0,)))  # Default to empty array if not provided
    actuator_model: field(default_factory=lambda: jnp.zeros((0,)))  # Default to empty array if not provided
    h: field(default_factory=lambda: jnp.zeros((0,)))  # Default to empty array if not provided

    def __str__(self):
        return f"States(model={self.model.shape}, gc_model={self.gc_model.shape}, actuator_model={self.actuator_model.shape}, h={self.h.shape if self.h is not None else 'None'})"

    def size(self):
        """
        Calculate the total number of elements across all state fields.
        
        Returns
        -------
        int
            Total number of scalar elements in all arrays of this States instance.
            
        Notes
        -----
        Uses JAX tree utilities to efficiently count elements across all fields.
        Useful for memory estimation and optimization problem sizing.
        """
        return sum(x.size for x in jax.tree_util.tree_leaves(self))

    def flatten(self):
        """
        Flatten all state arrays into a single 1D array.
        
        Returns
        -------
        jnp.ndarray
            1D array containing all state values concatenated together.
            Order follows the field declaration order: model, gc_model, actuator_model, h.
            
        Notes
        -----
        Essential for interfacing with optimization algorithms that require
        flat parameter vectors. The inverse operation can be performed using
        model-specific unflatten methods.
        """
        flat_states = jax.tree_util.tree_leaves(self)
        return jnp.concatenate([x.flatten() if isinstance(x, jnp.ndarray) else x for x in flat_states], axis=0)

    def __getitem__(self, index):
        """
        Extract states at specific time indices.
        
        Parameters
        ----------
        index : int, slice, or array-like
            Index or indices to extract from the time dimension.
            
        Returns
        -------
        States
            New States instance containing only the specified time points.
            
        Notes
        -----
        Supports standard Python indexing including slices and boolean masks.
        Essential for analyzing specific time points in optimization trajectories.
        """

        def slice_fn(x):
            return x[index] if isinstance(x, jnp.ndarray) else x

        return jax.tree_util.tree_map(slice_fn, self)
    
    def __len__(self):
        """
        Get the number of time steps in the trajectory.
        
        Returns
        -------
        int
            Number of time steps, determined by the first dimension of state arrays.
            Returns 1 for single-timestep data.
            
        Notes
        -----
        Based on the shape of the model field, which is assumed to be the primary
        state variable defining trajectory length.
        """
        return self.model.shape[0] if self.model.ndim > 1 else 1


@dataclass
class Constants:
    """
    Time-invariant model parameters and constants.
    
    This dataclass stores parameters that remain constant throughout
    a simulation or optimization, such as masses, lengths, and gains.
    These values define the physical properties and configuration
    of the biomechanical model.
    
    Attributes
    ----------
    model : jnp.ndarray
        Primary model constants (masses, lengths, inertias, gains).
        Shape typically (n_model_constants,).
    gc_model : jnp.ndarray
        Ground contact model constants (stiffness, damping, friction).
        Shape typically (n_gc_constants,).
    actuator_model : jnp.ndarray
        Actuator/muscle model constants (maximum forces, time constants).
        Shape typically (n_actuator_constants,).
        
    Notes
    -----
    - Constants are typically shared across all time points in a trajectory
    - Used for model parameterization and sensitivity analysis
    - Can be optimization variables in parameter identification problems
    """
    model: field(default_factory=lambda: jnp.zeros((0,)))
    gc_model: field(default_factory=lambda: jnp.zeros((0,)))
    actuator_model: field(default_factory=lambda: jnp.zeros((0,)))

    def __str__(self):
        return f"Constants(model={self.model.shape}, gc_model={self.gc_model.shape}, actuator_model={self.actuator_model.shape})"

    def multiply(self, other):
        """
        Element-wise multiplication of constants.
        
        Parameters
        ----------
        other : float or int
            Scalar value to multiply with all constant arrays.
            
        Returns
        -------
        Constants
            New Constants instance with all values multiplied by the scalar.
            
        Raises
        ------
        NotImplementedError
            If other is not a scalar (int or float).
            
        Notes
        -----
        Useful for scaling model parameters or sensitivity analysis.
        """
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        raise NotImplementedError("biosym.utils.states.Constants.multiply.notfloat")


@dataclass
class Globals:
    """
    Global optimization parameters for optimal control problems.
    
    This dataclass stores problem-level parameters that are optimized
    globally across the entire trajectory, such as movement duration
    and average speed. These parameters affect the entire motion pattern.
    
    Attributes
    ----------
    dur : jnp.ndarray
        Movement duration in seconds. Shape typically (1,).
        Controls the total time span of the optimized trajectory.
    speed : jnp.ndarray
        Average movement speed. Shape typically (1,).
        Used for speed-based constraints and objectives.
        
    Notes
    -----
    - Global parameters are shared across all time points
    - Essential for time-optimal and speed-constrained problems
    - Often used as optimization variables in trajectory optimization
    """
    dur: jnp.ndarray = field(default_factory=lambda: jnp.zeros((1,)))
    speed: jnp.ndarray = field(default_factory=lambda: jnp.zeros((1,)))

    def size(self):
        """
        Calculate the total number of global parameters.
        
        Returns
        -------
        int
            Total number of scalar elements in all global parameter arrays.
            
        Notes
        -----
        Typically returns 2 (duration + speed) for standard problems.
        """
        return sum(x.size for x in jax.tree_util.tree_leaves(self))

    def multiply(self, other):
        """
        Element-wise multiplication of global parameters.
        
        Parameters
        ----------
        other : float or int
            Scalar value to multiply with all global parameter arrays.
            
        Returns
        -------
        Globals
            New Globals instance with all values multiplied by the scalar.
            
        Raises
        ------
        NotImplementedError
            If other is not a scalar (int or float).
            
        Notes
        -----
        Used for scaling global parameters during optimization.
        """
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        raise NotImplementedError("biosym.utils.states.Constants.multiply.notfloat")


@dataclass
class StatesDict:
    """
    Complete representation of a biomechanical model's state and parameters.
    
    This is the primary data structure in biosym, combining time-varying states
    with time-invariant constants to provide a complete model representation.
    It serves as the main interface for simulations, optimizations, and analysis.
    
    Attributes
    ----------
    states : States
        Time-varying state variables (positions, velocities, activations, etc.).
        Contains trajectory data with shape (n_timesteps, n_dofs_per_field).
    constants : States
        Time-invariant model parameters (masses, lengths, gains, etc.).
        Contains constant values used throughout the simulation/optimization.
        
    Notes
    -----
    - Primary data structure for all biosym operations
    - Supports vectorized operations for efficient batch processing
    - Compatible with JAX transformations (jit, grad, vmap)
    - Essential for optimal control problem formulation
    - Provides unified interface for states and parameters
    
    Examples
    --------
    >>> # Create a StatesDict for a 10-timestep trajectory
    >>> states = States(model=jnp.zeros((10, 6)), gc_model=jnp.zeros((10, 3)))
    >>> constants = States(model=jnp.ones((20,)), gc_model=jnp.ones((5,)))
    >>> states_dict = StatesDict(states=states, constants=constants)
    >>> print(len(states_dict))  # Returns 10 (number of timesteps)
    """
    states: States
    constants: States

    def __getitem__(self, index):
        """
        Extract data at specific time indices.
        
        Parameters
        ----------
        index : int, slice, or array-like
            Index or indices to extract from the time dimension.
            
        Returns
        -------
        StatesDict
            New StatesDict containing only the specified time points.
            Constants are preserved (first element taken if multi-dimensional).
            
        Notes
        -----
        Essential for trajectory analysis and extracting specific time points.
        Constants are handled specially to maintain their time-invariant nature.
        """

        def slice_fn(x):
            return x[index] if isinstance(x, jnp.ndarray) else x

        sliced_states = jax.tree_util.tree_map(slice_fn, self.states)
        sliced_constants = jax.tree_util.tree_map(lambda x: x[0, :] if x.ndim > 1 else x, self.constants)
        return StatesDict(states=sliced_states, constants=sliced_constants)

    def multiply(self, other):
        """
        Element-wise multiplication with another StatesDict or scalar.
        
        Parameters
        ----------
        other : StatesDict, float, or int
            Value to multiply with. If StatesDict, performs element-wise 
            multiplication. If scalar, multiplies all arrays by the scalar.
            
        Returns
        -------
        StatesDict
            New StatesDict with multiplied values.
            
        Notes
        -----
        Used for scaling operations, sensitivity analysis, and mathematical
        operations on state trajectories. Preserves the structure of both
        states and constants.
        """
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        if isinstance(other, StatesDict):
            return jax.tree_util.tree_map(lambda x, y: x * y, self, other)

    def size(self):
        """
        Calculate total number of elements in states and constants.
        
        Returns
        -------
        int
            Total number of scalar elements across all arrays in the StatesDict.
            
        Notes
        -----
        Includes both time-varying states and time-invariant constants.
        Useful for memory estimation and optimization problem sizing.
        """
        return sum(x.size for x in jax.tree_util.tree_leaves(self))

    def __len__(self):
        """
        Get the number of time steps in the trajectory.
        
        Returns
        -------
        int
            Number of time steps, determined by the first dimension of state arrays.
            Returns 1 for single-timestep data.
            
        Notes
        -----
        Based on the shape of the model field in states, which is assumed
        to be the primary state variable defining trajectory length.
        """
        return self.states.model.shape[0] if self.states.model.ndim > 1 else 1

    def flat_at(self, idx):
        """
        Get a flattened view of all data at a specific time index.
        
        Parameters
        ----------
        idx : int
            Time index to extract and flatten.
            
        Returns
        -------
        jnp.ndarray
            1D array containing all state values at the specified time point.
            
        Notes
        -----
        Useful for interfacing with optimization algorithms that require
        flat parameter vectors at specific time points.
        """
        curr_state = self[idx]
        flat_states = jax.tree_util.tree_leaves(curr_state)
        return jnp.concatenate([x[idx] for x in flat_states if isinstance(x, jnp.ndarray)], axis=0)

    def replace_vector(self, section: Literal["states", "constants"], name: str, value: jnp.ndarray) -> "StatesDict":
        """
        Replace a specific field with a new value.
        
        Parameters
        ----------
        section : {"states", "constants"}
            Which section to modify (time-varying states or constants).
        name : str
            Name of the field to replace (e.g., "model", "gc_model").
        value : jnp.ndarray
            New array value to assign to the field.
            
        Returns
        -------
        StatesDict
            New StatesDict with the specified field updated.
            
        Raises
        ------
        ValueError
            If the specified field name doesn't exist in the section.
            
        Notes
        -----
        Creates a new instance rather than modifying in-place, following
        functional programming principles. Essential for updating specific
        model components during optimization or simulation.
        """
        target = getattr(self, section)
        if not hasattr(target, name):
            raise ValueError(f"'{name}' is not a field of {section}.")
        updated = replace(target, **{name: value})
        return replace(self, **{section: updated})

    def add(self, scalar):
        """
        Add a scalar value to all arrays in the StatesDict.
        
        Parameters
        ----------
        scalar : float or int
            Scalar value to add to all arrays.
            
        Returns
        -------
        StatesDict
            New StatesDict with the scalar added to all array elements.
            
        Notes
        -----
        Useful for bias operations and mathematical transformations.
        Applied to both states and constants uniformly.
        """
        return jax.tree_util.tree_map(lambda x: x + scalar, self)

    def __str__(self):
        s0 = "StatesDict:\n\tStates:\n"
        s1 = f"\t\tmodel: {self.states.model.shape}\n"
        s2 = f"\t\tgc_model: {self.states.gc_model.shape}\n"
        s3 = f"\t\tactuator_model: {self.states.actuator_model.shape}\n"
        s4 = f"\t\th: {self.states.h.shape if self.states.h is not None else 'None'}\n"
        s5 = "\tConstants:\n"
        s6 = f"\t\tmodel: {self.constants.model.shape}\n"
        s7 = f"\t\tgc_model: {self.constants.gc_model.shape}\n"
        s8 = f"\t\tactuator_model: {self.constants.actuator_model.shape}\n"
        return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8


def stack_dataclasses(instances):
    """
    Stack multiple StatesDict instances into a single batched instance.
    
    This function combines individual time points or single-state instances
    into a batched trajectory representation. Essential for creating
    optimization problems from collections of states.
    
    Parameters
    ----------
    instances : list or tuple
        List of StatesDict instances to stack along the time dimension.
        All instances must have compatible shapes for stacking.
        
    Returns
    -------
    StatesDict
        New StatesDict with states stacked along the first dimension.
        Constants are taken from the first instance (assumed identical).
        
    Raises
    ------
    ValueError
        If the input list is empty.
    TypeError
        If input is not a list or tuple of dataclass instances.
        
    Notes
    -----
    - Only the first instance's constants are used (others assumed identical)
    - Essential for trajectory construction in optimal control problems
    - Used to convert lists of individual states into batch format
    
    Examples
    --------
    >>> # Stack 3 individual states into a trajectory
    >>> state_list = [state1, state2, state3]
    >>> trajectory = stack_dataclasses(state_list)
    >>> print(len(trajectory))  # Returns 3
    """
    if not instances:
        raise ValueError("Cannot stack an empty list")
    if type(instances) not in [list, tuple]:
        raise TypeError("Input must be a list of dataclass instances")

    # jax.tree_util.tree_map applies a function over the corresponding fields
    dict_0 = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *instances)
    constants = instances[0].constants
    return StatesDict(dict_0.states, constants)


def reduce_dataclasses(instances, fn=None, weights=None):
    """
    Apply reduction operations across multiple dataclass instances.
    
    This function combines multiple StatesDict instances using mathematical
    operations like sum, mean, or max. Useful for ensemble operations,
    weighted averaging, and statistical analysis of trajectories.
    
    Parameters
    ----------
    instances : list
        List of StatesDict instances to reduce.
    fn : callable, optional
        Reduction function to apply (e.g., jnp.mean, jnp.sum, jnp.max).
        If None, returns weighted instances without reduction.
    weights : list, optional
        Weighting factors for each instance. If None, uses equal weights.
        Must match the length of instances if provided.
        
    Returns
    -------
    StatesDict or list
        If fn is provided, returns a single reduced StatesDict.
        If fn is None, returns list of weighted instances.
        
    Raises
    ------
    ValueError
        If instances is empty or weights length doesn't match instances.
        
    Notes
    -----
    - Weights are applied before reduction operation
    - Useful for ensemble methods and statistical analysis
    - Supports any JAX-compatible reduction function
    
    Examples
    --------
    >>> # Compute weighted average of multiple trajectories
    >>> trajectories = [traj1, traj2, traj3]
    >>> weights = [0.5, 0.3, 0.2]
    >>> avg_traj = reduce_dataclasses(trajectories, jnp.mean, weights)
    """
    if not instances:
        raise ValueError("Cannot reduce an empty list")
    if weights is None:
        weights = [1] * len(instances)
    else:
        if len(weights) != len(instances):
            raise ValueError("Weights must match the number of instances")
        for i, weight in enumerate(weights):
            instances[i] = instances[i].multiply(weight)
    if fn is None:
        return instances

    # Stack all corresponding fields: shape becomes (N, ...)
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *instances)

    # Apply reduction function along axis 0
    return jax.tree_util.tree_map(lambda x: fn(x, axis=0), stacked)


def dict_to_dataclass(states_dict):
    """
    Convert a dictionary representation to StatesDict dataclass.
    
    This function provides backwards compatibility and conversion from
    dictionary-based state representations to the structured dataclass format.
    Handles missing fields gracefully by setting them to None.
    
    Parameters
    ----------
    states_dict : dict
        Dictionary containing nested state and constant data.
        Expected structure: {"states": {...}, "constants": {...}}
        
    Returns
    -------
    StatesDict
        Converted dataclass representation with proper structure.
        Missing fields are set to None for graceful handling.
        
    Notes
    -----
    - Provides conversion from legacy dictionary formats
    - Handles missing fields gracefully with None defaults
    - Essential for data loading and compatibility layers
    - Nested dictionary access is handled safely
    """

    def get_value(d, *keys):
        """Safely get a nested value or return None."""
        for key in keys:
            d = d.get(key, None)
            if d is None:
                return None
        return d

    states = States(
        model=get_value(states_dict, "states", "model"),
        gc_model=get_value(states_dict, "states", "gc_model"),
        actuator_model=get_value(states_dict, "states", "actuator_model"),
        h=get_value(states_dict, "states", "h"),
    )
    constants = States(
        model=get_value(states_dict, "constants", "model"),
        gc_model=get_value(states_dict, "constants", "gc_model"),
        actuator_model=get_value(states_dict, "constants", "actuator_model"),
        h=get_value(states_dict, "constants", "h"),
    )
    return StatesDict(states=states, constants=constants)
