"""
Core data structures for biomechanical modeling and optimization in biosym.

This module defines the fundamental data structures used throughout the biosym
framework for representing states, constants, and global parameters in
biomechanical simulations and optimal control problems. These structures are
built on JAX for efficient computation and automatic differentiation.

Key Features:
- JAX-compatible data structures for efficient computation
- Decoupled physical state variables (q, dq, ddq, tau, etc.) and constants (g, masses, etc.)
- Automatic differentiation support through custom PyTree registration
- Vectorized operations for batch processing
- Indexing and slicing operations for trajectory analysis
- Backward-compatible property interface for legacy model slicing
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Any

import jax
import jax.numpy as jnp
import numpy as np


class FrozenDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))
    def __repr__(self):
        return f"FrozenDict({super().__repr__()})"


def _freeze(x):
    if isinstance(x, dict):
        return FrozenDict({k: _freeze(v) for k, v in x.items()})
    if isinstance(x, list):
        return tuple(_freeze(v) for v in x)
    return x


def _thaw(x):
    if isinstance(x, FrozenDict):
        return {k: _thaw(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return [_thaw(v) for v in x]
    return x


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class States:
    """
    Time-varying state variables for biomechanical models.
    
    This dataclass represents the dynamic state of a biomechanical system
    at one or more time points. Natively stores physical components in separate arrays,
    reducing tracing overhead during symbolic JAX lambdification.
    """
    # Separate physical vectors (optional, defaulting to None)
    q: jnp.ndarray | None = None
    qd: jnp.ndarray | None = None
    qdd: jnp.ndarray | None = None
    tau: jnp.ndarray | None = None
    ext_forces: jnp.ndarray | None = None
    ext_torques: jnp.ndarray | None = None

    gc_model: jnp.ndarray = None
    actuator_model: jnp.ndarray = None
    h: jnp.ndarray = None

    # Metadata and names
    names: list | None = None
    metadata: dict | None = None

    def __init__(
        self,
        q: jnp.ndarray | None = None,
        qd: jnp.ndarray | None = None,
        qdd: jnp.ndarray | None = None,
        tau: jnp.ndarray | None = None,
        ext_forces: jnp.ndarray | None = None,
        ext_torques: jnp.ndarray | None = None,
        gc_model: jnp.ndarray | None = None,
        actuator_model: jnp.ndarray | None = None,
        h: jnp.ndarray | None = None,
        names: list | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "qd", qd)
        object.__setattr__(self, "qdd", qdd)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "ext_forces", ext_forces)
        object.__setattr__(self, "ext_torques", ext_torques)
        object.__setattr__(self, "gc_model", gc_model)
        object.__setattr__(self, "actuator_model", actuator_model)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "metadata", metadata)


    def __setstate__(self, state):
        # Handle legacy dq/ddq
        if "dq" in state and "qd" not in state:
            state["qd"] = state.pop("dq")
        if "ddq" in state and "qdd" not in state:
            state["qdd"] = state.pop("ddq")

        for k, v in state.items():
            if k in ["slices"]:
                continue
            object.__setattr__(self, k, v)
        for field_name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model", "h"]:
            if not hasattr(self, field_name):
                object.__setattr__(self, field_name, None)
        if not hasattr(self, "names"):
            object.__setattr__(self, "names", None)
        if not hasattr(self, "metadata"):
            object.__setattr__(self, "metadata", None)

    def replace(self, **updates) -> "States":
        """Replace fields while keeping physical components structured."""
        if "dq" in updates:
            updates["qd"] = updates.pop("dq")
        if "ddq" in updates:
            updates["qdd"] = updates.pop("ddq")

        return dataclasses.replace(self, **updates)

    def __str__(self):
        parts = []
        for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model", "h"]:
            val = getattr(self, name)
            if val is not None:
                parts.append(f"{name}={val.shape if hasattr(val, 'shape') else type(val)}")
        return f"States({', '.join(parts)})"

    def size(self):
        return sum(x.size for x in jax.tree_util.tree_leaves(self))

    def flatten(self):
        flat_states = jax.tree_util.tree_leaves(self)
        return jnp.concatenate([x.flatten() if isinstance(x, jnp.ndarray) else x for x in flat_states], axis=0)

    def __getitem__(self, index):
        def slice_fn(x):
            if isinstance(x, jnp.ndarray):
                if x.shape[0] == 0:
                    return x
                return x[index]
            return x

        return jax.tree_util.tree_map(slice_fn, self)
    
    def __len__(self):
        for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model", "h"]:
            val = getattr(self, name)
            if val is not None and hasattr(val, "ndim"):
                return val.shape[0] if val.ndim > 1 else 1
        return 1

    def tree_flatten(self):
        active_fields = []
        children = []
        for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model", "h"]:
            val = getattr(self, name)
            if val is not None:
                active_fields.append(name)
                children.append(val)
        aux_data = (tuple(active_fields), _freeze(self.names), _freeze(self.metadata))
        return tuple(children), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        active_fields, names, metadata = aux_data
        names = _thaw(names)
        metadata = _thaw(metadata)

        kwargs = {
            "names": names,
            "metadata": metadata,
        }
        for name, val in zip(active_fields, children):
            kwargs[name] = val
        return cls(**kwargs)



@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class Constants:
    """
    Time-invariant model parameters and constants.
    
    Stores physical constants (gravity, masses, inertia, etc.) natively in separate arrays,
    while maintaining flat model backward compatibility.
    """
    # Separate physical constants
    g: jnp.ndarray = None
    mass: jnp.ndarray = None
    inertia: jnp.ndarray = None
    com: jnp.ndarray = None
    offset: jnp.ndarray = None

    gc_model: jnp.ndarray = None
    actuator_model: jnp.ndarray = None


    def __init__(
        self,
        g: jnp.ndarray = None,
        mass: jnp.ndarray = None,
        inertia: jnp.ndarray = None,
        com: jnp.ndarray = None,
        offset: jnp.ndarray = None,
        gc_model: jnp.ndarray = None,
        actuator_model: jnp.ndarray = None,

    ):

        if gc_model is None:
            gc_model = jnp.zeros((0,))
        if actuator_model is None:
            actuator_model = jnp.zeros((0,))
        if g is None:
            g = jnp.zeros((0,))
        if mass is None:
            mass = jnp.zeros((0,))
        if inertia is None:
            inertia = jnp.zeros((0,))
        if com is None:
            com = jnp.zeros((0,))
        if offset is None:
            offset = jnp.zeros((0,))

        object.__setattr__(self, "gc_model", gc_model)
        object.__setattr__(self, "actuator_model", actuator_model)
        object.__setattr__(self, "g", g)
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "inertia", inertia)
        object.__setattr__(self, "com", com)
        object.__setattr__(self, "offset", offset)


    def __setstate__(self, state):
        for k, v in state.items():
            object.__setattr__(self, k, v)
        for field_name in ["g", "mass", "inertia", "com", "offset"]:
            if not hasattr(self, field_name):
                object.__setattr__(self, field_name, jnp.zeros((0,)))
        if not hasattr(self, "slices"):
            object.__setattr__(self, "slices", {})
        if not hasattr(self, "names"):
            object.__setattr__(self, "names", None)
        if not hasattr(self, "metadata"):
            object.__setattr__(self, "metadata", None)

    def replace(self, **updates) -> "Constants":
        """Replace fields while keeping model and separate physical components in sync."""
        #for field, value in updates.items():
        #    return dataclasses.replace(self, **{field:value})
        return dataclasses.replace(self, **updates)

    def __str__(self):
        parts = []
        for name in ["g", "mass", "inertia", "com", "offset", "gc_model", "actuator_model"]:
            val = getattr(self, name)
            if val is not None:
                parts.append(f"{name}={val.shape if hasattr(val, 'shape') else type(val)}")
        return f"Constants({', '.join(parts)})"

    def multiply(self, other):
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        raise NotImplementedError("biosym.utils.states.Constants.multiply.notfloat")

    def tree_flatten(self):
        children = (self.g, self.mass, self.inertia, self.com, self.offset, self.gc_model, self.actuator_model)
        aux_data = (self.slices, _freeze(self.names), _freeze(self.metadata))
        return children, aux_data

    def flatten(self):
        return jnp.concatenate([x.flatten() if isinstance(x, jnp.ndarray) else x for x in jax.tree_util.tree_leaves(self)], axis=0)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        if len(aux_data) == 1:
            slices, = aux_data
            names = None
            metadata = None
        else:
            slices, names, metadata = aux_data

        names = _thaw(names)
        metadata = _thaw(metadata)

        g, masses, inertia, com, offset, gc_model, actuator_model = children
                           
        return cls(
            gc_model=gc_model, actuator_model=actuator_model,
            g=g, masses=masses, inertia=inertia, com=com, offset=offset,
            slices=slices, names=names, metadata=metadata
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Globals:
    """
    Global optimization parameters for optimal control problems.
    """
    dur: jnp.ndarray = field(default_factory=lambda: jnp.zeros((1,)))
    speed: jnp.ndarray = field(default_factory=lambda: jnp.zeros((1,)))

    def replace(self, **updates) -> "Globals":
        return dataclasses.replace(self, **updates)

    def size(self):
        return sum(x.size for x in jax.tree_util.tree_leaves(self))

    def multiply(self, other):
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        raise NotImplementedError("biosym.utils.states.Globals.multiply.notfloat")

    def tree_flatten(self):
        return (self.dur, self.speed), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dur, speed = children
        return cls(dur=dur, speed=speed)

def stack_dataclasses(instances):
    if not instances:
        raise ValueError("Cannot stack an empty list")
    if type(instances) not in [list, tuple]:
        raise TypeError("Input must be a list of dataclass instances")

    # Stack States and Constants individually to avoid structural/metadata mismatch on StatesDict wrappers
    stacked_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[inst.states for inst in instances])
    constants = instances[0].constants
    
    names = instances[0].names if hasattr(instances[0], "names") else None
    metadata = instances[0].metadata if hasattr(instances[0], "metadata") else None
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return StatesDict(stacked_states, constants, names=names, metadata=metadata)


def reduce_dataclasses(instances, fn=None, weights=None):
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

    # Reduce States and Constants individually to avoid structural/metadata mismatch on StatesDict wrappers
    stacked_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[inst.states for inst in instances])
    reduced_states = jax.tree_util.tree_map(lambda x: fn(x, axis=0), stacked_states)
    
    constants = instances[0].constants
    names = instances[0].names if hasattr(instances[0], "names") else None
    metadata = instances[0].metadata if hasattr(instances[0], "metadata") else None
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return StatesDict(states=reduced_states, constants=constants, names=names, metadata=metadata)


def dict_to_dataclass(states_dict):
    def get_value(d, *keys):
        for key in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(key, None)
            if d is None:
                return None
        return d

    qd_val = get_value(states_dict, "states", "qd")
    if qd_val is None:
        qd_val = get_value(states_dict, "states", "dq")

    qdd_val = get_value(states_dict, "states", "qdd")
    if qdd_val is None:
        qdd_val = get_value(states_dict, "states", "ddq")

    states = States(
        q=get_value(states_dict, "states", "q"),
        qd=qd_val,
        qdd=qdd_val,
        tau=get_value(states_dict, "states", "tau"),
        ext_forces=get_value(states_dict, "states", "ext_forces"),
        ext_torques=get_value(states_dict, "states", "ext_torques"),
        gc_model=get_value(states_dict, "states", "gc_model"),
        actuator_model=get_value(states_dict, "states", "actuator_model"),
        h=get_value(states_dict, "states", "h"),
        names=get_value(states_dict, "states", "names"),
        metadata=get_value(states_dict, "states", "metadata"),
    )
    constants = Constants(
        gc_model=get_value(states_dict, "constants", "gc_model"),
        actuator_model=get_value(states_dict, "constants", "actuator_model"),
        g=get_value(states_dict, "constants", "g"),
        masses=get_value(states_dict, "constants", "masses"),
        inertia=get_value(states_dict, "constants", "inertia"),
        com=get_value(states_dict, "constants", "com"),
        offset=get_value(states_dict, "constants", "offset"),
        slices=get_value(states_dict, "constants", "slices"),
        names=get_value(states_dict, "constants", "names"),
        metadata=get_value(states_dict, "constants", "metadata"),
    )
    
    sd_names = get_value(states_dict, "names")
    sd_metadata = get_value(states_dict, "metadata")
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return StatesDict(states=states, constants=constants, names=sd_names, metadata=sd_metadata)


def get_states_offsets(states) -> dict:
    offsets = {}
    current = 0
    for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model", "h"]:
        val = getattr(states, name)
        if val is not None:
            offsets[name] = current
            current += val.size
        else:
            offsets[name] = None
    return offsets
