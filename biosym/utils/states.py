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


def _resample_array(arr, N):
    if arr is None:
        return None
    arr = jnp.asarray(arr)
    # If ndim is 1, treat it as a single node (length 1)
    if arr.ndim == 1:
        return jnp.tile(arr, (N, 1))
    
    M = arr.shape[0]
    if M <= 1:
        return jnp.tile(arr, (N, 1)) if arr.ndim > 1 else jnp.tile(arr[jnp.newaxis, :], (N, 1))
        
    # Standard resampling using interpolation
    xp = jnp.linspace(0.0, 1.0, M)
    x = jnp.linspace(0.0, 1.0, N)
    
    # We interpolate each coordinate/dimension independently
    resampled = jnp.stack([jnp.interp(x, xp, arr[:, i]) for i in range(arr.shape[1])], axis=1)
    return resampled


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

    # Metadata, names, and constants
    names: list | None = None
    metadata: dict | None = None
    constants: Any = None

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
        constants: Any = None,
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
        # Make a list of the names, for each set attribute
        n = []
        if q is not None: n.append('q')
        if qd is not None: n.append('qd')
        if qdd is not None: n.append('qdd')
        if tau is not None: n.append('tau')
        if ext_forces is not None: n.append('ext_forces')
        if ext_torques is not None: n.append('ext_torques')
        if gc_model is not None: n.append('gc_model')
        if actuator_model is not None: n.append('actuator_model')
        if h is not None: n.append('h')
        object.__setattr__(self, "names", n)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "constants", constants)

    @property
    def states(self):
        """Self-reference property for backward compatibility with StatesDict wrapper."""
        return self

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
        for field_name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model", "h", "constants"]:
            if not hasattr(self, field_name):
                object.__setattr__(self, field_name, None)
        if not hasattr(self, "names"):
            object.__setattr__(self, "names", None)
        if not hasattr(self, "metadata"):
            object.__setattr__(self, "metadata", None)

    def replace(self, name=None, value=None, **kwargs) -> "States":
        """Replace fields while keeping physical components structured."""
        if name is not None and value is not None:
            return dataclasses.replace(self, **{name:value})
        else:
            return dataclasses.replace(self, **kwargs)

    def __str__(self):
        parts = []
        for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model", "h"]:
            val = getattr(self, name)
            if val is not None:
                parts.append(f"{name}={val.shape if hasattr(val, 'shape') else type(val)}")
        return f"States({', '.join(parts)})"

    def __repr__(self):
        return self.__str__()

    def size(self):
        return sum(x.size for x in jax.tree_util.tree_leaves(self))

    def flatten(self):
        flat_states = jax.tree_util.tree_leaves(self)
        return jnp.concatenate([x.flatten() if isinstance(x, jnp.ndarray) else x for x in flat_states], axis=0)

    def filter(self, names: list[str]) -> "States":
        """Filter states by name."""
        if names=="model": names=['q','qd','qdd','tau','ext_forces','ext_torques']
        return States(**{name: getattr(self, name) for name in names})

    @property
    def model(self) -> jnp.ndarray:
        """Flat model vector representation for backward compatibility."""
        return self.filter('model').to_array()

    def resample(self, N: int) -> "States":
        """Resample the States object to a new number of nodes N."""
        kwargs = {}
        kwargs["names"] = self.names
        kwargs["metadata"] = self.metadata
        
        fields = ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model"]
        for field in fields:
            val = getattr(self, field)
            if val is None:
                kwargs[field] = None
            else:
                kwargs[field] = _resample_array(val, N)
                
        # Handle h specifically to adjust it
        if self.h is None:
            kwargs["h"] = None
        else:
            h_arr = jnp.asarray(self.h)
            if h_arr.shape[-1] == 0:
                kwargs["h"] = jnp.zeros((N, 0))
            else:
                old_sum = jnp.sum(h_arr)
                resampled_h = _resample_array(h_arr, N)
                new_sum = jnp.sum(resampled_h)
                kwargs["h"] = jnp.where(new_sum > 0, resampled_h * (old_sum / new_sum), resampled_h)
                    
        return States(**kwargs)

    def to_array(self):
        return jnp.concatenate([getattr(self, name) for name in self.names if getattr(self, name) is not None], axis=-1)

    def __getitem__(self, index):
        if type(index)==str:
            return getattr(self, index)
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
        aux_data = (
            tuple(active_fields),
            _freeze(getattr(self, "names", None)),
            _freeze(getattr(self, "metadata", None)),
            _freeze(getattr(self, "constants", None))
        )
        return tuple(children), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        active_fields, names, metadata, constants = aux_data
        names = _thaw(names)
        metadata = _thaw(metadata)
        constants = _thaw(constants)

        kwargs = {
            "names": names,
            "metadata": metadata,
            "constants": constants,
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
        **kwargs
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

    def replace(self, name=None, value=None, **kwargs) -> "Constants":
        """Replace fields while keeping model and separate physical components in sync."""
        #for field, value in updates.items():
        #    return dataclasses.replace(self, **{field:value})
        if name is not None and value is not None:
            return dataclasses.replace(self, **{name:value})
        else:
            return dataclasses.replace(self, **kwargs)

    def __str__(self):
        parts = []
        for name in ["g", "mass", "inertia", "com", "offset", "gc_model", "actuator_model"]:
            val = getattr(self, name)
            if val is not None:
                parts.append(f"{name}={val.shape if hasattr(val, 'shape') else type(val)}")
        return f"Constants({', '.join(parts)})"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, index):
        if type(index)==str:
            return getattr(self, index)

    def multiply(self, other):
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        raise NotImplementedError("biosym.utils.states.Constants.multiply.notfloat")
    
    def filter(self, names: list[str]) -> "Constants":
        """Filter states by name."""
        if names=="model": names=['g','mass','inertia','com','offset']
        return Constants(**{name: getattr(self, name) for name in names})

    @property
    def model(self) -> jnp.ndarray:
        """Flat model vector representation for backward compatibility."""
        return self.filter('model').to_array()


    def tree_flatten(self):
        active_fields = []
        children = []
        for name in ["g", "mass", "inertia", "com", "offset", "gc_model", "actuator_model"]:
            val = getattr(self, name)
            if val is not None:
                active_fields.append(name)
                children.append(val)
        aux_data = (tuple(active_fields),)
        return tuple(children), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        active_fields, = aux_data
        kwargs = {}
        for name, val in zip(active_fields, children):
            kwargs[name] = val
        return cls(**kwargs)
                
    def flatten(self):
        return jnp.concatenate([x.flatten() if isinstance(x, jnp.ndarray) else x for x in jax.tree_util.tree_leaves(self)], axis=0)

    def to_array(self):
        return jnp.concatenate([x if isinstance(x, jnp.ndarray) else x for x in jax.tree_util.tree_leaves(self)], axis=0)

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


def concatenate(states_list: list[States]) -> States:
    """Concatenate a list of States objects along the time dimension."""
    if not states_list:
        raise ValueError("Cannot concatenate an empty list of States.")
    
    kwargs = {}
    kwargs["names"] = states_list[0].names
    kwargs["metadata"] = states_list[0].metadata
    
    fields = ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model", "h"]
    for field in fields:
        vals = [getattr(s, field) for s in states_list]
        if all(v is None for v in vals):
            kwargs[field] = None
            continue
            
        processed_vals = []
        for v in vals:
            if v is None:
                continue
            arr = jnp.asarray(v)
            if arr.ndim == 1:
                arr = jnp.expand_dims(arr, axis=0)
            processed_vals.append(arr)
            
        if processed_vals:
            kwargs[field] = jnp.concatenate(processed_vals, axis=0)
        else:
            kwargs[field] = None
            
    return States(**kwargs)


def resample(states_obj: States, N: int) -> States:
    """Resample a States object to a new number of nodes N."""
    if not isinstance(states_obj, States):
        raise TypeError(f"Expected States object, got {type(states_obj)}")
    return states_obj.resample(N)

