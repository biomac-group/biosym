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
        names: list | None = None,
        metadata: dict | None = None,
        **kwargs,
    ):
        if "h" in kwargs:
            # TODO(deprecation): Remove states.h support in 0.3.0
            import warnings
            warnings.warn(
                "Passing 'h' to States is deprecated. 'h' has been moved to Globals.",
                DeprecationWarning,
                stacklevel=2,
            )

        object.__setattr__(self, "q", q)
        object.__setattr__(self, "qd", qd)
        object.__setattr__(self, "qdd", qdd)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "ext_forces", ext_forces)
        object.__setattr__(self, "ext_torques", ext_torques)
        object.__setattr__(self, "gc_model", gc_model)
        object.__setattr__(self, "actuator_model", actuator_model)
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
        object.__setattr__(self, "names", n)
        object.__setattr__(self, "metadata", metadata)

    @property
    def h(self):
        # TODO(deprecation): Remove states.h support in 0.3.0
        import warnings
        warnings.warn(
            "Accessing 'h' from States is deprecated. 'h' has been moved to Globals.",
            DeprecationWarning,
            stacklevel=2,
        )
        return None

    def __setstate__(self, state):
        # Handle legacy dq/ddq
        if "dq" in state and "qd" not in state:
            state["qd"] = state.pop("dq")
        if "ddq" in state and "qdd" not in state:
            state["qdd"] = state.pop("ddq")

        for k, v in state.items():
            if k in ["slices", "h", "constants", "states"]:
                continue
            object.__setattr__(self, k, v)
        for field_name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model"]:
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

    def reshape(self, shape: tuple[int, ...] | int) -> "States":
        """
            Reshape a states-object to the requested shape. The last dimension is always retained, 
            therefore the requested shape should only contain the first two desired dimensions.
        """
        if isinstance(shape, int):
            shape = (shape,)

        return self.replace(
            q=self.q.reshape(shape + self.q.shape[-1:] if self.q is not None else None),
            qd=self.qd.reshape(shape + self.qd.shape[-1:] if self.qd is not None else None),
            qdd=self.qdd.reshape(shape + self.qdd.shape[-1:] if self.qdd is not None else None),
            tau=self.tau.reshape(shape + self.tau.shape[-1:] if self.tau is not None else None),
            ext_forces=self.ext_forces.reshape(shape + self.ext_forces.shape[-1:] if self.ext_forces is not None else None),
            ext_torques=self.ext_torques.reshape(shape + self.ext_torques.shape[-1:] if self.ext_torques is not None else None),
            gc_model=self.gc_model.reshape(shape + self.gc_model.shape[-1:] if self.gc_model is not None else None),
            actuator_model=self.actuator_model.reshape(shape + self.actuator_model.shape[-1:] if self.actuator_model is not None else None),
        )
    
    def shape(self):
        """Returns the non-model-axis shape for the states-object.
        """
        return self.q.shape[:-1]

    def squeeze(self):
        """
        Squeeze the states-object.
        """
        return self.replace(
            q=self.q.squeeze() if self.q is not None else None,
            qd=self.qd.squeeze() if self.qd is not None else None,
            qdd=self.qdd.squeeze() if self.qdd is not None else None,
            tau=self.tau.squeeze() if self.tau is not None else None,
            ext_forces=self.ext_forces.squeeze() if self.ext_forces is not None else None,
            ext_torques=self.ext_torques.squeeze() if self.ext_torques is not None else None,
            gc_model=self.gc_model.squeeze() if self.gc_model is not None else None,
            actuator_model=self.actuator_model.squeeze() if self.actuator_model is not None else None,
        )

    def __str__(self):
        parts = []
        for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model"]:
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
        for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model"]:
            val = getattr(self, name)
            if val is not None and hasattr(val, "ndim"):
                return val.shape[0] if val.ndim > 1 else 1
        return 1

    def get_n_states(self):
        return self.to_array().shape[-1]

    def tree_flatten(self):
        active_fields = []
        children = []
        for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model"]:
            val = getattr(self, name)
            if val is not None:
                active_fields.append(name)
                children.append(val)
        aux_data = (tuple(active_fields), _freeze(getattr(self, "names", None)), _freeze(getattr(self, "metadata", None)))
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
    
    Stores physical constants (gravity, masses, inertia, etc.) natively in separate arrays.
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
    h: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))

    def replace(self, **updates) -> "Globals":
        return dataclasses.replace(self, **updates)

    def size(self):
        return sum(x.size for x in jax.tree_util.tree_leaves(self))

    def multiply(self, other):
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        raise NotImplementedError("biosym.utils.states.Globals.multiply.notfloat")

    def tree_flatten(self):
        return (self.dur, self.speed, self.h), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dur, speed, h = children
        return cls(dur=dur, speed=speed, h=h)


def get_states_offsets(states) -> dict:
    offsets = {}
    current = 0
    for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model"]:
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
    
    fields = ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model"]
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


def sum(obj, weights: jnp.ndarray | None = None):
    """Sum a pytree (States, Constants, Globals, …) over axis 0.

    Parameters
    ----------
    obj : any JAX-registered pytree
        The object to reduce.  All array leaves are summed along axis 0.
    weights : array_like of shape ``(N,)``, optional
        Per-node weights.  Each leaf is multiplied by the weight vector
        (broadcast to the leaf shape) before summing, giving a
        **weighted sum**.

    Returns
    -------
    Same type as *obj*, with the node axis removed from every leaf.
    """
    def _reduce(x):
        if not isinstance(x, jnp.ndarray) or x.ndim == 0:
            return x
        if weights is not None:
            w = jnp.asarray(weights)
            for _ in range(x.ndim - 1):
                w = w[..., jnp.newaxis]
            return jnp.sum(x * w, axis=0, keepdims=True)
        return jnp.sum(x, axis=0, keepdims=True)

    return jax.tree_util.tree_map(_reduce, obj)


def mean(obj, weights: jnp.ndarray | None = None):
    """Mean of a pytree (States, Constants, Globals, …) over axis 0.

    Parameters
    ----------
    obj : any JAX-registered pytree
        The object to reduce.  All array leaves are averaged along axis 0.
    weights : array_like of shape ``(N,)``, optional
        Per-node weights.  The result is ``Σ wᵢ·xᵢ / Σ wᵢ`` for each leaf
        (weighted mean).  Implemented by normalising *weights* and delegating
        to :func:`sum`.

    Returns
    -------
    Same type as *obj*, with the node axis removed from every leaf.
    """
    if weights is not None:
        w = jnp.asarray(weights)
        return sum(obj, weights=w / jnp.sum(w))

    def _reduce(x):
        if not isinstance(x, jnp.ndarray) or x.ndim == 0:
            return x
        return jnp.mean(x, axis=0, keepdims=True)

    return jax.tree_util.tree_map(_reduce, obj)
