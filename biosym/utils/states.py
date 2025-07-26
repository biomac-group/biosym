import jax
import jax.numpy as jnp
import os
from flax.struct import dataclass
from dataclasses import replace
from typing import Literal

@dataclass
class States:
    model: jnp.ndarray
    gc_model: jnp.ndarray = jnp.zeros((0,))  # Default to empty array if not provided
    # Assuming actuator_model and
    actuator_model: jnp.ndarray = jnp.zeros((0,))  # Default to empty array if not provided
    h: jnp.ndarray = jnp.zeros((0,))  # Default to empty array if not provided

    def __str__(self):
        return f"States(model={self.model.shape}, gc_model={self.gc_model.shape}, actuator_model={self.actuator_model.shape}, h={self.h.shape if self.h is not None else 'None'})"

    def size(self):
        """
        Get the number of elements in the StatesDict.
        :return: Total number of elements across all fields.    
        """
        return sum(x.size for x in jax.tree_util.tree_leaves(self))
    
    def flatten(self):
        """
        Flatten the States dataclass into a single array.
        :return: A flattened array of all fields.
        """
        flat_states = jax.tree_util.tree_leaves(self)
        return jnp.concatenate([x.flatten() if isinstance(x, jnp.ndarray) else x for x in flat_states], axis=0)

@dataclass
class Constants:
    model: jnp.ndarray
    gc_model: jnp.ndarray = jnp.zeros((0,))  # Default to empty array if not provided
    actuator_model: jnp.ndarray = jnp.zeros((0,))  # Default to empty array if not provided

    def __str__(self):
        return f"Constants(model={self.model.shape}, gc_model={self.gc_model.shape}, actuator_model={self.actuator_model.shape})"

    def multiply(self, other):
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        else:
            raise NotImplementedError('biosym.utils.states.Constants.multiply.notfloat')

@dataclass
class Globals:
    dur: jnp.ndarray = jnp.zeros((1,))  # Default to empty array if not provided
    speed: jnp.ndarray = jnp.zeros((1,))  # Default to empty array if not provided

    def size(self):
        """
        Get the size of the Globals dataclass.
        :return: Total number of elements in the Globals dataclass.
        """
        return sum(x.size for x in jax.tree_util.tree_leaves(self))

    def multiply(self, other):
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        else:
            raise NotImplementedError('biosym.utils.states.Constants.multiply.notfloat')

@dataclass
class StatesDict:
    states: States
    constants: States

    def __getitem__(self, index):
        """
        Get the state at a specific index.
        :param index: Index to access the state.
        :return: A new StatesDict with the state at the specified index.
        """
        def slice_fn(x):
            return x[index] if isinstance(x, jnp.ndarray) else x
        sliced_states = jax.tree_util.tree_map(slice_fn, self.states)
        sliced_constants = jax.tree_util.tree_map(lambda x: x[0,:] if x.ndim > 1 else x, self.constants)
        return StatesDict(states=sliced_states, constants=sliced_constants)

    def multiply(self, other):
        """
        Multiply two by a StatesDict instances element-wise or by a scalar.
        :param other: A scalar or another StatesDict instance.
        :return: A new StatesDict with multiplied values.
        """
        if isinstance(other, (int, float)):
            return jax.tree_util.tree_map(lambda x: x * other, self)
        elif isinstance(other, StatesDict):
            return jax.tree_util.tree_map(lambda x, y: x * y, self, other)

    def size(self):
        """
        Get the number of elements in the StatesDict.
        :return: Total number of elements across all fields.    
        """
        return sum(x.size for x in jax.tree_util.tree_leaves(self))
    
    def __len__(self):
        """ Get the number of states in the StatesDict. 
        Here, we assume that the number of states is determined by the first field (model). """
        return self.states.model.shape[0] if self.states.model.ndim > 0 else 1
    
    def flat_at(self, idx):
        """
        Get a flattened view of the StatesDict at a specific index.
        :param idx: Index to access the state.
        :return: A flattened array of the state at the specified index.
        """
        curr_state = self[idx]
        flat_states = jax.tree_util.tree_leaves(curr_state)
        return jnp.concatenate([x[idx] for x in flat_states if isinstance(x, jnp.ndarray)], axis=0)

    def replace_vector(
        self,
        section: Literal["states", "constants"],
        name: str,
        value: jnp.ndarray
    ) -> "StatesDict":
        target = getattr(self, section)
        if not hasattr(target, name):
            raise ValueError(f"'{name}' is not a field of {section}.")
        updated = replace(target, **{name: value})
        return replace(self, **{section: updated})
    
    def add(self, scalar):
        """
        Add a scalar value to all fields in the StatesDict.
        :param scalar: Scalar value to add.
        :return: A new StatesDict with added values.
        """
        return jax.tree_util.tree_map(lambda x: x + scalar, self)
    
    def __str__(self):
        s0 = f"StatesDict:\n\tStates:\n"
        s1 = f"\t\tmodel: {self.states.model.shape}\n"
        s2 = f"\t\tgc_model: {self.states.gc_model.shape}\n"
        s3 = f"\t\tactuator_model: {self.states.actuator_model.shape}\n"
        s4 = f"\t\th: {self.states.h.shape if self.states.h is not None else 'None'}\n"
        s5 = f"\tConstants:\n"
        s6 = f"\t\tmodel: {self.constants.model.shape}\n"
        s7 = f"\t\tgc_model: {self.constants.gc_model.shape}\n"
        s8 = f"\t\tactuator_model: {self.constants.actuator_model.shape}\n"
        return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8



def stack_dataclasses(instances):
    """Stack a list of flax.struct.dataclass instances into a single instance.
    Only the first occurence of constants will be used
    """
    if not instances:
        raise ValueError("Cannot stack an empty list")
    if not type(instances) in [list, tuple]:
        raise TypeError("Input must be a list of dataclass instances")

    # jax.tree_util.tree_map applies a function over the corresponding fields
    dict_0 = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *instances)
    constants = instances[0].constants
    return StatesDict(dict_0.states, constants)

def reduce_dataclasses(instances, fn=None, weights=None):
    """Apply a reduction function (e.g., mean, sum, max) across a list of dataclass instances."""
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
    Convert a states dictionary to a dataclass.
    If a field is missing, it will be set to None.
    :param states_dict: Dictionary of states.
    :return: A dataclass with states as fields.
    """
    def get_value(d, *keys):
        """Safely get a nested value or return None."""
        for key in keys:
            d = d.get(key, None)
            if d is None:
                return None
        return d

    states = States(
        model=get_value(states_dict, 'states', 'model'),
        gc_model=get_value(states_dict, 'states', 'gc_model'),
        actuator_model=get_value(states_dict, 'states', 'actuator_model'),
        h=get_value(states_dict, 'states', 'h')
    )
    constants = States(
        model=get_value(states_dict, 'constants', 'model'),
        gc_model=get_value(states_dict, 'constants', 'gc_model'),
        actuator_model=get_value(states_dict, 'constants', 'actuator_model'), 
        h=get_value(states_dict, 'constants', 'h')
    )
    return StatesDict(states=states, constants=constants)
