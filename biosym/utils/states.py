import jax
import jax.numpy as jnp
import os
from flax.struct import dataclass

@dataclass
class States:
    model: jnp.ndarray
    gc_model: jnp.ndarray
    actuator_model: jnp.ndarray
    h: jnp.ndarray

@dataclass
class Globals:
    dur: jnp.ndarray
    speed: jnp.ndarray

@dataclass
class StatesDict:
    states: States
    constants: States
    globals: Globals

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
    globals_ = Globals(
        dur=get_value(states_dict, 'globals', 'dur'),
        speed=get_value(states_dict, 'globals', 'speed')
    )
    return StatesDict(states=states, constants=constants, globals=globals_)
