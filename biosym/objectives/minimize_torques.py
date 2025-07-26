import jax
import jax.numpy as jnp
import os
from biosym.objectives.base_objective import BaseObjective
from functools import partial

class Objective(BaseObjective):
    """
        Objective term for minimizing torques. 
    """
    def __init__(self, model, settings, **kwargs):
        """
        Initialize the BaseObjective class with a model and settings.
        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the objective function.
        """
        self.model = model
        self.settings = settings

        if "exponent" in kwargs:
            self.exponent = kwargs["exponent"]
        else:
            self.exponent = 2

        self.speedweighting = kwargs.get("speedweighting", False)

    def _get_info(self):
        """
        Get information about the objective function.
        This method can be overridden in subclasses to provide specific information.
        """
        return {
            'name': os.path.splitext(os.path.basename(__file__))[0],
            'description': 'Objective term for minimizing torques.',
            'required_variables': {'states': ["model"], "constants": ["model"]},
            'idx_int_forces': self.model.forces['idx'],
            'n_int_forces': self.model.forces['n'],
            'exponent': self.exponent,
            'speedweighting': self.speedweighting,
        }

    def get_objfun(self):
        """ :return: The objective function. """
        fun = partial(objfun, settings = self.settings, info=self._get_info())
        return jax.jit(fun)

    def get_gradient(self):
        """ :return: The gradient of the objective function. """
        fun = partial(objfun, settings = self.settings, info=self._get_info())
        return jax.jit(jax.grad(fun, argnums=[0,1]))
    
def objfun(states_list, globals_dict, settings, info):
    """
    Evaluate the objective function.
    
    :param model: biosym model object.
    :param states_list: Dictionary containing the current states.
    :param settings: Settings for the objective function.
    :param info: Information about the objective function.
    :return: The evaluated value of the objective function.
    """
    if globals_dict is not None:
        if info['speedweighting']:
            # Apply speed weighting to the forces
            forces = states_list.states.model[:settings['nnodes'], info['idx_int_forces']:info['idx_int_forces'] + info['n_int_forces']] / settings['nnodes'] * globals_dict.dur / globals_dict.speed
        else:
            forces = states_list.states.model[:settings['nnodes'],info['idx_int_forces']:info['idx_int_forces'] + info['n_int_forces']] / settings['nnodes'] * globals_dict.dur
    else:
        forces = states_list.states.model[:settings['nnodes'],info['idx_int_forces']:info['idx_int_forces'] + info['n_int_forces']] / settings['nnodes']
    # Compute the objective value (e.g., L2 norm of the forces)
    return jnp.sum(jnp.abs(jnp.power(forces, info['exponent'])))

