import jax
import jax.numpy as jnp
import os
from biosym.objectives.base_objective import BaseObjective

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
            'exponent': self.exponent
        }

    def get_objfun(self):
        """ :return: The objective function. """
        fun = lambda states_list, globals_dict: objfun(self.model, states_list, globals_dict, self.settings, self._get_info())
        return jax.jit(fun)

    def get_gradient(self):
        """ :return: The gradient of the objective function. """
        fun = lambda states_list, globals_dict: objfun(self.model, states_list, globals_dict, self.settings, self._get_info())
        return jax.jit(jax.grad(fun))
    
def objfun(model, states_list, globals_dict, settings, info):
    """
    Evaluate the objective function.
    
    :param model: biosym model object.
    :param states_list: Dictionary containing the current states.
    :param settings: Settings for the objective function.
    :param info: Information about the objective function.
    :return: The evaluated value of the objective function.
    """
    forces = states_list['states']['model'][info['idx_int_forces']:info['idx_int_forces'] + info['n_int_forces']]

    # Compute the objective value (e.g., L2 norm of the forces)
    return jnp.sum(jnp.power(forces, info['exponent']))

