import jax
import jax.numpy as jnp
from biosym.objectives.base_objective import BaseObjective
import os
from functools import partial

class Objective(BaseObjective):
    """
    Abstract base class for objectives in the biosym package.
    All objectives should inherit from this class and implement the required methods.
    """
    def __init__(self, model, settings):
        """
        Initialize the BaseObjective class with a model and settings.
        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the objective function.
        """
        self.model = model
        self.settings = settings
        self.settings['idx_acc'] = jnp.arange(model.accs['idx'], model.accs['idx'] + model.accs['n'])

    def _get_info(self):
        """
        Get information about the objective function.
        This method can be overridden in subclasses to provide specific information.
        """
     
        return {
            'name': os.path.splitext(os.path.basename(__file__))[0],
            'description': 'Regularization objective.',
            'required_variables': None,
        }

    def get_objfun(self):
        """ :return: The objective function. """
        fun = partial(objfun, settings = self.settings, info=self._get_info())
        return jax.jit(fun)

    def get_gradient(self):
        """ :return: The gradient of the objective function. """
        fun = partial(objfun, settings = self.settings, info=self._get_info())
        return jax.jit(jax.grad(fun, argnums = [0,1]))

def objfun(states_list, globals_dict, settings, info):
    """
    Evaluate the objective function.
    
    :param model: biosym model object.
    :param states_list: Dictionary containing the current states.
    :param settings: Settings for the objective function.
    :param info: Information about the objective function.
    :return: The evaluated value of the objective function.
    """
    # Goal: reduce jerk, e.g. diff(qd)
    qd = states_list.states.model[:, settings['idx_acc']]
    jerk = jnp.diff(qd, axis=0)
    return jnp.mean(jnp.square(jerk))

