import jax
import jax.numpy as jnp
from biosym.objectives.base_objective import BaseObjective
import os

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

    def _get_info(self):
        """
        Get information about the objective function.
        This method can be overridden in subclasses to provide specific information.
        """
     
        return {
            'name': os.path.splitext(os.path.basename(__file__))[0],
            'description': 'Base objective class for biosym objectives.',
            'required_variables': None,
        }

    def get_objfun(self):
        """ :return: The objective function. """
        fun = lambda states_list, globals_dict: objfun(self.model, states_list, globals_dict, self.settings, self.get_info())
        return jax.jit(fun)

    def get_gradient(self):
        """ :return: The gradient of the objective function. """
        fun = lambda states_list, globals_dict: objfun(self.model, states_list, globals_dict, self.settings, self.get_info())
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

    for i in settings['nnodes']:
        for key, value in states_list['states'].items():
            pass
    return 0
        