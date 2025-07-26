from biosym.objectives import *
from biosym.ocp import utils
from biosym.utils import states
import jax.numpy as jnp
import jax
from functools import partial

class ObjectiveFunction:
    """
    Base class for objective functions in the biosym optimization framework.
    """
    def __init__(self, model, settings):
        """
        Initialize the ObjectiveFunction class with a model and settings.
        
        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the objective function.
        """
        self.model = model
        self.settings = settings
        self.objective_functions = []
        self.objective_gradients = []
        self.required_variables = {'states': [], 'constants': [], 'calculated': []}
        self.weights = settings.get('weights', [])
        for objective in settings.get('objectives', []):
            self.add_objective(objective.get('name'), objective.get('weight'), objective.get('args', None))

        self.objfun = jax.jit(partial(evaluate_objectives, self.objective_functions, self.weights))
        self.objfun.__name__ = 'evaluate_objectives'
        self.gradfun = jax.jit(partial(evaluate_gradients, self.objective_gradients, self.weights))
        self.gradfun.__name__ = 'evaluate_gradients'

    def add_objective(self, name, weight, kwargs=None):
        """
        Add an objective function to the optimization problem.
        
        :param name: Name of the objective function class or instance.
        :param weight: Weight for the objective function.
        :param args: Additional arguments for the objective function.
        """
        if isinstance(name, str):
            objective_class = globals().get(name)
            if objective_class:
                objective = objective_class.Objective(self.model, self.settings, **kwargs if kwargs else {})
            else:
                raise ValueError(f"Objective class '{name}' not found.")
        elif hasattr(name, '__init__'):
            objective = name(self.model, self.settings, **kwargs if kwargs else {})
        else:
            raise ValueError("Invalid objective object provided.")
        
        info = objective._get_info()
        print(f"Adding objective: {info.get('name')} with weight {weight}")

        self.objective_functions.append(objective.get_objfun())
        self.objective_gradients.append(objective.get_gradient())
        self.weights.append(weight)

        # Update required variables
        if info['required_variables']:
            for var_type, vars in info['required_variables'].items():
                if var_type not in self.required_variables:
                    self.required_variables[var_type] = []
                self.required_variables[var_type].extend(vars)

def evaluate_objectives(objective_functions, weights, states_list, globals_dict=None):
    """
    Evaluate the objective functions.
    
    :param states_list: Dictionary containing the current states.
    :param globals_dict: Dictionary containing global variables (optional).
    :return: The evaluated values of the objective functions.
    """
    results = 0
    for i, obj_fun in enumerate(objective_functions):
        result = obj_fun(states_list, globals_dict)
        results += result * weights[i]  # Apply the corresponding weight
    return results

def evaluate_gradients(objective_gradients, weights, states_list, globals_dict=None):
    """
    Evaluate the gradients of the objective functions.
    
    :param states_list: Dictionary containing the current states.
    :param globals_dict: Dictionary containing global variables (optional).
    :return: The evaluated gradients of the objective functions.
    """
    gradients = []
    globals_gradients = []
    for i, grad_fun in enumerate(objective_gradients):
        gradient, globals_gradient = grad_fun(states_list, globals_dict)
        gradients.append(gradient)
        globals_gradients.append(globals_gradient)
    # Add all gradients together
    gradients = states.reduce_dataclasses(gradients, jnp.sum, weights)
    # Remove all None entries from globals_gradients
    globals_gradients = [g for g in globals_gradients if g is not None]
    if globals_gradients:
        # Sum the global gradients if they exist
        globals_gradients = states.reduce_dataclasses(globals_gradients, jnp.sum, weights)
    else:
        globals_gradients = None
    return gradients, globals_gradients