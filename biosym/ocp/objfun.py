"""
Objective function handling for optimal control problems in the biosym framework.

This module provides the ObjectiveFunction class for managing multiple objective
terms in optimization problems, including their evaluation and gradient computation.
"""

from functools import partial

import jax
import jax.numpy as jnp

from biosym.objectives import *
from biosym.utils import states


class ObjectiveFunction:
    """
    Manager for objective functions in optimal control problems.

    This class handles multiple objective terms, their weights, and provides
    efficient evaluation methods for optimization algorithms. It supports
    adding various types of objectives and compiles them into JIT-optimized
    functions for performance during optimization.

    Attributes
    ----------
    model : BiosymModel
        The biomechanical model being optimized.
    settings : dict
        Configuration settings for objective function setup.
    objective_functions : list
        List of objective function callables.
    objective_gradients : list
        List of gradient function callables.
    required_variables : dict
        Dictionary tracking required state variables for all objectives.
    weights : list
        Weighting factors for each objective term.
    objfun : callable
        JIT-compiled function for objective evaluation.
    gradfun : callable
        JIT-compiled function for gradient evaluation.
    """

    def __init__(self, model, settings):
        """
        Initialize the objective function manager.

        Parameters
        ----------
        model : BiosymModel
            The biomechanical model object representing the system to be optimized.
        settings : dict
            Configuration dictionary containing objective function settings.
            Expected to contain 'objectives' list and optional 'weights' list.

        Notes
        -----
        - Automatically processes objectives from settings during initialization
        - Creates JIT-compiled functions for efficient evaluation during optimization
        - Tracks required variables across all objective terms
        """
        self.model = model
        self.settings = settings
        self.objective_functions = []
        self.objective_gradients = []
        self._objectives = []
        self.required_variables = {"states": [], "constants": [], "calculated": []}
        self.weights = settings.get("weights", [])
        for objective in settings.get("objectives", []):
            self.add_objective(
                objective.get("name"),
                objective.get("weight"),
                objective.get("args", None),
            )

    def _compile_callables(self):
        """
        Compile the objective and gradient functions into JIT-optimized callables.
        """
        self.objfun = jax.jit(
            partial(evaluate_objectives, self.objective_functions, self.weights)
        )
        self.objfun.__name__ = "evaluate_objectives"
        self.gradfun = jax.jit(
            partial(evaluate_gradients, self.objective_gradients, self.weights)
        )
        self.gradfun.__name__ = "evaluate_gradients"

    def add_objective(self, name, weight, kwargs=None):
        """
        Add an objective function to the optimization problem.

        Parameters
        ----------
        name : str or class
            Name of the objective function class (string) or objective class instance.
            If string, the class will be looked up in the global namespace.
        weight : float or int
            Weighting factor for this objective in the total cost function.
            Higher weights emphasize this objective more in optimization.
        kwargs : dict, optional
            Additional keyword arguments to pass to the objective constructor.

        Raises
        ------
        ValueError
            If the objective class name is not found in the global namespace.

        Notes
        -----
        - Objectives are instantiated with model and settings
        - Objective functions and gradients are stored for later compilation
        - Required variables are tracked and accumulated across objectives
        """
        weight = float(weight)
        if isinstance(name, str):
            objective_class = globals().get(name)
            if objective_class:
                objective = objective_class.Objective(
                    self.model, self.settings, **kwargs if kwargs else {}
                )
            else:
                raise ValueError(f"Objective class '{name}' not found.")
        elif hasattr(name, "__init__"):
            objective = name(self.model, self.settings, **kwargs if kwargs else {})
        else:
            raise ValueError("Invalid objective object provided.")

        info = objective._get_info()
        print(f"Adding objective: {info.get('name')} with weight {weight}")

        self.objective_functions.append(objective.get_objfun())
        self.objective_gradients.append(objective.get_gradient())
        self._objectives.append(objective)
        self.weights.append(weight)

        # Update required variables
        if info["required_variables"]:
            for var_type, vars in info["required_variables"].items():
                if var_type not in self.required_variables:
                    self.required_variables[var_type] = []
                self.required_variables[var_type].extend(vars)
        self._compile_callables()


def evaluate_objectives(objective_functions, weights, states_list, globals_dict=None):
    """
    Evaluate all objective functions and compute weighted total cost.

    This function efficiently evaluates all objective terms and combines them
    into a single scalar cost value using the specified weights. It's designed
    to be JIT-compiled for performance during optimization.

    Parameters
    ----------
    objective_functions : list
        List of objective function callables to evaluate.
    weights : list
        Weighting factors for each objective function.
    states_list : dict
        Dictionary containing current state variables (positions, velocities, etc.).
    globals_dict : dict, optional
        Dictionary containing global variables or parameters.

    Returns
    -------
    float
        Weighted sum of all objective function values - the total cost to minimize.

    Notes
    -----
    - This function is typically JIT-compiled for optimal performance
    - Each objective is weighted before summing to form the total cost
    - Used at each optimization iteration to evaluate solution quality
    """
    results = 0
    for i, obj_fun in enumerate(objective_functions):
        result = obj_fun(states_list, globals_dict)
        results += result * weights[i]  # Apply the corresponding weight
    return results


def evaluate_gradients(objective_gradients, weights, states_list, globals_dict=None):
    """
    Evaluate gradients of all objective functions for gradient-based optimization.

    This function computes the gradients of all objective terms with respect to
    state and global variables, combining them with appropriate weighting to form
    the total cost gradient needed for optimization algorithms.

    Parameters
    ----------
    objective_gradients : list
        List of gradient function callables for each objective.
    weights : list
        Weighting factors for each objective function.
    states_list : dict
        Dictionary containing current state variables.
    globals_dict : dict, optional
        Dictionary containing global variables or parameters.

    Returns
    -------
    tuple
        Tuple containing:
        - gradients: Weighted sum of state variable gradients
        - globals_gradients: Weighted sum of global variable gradients (or None)

    Notes
    -----
    - Gradients are combined using weighted summation across objectives
    - State gradients are combined using dataclass reduction operations
    - Global gradients are filtered for None values before combining
    - Used by optimization algorithms to compute search directions
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
        globals_gradients = states.reduce_dataclasses(
            globals_gradients, jnp.sum, weights
        )
    else:
        globals_gradients = None
    return gradients, globals_gradients
