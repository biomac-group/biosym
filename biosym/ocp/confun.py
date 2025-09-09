"""
Constraint functions for optimal control problems in biosym.

This module provides constraint handling for optimal control problems,
including constraint evaluation, Jacobian computation, and integration
with collocation methods for biomechanical motion optimization.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from biosym.constraints import *


class Constraints:
    """Handle constraints for optimal control problems.
    
    This class manages constraint evaluation, Jacobian computation, and
    constraint structure analysis for biomechanical optimal control problems.
    It integrates various constraint types including dynamics, periodicity,
    discretization, and custom user constraints.
    
    Attributes
    ----------
    model : BiosymModel
        The biomechanical model containing system dynamics
    settings : dict
        Configuration settings for constraint handling
    constraint_list : list
        List of active constraint objects
    ncon : int
        Total number of constraint equations
    """

    def __init__(self, model, settings):
        """Initialize the Constraints class.
        
        Parameters
        ----------
        model : BiosymModel
            Biosym model object representing the system to be controlled
        settings : dict
            Dictionary containing settings for the constraints including
            constraint types, weights, and parameters
        """
        self.model = model
        self.settings = settings
        self.nvar = settings.get("nvar")
        self.ncons_model = len(self.model.fr)

        # Initialize constraints and jacobian building
        self.constraint_functions = []
        self.jacobian_functions = []
        self.required_variables = {"states": [], "constants": [], "calculated": []}
        self.weights = []
        self.c_start, self.nnz_start = [0], [0]  # First constraint starts at 0
        for constraint in settings.get("constraints", []):
            self.add_constraint(constraint.get("name"), constraint.get("weight"), constraint.get("args"))
        self.nnz_start, self.c_start = (
            np.cumsum(self.nnz_start, dtype=np.int32).tolist(),
            np.cumsum(self.c_start, dtype=np.int32).tolist(),
        )
        self.ncon = self.c_start[-1]  # last entry is only total ncon
        self.nnz = self.nnz_start[-1]
        self._compile_callables()

    def _compile_callables(self):
        """
        Compile constraint and Jacobian functions using JAX JIT compilation.
        
        This method creates JIT-compiled versions of the constraint evaluation
        and Jacobian computation functions for optimal performance during
        optimization. The functions are wrapped with partial application
        to include necessary parameters like weights and indices.
        
        Notes
        -----
        - Uses JAX JIT compilation for fast execution during optimization
        - Creates partially applied functions with pre-configured parameters
        - Sets custom function names for better debugging and profiling
        """
        self.confun = jax.jit(
            partial(evaluate_constraints, self.constraint_functions, self.weights, self.ncon, self.c_start)
        )
        self.confun.__name__ = "evaluate_constraints"
        self.jacobian = jax.jit(
            partial(evaluate_jacobian, self.jacobian_functions, self.weights, self.nnz, self.c_start, self.nnz_start)
        )
        self.jacobian.__name__ = "evaluate_jacobian"

    def add_constraint(self, name, weight, args=None):
        """
        Add a constraint to the optimal control problem.
        
        Parameters
        ----------
        name : str or class
            Either a string name of a constraint class or a constraint class object.
            If string, the class will be looked up in the global namespace.
        weight : float, int, or str
            Weighting factor for the constraint in the optimization.
            Can be numeric value or special string like "1/BW" for bodyweight normalization.
        args : dict, optional
            Additional arguments to pass to the constraint constructor.
            
        Raises
        ------
        ValueError
            If the constraint class name is not found or invalid constraint object provided.
            
        Notes
        -----
        - Constraints are instantiated with the model and settings objects
        - Weights are stored for later use in constraint evaluation
        - Constraint information (number of constraints, Jacobian sparsity) is recorded
        """
        # If the constraint is a string, instantiate it; if it's a class handle, instantiate it with model and settings.
        if isinstance(name, str):
            constraint_class = globals().get(name)
            if constraint_class:
                constraint = constraint_class.Constraint(self.model, self.settings, args)
            else:
                raise ValueError(f"Constraint class '{name}' not found.")
        elif hasattr(name, "__init__"):
            name.__init__(self.model, self.settings, args)
            constraint = name
        else:
            raise ValueError("Invalid constraint object provided.")

        info = constraint._get_info()
        print(f"Adding constraint: {info.get('name')} with weight {weight}")
        self.c_start.append(info["ncons"])
        self.nnz_start.append(info["nnz"])
        self.constraint_functions.append(constraint.get_confun())
        self.jacobian_functions.append(constraint.get_jacobian())
        if isinstance(weight, (int, float)):
            self.weights.append(float(weight))
        elif isinstance(weight, str):
            if weight == "1/BW":
                self.weights.append(1 / jnp.sum(jnp.array([body["mass"] for body in self.model.dicts["bodies"]])))
            else:
                raise ValueError(f"Weight '{weight}' is not a valid number or setting. Valid settings are: '1/BW'.")
            print(weight)
        else:
            raise ValueError("Weight must be a number or a string referring to a setting.")

        # Update required variables
        if info["required_variables"]:
            for var_type, vars in info["required_variables"].items():
                if var_type not in self.required_variables:
                    self.required_variables[var_type] = []
                self.required_variables[var_type].extend(vars)


def evaluate_constraints(constraint_functions, weights, ncon, c_start, states_list, globals_dict=None):
    """
    Evaluate all constraints for the current state of the optimal control problem.
    
    This function efficiently evaluates all constraint functions and assembles
    them into a single constraint vector with appropriate weighting. It's designed
    to be JIT-compiled for performance during optimization.
    
    Parameters
    ----------
    constraint_functions : list
        List of constraint functions to evaluate.
    weights : list
        Weighting factors for each constraint type.
    ncon : int
        Total number of constraints.
    c_start : list
        Starting indices for each constraint type in the output vector.
    states_list : dict
        Dictionary containing current state variables (positions, velocities, etc.).
    globals_dict : dict, optional
        Dictionary containing global variables or parameters.
        
    Returns
    -------
    jnp.ndarray
        Constraint vector of shape (ncon,) containing all weighted constraint values.
        
    Notes
    -----
    - This function is typically JIT-compiled for optimal performance
    - Constraints are weighted and concatenated into a single vector
    - Used during optimization iterations to evaluate constraint violations
    """
    c_vec = jnp.empty((ncon,), dtype=jnp.float32)
    for i, con in enumerate(constraint_functions):
        c_vec = c_vec.at[c_start[i] : c_start[i + 1]].set(con(states_list, globals_dict) * weights[i])
    return c_vec


def evaluate_jacobian(jacobian_functions, weights, nnz, c_start, nnz_start, states_list, globals_dict=None):
    """
    Evaluate the Jacobian matrix of all constraints for gradient-based optimization.
    
    This function computes the sparse Jacobian matrix of all constraints with respect
    to the state variables. The Jacobian is essential for efficient gradient-based
    optimization algorithms used in optimal control.
    
    Parameters
    ----------
    jacobian_functions : list
        List of Jacobian functions for each constraint type.
    weights : list
        Weighting factors for each constraint type.
    nnz : int
        Total number of non-zero entries in the sparse Jacobian.
    c_start : list
        Starting row indices for each constraint type in the Jacobian.
    nnz_start : list
        Starting indices for non-zero entries of each constraint type.
    states_list : dict
        Dictionary containing current state variables.
    globals_dict : dict, optional
        Dictionary containing global variables or parameters.
        
    Returns
    -------
    tuple of jnp.ndarray
        Sparse Jacobian matrix in COO format as (rows, cols, data) where:
        - rows: Row indices of non-zero entries
        - cols: Column indices of non-zero entries  
        - data: Values of non-zero entries (weighted)
        
    Notes
    -----
    - Returns sparse matrix in COO (coordinate) format for efficiency
    - Jacobian entries are weighted according to constraint weights
    - Row indices are adjusted to account for constraint concatenation
    - Used by optimization algorithms for computing search directions
    """
    rows, cols, data = (
        jnp.empty((nnz,), dtype=jnp.int32),
        jnp.empty((nnz,), dtype=jnp.int32),
        jnp.empty((nnz,), dtype=jnp.float32),
    )
    for i, jac in enumerate(jacobian_functions):
        r, c, d = jac(states_list, globals_dict)
        r = r + c_start[i]
        rows = rows.at[nnz_start[i] : nnz_start[i + 1]].set(r)
        cols = cols.at[nnz_start[i] : nnz_start[i + 1]].set(c)
        data = data.at[nnz_start[i] : nnz_start[i + 1]].set(d * weights[i])
    return rows, cols, data
