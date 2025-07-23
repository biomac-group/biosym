
from biosym.constraints import *
import jax.numpy as jnp
import jax
from functools import partial

class Constraints():
    """
    A class to handle constraints for optimal control problems.
    This class is a placeholder and should be implemented with specific methods
    for constraint handling.
    """

    def __init__(self, model, settings):
        """
        Initialize the Constraints class with a model and settings.
        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the constraints.
        """
        self.model = model
        self.settings = settings
        self.nvar = settings.get('nvar')
        self.ncons_model = len(self.model.fr)

        # Initialize constraints and jacobian building
        self.constraint_functions = []
        self.jacobian_functions = []
        self.required_variables = {'states': [], 'constants': [], 'calculated': []}
        self.weights = []
        self.c_start, self.nnz_start = [0], [0] # First constraint starts at 0
        for constraint in settings.get('constraints', []):
            self.add_constraint(constraint.get('name'), constraint.get('weight'), constraint.get('args'))
        self.ncon = self.c_start[-1] # last entry is only total ncon
        self.nnz = self.nnz_start[-1]
        self._compile_callables()

    def _compile_callables(self):
        """
        Compile the constraint functions and Jacobian functions using JAX.
        """
        self.confun = jax.jit(partial(evaluate_constraints, self.constraint_functions, self.weights, self.ncon, self.c_start))
        self.confun.__name__ = 'evaluate_constraints'
        self.jacobian = jax.jit(partial(evaluate_jacobian, self.jacobian_functions, self.weights, self.nnz, self.c_start, self.nnz_start))
        self.jacobian.__name__ = 'evaluate_jacobian'


    def add_constraint(self, name, weight, args=None):
        """
        Add a constraint to the problem.
        :param constraint: A constraint object to be added.
        """
        # If the constraint is a string, instantiate it; if it's a class handle, instantiate it with model and settings.
        if isinstance(name, str):
            constraint_class = globals().get(name)
            if constraint_class:
                constraint = constraint_class.Constraint(self.model, self.settings)
            else:
                raise ValueError(f"Constraint class '{name}' not found.")
        elif hasattr(name, '__init__'):
            name.__init__(self.model, self.settings)
            constraint = name
        else:
            raise ValueError("Invalid constraint object provided.")

        info = constraint._get_info()
        print(f"Adding constraint: {info.get('name')} with weight {weight}")
        self.c_start.append(info['ncons'] + sum(self.c_start))
        self.nnz_start.append(info['nnz'] + sum(self.nnz_start))
        self.constraint_functions.append(constraint.get_confun())
        self.jacobian_functions.append(constraint.get_jacobian())
        if isinstance(weight, (int, float)):
            self.weights.append(weight)
        elif isinstance(weight, str):
            if weight == "1/BW":
                self.weights.append(1 / jnp.sum(jnp.array(
            [body["mass"] for body in self.model.dicts["bodies"]])))
            else:
                raise ValueError(f"Weight '{weight}' is not a valid number or setting. Valid settings are: '1/BW'.")
        else:
            raise ValueError("Weight must be a number or a string referring to a setting.")

        # Update required variables
        if info['required_variables']:
            for var_type, vars in info['required_variables'].items():
                if var_type not in self.required_variables:
                    self.required_variables[var_type] = []
                self.required_variables[var_type].extend(vars)

def evaluate_constraints(constraint_functions, weights, ncon, c_start, states_list, globals_dict=None):
    """
    Evaluate the constraints for the current state of the model.
    This method loops through all constraint functions and evaluates them.
    :param states_list: Dictionary containing the current states.
    :param globals_dict: Dictionary containing global variables, if any.
    :return: Vector of evaluated constraints
    """
    c_vec = jnp.empty((ncon,), dtype = jnp.float32)
    for i, con in enumerate(constraint_functions):
        c_vec = c_vec.at[c_start[i]:c_start[i+1]].set(con(states_list, globals_dict)*weights[i])
    return c_vec

def evaluate_jacobian(jacobian_functions, weights, nnz, c_start, nnz_start, states_list, globals_dict=None):
    """
    Evaluate the Jacobian of the constraints for the current state of the model.
    This method loops through all Jacobian functions and evaluates them.
    :param states_list: Dictionary containing the current states.
    :param globals_dict: Dictionary containing global variables, if any.
    :return: Tuple of (rows, cols, data) representing the sparse Jacobian matrix
    """
    rows, cols, data = jnp.empty((nnz,), dtype=jnp.int32), jnp.empty((nnz,), dtype=jnp.int32), jnp.empty((nnz,), dtype=jnp.float32)
    for i, jac in enumerate(jacobian_functions):
        r, c, d = jac(states_list, globals_dict)
        r = r + c_start[i]
        rows = rows.at[nnz_start[i]:nnz_start[i+1]].set(r)
        cols = cols.at[nnz_start[i]:nnz_start[i+1]].set(c)
        data = data.at[nnz_start[i]:nnz_start[i+1]].set(d*weights[i])
    return rows, cols, data
    

