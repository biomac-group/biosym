import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.constraints.base_constraint import BaseConstraint


# any constraint needs to be named Constraint, otherwise it will not be found by the OCP class
class Constraint(BaseConstraint):
    """
    Base class for dynamics constraints in the biosym package.

    This class provides a template for implementing specific dynamics constraints.
    It includes methods for evaluating the constraint function, computing the Jacobian,
    and retrieving information about the constraint.
    """

    def __init__(self, model, settings, args):
        """
        Initialize the DynamicsConstraint class with a model and settings.

        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the dynamics constraint.
        """
        self.model = model
        self.settings = settings.copy()
        self.settings["nvpn"] = len(model.state_vector)
        self.nvar = settings.get("nvar")
        self.ncons_model = len(self.model.fr)

    def _get_info(self):
        """
        Get information about the dynamics constraint.

        This method can be overridden in subclasses to provide specific information.
        """
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Base dynamics constraint class for biosym constraints.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "nnz": self.get_nnz(),
            "ncons": self.get_n_constraints(),
            "ncons_pernode": self.ncons_model,
        }

    def get_confun(self):
        """
        Evaluate the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The dynamics constraint function.
        """
        return jax.jit(partial(confun, self.model.run["confun"], settings=self.settings, info=self._get_info()))

    def get_jacobian(self):
        """
        Get the Jacobian of the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The Jacobian of the dynamics constraint function.
        """
        return jax.jit(partial(jacobian, self.model.run["jacobian"], settings=self.settings, info=self._get_info()))

    def get_n_constraints(self):
        """
        Get the number of constraints defined by this dynamics constraint.

        :return: The number of constraints.
        """
        return self.ncons_model * self.settings.get("nnodes")

    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian of the dynamics constraint.

        :return: The number of non-zero entries.
        """
        return self.get_n_constraints() * self.settings.get("nvpn")


# @partial(jax.grad, argnums=(1, 2))
# @partial(jax.jit, static_argnums=(0))
def confun(modelfn, states_list, globals_dict, settings, info):
    """
    Placeholder for the constraint function.

    This function should be implemented in subclasses to evaluate the dynamics constraints.

    :param states_list: List containing the current states.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The evaluated value of the constraint function.
    """
    data_out = jnp.empty((info["ncons"],), dtype=float)
    nnodes = settings.get("nnodes")
    ncons_sympy = info["ncons_pernode"]

    def body_fun(n, carry):
        data_out = carry
        state_ = states_list[n]
        val = modelfn(state_.states, state_.constants).squeeze()
        start = n * ncons_sympy
        data_out = jax.lax.dynamic_update_slice(data_out, val, (start,))
        return data_out

    data_out = jax.lax.fori_loop(0, nnodes, body_fun, data_out)
    return data_out


def jacobian(modelfn, states_list, globals_dict, settings, info):
    """
    Placeholder for the Jacobian of the constraint function.

    This function should be implemented in subclasses to compute the Jacobian of the dynamics constraints.

    :param states_list: List containing the current states.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The Jacobian of the constraint function.
    """
    nnz = info["nnz"]
    nvpn = settings.get("nvpn")
    nnodes = settings.get("nnodes")
    ncons_sympy = info["ncons_pernode"]
    rows_out = jnp.empty((nnz,), dtype=int)
    cols_out = jnp.empty((nnz,), dtype=int)
    data_out = jnp.empty((nnz,), dtype=float)



    jac_all = jax.vmap(modelfn, in_axes=(0, None))(states_list[:nnodes].states, states_list.constants)

    # Vectorized computation for all nodes at once
    # Create node indices for all blocks
    node_indices = jnp.arange(nnodes)  # [0, 1, 2, ..., nnodes-1]
    
    # Compute row blocks [shape (nnodes, ncons_sympy)] and column blocks [shape (nnodes, nvpn)]
    row_blocks = node_indices[:, None] * ncons_sympy + jnp.arange(ncons_sympy)[None, :]
    col_blocks = node_indices[:, None] * states_list[0].states.size() + jnp.arange(nvpn)[None, :]
    
    # Create rows indices by repeating each row block nvpn times: shape (nnodes, ncons_sympy * nvpn)
    rows_blocks = jnp.repeat(row_blocks, nvpn, axis=1)
    cols_blocks = jnp.tile(col_blocks, (1, ncons_sympy))
    data_blocks = jac_all.model.reshape(nnodes, -1)
    
    # Flatten all blocks to create final arrays
    rows_out = rows_blocks.flatten()
    cols_out = cols_blocks.flatten()
    data_out = data_blocks.flatten()

    return rows_out, cols_out, data_out
