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

    block_size = ncons_sympy * nvpn

    def body_fun(n, carry):
        rows_out, cols_out, data_out = carry
        state_ = states_list[n]
        jac = modelfn(state_.states, state_.constants)

        row_block = n * ncons_sympy + jnp.arange(ncons_sympy)
        col_block = state_.states.size() * n + jnp.arange(nvpn)

        rows_block = jnp.repeat(row_block, nvpn)  # Shape: (ncons_sympy * nvpn,)
        cols_block = jnp.tile(col_block, ncons_sympy)  # Shape: (ncons_sympy * nvpn,)
        data_block = jac.model.flatten()  # Flatten the block

        start = n * block_size  # Calculate where to insert this block

        rows_out = jax.lax.dynamic_update_slice(rows_out, rows_block, (start,))
        cols_out = jax.lax.dynamic_update_slice(cols_out, cols_block, (start,))
        data_out = jax.lax.dynamic_update_slice(data_out, data_block, (start,))

        return (rows_out, cols_out, data_out)

    rows_out, cols_out, data_out = jax.lax.fori_loop(0, nnodes, body_fun, (rows_out, cols_out, data_out))
    return rows_out, cols_out, data_out
