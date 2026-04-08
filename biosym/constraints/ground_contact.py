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
        Initialize the Ground Contact Constraint class with a model and settings.

        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the dynamics constraint.
        """
        self.model = model
        self.settings = settings.copy()
        self.settings["nvpn"] = len(model.state_vector)
        self.nvar = settings.get("nvar")
        self.nf = model.ext_forces["n"] + model.ext_torques["n"]
        self.ncons_model = len(self.model.fr)

    def _get_info(self):
        """
        Get information about the dynamics constraint.

        This method can be overridden in subclasses to provide specific information.
        """
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Ground contact constraint class for biosym ocp.",
            "required_variables": {"states": ["model", "gc_model"], "constants": ["model", "gc_model"]},
            "nnz": self.get_nnz(),
            "ncons": self.get_n_constraints(),
            "ncons_pernode": self.nf,
            "idx_ext_forces": self.model.ext_forces["idx"],
            "idx_ext_torques": self.model.ext_torques["idx"],
            "n_ext_forces": self.model.ext_forces["n"],
            "n_ext_torques": self.model.ext_torques["n"],
        }

    def get_confun(self):
        """
        Evaluate the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The dynamics constraint function.
        """
        return jax.jit(partial(confun, self.model, settings=self.settings, info=self._get_info()))

    def get_jacobian(self):
        """
        Get the Jacobian of the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The Jacobian of the dynamics constraint function.
        """
        return jax.jit(partial(jacobian, self.model, settings=self.settings, info=self._get_info()))

    def get_n_constraints(self):
        """
        Get the number of constraints defined by this dynamics constraint.

        :return: The number of constraints.
        """
        return self.nf * self.settings.get("nnodes")

    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian of the dynamics constraint.

        :return: The number of non-zero entries.
        """
        return self.get_n_constraints() * self.settings.get("nvpn")


def confun(model, states_list, globals_dict, settings, info):
    """
    Placeholder for the constraint function.

    This function should be implemented in subclasses to evaluate the dynamics constraints.

    :param states_list: List containing the current states.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The evaluated value of the constraint function.

    Todo: there is some non-jax logic in here, which could be replaced with a static function
    """
    data_out = jnp.empty((info["ncons"],), dtype=float)
    nnodes = settings.get("nnodes")
    ncons = info["ncons_pernode"]

    def body_fun(n, carry):
        data_out = carry
        state_ = states_list[n]
        forces_gc, moments_gc = model.run["gc_model"](
            state_.states, state_.constants
        )  # Get ground contact forces and moments
        # Find forces and moments in the model
        forces_model = state_.states.model[model.ext_forces["idx"] : model.ext_forces["idx"] + model.ext_forces["n"]]
        moments_model = state_.states.model[
            model.ext_torques["idx"] : model.ext_torques["idx"] + model.ext_torques["n"]
        ]

        val = jnp.concatenate((forces_gc.flatten() - forces_model, moments_gc.flatten() - moments_model), axis=0)
        start = n * ncons
        data_out = jax.lax.dynamic_update_slice(data_out, val, (start,))
        return data_out

    data_out = jax.lax.fori_loop(0, nnodes, body_fun, data_out)
    return data_out


def jacobian(model, states_list, globals_dict, settings, info):
    """
    Placeholder for the Jacobian of the constraint function.

    This function should be implemented in subclasses to compute the Jacobian of the dynamics constraints.

    param states_list: List containing the current states.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The Jacobian of the constraint function.
    """
    nnz = info["nnz"]
    nvpn = settings.get("nvpn")
    nnodes = settings.get("nnodes")
    ncons = info["ncons_pernode"]
    rows_out = jnp.empty((nnz,), dtype=int)
    cols_out = jnp.empty((nnz,), dtype=int)
    data_out = jnp.empty((nnz,), dtype=float)

    block_size = ncons * nvpn

    def body_fun(n, carry):
        rows_out, cols_out, data_out = carry
        state_ = states_list[n]
        jac = model.run["gc_model_jacobian"](state_.states, state_.constants)

        jac_model = jnp.vstack((jac[0].model, jac[1].model))
        jac_gc_model = jnp.vstack((jac[0].gc_model, jac[1].gc_model))

        if jac_gc_model.shape[-1] != 0:
            raise NotImplementedError("Jacobian for ground contact model states not implemented yet.")

        # Jacobian block for the model
        row_block = n * ncons + jnp.arange(ncons)
        col_block = state_.states.size() * n + jnp.arange(nvpn)

        rows_block = jnp.repeat(row_block, nvpn)  # Shape: (ncons * nvpn,)
        cols_block = jnp.tile(col_block, ncons)  # Shape: (ncons * nvpn,)

        # Add -1 to the forces and moments in the model
        # Forces
        rows = jnp.repeat(jnp.arange(info["n_ext_forces"] // 3), 3)  # [0, 0, 0, 1, 1, 1] for gait2d
        cols = jnp.tile(jnp.arange(info["n_ext_forces"] // 2), 2)  # [0, 1, 2, 0, 1, 2] for gait2d
        depth = model.ext_forces["idx"] + jnp.arange(6)
        jac_model = jac_model.at[rows, cols, depth].add(-1)  # Subtract 1 from the forces and moments in the model
        # Moments
        rows = (
            jnp.repeat(jnp.arange(info["n_ext_torques"] // 3), 3) + info["n_ext_forces"] // 3
        )  # [0, 0, 0, 1, 1, 1] for gait2d
        cols = jnp.tile(jnp.arange(info["n_ext_torques"] // 2), 2)  # [0, 1, 2, 0, 1, 2] for gait2d
        depth = model.ext_torques["idx"] + jnp.arange(6)
        jac_model = jac_model.at[rows, cols, depth].add(-1)

        data_block = jac_model.flatten()  # Flatten the block

        start = n * block_size  # Calculate where to insert this block

        rows_out = jax.lax.dynamic_update_slice(rows_out, rows_block, (start,))
        cols_out = jax.lax.dynamic_update_slice(cols_out, cols_block, (start,))
        data_out = jax.lax.dynamic_update_slice(data_out, data_block, (start,))

        return (rows_out, cols_out, data_out)

    rows_out, cols_out, data_out = jax.lax.fori_loop(0, nnodes, body_fun, (rows_out, cols_out, data_out))
    return rows_out, cols_out, data_out
