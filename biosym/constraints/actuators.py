import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.constraints.base_constraint import BaseConstraint


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
        self.settings["nvpn_model"] = len(model.state_vector)
        self.settings["nvpn"] = len(model.state_vector) + model.actuators.get_n_states()
        self.settings["nact"] = model.actuators.get_n_states()
        self.nvar = settings.get("nvar")
        self.nf = model.forces["n"]
        self.ncons_model = self.model.forces["n"]

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
            "idx_forces": self.model.forces["idx"],
            "n_forces": self.model.forces["n"],
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
        return (self.nf + self.model.actuators.get_n_constraints()) * self.settings.get("nnodes")

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
    model.run["actuator_model"](states_list.states, states_list.constants)  # Test full function

    forces_act = model.run["actuator_model"](states_list.states, states_list.constants)  # Get ground contact forces and moments
    forces_model = states_list.states.model[:, model.forces["idx"] : model.forces["idx"] + model.forces["n"]]
    print(forces_act.shape, forces_model.shape)
    data_out = (forces_act - forces_model).flatten()
    return data_out

    # Find
    def body_fun(n, carry):
        data_out = carry
        state_ = states_list[n]
        # Find forces and moments in the model
        forces_model = state_.states.model[model.forces["idx"] : model.forces["idx"] + model.forces["n"]]

        val = forces_act - forces_model

        start = n * ncons
        data_out = jax.lax.dynamic_update_slice(data_out, val, (start,))
        return data_out
    
    data_out = jax.lax.fori_loop(0, nnodes, body_fun, data_out)
    
    if model.actuator_model.get_n_constraints() > 0:
        pass
    
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
    nvpn_model = settings.get("nvpn_model")
    nact = settings.get("nact")
    nnodes = settings.get("nnodes")
    ncons = info["ncons_pernode"]
    rows_out = jnp.empty((nnz,), dtype=int)
    cols_out = jnp.empty((nnz,), dtype=int)
    data_out = jnp.empty((nnz,), dtype=float)

    block_size = ncons * nvpn

    def body_fun(n, carry):
        rows_out, cols_out, data_out = carry
        state_ = states_list[n]
        jac = model.run["actuator_model_jacobian"](state_.states, state_.constants)

        jac_model = jac.model
        jac_actuators = jac.actuator_model

        # Jacobian block for the model
        row_block = n * ncons + jnp.arange(ncons)
        col_block = state_.states.size() * n + jnp.arange(nvpn_model)

        rows_block = jnp.repeat(row_block, nvpn_model)  # Shape: (ncons * nvpn,)
        cols_block = jnp.tile(col_block, ncons)  # Shape: (ncons * nvpn,)

        # Add -1 to the forces and moments in the model
        # Forces
        rows = jnp.arange(info["n_forces"])
        depth = model.forces["idx"] + jnp.arange(info["n_forces"])
        jac_model = jac_model.at[rows, depth].add(-1)  # Subtract 1 from the forces and moments in the model
        data_block = jac_model.flatten()  # Flatten the block

        start = n * block_size  # Calculate where to insert this block
        rows_out = jax.lax.dynamic_update_slice(rows_out, rows_block, (start,))
        cols_out = jax.lax.dynamic_update_slice(cols_out, cols_block, (start,))
        data_out = jax.lax.dynamic_update_slice(data_out, data_block, (start,))

        row_block_act = n * ncons + jnp.arange(ncons)
        col_block_act = state_.states.size() * (n + 1) - nact + jnp.arange(nact)
        data_block = jac_actuators.flatten()  # Flatten the block

        row_block_act = jnp.repeat(row_block_act, nact)  # Shape: (ncons * nvpn,)
        col_block_act = jnp.tile(col_block_act, ncons)  # Shape: (ncons * nvpn,)

        start = (n + 1) * block_size - nact * ncons  # Calculate where to insert this block
        rows_out = jax.lax.dynamic_update_slice(rows_out, row_block_act, (start,))
        cols_out = jax.lax.dynamic_update_slice(cols_out, col_block_act, (start,))
        data_out = jax.lax.dynamic_update_slice(data_out, data_block, (start,))

        return (rows_out, cols_out, data_out)

    rows_out, cols_out, data_out = jax.lax.fori_loop(0, nnodes, body_fun, (rows_out, cols_out, data_out))
    return rows_out, cols_out, data_out
