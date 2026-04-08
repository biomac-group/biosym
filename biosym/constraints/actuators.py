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
        return self.nf * self.settings.get("nnodes") + self.model.actuators.get_n_constraints(self.model, self.settings)

    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian of the dynamics constraint.

        :return: The number of non-zero entries.
        """
        if self.model.actuators.get_n_constraints(self.model, self.settings) > 0:
            return (
                self.model.actuators.get_nnz(self.model, self.settings)
                + self.nf * self.settings.get("nvpn") * self.settings.get("nnodes")
            )
        else:   
            return self.nf * self.settings.get("nvpn") * self.settings.get("nnodes")

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

    forces_act = model.run["actuator_model"](states_list.states, states_list.constants)[:nnodes]  # Get ground contact forces and moments
    forces_model = states_list.states.model[:nnodes, model.forces["idx"] : model.forces["idx"] + model.forces["n"]]
    data_out = (forces_act - forces_model).flatten().squeeze()

    if model.actuator_model.get_n_constraints(model, settings) > 0:
        c_act = model.actuator_model.constraints((states_list.states, globals_dict), states_list.constants, model, settings)
        data_out = jnp.concatenate((data_out, c_act.flatten().squeeze()), axis=0)
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
    nnodes_dur = settings.get("nnodes_dur")
    ncons = info["ncons_pernode"]

    # Vectorized computation for all nodes at once
    jac_all = jax.vmap(model.run["actuator_model_jacobian"], in_axes=(0, None))(
        states_list[:nnodes].states, states_list.constants
    )

    # Extract model and actuator jacobians
    jac_model_all = jac_all.model  # Shape: (nnodes, ncons, nvpn_model)
    jac_actuators_all = jac_all.actuator_model  # Shape: (nnodes, ncons, nact)

    # Add -1 to the forces and moments in the model jacobian
    forces_rows = jnp.arange(info["n_forces"])
    forces_cols = model.forces["idx"] + jnp.arange(info["n_forces"])
    jac_model_all = jac_model_all.at[..., forces_rows, forces_cols].add(-1)

    # Create node indices for all blocks
    node_indices = jnp.arange(nnodes)

    # Model jacobian blocks
    row_blocks_model = node_indices[:, None] * ncons + jnp.arange(ncons)[None, :]
    col_blocks_model = node_indices[:, None] * states_list[0].states.size() + jnp.arange(nvpn_model)[None, :]
    
    rows_model = jnp.repeat(row_blocks_model, nvpn_model, axis=1)
    cols_model = jnp.tile(col_blocks_model, (1, ncons))
    data_model = jac_model_all.reshape(nnodes, -1)

    # Actuator jacobian blocks
    row_blocks_act = node_indices[:, None] * ncons + jnp.arange(ncons)[None, :]
    col_blocks_act = (node_indices[:, None] + 1) * states_list[0].states.size() - nact + jnp.arange(nact)[None, :]
    
    rows_act = jnp.repeat(row_blocks_act, nact, axis=1)
    cols_act = jnp.tile(col_blocks_act, (1, ncons))
    data_act = jac_actuators_all.reshape(nnodes, -1)

    # Concatenate model and actuator parts
    rows_out = jnp.concatenate([rows_model.flatten(), rows_act.flatten()])
    cols_out = jnp.concatenate([cols_model.flatten(), cols_act.flatten()])
    data_out = jnp.concatenate([data_model.flatten(), data_act.flatten()])


    # Get actuator constraints jacobian if applicable
    if model.actuator_model.get_n_constraints(model, settings) > 0:
        rows_act_con, cols_act_con, data_act_con = model.actuator_model.jacobian(
            (states_list.states, globals_dict), states_list.constants, model, settings
        )
        rows_act_con = rows_act_con + (info.get('ncons_pernode')*settings.get('nnodes')) # Shift row indices to avoid overlap
        rows_out = jnp.concatenate([rows_out, rows_act_con], axis=0)
        cols_out = jnp.concatenate([cols_out, cols_act_con], axis=0)
        data_out = jnp.concatenate([data_out, data_act_con], axis=0)

    return rows_out, cols_out, data_out

