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
        self.settings["nvpn_model"] = model.default_states.get_n_states()
        self.settings["nvpn"] = model.default_states.get_n_states()
        self.settings["nact"] = model.actuators.get_n_states()
        self.nvar = settings.get("nvar")
        self.nf = model.tau.n
        self.ncons_model = self.model.tau.n

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
            "idx_forces": self.model.tau.combined_idx,
            "n_forces": self.model.tau.n,
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
    constants = model.default_constants
    forces_act = model.run["actuator_model"](states_list, constants)
    forces_model = states_list.tau
    data_out = (forces_act - forces_model).flatten()

    if model.actuator_model.get_n_constraints(model, settings) > 0:
        c_act = model.actuator_model.constraints((states_list, globals_dict), constants, model, settings)
        data_out = jnp.concatenate((data_out, c_act.flatten()), axis=0)
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

    constants = model.default_constants

    # Vectorized computation for all nodes at once
    jac_all = jax.vmap(model.run["actuator_model_jacobian"], in_axes=(0, None))(
        states_list, constants
    ).to_array()

    from biosym.utils.states import get_states_offsets
    offsets = get_states_offsets(model.default_states)
    tau_start = offsets["tau"]
    forces_idx = jnp.arange(ncons)
    jac_all = jac_all.at[..., forces_idx, tau_start + forces_idx].add(-1)

    # Create node indices for all blocks
    node_indices = jnp.arange(nnodes)

    # Model jacobian blocks
    row_blocks_model = node_indices[:, None] * ncons + jnp.arange(ncons)[None, :]
    col_blocks_model = node_indices[:, None] * nvpn + jnp.arange(nvpn)[None, :]
    
    rows_out = jnp.repeat(row_blocks_model, nvpn, axis=1).flatten()
    cols_out = jnp.tile(col_blocks_model, (1, ncons)).flatten()
    data_out = jac_all.reshape(nnodes, -1).flatten()


    # Get actuator constraints jacobian if applicable
    if model.actuator_model.get_n_constraints(model, settings) > 0:
        rows_act_con, cols_act_con, data_act_con = model.actuator_model.jacobian(
            (states_list, globals_dict), constants, model, settings
        )
        rows_act_con = rows_act_con + (info.get('ncons_pernode') * settings.get('nnodes'))  # Shift row indices to avoid overlap
        rows_out = jnp.concatenate([rows_out, rows_act_con], axis=0)
        cols_out = jnp.concatenate([cols_out, cols_act_con], axis=0)
        data_out = jnp.concatenate([data_out, data_act_con], axis=0)

    return rows_out, cols_out, data_out

