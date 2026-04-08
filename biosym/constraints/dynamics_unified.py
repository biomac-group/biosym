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
        self.settings["nvpn"] = model.default_inputs.states.size()
        self.nvar = settings.get("nvar")
        self.ncons_model = len(self.model.fr)
        self.bodymass = model.variables[(model.variables['name'].str.startswith('m_')) & (model.variables['type'] == "constant")]['x0'].sum()

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
            "nnz_model": self.ncons_model * self.settings.get("nvpn") * self.settings.get("nnodes"),
            "ncons": self.get_n_constraints(),
            "ncons_pernode": self.ncons_model,
            "bodymass": self.bodymass,
        }

    def get_confun(self):
        """
        Evaluate the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The dynamics constraint function.
        """
        modelfn = partial(conf, self.model)
        return jax.jit(partial(confun, modelfn, settings=self.settings, info=self._get_info(), model=self.model))

    def get_jacobian(self):
        """
        Get the Jacobian of the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The Jacobian of the dynamics constraint function.
        """
        modelfn = jax.jacobian(partial(conf, self.model))
        return jax.jit(partial(jacobian, modelfn, settings=self.settings, info=self._get_info(), model=self.model))

    def get_n_constraints(self):
        """
        Get the number of constraints defined by this dynamics constraint.

        :return: The number of constraints.
        """
        return self.ncons_model * self.settings.get("nnodes") + self.model.actuators.get_n_constraints(self.model, self.settings) + self.model.gc_model.get_n_constraints(self.model, self.settings)

    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian of the dynamics constraint.

        :return: The number of non-zero entries.
        """
        nnz = 0
        if self.model.actuators.get_n_constraints(self.model, self.settings) > 0:
            nnz += (
                self.model.actuators.get_nnz(self.model, self.settings)
            )
        if self.model.gc_model.get_n_constraints(self.model, self.settings) > 0:
            nnz += (
                self.model.gc_model.get_nnz(self.model, self.settings)
            )
        nnz += self.ncons_model * self.settings.get("nvpn") * self.settings.get("nnodes")
        return nnz
    
def conf(model, states, constants):
    external_forces = model.run["gc_model"](states, constants)
    internal_forces = model.run["actuator_model"](states, constants)
    states_ = states.model
    states_ = states_.at[model.forces["idx"]:model.forces["idx"]+model.forces["n"]].set(internal_forces.flatten())
    states_ = states_.at[model.ext_forces["idx"]:model.ext_forces["n"]+model.ext_forces['idx']].set(external_forces[0].flatten())
    states_ = states_.at[model.ext_torques["idx"]:model.ext_torques["n"]+model.ext_torques['idx']].set(external_forces[1].flatten())
    states = states.replace(model=states_)

    return model.run["confun"](states, constants)


# @partial(jax.grad, argnums=(1, 2))
# @partial(jax.jit, static_argnums=(0))
def confun(modelfn, states_list, globals_dict, settings, info, model):
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
        data_out = jax.lax.dynamic_update_slice(data_out, 1/info['bodymass']*val, (start,))
        return data_out

    data_out = jax.lax.fori_loop(0, nnodes, body_fun, data_out)

    if model.actuator_model.get_n_constraints(model, settings) > 0:
        c_act = model.actuator_model.constraints((states_list.states, globals_dict), states_list.constants, model, settings)
        data_out = data_out.at[-c_act.size :].set(c_act.flatten().squeeze())

    if model.gc_model.get_n_constraints(model, settings) > 0:
        raise NotImplementedError("Ground contact constraints in unified dynamics not yet implemented.")
    return data_out


def jacobian(modelfn, states_list, globals_dict, settings, info, model):
    """
    Placeholder for the Jacobian of the constraint function.

    This function should be implemented in subclasses to compute the Jacobian of the dynamics constraints.

    :param states_list: List containing the current states.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The Jacobian of the constraint function.
    """
    nnz_model = info["nnz_model"]
    nvpn = states_list[0].states.size()
    nnodes = settings.get("nnodes")
    ncons_sympy = info["ncons_pernode"]
    rows_out = jnp.empty((nnz_model,), dtype=int)
    cols_out = jnp.empty((nnz_model,), dtype=int)
    data_out = jnp.empty((nnz_model,), dtype=float)

    block_size = ncons_sympy * nvpn

    jac_ = jax.vmap(modelfn, in_axes=(0, None))(states_list.states, states_list.constants)

    def body_fun(n, carry):
        rows_out, cols_out, data_out = carry
        state_ = states_list[n]
        #jac = modelfn(state_.states, state_.constants)
        jac = jac_[n]


        row_block = n * ncons_sympy + jnp.arange(ncons_sympy)
        col_block = state_.states.size() * n + jnp.arange(nvpn)

        rows_block = jnp.repeat(row_block, nvpn)  # Shape: (ncons_sympy * nvpn,)
        cols_block = jnp.tile(col_block, ncons_sympy)  # Shape: (ncons_sympy * nvpn,)
        data_block = jnp.concatenate((
            jac.model,
            jac.gc_model,
            jac.actuator_model,
        ), axis=-1).flatten()  # Flatten the block

        start = n * (block_size) # Calculate where to insert this block

        rows_out = jax.lax.dynamic_update_slice(rows_out, rows_block, (start,))
        cols_out = jax.lax.dynamic_update_slice(cols_out, cols_block, (start,))
        data_out = jax.lax.dynamic_update_slice(data_out, 1/info['bodymass']*data_block, (start,))

        return (rows_out, cols_out, data_out)

    rows_out, cols_out, data_out = jax.lax.fori_loop(0, nnodes, body_fun, (rows_out, cols_out, data_out))

    if model.actuator_model.get_n_constraints(model, settings) > 0:
        rows_act_con, cols_act_con, data_act_con = model.actuator_model.jacobian(
            (states_list.states, globals_dict), states_list.constants, model, settings
        )
        rows_act_con = rows_act_con + (info.get('ncons_pernode')*settings.get('nnodes')) # Shift row indices to avoid overlap
        rows_out = jnp.concatenate([rows_out, rows_act_con], axis=0)
        cols_out = jnp.concatenate([cols_out, cols_act_con], axis=0)
        data_out = jnp.concatenate([data_out, data_act_con], axis=0)
    return rows_out, cols_out, data_out
