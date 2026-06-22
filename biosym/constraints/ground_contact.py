import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.constraints.base_constraint import BaseConstraint


# any constraint needs to be named Constraint, otherwise it will not be found by the OCP class
class Constraint(BaseConstraint):
    """
    Ground contact constraint for biosym OCP.

    Enforces that the model's external forces and moments equal those computed
    by the ground contact model: gc_model(states) - ext_forces = 0.
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
        self.settings["nvpn_gc_model"] = model.gc_model.get_n_states()
        self.settings["nvpn_actuator_model"] = model.actuators.get_n_states()
        self.settings["nvpn"] = (
            self.settings["nvpn_model"]
            + self.settings["nvpn_gc_model"]
            + self.settings["nvpn_actuator_model"]
        )
        self.nvar = settings.get("nvar")
        self.nf = model.ext_forces.n + model.ext_torques.n
        self.ncons_model = len(self.model.fr)

    def _get_info(self):
        """
        Get information about the ground contact constraint.
        """
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Ground contact constraint class for biosym ocp.",
            "required_variables": {"states": ["model", "gc_model"], "constants": ["model", "gc_model"]},
            "nnz": self.get_nnz(),
            "ncons": self.get_n_constraints(),
            "ncons_pernode": self.nf,
            "n_ext_forces": self.model.ext_forces.n,
            "n_ext_torques": self.model.ext_torques.n,
        }

    def get_confun(self):
        """
        Return the JIT-compiled constraint function.
        """
        return jax.jit(partial(confun, self.model, settings=self.settings, info=self._get_info()))

    def get_jacobian(self):
        """
        Return the JIT-compiled Jacobian function.
        """
        return jax.jit(partial(jacobian, self.model, settings=self.settings, info=self._get_info()))

    def get_n_constraints(self):
        """
        Get the number of constraints (nf * nnodes).
        """
        return self.nf * self.settings.get("nnodes")

    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian.
        """
        return self.get_n_constraints() * self.settings.get("nvpn")


def confun(model, states_list, globals_dict, settings, info):
    """
    Evaluate the ground contact constraint.

    Computes: gc_model(states) - ext_forces = 0 for forces and moments.

    :param model: BiosymModel
    :param states_list: batched States object of shape (nnodes, ...)
    :param globals_dict: global variables (unused)
    :param settings: constraint settings dict
    :param info: constraint info dict
    :return: flattened residual vector of shape (nnodes * nf,)
    """
    constants = model.default_constants
    nnodes = settings.get("nnodes")

    def _eval_single(states_):
        forces_gc, moments_gc = model.run["gc_model"](states_, constants)
        res_forces = forces_gc.flatten() - states_.ext_forces.flatten()
        res_moments = moments_gc.flatten() - states_.ext_torques.flatten()
        return jnp.concatenate([res_forces, res_moments])

    result = jax.vmap(_eval_single)(states_list[:nnodes])
    return result.flatten()



def jacobian(model, states_list, globals_dict, settings, info):
    """
    Compute the sparse COO Jacobian of the ground contact constraint.

    Uses model.run["gc_model_jacobian"] for the gc_model contribution, then
    adds the -I contribution for ext_forces and ext_torques.

    :param model: BiosymModel
    :param states_list: batched States object of shape (nnodes, ...)
    :return: (rows, cols, data) sparse COO arrays
    """
    nnz = info["nnz"]
    nvpn = settings.get("nvpn")
    nvpn_model = settings.get("nvpn_model")
    nvpn_gc_model = settings.get("nvpn_gc_model")
    nvpn_actuator_model = settings.get("nvpn_actuator_model")
    nnodes = settings.get("nnodes")
    ncons = info["ncons_pernode"]
    n_ext_forces = info["n_ext_forces"]
    n_ext_torques = info["n_ext_torques"]

    constants = model.default_constants

    def _jac_single(n, states_):
        """Compute the COO jacobian block for node n."""
        # gc_model jacobian w.r.t. states: returns (jac_forces, jac_moments) — each a States pytree
        # with fields of shape (n_bodies_per_force_dim, field_dim)
        jac = model.run["gc_model_jacobian"](states_, constants)
        jac_forces, jac_moments = jac  # each is a States with shape (..., field_dim)

        # Assemble dense jacobian blocks from model-state fields
        def _concat_model_fields(jac_struct):
            parts = []
            for name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques"]:
                val = getattr(jac_struct, name)
                if val is not None:
                    parts.append(val.reshape(-1, val.shape[-1]))
            return jnp.concatenate(parts, axis=-1)  # (n_force_components, nvpn_model)

        jac_model_forces = _concat_model_fields(jac_forces)   # (n_ext_forces, nvpn_model)
        jac_model_moments = _concat_model_fields(jac_moments)  # (n_ext_torques, nvpn_model)

        # Stack forces and moments: (ncons, nvpn_model)
        jac_model = jnp.vstack([jac_model_forces, jac_model_moments])

        # gc_model states contribution (if any)
        if nvpn_gc_model > 0:
            jac_gc_forces = jac_forces.gc_model.reshape(-1, jac_forces.gc_model.shape[-1])
            jac_gc_moments = jac_moments.gc_model.reshape(-1, jac_moments.gc_model.shape[-1])
            jac_gc = jnp.vstack([jac_gc_forces, jac_gc_moments])  # (ncons, nvpn_gc_model)

        # actuator_model contribution (if any)
        if nvpn_actuator_model > 0:
            jac_act_forces = jac_forces.actuator_model.reshape(-1, jac_forces.actuator_model.shape[-1])
            jac_act_moments = jac_moments.actuator_model.reshape(-1, jac_moments.actuator_model.shape[-1])
            jac_act = jnp.vstack([jac_act_forces, jac_act_moments])  # (ncons, nvpn_actuator_model)

        node_offset = n * nvpn
        row_block = n * ncons + jnp.arange(ncons)

        rows_blocks = []
        cols_blocks = []
        data_blocks = []

        # Model state jacobian block (also subtract -I on ext_forces/ext_torques columns)
        # The -ext_forces and -ext_torques terms contribute -I in the model part
        # ext_forces start at: nvpn_model - n_ext_forces - n_ext_torques
        # ext_torques start at: nvpn_model - n_ext_torques
        ef_start = nvpn_model - n_ext_forces - n_ext_torques
        et_start = nvpn_model - n_ext_torques
        jac_model = jac_model.at[:n_ext_forces, ef_start:ef_start + n_ext_forces].add(
            -jnp.eye(n_ext_forces)
        )
        jac_model = jac_model.at[n_ext_forces:, et_start:et_start + n_ext_torques].add(
            -jnp.eye(n_ext_torques)
        )

        rows_blocks.append(jnp.repeat(row_block, nvpn_model))
        cols_blocks.append(jnp.tile(node_offset + jnp.arange(nvpn_model), ncons))
        data_blocks.append(jac_model.flatten())

        if nvpn_gc_model > 0:
            rows_blocks.append(jnp.repeat(row_block, nvpn_gc_model))
            cols_blocks.append(jnp.tile(node_offset + nvpn_model + jnp.arange(nvpn_gc_model), ncons))
            data_blocks.append(jac_gc.flatten())

        if nvpn_actuator_model > 0:
            rows_blocks.append(jnp.repeat(row_block, nvpn_actuator_model))
            cols_blocks.append(
                jnp.tile(node_offset + nvpn_model + nvpn_gc_model + jnp.arange(nvpn_actuator_model), ncons)
            )
            data_blocks.append(jac_act.flatten())

        rows_block = jnp.concatenate(rows_blocks)
        cols_block = jnp.concatenate(cols_blocks)
        data_block = jnp.concatenate(data_blocks)
        return rows_block, cols_block, data_block

    rows_out = jnp.empty((nnz,), dtype=int)
    cols_out = jnp.empty((nnz,), dtype=int)
    data_out = jnp.empty((nnz,), dtype=float)

    block_size = ncons * nvpn

    def body_fun(n, carry):
        rows_out, cols_out, data_out = carry
        state_ = states_list[n]
        rows_block, cols_block, data_block = _jac_single(n, state_)
        start = n * block_size
        rows_out = jax.lax.dynamic_update_slice(rows_out, rows_block, (start,))
        cols_out = jax.lax.dynamic_update_slice(cols_out, cols_block, (start,))
        data_out = jax.lax.dynamic_update_slice(data_out, data_block, (start,))
        return (rows_out, cols_out, data_out)

    rows_out, cols_out, data_out = jax.lax.fori_loop(0, nnodes, body_fun, (rows_out, cols_out, data_out))
    return rows_out, cols_out, data_out
