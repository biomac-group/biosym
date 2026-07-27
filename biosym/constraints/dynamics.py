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
        self.model = model
        self.settings = settings.copy()
        self.settings["nvpn"] = model.default_states.size()
        self.nvar = settings.get("nvar")
        self.ncons_model = len(self.model.fr)
        self.bodymass = model.variables[(model.variables['name'].str.startswith('m_')) & (model.variables['type'] == "constant")]['x0'].sum()

    def _get_info(self):
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
        modelfn = partial(conf, self.model)
        return jax.jit(partial(confun, modelfn, settings=self.settings, info=self._get_info(), model=self.model))

    def get_jacobian(self):
        modelfn = jax.jacobian(partial(conf, self.model))
        return jax.jit(partial(jacobian, modelfn, settings=self.settings, info=self._get_info(), model=self.model))

    def get_n_constraints(self):
        return self.ncons_model * self.settings.get("nnodes") + self.model.actuator_model.get_n_constraints(self.model, self.settings) + self.model.gc_model.get_n_constraints(self.model, self.settings)

    def get_nnz(self):
        nnz = 0
        if self.model.actuator_model.get_n_constraints(self.model, self.settings) > 0:
            nnz += (self.model.actuator_model.get_nnz(self.model, self.settings))
        if self.model.gc_model.get_n_constraints(self.model, self.settings) > 0:
            nnz += (self.model.gc_model.get_nnz(self.model, self.settings))
        nnz += self.ncons_model * self.settings.get("nvpn") * self.settings.get("nnodes")
        return nnz


def calc_forces(states, model, constants=None):
    """
    First stage of the dynamics evaluation: run the ground-contact and actuator
    models' forward passes and inject the resulting forces/torques into states.
    """
    if constants is None:
        constants = model.default_constants
    external_forces, external_torques = model.run["gc_model"](states, constants)
    internal_forces = model.run["actuator_model"](states, constants)
    return states.replace(
        tau=internal_forces.flatten(),
        ext_forces=external_forces.flatten(),
        ext_torques=external_torques.flatten(),
    )


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def confun_mm_tau(states, constants, model):
    mm = model.run['mass_matrix'](states, constants)
    forcing = model.run['forcing'](states, constants)
    qdd = states.qdd

    # R = M(q)*qdd - forcing
    inertial_force = jnp.matmul(mm, qdd[..., None])[..., 0]
    residuals = inertial_force - forcing
    return residuals


@confun_mm_tau.defjvp
def jvpfun(model, primals, tangents):
    states, constants = primals
    dstates, dconstants = tangents

    mm = model.run['mass_matrix'](states, constants)
    forcing = model.run['forcing'](states, constants)
    qdd = states.qdd
    dqdd = dstates.qdd

    inertial_force = jnp.matmul(mm, qdd[..., None])[..., 0]
    residuals = inertial_force - forcing

    # Compute d(M(q)*qdd) and dforcing using JVP of vector-valued functions.
    # This avoids computing the tangent of the full mass matrix (O(n^2))
    # and instead computes the tangent of the contracted vector (O(n)).
    def mm_qdd_and_forcing(s, c):
        M = model.run['mass_matrix'](s, c)
        F = model.run['forcing'](s, c)
        M_qdd = jnp.matmul(M, qdd[..., None])[..., 0]
        return M_qdd, F

    _, (dmm_qdd, dforcing) = jax.jvp(
        lambda s, c: mm_qdd_and_forcing(s, c),
        (states, constants),
        (dstates, dconstants)
    )

    mm_dqdd = jnp.matmul(mm, dqdd[..., None])[..., 0]
    dresiduals = dmm_qdd + mm_dqdd - dforcing

    return residuals, dresiduals


def conf(model, states, constants):
    states_filled = calc_forces(states, model, constants)
    return confun_mm_tau(states_filled, constants, model)


def confun(modelfn, states_list, globals_dict, settings, info, model):
    nnodes = settings.get("nnodes")
    constants = model.default_constants
    vals = jax.vmap(modelfn, in_axes=(0, None))(states_list[:nnodes], constants)
    data_model = (1 / info['bodymass'] * vals.squeeze()).reshape(-1)

    if model.actuator_model.get_n_constraints(model, settings) > 0:
        c_act = model.actuator_model.constraints((states_list, globals_dict), constants, model, settings)
        data_out = jnp.concatenate([data_model, c_act.flatten().squeeze()])
    else:
        data_out = data_model

    if model.gc_model.get_n_constraints(model, settings) > 0:
        raise NotImplementedError("Ground contact constraints in unified dynamics not yet implemented.")
    return data_out


def jacobian(modelfn, states_list, globals_dict, settings, info, model):
    nvpn = states_list[0].size()
    nnodes = settings.get("nnodes")
    ncons_sympy = info["ncons_pernode"]
    constants = model.default_constants

    jac_ = jax.vmap(modelfn, in_axes=(0, None))(states_list[:nnodes], constants)

    def _node_block(n, jac):
        row_block = n * ncons_sympy + jnp.arange(ncons_sympy)
        col_block = nvpn * n + jnp.arange(nvpn)

        rows_block = jnp.repeat(row_block, nvpn)
        cols_block = jnp.tile(col_block, ncons_sympy)
        jac_model = jnp.concatenate((jac.q, jac.qd, jac.qdd, jac.tau, jac.ext_forces, jac.ext_torques), axis=-1)
        data_block = jnp.concatenate((jac_model, jac.gc_model, jac.actuator_model), axis=-1).flatten()
        return rows_block, cols_block, data_block

    rows_out, cols_out, data_out = jax.vmap(_node_block)(jnp.arange(nnodes), jac_)
    rows_out = rows_out.reshape(-1)
    cols_out = cols_out.reshape(-1)
    data_out = (1 / info['bodymass'] * data_out).reshape(-1)

    if model.actuator_model.get_n_constraints(model, settings) > 0:
        rows_act_con, cols_act_con, data_act_con = model.actuator_model.jacobian(
            (states_list, globals_dict), constants, model, settings
        )
        rows_act_con = rows_act_con + (info.get('ncons_pernode') * settings.get('nnodes'))  # Shift row indices to avoid overlap
        rows_out = jnp.concatenate([rows_out, rows_act_con], axis=0)
        cols_out = jnp.concatenate([cols_out, cols_act_con], axis=0)
        data_out = jnp.concatenate([data_out, data_act_con], axis=0)
    return rows_out, cols_out, data_out
