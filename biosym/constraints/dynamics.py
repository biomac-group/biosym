import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.constraints.base_constraint import BaseConstraint

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

        args = args or {}
        confun_type = args.get('dynamics_function', "newton-euler")

        if confun_type == "newton-euler":
            confun_mm = partial(confun_mm_tau, model=self.model)
            self.model._precompile_fn(confun_mm, (self.model.default_states, self.model.default_constants), "confun_mm", is_jax_fn=True)
            self.model._precompile_fn(confun_mm, (self.model.default_states, self.model.default_constants), "confun_mm_jacobian", jacobian=True, is_jax_fn=True)
            self.con_string = "confun_mm"
        elif confun_type == 'kane':
            self.con_string = "kane"
        elif confun_type in ['rnea', 'aba']:
            raise NotImplementedError("Dynamics function based on RNEA and ABA not implemented yet.")
        else:
            raise ValueError(f"Dynamics function {confun_type} not recognized.")
        

    def _get_info(self):
        """
        Get information about the dynamics constraint.

        This method can be overridden in subclasses to provide specific information.
        """
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Base dynamics constraint class for biosym constraints.",
            "required_variables": {"states": ["q","qd","qdd","tau","ext_forces","ext_torques"], "constants": ["model"]},
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
        return jax.jit(partial(confun, self.model.run[self.con_string], settings=self.settings, info=self._get_info(), model=self.model))

    def get_jacobian(self):
        """
        Get the Jacobian of the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The Jacobian of the dynamics constraint function.
        """
        return jax.jit(partial(jacobian, self.model.run[f"{self.con_string}_jacobian"], settings=self.settings, info=self._get_info(), model=self.model))

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


def confun(modelfn, states_list, globals_dict, settings, info, model):
    """
    Evaluate the dynamics constraint (equations of motion residuals).

    Calls the model-provided Kane function which handles batching internally.

    :param modelfn: model.run["kane"] — takes (states, constants) → residuals
    :param states_list: batched States object of shape (nnodes, ...)
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: Flattened residual vector of shape (nnodes * ncons_model,)
    """
    nnodes = settings.get("nnodes")
    return modelfn(states_list[:nnodes], model.default_constants).flatten()


def jacobian(modelfn, states_list, globals_dict, settings, info, model):
    """
    Compute the sparse COO Jacobian of the dynamics constraint.

    Uses model.run["kane_jacobian"] which returns a States-shaped jacobian
    (jacobian of residuals w.r.t. model states, vmapped over nodes).

    :param modelfn: model.run["kane_jacobian"] — takes (states, constants) → States-shaped jac
    :param states_list: batched States object of shape (nnodes, ...)
    :return: (rows, cols, data) sparse COO arrays
    """
    nnz = info["nnz"]
    nvpn = settings.get("nvpn")            # model-only state count (q,qd,qdd,tau,f,m)
    nvpn_total = states_list[0].size()     # full state per node (model + gc + actuator)
    nnodes = settings.get("nnodes")
    ncons_pernode = info["ncons_pernode"]

    # jac is a States-shaped pytree; each field has shape (nnodes, ncons_pernode, field_dim)
    # (or squeezed to (ncons_pernode, field_dim) for nnodes=1)
    jac = modelfn(states_list[:nnodes], model.default_constants).to_array().flatten()
    # Build COO indices
    node_indices = jnp.arange(nnodes)

    # Row indices: for node n, constraints are [n*ncons_pernode, ..., (n+1)*ncons_pernode - 1]
    row_base = node_indices[:, jnp.newaxis] * ncons_pernode + jnp.arange(ncons_pernode)[jnp.newaxis, :]
    # col indices: node stride is nvpn_total (full state), but dynamics only touches the
    # first nvpn (model) columns within each node block.
    col_base = node_indices[:, jnp.newaxis] * nvpn_total + jnp.arange(nvpn)[jnp.newaxis, :]

    # Expand to (nnodes, ncons_pernode, nvpn)
    rows = jnp.repeat(row_base[:, :, jnp.newaxis], nvpn, axis=2)      # (nnodes, ncons_pernode, nvpn)
    cols = jnp.repeat(col_base[:, jnp.newaxis, :], ncons_pernode, axis=1)  # (nnodes, ncons_pernode, nvpn)

    rows_out = rows.flatten()
    cols_out = cols.flatten()
    data_out = jac

    return rows_out, cols_out, data_out
