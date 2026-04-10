import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.constraints.base_constraint import BaseConstraint


# any constraint needs to be named Constraint, otherwise it will not be found by the OCP class
class Constraint(BaseConstraint):
    """
    Base class for discretization constraints in the biosym package.

    This class provides a template for implementing specific discretization constraints.
    It includes methods for evaluating the constraint function, computing the Jacobian,
    and retrieving information about the constraint.
    """

    def __init__(self, model, settings, args):
        """
        Initialize the DiscretizationConstraint class with a model and settings.
        """
        self.model = model
        self.settings = settings.copy()
        self.args = args
        self.settings["nvpn"] = len(model.state_vector)
        self.nvar = settings.get("nvar")
        self.vars = args.get("vars", "q")
        if self.vars == "q":
            self.n_var = self.model.coordinates["n"]
        else:
            self.n_var = len(self.model.body_origins)
            raise NotImplementedError("Only 'q' mode is currently implemented for discretization constraints.")

        self.adaptive_h = settings["discretization"]["args"].get("adaptive_h", False)

        self.sections, self.section_constraints = find_dependents(model)


    def _get_info(self):
        """
        Get information about the dynamics constraint.

        This method can be overridden in subclasses to provide specific information.
        """
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Discretization/Continuity constraint class for biosym constraints.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "nnz": self.get_nnz(),
            "ncons": self.get_n_constraints(),
            "nvar": self.n_var,
            "adaptive_h": self.adaptive_h,
            "mode": self.args.get("mode", "backward"),
            "sections": self.sections,
            "ncons_per_node": 2 * self.n_var + self.section_constraints
        }

    def get_confun(self):
        """
        Evaluate the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The dynamics constraint function.
        """
        return jax.jit(partial(confun_q, settings=self.settings, info=self._get_info()))

    def get_jacobian(self):
        """
        Get the Jacobian of the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The Jacobian of the dynamics constraint function.
        """
        return jax.jit(partial(jacobian_q, settings=self.settings, info=self._get_info()))

    def get_n_constraints(self):
        """
        Get the number of constraints defined by this dynamics constraint.

        :return: The number of constraints.
        """
        return (self.settings.get("nnodes_dur") - 1) * (self.n_var * 2 + self.section_constraints) # *2: for qd and qdd

    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian of the dynamics constraint.

        :return: The number of non-zero entries.
        """
        return self.get_n_constraints() * 4  # q(t), q(t+1), qd(t), h


def confun_at_node(states_list, next_states_list, globals_dict, settings, info, h):
    """
    Evaluate the constraint function at a specific node.

    :param states_list: List containing the current states.
    :param next_states_list: List containing the next states.
    :param globals_dict: Dictionary containing global variables.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The evaluated value of the constraint function at the node.
    """
    q_i = states_list.states.model[: 2 * info["nvar"]]
    q_i_next = next_states_list.states.model[: 2 * info["nvar"]]
    qd_0 = (q_i_next - q_i) / h
    qd_states = (
        next_states_list.states.model[info["nvar"] : 3 * info["nvar"]]
        if info["mode"] == "backward"
        else states_list.states.model[info["nvar"] : 3 * info["nvar"]]
    )
    
    # Contact model
    sec_ = info['sections']
    for section in ['contact_model', 'actuator_model']:
        if section in sec_:
            if len(sec_[section]['states']) > 0:
                if section == 'contact_model':
                    q_ic = states_list.states.gc_model[sec_[section]['states']]
                    q_ic_next = next_states_list.states.gc_model[sec_[section]['states']]
                    q_id = (
                            next_states_list.states.gc_model[sec_[section]['derivatives']]
                            if info["mode"] == "backward"
                            else states_list.states.gc_model[sec_[section]['derivatives']])
                else:
                    q_ic = states_list.states.actuator_model[sec_[section]['states']]
                    q_ic_next = next_states_list.states.actuator_model[sec_[section]['states']]
                    q_id = (
                            next_states_list.states.actuator_model[sec_[section]['derivatives']]
                            if info["mode"] == "backward"
                            else states_list.states.actuator_model[sec_[section]['derivatives']])
                qd_i0 = (q_ic_next - q_ic) / h
                qd_0 = jnp.concatenate((qd_0, qd_i0))
                qd_states = jnp.concatenate((qd_states, q_id))
    return qd_0 - qd_states


def jacobian_at_node(states_list, next_states_list, globals_dict, settings, info, h):
    """
    Compute the Jacobian of the constraint function at a specific node.

    :param states_list: List containing the current states.
    :param next_states_list: List containing the next states.
    :param globals_dict: Dictionary containing global variables.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The Jacobian of the constraint function at the node.
    """
    q_i = states_list.states.model[: 2 * info["nvar"]]
    q_i_next = next_states_list.states.model[: 2 * info["nvar"]]

    d1 = -jnp.ones(info["nvar"] * 2) / h  # df/dq_i
    d2 = jnp.ones(info["nvar"] * 2) / h  # df/dq_i_next
    d3 = -jnp.ones(info["nvar"] * 2)  # df/dqd_states
    d4 = -(q_i_next - q_i) / (h**2)  # df/dh

    r = jnp.arange(info["nvar"] * 2, dtype=int)

    c1 = jnp.arange(info["nvar"] * 2, dtype=int)
    c2 = jnp.arange(info["nvar"] * 2, dtype=int) + states_list.states.size()
    c3 = (
        jnp.arange(info["nvar"] * 2, dtype=int)
        + info["nvar"]
        + (states_list.states.size() if info["mode"] == "backward" else 0)
    )
    if info["adaptive_h"]:
        c4 = (
            jnp.ones(info["nvar"] * 2, dtype=int) * states_list.states.size() - 1
        )  # adaptive step size, h is always the last variable
    else:
        c4 = jnp.ones(info["nvar"] * 2, dtype=int)

    r, c, d = jnp.concatenate((r, r, r, r)), jnp.concatenate((c1, c2, c3, c4)), jnp.concatenate((d1, d2, d3, d4))

    sec_ = info['sections']
    for section in ['contact_model', 'actuator_model']:
        if section in sec_:
            if len(sec_[section]['states']) > 0:
                if section == 'contact_model':
                    q_ic = states_list.states.gc_model[sec_[section]['states']]
                    q_ic_next = next_states_list.states.gc_model[sec_[section]['states']]
                    n_curr = 0
                else:
                    q_ic = states_list.states.actuator_model[sec_[section]['states']]
                    q_ic_next = next_states_list.states.actuator_model[sec_[section]['states']]
                    n_curr = states_list.states.gc_model.size
                l_0 = len(sec_[section]['states'])
                n_model = states_list.states.model.size
                d1 = -jnp.ones(l_0) / h  # df/dq_i
                d2 = jnp.ones(l_0) / h  # df/dq_i_next
                d3 = -jnp.ones(l_0 )  # df/dqd_states
                d4 = -(q_ic_next - q_ic) / (h**2)  # df/dh

                ri = jnp.arange(l_0, dtype=int) + info["nvar"] * 2

                c1 = n_model + n_curr + sec_[section]['states']
                c2 = n_model + n_curr + sec_[section]['states'] + states_list.states.size()
                c3 = n_model + n_curr + sec_[section]['derivatives'] + (states_list.states.size() if info["mode"] == "backward" else 0)
                if info["adaptive_h"]:
                    c4 = (
                        jnp.ones(l_0, dtype=int) * states_list.states.size() - 1
                    )  # adaptive step size, h is always the last variable
                else:
                    c4 = jnp.ones(l_0, dtype=int)

                r, c, d = jnp.concatenate((r, ri, ri, ri, ri)), jnp.concatenate((c, c1, c2, c3, c4)), jnp.concatenate((d, d1, d2, d3, d4))
    return r, c, d


# @partial(jax.grad, argnums=(1, 2))
# @partial(jax.jit, static_argnums=(0))
def confun_q(states_list, globals_dict, settings, info):
    """
    Placeholder for the constraint function.

    This function should be implemented in subclasses to evaluate the dynamics constraints.

    :param states_list: List containing the current states.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The evaluated value of the constraint function.
    """
    data_out = jnp.empty((info["ncons"],), dtype=float)
    nnodes = settings.get("nnodes_dur")
    if info["adaptive_h"]:
        h = states_list.states.h[
            : nnodes - 1
        ]  # Adaptive step size for each node, h is always the step to the next node
    else:
        h = jnp.ones(nnodes - 1) * globals_dict.dur / (nnodes - 1)  # Constant step size
    nvar = info["nvar"]
    def body_fun(n, carry):
        data_out = carry
        cons = confun_at_node(states_list[n], states_list[n + 1], globals_dict, settings, info, h[n])
        data_out = jax.lax.dynamic_update_slice(data_out, cons, (n*info['ncons_per_node'],))
        return data_out

    data_out = jax.lax.fori_loop(0, nnodes - 1, body_fun, (data_out))
    return data_out


def jacobian_q(states_list, globals_dict, settings, info):
    """
    Placeholder for the Jacobian of the constraint function.

    This function should be implemented in subclasses to compute the Jacobian of the dynamics constraints.

    :param states_list: List containing the current states.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The Jacobian of the constraint function.
    """
    nnodes = settings.get("nnodes_dur")
    if info["adaptive_h"]:
        h = states_list.states.h[
            : nnodes - 1
        ]  # Adaptive step size for each node, h is always the step to the next node
    else:
        h = jnp.ones(nnodes - 1) * globals_dict.dur / (nnodes - 1)  # Constant step size
    nvar = info["nvar"]

    rows_out, cols_out, data_out = (
        jnp.empty((info["nnz"],), dtype=int),
        jnp.empty((info["nnz"],), dtype=int),
        jnp.empty((info["nnz"],), dtype=float),
    )

    def body_fun(n, carry):
        rows_out, cols_out, data_out = carry
        state_ = states_list[n]
        r, c, d = jacobian_at_node(state_, states_list[n + 1], globals_dict, settings, info, h[n])
        r = r + n * info['ncons_per_node'] # Adjust row indices for the current node
        c = c + n * states_list[n].states.size()

        # Divide by number of nodes and set the indices to the globals column
        if not info["adaptive_h"]:
            d = d.at[6*nvar:8*nvar].multiply(1 / (nnodes - 1))
            c = c.at[6*nvar:8*nvar].set(settings.get("nnodes_dur") * states_list[n].states.size())
            sec_ = info['sections']
            for section in ['contact_model', 'actuator_model']:
                if section in sec_:
                    if section == 'actuator_model':
                        extra = len(sec_['contact_model']['states']) if 'contact_model' in sec_ else 0
                        this = len(sec_['actuator_model']['states'])
                    else:
                        extra = 0
                        this = len(sec_['contact_model']['states'])
                    d = d.at[8*nvar+4*extra+3*this:8*nvar+4*extra+4*this].multiply(1 / (nnodes - 1))
                    c = c.at[8*nvar+4*extra+3*this:8*nvar+4*extra+4*this].set(settings.get("nnodes_dur") * states_list[n].states.size())
        start = n * 4 * info['ncons_per_node']  # Calculate where to insert this block

        rows_out = jax.lax.dynamic_update_slice(rows_out, r, (start,))
        cols_out = jax.lax.dynamic_update_slice(cols_out, c, (start,))
        data_out = jax.lax.dynamic_update_slice(data_out, d, (start,))

        return (rows_out, cols_out, data_out)

    rows_out, cols_out, data_out = jax.lax.fori_loop(0, nnodes - 1, body_fun, (rows_out, cols_out, data_out))
    return rows_out, cols_out, data_out


def find_dependents(model):
    results = {}
    n_constraints = 0

    for section in ['contact_model', 'actuator_model']:
        # Check if the section exists in the model
        if not hasattr(model, section):
            continue
        curr_model = getattr(model, section)
        if curr_model.get_n_states() > 1:
            state_vector = curr_model.state_vector

            states = []
            derivatives = []

            # Build a map from base names to their indices
            for i, name in enumerate(state_vector):
                if (name + '_dot') in state_vector:
                    states.append(i)
                    derivatives.append(state_vector.index(name+'_dot'))
                    if (name + '_ddot') in state_vector:
                        states.append(state_vector.index(name+'_dot'))
                        derivatives.append(state_vector.index(name+'_ddot'))


            results[section] = {
                'states': jnp.array(states),
                'derivatives': jnp.array(derivatives)
            }
            n_constraints += len(states)
    return results, n_constraints