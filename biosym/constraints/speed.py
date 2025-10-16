import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.constraints.base_constraint import BaseConstraint


# any constraint needs to be named Constraint, otherwise it will not be found by the OCP class
class Constraint(BaseConstraint):
    """
    Base class for speed constraints in the biosym package.
    """

    def __init__(self, model, settings, args):
        """
        Initialize the SpeedConstraint class with a model and settings.
        """
        self.model = model
        self.settings = settings.copy()
        self.args = args
        self.settings["nvpn"] = len(model.state_vector)
        self.nvar = settings.get("nvar")

    def _get_info(self):
        """
        Get information about the dynamics constraint.

        This method can be overridden in subclasses to provide specific information.
        """
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Speed constraint class for biosym constraints.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "nnz": self.get_nnz(),
            "ncons": self.get_n_constraints(),
            "speed_var_idx": self.model.speeds["idx"],
        }

    def get_confun(self):
        """
        Evaluate the speed constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The speed constraint function.
        """
        return jax.jit(partial(confun, settings=self.settings, info=self._get_info()))

    def get_jacobian(self):
        """
        Get the Jacobian of the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The Jacobian of the dynamics constraint function.
        """
        return jax.jit(partial(jacobian, settings=self.settings, info=self._get_info()))

    def get_n_constraints(self):
        """
        Get the number of constraints defined by this dynamics constraint.

        :return: The number of constraints.
        """
        return 1

    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian of the dynamics constraint.

        :return: The number of non-zero entries.
        """
        return self.settings.get("nnodes_dur")


def confun(states_list, globals_dict, settings, info):
    """
    Evaluate the constraint function for adaptive step sizes.
    :param states_list: Dictionary containing the current states.
    :param globals_dict: Dictionary containing global variables (e.g., duration).
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The evaluated constraint function.
    """
    print(states_list.states.model)
    return globals_dict.speed - jnp.mean(
        states_list.states.model[: settings.get("nnodes_dur") - 1, info["speed_var_idx"]]
    )  # Ensure the sum of step sizes equals the total duration


def jacobian(states_list, globals_dict, settings, info):
    """
    Placeholder for the Jacobian of the constraint function.

    This function should be implemented in subclasses to compute the Jacobian of the dynamics constraints.

    :param states_list: List containing the current states.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The Jacobian of the constraint function.
    """
    r = jnp.zeros((info["nnz"],), dtype=int)
    c = (
        jnp.arange(info["nnz"], dtype=int) * states_list[0].states.size() + info["speed_var_idx"] - 1
    )  # The -1 seems pretty wrong, but we keep it here
    c = c.at[-1].set(settings.get("nnodes_dur") * states_list[0].states.size()) + 1
    d = -jnp.ones((info["nnz"],), dtype=float) / (settings.get("nnodes_dur") - 1)
    d = d.at[-1].set(1.0)  # The last entry corresponds to the total duration constraint
    return r, c, d
