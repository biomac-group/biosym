import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.objectives.base_objective import BaseObjective


class Objective(BaseObjective):
    """
    Objective term for minimizing torques.
    """

    def __init__(self, model, settings, **kwargs):
        """
        Initialize the BaseObjective class with a model and settings.
        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the objective function.
        :param weighting: "volumeweighted" (default, for muscle-driven models) or
            "equal". Per-actuator weights used in the weighted mean over
            actuators, matching BioMAC-Sim-Toolbox's effortTermMuscles.m:
            "volumeweighted" weights each muscle by fmax * lceopt (a proxy for
            muscle volume/mass, since muscle mass ~ (fmax/sigma) * rho * lceopt),
            normalized to sum to 1; "equal" weights every actuator 1/n_actuators.
            Falls back to "equal" (with a warning) for non-muscle actuator
            models that have no fmax/lceopt (e.g. pure torque-driven models).
        """
        self.model = model
        self.settings = settings

        if "exponent" in kwargs:
            self.exponent = kwargs["exponent"]
        else:
            self.exponent = 2

        self.speedweighting = kwargs.get("speedweighting", False)

        n_actuators = self.model.actuators.get_n_actuators()
        self.n_actuators = n_actuators
        muscle_constants = getattr(self.model.actuators, "muscle_constants", None)
        has_muscle_volume = (
            muscle_constants is not None and "fmax" in muscle_constants and "lceopt" in muscle_constants
        )
        weighting = kwargs.get("weighting", "volumeweighted" if has_muscle_volume else "equal")

        if weighting == "volumeweighted":
            if not has_muscle_volume:
                raise ValueError(
                    "weighting='volumeweighted' requires a muscle actuator model exposing "
                    "'fmax' and 'lceopt' (e.g. Hill2d); this model's actuators don't have those."
                )
            # Muscle "volume" proxy: mass ~ (fmax / sigma) * rho * lceopt, sigma/rho constant
            # across muscles, so fmax * lceopt is proportional to muscle volume/mass.
            volume = jnp.asarray(muscle_constants["fmax"]).flatten() * jnp.asarray(muscle_constants["lceopt"]).flatten()
            weights = volume / jnp.sum(volume)
        elif weighting == "equal":
            weights = jnp.ones(n_actuators) / n_actuators
        else:
            raise ValueError(f"Unknown weighting '{weighting}'. Valid options: 'volumeweighted', 'equal'.")
        self.weighting = weighting
        self.weights = weights

    def _get_info(self):
        """
        Get information about the objective function.
        This method can be overridden in subclasses to provide specific information.
        """
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Objective term for minimizing effort.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "idx_int_forces": self.model.tau.combined_idx,
            "n_int_forces": self.model.tau.n,
            # Hill2d/Millard2012 expose an "a" (activation) slot per actuator
            # within a larger per-actuator state block (Lce, a, e, ... --
            # Hill2d's Lce_dot is derived, not a state, see hill2d.py);
            # CoordinateActuator/TorqueActuator have no such split -- their
            # entire actuator_model state block *is* one value per actuator
            # (the commanded generalized force itself), so every slot is the
            # quantity to penalize.
            "range_actuators": (
                self.model.actuators.idx["a"] if hasattr(self.model.actuators, "idx")
                else jnp.arange(self.n_actuators)
            ),
            "exponent": self.exponent,
            "speedweighting": self.speedweighting,
            "weights": self.weights,
        }

    def get_objfun(self):
        """:return: The objective function."""
        fun = partial(objfun, settings=self.settings, info=self._get_info())
        return jax.jit(fun)

    def get_gradient(self):
        """:return: The gradient of the objective function."""
        fun = partial(objfun, settings=self.settings, info=self._get_info())
        return jax.jit(jax.grad(fun, argnums=[0, 1]))


def objfun(states_list, globals_dict, settings, info):
    """
    Evaluate the objective function.

    :param model: biosym model object.
    :param states_list: Dictionary containing the current states.
    :param settings: Settings for the objective function.
    :param info: Information about the objective function.
    :return: The evaluated value of the objective function.
    """
    # Per-node average control cost (no `dur` folded in here -- unlike a
    # stray `* dur` before the exponent, which would spuriously raise dur to
    # the exponent power in the total cost and reward shrinking it). The
    # nnodes normalization must happen *after* the sum/power, not before --
    # dividing forces by nnodes first would raise nnodes to `exponent`'s
    # power too, the exact bug this comment warns against for dur.
    forces = states_list.actuator_model[: settings["nnodes"], info["range_actuators"]]
    # info["weights"] sums to 1 over actuators (volume-weighted or equal, see Objective.__init__),
    # matching BioMAC-Sim-Toolbox's effortTermMuscles.m: sum(weights' * X.^exponent) / nNodes.
    output = jnp.sum(jnp.abs(jnp.power(forces, info["exponent"])) * info["weights"][None, :]) / settings["nnodes"]

    if globals_dict is not None and info["speedweighting"]:
        # Normalize the already-summed cost by speed^exponent (not per-node).
        output = output / jnp.power(globals_dict.speed, info["exponent"])

    return output
