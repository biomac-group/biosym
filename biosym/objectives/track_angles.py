import os
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd

from biosym.objectives.base_objective import BaseObjective


class Objective(BaseObjective):
    """
    Objective term for tracking experimental joint angles.
    """

    def __init__(self, model, settings, **kwargs):
        """
        Initialize the objective with experimental joint angle data.

        :param model: biosym model object.
        :param settings: Dictionary containing settings for the objective function.
        :param datafile: Path to CSV file with mean and variance joint angles.
        """
        self.model = model
        self.settings = settings

        if "file" not in kwargs:
            raise ValueError("TrackAnglesObjective requires 'file' in args from YAML.")
        filepath = kwargs["file"]
        joint_angles = pd.read_csv(filepath)
        joint_angles_mean = joint_angles.filter(like="mean")
        joint_angles_var = joint_angles.filter(like="var")

        eps = 1e-8  # avoid division by zero
        self.q_exp = jnp.array(joint_angles_mean.values)
        self.q_var = jnp.array(joint_angles_var.values) + eps
        # attach arrays into a settings dict passed to objfun so signature matches others
        self.obj_settings = {"q_exp": self.q_exp, "q_var": self.q_var}

        self.n_nodes = self.settings["nnodes"]
        self.n_joints = len(joint_angles_mean.columns)
        self.norm_factor = self.n_nodes * self.n_joints

    def _get_info(self):
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Objective term for tracking joint angles against experimental data.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "n_nodes": self.n_nodes,
            "n_joints": self.n_joints,
            "norm_factor": self.norm_factor,
        }

    def get_objfun(self):
        """Return the objective function."""
        fun = partial(
            objfun,
            settings=self.obj_settings,
            info=self._get_info(),
        )
        return jax.jit(fun)

    def get_gradient(self):
        """Return the gradient of the objective function."""
        fun = partial(
            objfun,
            settings=self.obj_settings,
            info=self._get_info(),
        )
        # argnums: 0=states, 1=globals_dict, so differntiate w.r.t. states and globals
        return jax.jit(jax.grad(fun, argnums=[0, 1]))


def objfun(states_list, globals_dict, settings, info):
    """
    Objective function: Track joint angles vs experimental mean.
    """
    # Extract simulated joint angles from states
    q_sim = states_list.states.model[: info["n_nodes"], : info["n_joints"]]

    # read expected values from settings passed in
    q_exp = settings["q_exp"]
    q_var = settings["q_var"]
    # Weighted squared error
    error = (q_sim - q_exp) ** 2 / q_var
    return jnp.sum(error) / info["norm_factor"]
