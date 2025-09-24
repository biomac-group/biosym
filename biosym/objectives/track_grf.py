import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.objectives.base_objective import BaseObjective
from biosym.utils import read_mot, segment_gait_averages


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
        self.n_nodes = self.settings["nnodes"]
        self.n_grfs = self.model.ext_forces["n"]
        self.norm_factor = self.n_nodes * self.n_grfs

        if "file" not in kwargs:
            raise ValueError("TrackGRFSObjective requires 'file' in args from YAML.")

        # read grf file from yaml either as pre-segmented mean/var or raw data
        preseg_file_path = kwargs.get("presegmented_file", None)
        preseg = bool(kwargs.get("presegmented"))

        if preseg:
            # Treat file_path as a CSV that already contains mean/var columns
            grf_df = read_mot(preseg_file_path)
            grf_mean_df = grf_df.filter(like="_mean")
            grf_var_df = grf_df.filter(like="_var")
        else:
            _, gait_grfs = segment_gait_averages(n_points=self.n_nodes)
            grf_mean_df = gait_grfs.filter(like="_mean")
            grf_var_df = gait_grfs.filter(like="_var")

        # number of rows (time points) must match n_nodes
        if grf_mean_df.shape[0] != int(self.n_nodes):
            raise NotImplementedError(
                f"Tracking data length mismatch: objective n_nodes={self.n_nodes} "
                f"but grf tracking data has {grf_mean_df.shape[0]} rows."
            )
        self.grf_exp = jnp.asarray(grf_mean_df.values)
        self.grf_var = jnp.asarray(grf_var_df.values) + 1e-8

        # attach arrays into a settings dict passed to objfun so signature matches others
        self.obj_settings = {"grf_exp": self.grf_exp, "grf_var": self.grf_var}

    def _get_info(self):
        # Provide info used by objfun and gradient builder
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Objective term for tracking GRFs against experimental data.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "n_nodes": self.n_nodes,
            "idx_grfs": self.model.ext_forces["idx"],
            "n_grfs": self.n_grfs,
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
    Objective function: Track GRFs vs experimental mean/var.

    Behavior:
      - If settings contains 'grf_exp'/'grf_var' (precomputed), use them (fast).
      - Elif globals_dict contains 'grf_exp'/'grf_var', use those.
      - Else call biosym.utils.segment_gait_cycles.segment_gait_averages()
        and use the returned gait_avg_grfs (second element).
    """

    # read expected values from settings passed in
    grf_exp = settings["grf_exp"]
    grf_var = settings["grf_var"]

    # simulated GRFs from state vector
    grf_sim = states_list.states.model[
        : info["n_nodes"], info["idx_grfs"] : info["idx_grfs"] + info["n_grfs"]
    ]

    # Weighted squared error
    error = (grf_sim - grf_exp) ** 2 / grf_var

    return jnp.mean(error)
