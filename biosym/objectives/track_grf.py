import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from biosym.constraints.dynamics import calc_forces
from biosym.objectives.base_objective import BaseObjective
from biosym.utils import read_mot, segment_gait_averages


class Objective(BaseObjective):
    """
    Objective term for tracking experimental ground reaction forces (GRFs).
    """

    def __init__(self, model, settings, **kwargs):
        """
        Initialize the objective with experimental GRF data.

        :param model: biosym model object.
        :param settings: Dictionary containing settings for the objective function.
        :param datafile: Path to CSV file with mean and variance GRFs.
        """
        self.model = model
        self.settings = settings
        self.n_nodes = self.settings["nnodes"]
        self.n_grfs = self.model.ext_forces.n
        self.norm_factor = self.n_nodes * self.n_grfs

        # read grf file from yaml either as pre-segmented mean/var or raw data
        preseg_file_path = kwargs.get("presegmented_file", None)
        preseg = bool(kwargs.get("presegmented"))

        if preseg:
            # Treat file_path as a CSV that already contains mean/var columns
            grf_df = read_mot(preseg_file_path)
            grf_mean_df = grf_df.filter(like="_mean")
            grf_var_df = grf_df.filter(like="_var")
        else:
            _, gait_grfs, _ = segment_gait_averages(n_points=self.n_nodes)
            grf_mean_df = gait_grfs.filter(like="_mean")
            grf_var_df = gait_grfs.filter(like="_var")

        # No variance columns present (e.g. reference data that intentionally
        # doesn't track variance) -- fall back to uniform (unweighted) error.
        if grf_var_df.shape[1] == 0:
            grf_var_df = pd.DataFrame(
                np.ones_like(grf_mean_df.values),
                columns=[c.replace("_mean", "_var") for c in grf_mean_df.columns],
                index=grf_mean_df.index,
            )

        # The tracking data is always laid out left-foot-then-right-foot
        # (see segment_gait_cycles.create_averaged_gait_forces), but the
        # model's ext_forces slots are ordered by contact-body definition
        # order in the model file, which need not be left-then-right (e.g.
        # gait2d test models define the right foot's contact points first).
        # Reindex the data columns to match the model's actual body order
        # so each ext_forces slot is compared against the correct foot.
        grf_mean_df = grf_mean_df[self._matching_columns(grf_mean_df.columns)]
        grf_var_df = grf_var_df[self._matching_columns(grf_var_df.columns)]

        # number of rows (time points) must match n_nodes
        if grf_mean_df.shape[0] != int(self.n_nodes):
            raise NotImplementedError(
                f"Tracking data length mismatch: objective n_nodes={self.n_nodes} "
                f"but grf tracking data has {grf_mean_df.shape[0]} rows."
            )
        self.grf_exp = jnp.asarray(grf_mean_df.values)
        self.grf_var = jnp.asarray(grf_var_df.values) + 1e-8

        # attach arrays into a settings dict passed to objfun so signature matches others
        self.obj_settings = {
            "grf_exp": self.grf_exp,
            "grf_var": self.grf_var,
            "constants": self.model.default_constants,
        }

    def _matching_columns(self, columns):
        """
        Reorder tracking-data columns (named '<1_ground_force|ground_force>_v<dim>_<mean|var>')
        so they line up with self.model.ext_forces.names ('f_<body>_<dim>'), body by body,
        instead of assuming the data's left-then-right layout matches the model's contact-body
        order.
        """
        suffix = "_mean" if any(c.endswith("_mean") for c in columns) else "_var"
        available = set(columns)
        ordered = []
        for name in self.model.ext_forces.names[::3]:
            body = name[2:-2]  # strip "f_" prefix and "_x" dim suffix
            if body.endswith(("_l", "_L")):
                prefix = "1_ground_force"
            elif body.endswith(("_r", "_R")):
                prefix = "ground_force"
            else:
                raise NotImplementedError(
                    f"Cannot infer left/right side of contact body '{body}' from its name; "
                    "expected it to end with '_r'/'_l' (case-insensitive)."
                )
            for dim in ["x", "y", "z"]:
                col = f"{prefix}_v{dim}{suffix}"
                if col not in available:
                    raise NotImplementedError(
                        f"Tracking data is missing expected column '{col}' for contact body '{body}'."
                    )
                ordered.append(col)
        return ordered

    def _get_info(self):
        # Provide info used by objfun and gradient builder
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Objective term for tracking GRFs against experimental data.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "n_nodes": self.n_nodes,
            "n_grfs": self.n_grfs,
            "norm_factor": self.norm_factor,
        }

    def get_objfun(self):
        """Return the objective function."""
        fun = partial(
            objfun,
            model=self.model,
            settings=self.obj_settings,
            info=self._get_info(),
        )
        return jax.jit(fun)

    def get_gradient(self):
        """Return the gradient of the objective function."""
        fun = partial(
            objfun,
            model=self.model,
            settings=self.obj_settings,
            info=self._get_info(),
        )
        # argnums: 0=states, 1=globals_dict, so differntiate w.r.t. states and globals
        return jax.jit(jax.grad(fun, argnums=[0, 1]))


def objfun(states_list, globals_dict, model, settings, info):
    """
    Objective function: Track GRFs vs experimental mean/var.

    The simulated GRFs are computed by running the ground-contact model's
    forward pass on the current states (the same computation the dynamics
    constraint uses), rather than reading the ext_forces state slot -- that
    slot is a free decision variable that is only tied to the true contact
    force indirectly through the (soft) dynamics constraint, so it need not
    match the physically-consistent GRF away from a fully converged solution.
    """

    # read expected values from settings passed in
    grf_exp = settings["grf_exp"]
    grf_var = settings["grf_var"]
    constants = settings["constants"]
    n_nodes = info["n_nodes"]

    # Evaluate the contact model's forward pass per node (it is defined for a
    # single node, so vmap it over the leading node axis of the trajectory).
    nodes = states_list[:n_nodes]
    filled = jax.vmap(calc_forces, in_axes=(0, None, None))(nodes, model, constants)
    grf_sim = filled.ext_forces

    # Weighted squared error
    error = (grf_sim - grf_exp) ** 2 / grf_var

    return jnp.mean(error)
