import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.objectives.base_objective import BaseObjective
from biosym.utils import segment_gait_averages


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
        self.initial_joints = self.model.coordinates["names"]
        self.n_nodes = self.settings["nnodes"]

        eps = 1e-8  # avoid division by zero

        if "file" not in kwargs:
            raise ValueError("TrackAnglesObjective requires 'file' in args from YAML.")

        # segment_gait_averages returns (gait_avg_joint_angles, gait_avg_qs)
        gait_joint_angles, _ = segment_gait_averages(n_points=self.n_nodes)
        # gait_joint_angles expected to have "<channel>_mean" and "<channel>_var" columns
        q_mean_df = gait_joint_angles.filter(like="_mean")
        q_var_df = gait_joint_angles.filter(like="_var")

        # number of rows (time points) must match n_nodes
        if q_mean_df.shape[0] != int(self.n_nodes):
            raise NotImplementedError(
                f"Tracking data length mismatch: objective n_nodes={self.n_nodes} "
                f"but  angle tracking data has {q_mean_df.shape[0]} rows."
            )

        # # column names (stripped of '_mean') must match the model coordinate names
        # tracking_cols = [c.replace("_mean", "") for c in q_mean_df.columns.tolist()]
        # model_coord_names = list(self.joints)
        # if tracking_cols != model_coord_names:
        #     raise NotImplementedError(
        #         "Tracking data joint names do not match model coordinates"
        #     )

        # exclude certain coordinates from tracking
        exclude = kwargs.get("exclude", None)
        exclude_names_list = []
        if exclude is not None:
            for e in exclude:
                exclude_names_list.append(e)

        # build list of tracked indices (relative to coordinate slice)
        tracked_indices = [
            i
            for i, name in enumerate(self.initial_joints)
            if name not in exclude_names_list
        ]
        self.tracked_indices = tuple(tracked_indices)  # store as tuple for immutability

        # index pandas DataFrame by integer positions
        cols = list(self.tracked_indices)
        self.q_exp = jnp.asarray(
            q_mean_df.iloc[:, cols].values
        )  # shape (n_points, n_tracked)
        self.q_var = jnp.asarray(q_var_df.iloc[:, cols].values) + eps

        self.n_joints = len(self.tracked_indices)
        self.joints = [self.initial_joints[i] for i in self.tracked_indices]
        self.norm_factor = self.n_nodes * self.n_joints

        # attach arrays into a settings dict passed to objfun so signature matches others
        self.obj_settings = {"q_exp": self.q_exp, "q_var": self.q_var}

    def _get_info(self):
        # strip trailing '_mean' from column names to give cleaner joint labels
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Objective term for tracking joint angles against experimental data.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "n_nodes": self.n_nodes,
            "n_joints": self.n_joints,
            "tracked_indices": self.tracked_indices,
            "joints": self.joints,
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

    # read expected values from settings passed in
    q_exp = settings["q_exp"]
    q_var = settings["q_var"]

    # Extract simulated joint angles from states
    q_sim = states_list.states.model[: info["n_nodes"], info["tracked_indices"]]

    # Weighted squared error
    error = (q_sim - q_exp) ** 2 / q_var

    return jnp.mean(error)
