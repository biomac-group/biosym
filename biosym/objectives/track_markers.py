import os
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from biosym.ocp.utils import get_row_FK

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
        self.markers = [f"marker_{n}" for n in self.model.markers_parsed.get("base_names", [])]
        self.n_nodes = self.settings["nnodes"]

        eps = 1e-8  # avoid division by zero

        
        # If YAML has `2d: true`, treat Z as zero in expected data
        self.movement_2d = bool(kwargs.get("2d", True))

        if "file" not in kwargs:
            raise ValueError("TrackMarkersObjective requires 'file' in args from YAML.")
        
        # Treadmill speed must be provided explicitly now (no longer pulled from model/settings bounds)
        treadmill_speed = kwargs.get("treadmill_speed", None)
        if treadmill_speed is None:
            treadmill_speed = self.settings.get("treadmill_speed", None)
        if treadmill_speed is None:
            raise ValueError(
                "treadmill_speed must be provided (add 'treadmill_speed' under the objective's YAML kwargs)."
            )
        
        # Get averaged marker data (columns like CLAV_X_mean, ..., CLAV_X_var, ...)
        _, _, gait_marker_angles = segment_gait_averages(n_points=self.n_nodes,  treadmill_speed=treadmill_speed)
        markers_mean_df = gait_marker_angles.filter(like="_mean")
        markers_var_df = gait_marker_angles.filter(like="_var")

        # Validate length
        if markers_mean_df.shape[0] != int(self.n_nodes):
            raise NotImplementedError(
                f"Tracking data length mismatch: objective n_nodes={self.n_nodes} "
                f"but marker tracking data has {markers_mean_df.shape[0]} rows."
            )

        # Build base marker names in model order, strip 'marker_' prefix
        base_names = [n.replace("marker_", "") for n in self.markers]
        n_nodes = int(self.n_nodes)

        # Helper to fetch a mean/var column or fallback (Z->0 mean, 1 var)
        def get_mean_col(marker, ax):
            col = f"{marker}_{ax}_mean"
            if col in markers_mean_df.columns:
                return jnp.asarray(markers_mean_df[col].to_numpy())
            # 2D fallback for missing axis
            return jnp.zeros((n_nodes,))

        def get_var_col(marker, ax):
            col = f"{marker}_{ax}_var"
            if col in markers_var_df.columns:
                return jnp.asarray(markers_var_df[col].to_numpy())
            # Neutral variance when missing axis
            return jnp.ones((n_nodes,))

        # Build [X | Y | Z] blocks in model marker order so they match state layout
        exp_X = jnp.stack([get_mean_col(m, "X") for m in base_names], axis=1)
        exp_Y = jnp.stack([get_mean_col(m, "Y") for m in base_names], axis=1)
        exp_Z = jnp.stack([get_mean_col(m, "Z") for m in base_names], axis=1)

        var_X = jnp.stack([get_var_col(m, "X") for m in base_names], axis=1)
        var_Y = jnp.stack([get_var_col(m, "Y") for m in base_names], axis=1)
        var_Z = jnp.stack([get_var_col(m, "Z") for m in base_names], axis=1)

        # If 2D, set expected Z to zero
        if self.movement_2d:
            exp_Z = jnp.zeros_like(exp_Z)

        self.markers_exp = jnp.concatenate([exp_X, exp_Y, exp_Z], axis=1)  # (N, 3*n_markers)
        self.markers_var = jnp.concatenate([var_X, var_Y, var_Z], axis=1) + eps

        # Number of model markers (tracked in model order)
        self.n_markers = len(base_names)
        self.tracked_markers = base_names
        self.norm_factor = self.n_nodes * self.n_markers

        # Settings passed to objfun
        self.obj_settings = {
            "markers_exp": self.markers_exp,
            "markers_var": self.markers_var,
            "force_2d": self.movement_2d,
            "eps": eps,
            # Provide FK function and layout info so objfun can compute sim markers
            "fk_marker": self.model.run.get("FK_marker"),
            "n_bodies": len(self.model.dicts.get("bodies", [])),
            "n_markers": self.n_markers,
        }

    def _get_info(self):
        # Strip trailing '_mean' from column names to give cleaner marker labels
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Objective term for tracking markers against experimental data.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "n_nodes": self.n_nodes,
            "n_markers": self.n_markers,
            "tracked_markers": self.tracked_markers,
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
    Objective function: Track marker positions (X, Y, Z) vs experimental mean.
    """
    markers_exp = settings["markers_exp"]  # (N, 3*n_markers)
    markers_var = settings["markers_var"]  # (N, 3*n_markers)
    force_2d    = settings.get("force_2d", False)
    eps         = settings.get("eps", 1e-8)

    N = info["n_nodes"]
    n_markers = info["n_markers"]

    # Compute simulated marker positions via FK (bodies + markers), then take marker rows
    fk_marker = settings.get("fk_marker", None)
    n_bodies = settings.get("n_bodies", 0)
    n_markers = settings.get("n_markers", n_markers)

    def fk_step(i):
        # Extract single-timepoint states/constants tailored for FK_marker (1D constants untouched)
        single = get_row_FK(states_list, i)
        # fk_marker returns array of shape (n_bodies + n_markers, 3)
        return fk_marker(single.states, single.constants)

    # Map FK over time points
    pos_all = lax.map(lambda i: fk_step(i), jnp.arange(N, dtype=jnp.int32))
    # Keep only marker rows
    markers_pos = pos_all[:, n_bodies : n_bodies + n_markers, :]  # (N, n_markers, 3)
    sim_X = markers_pos[:, :, 0]
    sim_Y = markers_pos[:, :, 1]
    sim_Z = markers_pos[:, :, 2]

    exp_X = markers_exp[:, 0:n_markers]
    exp_Y = markers_exp[:, n_markers:2 * n_markers]
    exp_Z = markers_exp[:, 2 * n_markers:3 * n_markers]

    var_X = markers_var[:, 0:n_markers]
    var_Y = markers_var[:, n_markers:2 * n_markers]
    var_Z = markers_var[:, 2 * n_markers:3 * n_markers]

    # 2D: track Z to zero target
    if force_2d:
        exp_Z = jnp.zeros_like(exp_Z)

    # Safe variance
    var_X = jnp.clip(jnp.where(jnp.isfinite(var_X), var_X, 1.0), eps, jnp.inf)
    var_Y = jnp.clip(jnp.where(jnp.isfinite(var_Y), var_Y, 1.0), eps, jnp.inf)
    var_Z = jnp.clip(jnp.where(jnp.isfinite(var_Z), var_Z, 1.0), eps, jnp.inf)

    # Weighted squared error per axis
    err_X = (sim_X - exp_X) ** 2 / var_X
    err_Y = (sim_Y - exp_Y) ** 2 / var_Y
    err_Z = (sim_Z - exp_Z) ** 2 / var_Z

    return jnp.mean(err_X) + jnp.mean(err_Y) + jnp.mean(err_Z)