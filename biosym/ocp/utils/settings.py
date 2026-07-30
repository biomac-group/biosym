"""
Utilities for processing and validating collocation settings.
"""

import numpy as np
import jax.numpy as jnp
from biosym.utils import states


def flat_model_to_states(model, v, states_template=None):
    q_len = model.coordinates.n
    qd_len = model.speeds.n
    qdd_len = model.accs.n
    tau_len = model.tau.n
    ext_f_len = model.ext_forces.n
    ext_t_len = model.ext_torques.n
    
    i = 0
    q = v[i : i + q_len]
    i += q_len
    qd = v[i : i + qd_len]
    i += qd_len
    # qdd/tau/ext_forces/ext_torques are no longer free decision variables
    # (they're derived on the fly), so their bound values are unused here.
    i += qdd_len + tau_len + ext_f_len + ext_t_len

    if states_template is None:
        states_template = model.opt_states

    return states_template.replace(
        q=q,
        qd=qd,
    )


def process_collocation_settings(model, settings):
    """
    Process and validate collocation settings for optimal control problems.

    This function takes raw settings from configuration files or dictionaries
    and processes them into a standardized format suitable for collocation-based
    optimal control. It validates required keys, sets up discretization parameters,
    and handles data type conversions.

    Parameters
    ----------
    model : BiosymModel
        The biomechanical model for which settings are being processed.
    settings : dict
        Dictionary containing collocation method configuration including:
        - settings: solver parameters, nnodes, discretization
        - constraints: constraint specifications
        - objectives: objective function specifications
        - bounds: variable bounds for optimization

    Returns
    -------
    dict
        Processed and validated settings dictionary with additional computed
        fields like dtype conversions, node handling, and discretization setup.

    Raises
    ------
    ValueError
        If required settings keys are missing or invalid values are provided.

    Notes
    -----
    - Validates required configuration structure for collocation methods
    - Handles both single-node and multi-node (discretized) problems
    - Converts data types appropriately for JAX compatibility
    - Sets up adaptive time stepping if specified
    """
    # Assert that required settings are present
    required_keys = {
        "settings": ["model", "nnodes"],
        "constraints": [],
        "objectives": [],
        "bounds": [],
    }
    for key, value in required_keys.items():
        if key not in settings:
            raise ValueError(f"Missing required key: {key}")
        for sub_key in value:
            if sub_key not in settings[key]:
                raise ValueError(f"Missing required sub-key: {sub_key} in {key}")

    # Check nnodes is a positive integer
    print(
        "Collocation warning in process collocation settings: Bounds are not correctly handled"
    )
    if settings["settings"]["nnodes"] > 1:
        pass

    settings["nnodes"] = settings["settings"]["nnodes"]

    # Nnodes handling
    if (
        not isinstance(settings["settings"]["nnodes"], int)
        or settings["settings"]["nnodes"] <= 0
    ):
        raise ValueError("nnodes must be a positive integer.")
    if settings["settings"]["nnodes"] > 1:  # Discretization is needed
        if "discretization" not in settings["settings"]:
            raise ValueError("Discretization settings are required when nnodes > 1.")

        settings["discretization"] = settings["settings"]["discretization"]
        required_keys = ["type"]
        for key in required_keys:
            if key not in settings["settings"]["discretization"]:
                raise ValueError(f"Missing required discretization setting: {key}")
        # Allowed types:
        if settings["discretization"]["type"] not in ["euler", "rk4"]:
            raise ValueError(
                f"Invalid discretization type: {settings["discretization"]["type"]}. Allowed types are 'euler', 'rk4'."
            )
        if settings["discretization"]["type"] not in ["euler"]:
            raise NotImplementedError(
                f"Discretization type {settings["discretization"]["type"]} is not implemented yet."
            )

        # Add the discretization to the constraints:
        settings["constraints"].append(
            {
                "name": "discretization",
                "weight": settings["settings"]["discretization"]["args"].get(
                    "weight", 1
                ),
                "args": settings["settings"]["discretization"].get("args", {}),
            }
        )
        settings["constraints"].append(
            {
                "name": "speed",
                "weight": settings["settings"]["discretization"]["args"].get(
                    "weight", 1
                ),
                "args": settings["settings"]["discretization"].get("args", {}),
            }
        )

        if settings["discretization"]["args"]["adaptive_h"]:
            settings["constraints"].append(
                {
                    "name": "adaptive_h",
                    "weight": settings["discretization"].get("weight", 1),
                    "args": None,
                }
            )
            # The initial guess needs to include h in this case, otherwise we leave it
            Warning(
                "Initial guess handling for adaptive step size is not implemented yet."
            )

        if any(
            constraint["name"] == "periodicity"
            for constraint in settings["constraints"]
        ):
            settings["nnodes_dur"] = settings["nnodes"] + 1
        else:
            settings["nnodes_dur"] = settings["nnodes"]

        # set globals and bounds for dur and speed, and verify inputs being set correctly
        if "speed" not in settings["bounds"]:
            raise ValueError(
                "Speed bounds are required for collocation with multiple nodes."
            )
        if "dur" not in settings["bounds"]:
            raise ValueError(
                "Duration bounds are required for collocation with multiple nodes."
            )
        if isinstance(settings["bounds"]["speed"], (int, float)):
            settings["bounds"]["speed"] = [
                settings["bounds"]["speed"],
                settings["bounds"]["speed"],
            ]
        if isinstance(settings["bounds"]["dur"], (int, float)):
            settings["bounds"]["dur"] = [
                settings["bounds"]["dur"],
                settings["bounds"]["dur"],
            ]
        if len(settings["bounds"]["speed"]) != 2 or len(settings["bounds"]["dur"]) != 2:
            raise ValueError(
                "Speed and duration bounds must be a list of two values [min, max]."
            )
        adaptive_h = settings["discretization"]["args"]["adaptive_h"]
        nn_con = settings["nnodes_dur"] - 1
        h_min = jnp.zeros((nn_con,)) if adaptive_h else jnp.zeros((0,))
        h_max = jnp.ones((nn_con,)) * settings["bounds"]["dur"][1] if adaptive_h else jnp.zeros((0,))

        settings["bounds"]["global_min"] = states.Globals(
            dur=settings["bounds"]["dur"][0], speed=settings["bounds"]["speed"][0], h=h_min
        )
        settings["bounds"]["global_max"] = states.Globals(
            dur=settings["bounds"]["dur"][1], speed=settings["bounds"]["speed"][1], h=h_max
        )
    else:
        settings["nnodes_dur"] = settings["nnodes"]

    states_variables = model.variables[model.variables.type == "state"]
    min_generic = flat_model_to_states(model, jnp.array(states_variables.xmin), model.opt_states)
    max_generic = flat_model_to_states(model, jnp.array(states_variables.xmax), model.opt_states)

    # Placeholders for gc and actuator model
    min_generic = min_generic.replace(
        actuator_model=min_generic.actuator_model - 1e3
    )
    max_generic = max_generic.replace(
        actuator_model=max_generic.actuator_model + 1e3
    )
    min_generic = min_generic.replace(
        gc_model=min_generic.gc_model - 1e3
    )
    max_generic = max_generic.replace(
        gc_model=max_generic.gc_model + 1e3
    )

    settings["bounds"]["min"] = states.concatenate(
        [min_generic] * settings["nnodes_dur"]
    )
    settings["bounds"]["max"] = states.concatenate(
        [max_generic] * settings["nnodes_dur"]
    )

    if settings["settings"]["nnodes"] == 1:  # Bounds on ddot and dot values set to zero
        for section in ["min", "max"]:
            qd_val = settings["bounds"][section].qd.at[0].set(
                jnp.zeros(model.speeds.n, dtype=float) + (1e-8 if section == "max" else -1e-8)
            )
            settings["bounds"][section] = settings["bounds"][section].replace(qd=qd_val)

    if settings["bounds"]["start_at_origin"]:
        q_min = settings["bounds"]["min"].q.at[0, 0].set(0)
        settings["bounds"]["min"] = settings["bounds"]["min"].replace(q=q_min)
        q_max = settings["bounds"]["max"].q.at[0, 0].set(0)
        settings["bounds"]["max"] = settings["bounds"]["max"].replace(q=q_max)

    if model.actuator_model.get_n_states() > 0:
        for bound in ["min", "max"]:
            settings["bounds"][bound] = settings["bounds"][bound].replace(
                actuator_model=jnp.tile(
                    model.actuator_model.bounds["states"][bound][jnp.newaxis, :],
                    (settings["nnodes_dur"], 1)
                ),
            )
    return settings
