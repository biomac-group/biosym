"""
Utilities for creating and loading initial guesses.
"""

import os
import cloudpickle
import numpy as np
import jax.numpy as jnp
from biosym.utils import states


def make_initial_guess(model, settings, initial_guess):
    """
    Set up the initial guess for the collocation problem.
    """
    initial_guess_states = None
    initial_guess_globals = None

    if initial_guess is None:
        initial_guess_states = states.concatenate(
            [model.default_states] * settings["nnodes_dur"]
        )

    elif isinstance(initial_guess, dict):
        if initial_guess["type"] == "random":
            if isinstance(settings["bounds"]["min"], states.States):
                states_min = settings["bounds"]["min"]
                states_max = settings["bounds"]["max"]
                initial_guess_states = states.States(
                    **{
                        name: np.random.uniform(low=states_min[name], high=states_max[name], size=(states_max[name].shape))
                        for name in states_min.__dict__['names']
                    }
                )
            else: 
                raise NotImplementedError(
                    "Bounds for states must be provided as a States object when using random initial guess. This might not cover gait problems where this is a tuple"
                )
        elif initial_guess["type"] == "default":
            initial_guess_states = states.concatenate(
                [model.default_states] * settings["nnodes_dur"]
            )
        elif initial_guess["type"] == "mid":
            pass
        elif initial_guess["type"] == "from_file":
            print("Loading initial guess from file")
            ig_file = os.path.expanduser(initial_guess["file"])
            with open(ig_file, "rb") as f:
                (x, globals_), info, ig_settings = cloudpickle.load(f)
            if ig_settings["nnodes"] == 1:
                x = x[0]
                initial_guess_states = states.concatenate(
                    [x] * settings["nnodes_dur"]
                )
            elif ig_settings["nnodes"] == settings["nnodes"]:
                # If x is tuple, it will be (states, globals), otherwise it's just states
                if isinstance(x, tuple):
                    states_ig, globals_ig = x
                else:
                    states_ig = x
                    globals_ig = globals_
                initial_guess_states = states_ig
                if globals_ig is not None:
                    if not hasattr(globals_ig, "h"):
                        adaptive_h = settings["discretization"]["args"]["adaptive_h"]
                        h_init = jnp.ones((settings["nnodes_dur"] - 1,)) * globals_ig.dur / (settings["nnodes_dur"] - 1) if adaptive_h else jnp.zeros((0,))
                        initial_guess_globals = states.Globals(dur=globals_ig.dur, speed=globals_ig.speed, h=h_init)
                    else:
                        initial_guess_globals = globals_ig
                return initial_guess_states, initial_guess_globals
            else:
                if isinstance(x, tuple):
                    states_ig, globals_ig = x
                else:
                    states_ig = x
                    globals_ig = globals_
                initial_guess_states = states_ig.resample(settings["nnodes_dur"])
                if globals_ig is not None:
                    if not hasattr(globals_ig, "h"):
                        adaptive_h = settings["discretization"]["args"]["adaptive_h"]
                        h_init = jnp.ones((settings["nnodes_dur"] - 1,)) * globals_ig.dur / (settings["nnodes_dur"] - 1) if adaptive_h else jnp.zeros((0,))
                        initial_guess_globals = states.Globals(dur=globals_ig.dur, speed=globals_ig.speed, h=h_init)
                    else:
                        initial_guess_globals = globals_ig
                return initial_guess_states, initial_guess_globals
        else:
            raise ValueError(
                f"Invalid initial guess type: {initial_guess['type']}. Allowed types are 'random', 'default', 'mid', 'from_file'."
            )
    elif isinstance(initial_guess, list):
        initial_guess_states = states.concatenate(initial_guess)
    elif isinstance(initial_guess, str):
        if initial_guess == "random":
            pass
        elif initial_guess == "default":
            initial_guess_states = states.concatenate(
                [model.default_states] * settings["nnodes_dur"]
            )
        elif initial_guess == "mid":
            pass
    else:
        raise ValueError(
            "Invalid initial guess type. Must be dict, list, str, or None."
        )

    if settings["nnodes"] > 1:
        dur_mean = 1.0
        speed_mean = 1.0
        if "bounds" in settings:
            if "dur" in settings["bounds"]:
                dur_ = np.array(settings["bounds"]["dur"])
                if len(dur_) == 1:
                    dur_ = np.array([dur_, dur_])
                dur_mean = jnp.mean(dur_)
            if "speed" in settings["bounds"]:
                speed_ = np.array(settings["bounds"]["speed"])
                if len(speed_) == 1:
                    speed_ = np.array([speed_, speed_])
                speed_mean = jnp.mean(speed_)

        adaptive_h = settings["discretization"]["args"]["adaptive_h"]
        h_init = jnp.ones((settings["nnodes_dur"] - 1,)) * dur_mean / (settings["nnodes_dur"] - 1) if adaptive_h else jnp.zeros((0,))

        initial_guess_globals = states.Globals(
            dur=dur_mean,
            speed=speed_mean,
            h=h_init,
        )

    return initial_guess_states, initial_guess_globals
