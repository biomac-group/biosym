import copy

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint

import biosym.utils.states as stat

# Caching functions in this module is unwanted.
jax.config.update("jax_persistent_cache_min_compile_time_secs", 200)


class SimulationEnvironment:
    def __init__(
        self,
        model,
        dt=0.01,
        integrator="RK4",
        n_steps=1000,
        stopping_criteria=None,
        initial_state="neutral",
        **kwargs,
    ):
        """
        Initialize the forward simulation.

        Parameters
        ----------
            model: The model to be simulated.
            dt: The time step for the simulation.
            integrator: The integrator to be used for the simulation.
            n_steps: The number of steps to simulate.
            stopping_criteria: A function that takes the current state and returns True if the simulation should stop.
        """
        self.model = model
        self.dt = dt
        self.integrator = integrator
        self.n_steps = n_steps
        self.initial_state = initial_state
        self.stopping_criteria = stopping_criteria

        self._create_simulation_ode()
        if type(initial_state) == str:
            if initial_state not in ["neutral", "random"]:
                raise ValueError(f"Simulation: initial state '{initial_state}' is not supported.")
            self.reset(mode=initial_state, **kwargs)
        elif type(initial_state) == dict:
            # Check if the initial state is a valid dictionary
            if "states" not in initial_state or "constants" not in initial_state:
                raise ValueError("Simulation: initial state must contain 'states' and 'constants' keys.")
            if (
                "model" not in initial_state["states"]
                or "gc_model" not in initial_state["states"]
                or "actuator_model" not in initial_state["states"]
            ):
                raise ValueError(
                    "Simulation: initial state must contain 'model', 'gc_model', and 'actuator_model' keys."
                )
            if (
                "model" not in initial_state["constants"]
                or "gc_model" not in initial_state["constants"]
                or "actuator_model" not in initial_state["constants"]
            ):
                raise ValueError(
                    "Simulation: initial state must contain 'model', 'gc_model', and 'actuator_model' keys."
                )
            self.state = initial_state

    def _create_simulation_ode(self):
        """
        Create the simulation ODE function.
        This function is used to compute the state derivatives for the simulation.
        """

        def simulation_ode(model_state_vector, t, others, n_extforce, runs):
            controls, constants_, actuator_states_, gc_states_, remaining_states_ = others
            a = model_state_vector
            states_ = stat.States(
                model=jnp.concatenate((a, remaining_states_)),
                gc_model=gc_states_,
                actuator_model=actuator_states_,
                h=None,
            )
            # Run actuator model
            if "actuator_model" in runs:
                actuator_signals = runs["actuator_model"](states_, constants_)
            else:
                actuator_signals = controls
            # Run ground contact model
            if "gc_model" in runs:
                contact_signals = runs["gc_model"](states_, constants_)
            else:
                contact_signals = jnp.zeros(n_extforce["n"]), jnp.zeros(n_extforce["n"])

            model_state_vector_ = jnp.concatenate(
                (
                    model_state_vector,
                    jnp.zeros(len(model_state_vector) // 2),
                    actuator_signals,
                    contact_signals[0].flatten(),
                    contact_signals[1].flatten(),
                )
            )
            states_ = stat.States(
                model=model_state_vector_,
                gc_model=gc_states_,
                actuator_model=actuator_states_,
                h=None,
            )

            M = runs["mass_matrix"](states_, constants_)
            F = runs["forcing"](states_, constants_)

            # Add numerical stability to matrix solve
            M_reg = M  # + jnp.eye(M.shape[0])
            try:
                acc = jnp.linalg.solve(M_reg, F)
            except:
                # Fallback to pseudoinverse if direct solve fails
                raise NotImplementedError("Matrix solve failed")

            return_states = jnp.concatenate((a[len(acc) : 2 * len(acc)], acc[:, 0]))
            return return_states

        # states = self.model.default_inputs['states']
        # constants = self.model.default_inputs['constants']
        self.simulation_ode = lambda states, t, others: simulation_ode(
            states, t, others, self.model.ext_forces, self.model.run
        )
        self.simulation_ode = jax.jit(self.simulation_ode)
        # Try to cache the jit-compilation - might not make a difference
        self.simulation_ode(
            jnp.zeros(2 * self.model.coordinates["n"]),
            0.0,
            (
                jnp.zeros(self.model.forces["n"]),
                self.model.default_inputs.constants,
                self.model.default_inputs.states.actuator_model,
                self.model.default_inputs.states.gc_model,
                self.model.default_inputs.states.model[2 * self.model.coordinates["n"] :],
            ),
        )

    def step(self, controls=None):
        """
        Perform a single step of the simulation.
        This function updates the state of the simulation by one time step.

        Parameters
        ----------
            controls: The control inputs for the simulation.

        Returns
        -------
            state: The updated state of the simulation.
            reward: The reward for the current step.
            terminated: A boolean indicating if the simulation has terminated.
            truncated: A boolean indicating if the simulation has been truncated.
            info: Additional information about the simulation step.
        """
        if controls is None:
            controls = jnp.zeros(self.model.forces["n"])

        def _step_internal(controls, state, dt, simulation_ode, n_coordinates=0):
            """
            This function runs 3 computations:
                1. Compute the torque signal from the actuator model.
                2. Compute the GRF signal from the contact model.
                3. Integrate the state using the ODE function.
            """
            assert state is not None, "Simulation environment not initialized. Call reset() first."
            # Integrate the state using the ODE function
            # Time array with two points: initial time and initial time + dt
            t = jnp.array([0.0, dt])

            # Purify inputs
            states_ = state.states.model
            constants_ = state.constants
            actuator_states_ = state.states.actuator_model
            gc_states_ = state.states.gc_model
            new_state_ = odeint(
                simulation_ode,
                states_[: 2 * n_coordinates],
                t,
                (
                    controls,
                    constants_,
                    actuator_states_,
                    gc_states_,
                    states_[2 * n_coordinates :],
                ),
                atol=1e-6,
                rtol=1e-6,
                mxstep=10000,
            )
            # The current state is only the last value of the integration
            return new_state_[-1]

        if not hasattr(self, "_step"):
            self._step = lambda controls, state, dt: _step_internal(
                controls, state, dt, self.simulation_ode, self.model.coordinates["n"]
            )
            self._step = jax.jit(self._step)

        # self.state['states']['model'] = self._step(controls, self.state, self.dt, self.simulation_ode)

        self.state = self.state.replace_vector(
            "states",
            "model",
            self.state.states.model.at[: 2 * self.model.coordinates["n"]].set(
                self._step(controls, self.state, self.dt)
            ),
        )
        # Check for callbacks for rewards and stopping criteria
        reward = self.reward_callback(self.state, controls, self.model) if hasattr(self, "reward_callback") else 0.0
        terminated = (
            self.termination_callback(self.state, self.model) if hasattr(self, "termination_callback") else False
        )
        truncated = self.truncation_callback(self.state, self.model) if hasattr(self, "truncation_callback") else False
        info = self.info_callback(self.state, controls, self.model) if hasattr(self, "info_callback") else {}

        return copy.deepcopy(self.state), reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        """
        Reset the simulation environment to the initial state.
        This function initializes the state of the simulation and prepares it for a new run.

        Parameters
        ----------
            seed: The random seed for the simulation.
            initial_state: The initial state of the simulation.
            mode: The mode of the simulation (e.g. "neutral", "random").
        """
        if "initial_state" in kwargs:
            mode = kwargs["initial_state"]
        else:
            mode = self.initial_state

        if type(mode) == str:
            if mode not in ["neutral", "random"]:
                raise ValueError(f"Simulation: initial state '{mode}' is not supported.")
            if seed is not None:
                np.random.seed(seed)
            if self.initial_state == "neutral":
                self.state = copy.deepcopy(self.model.default_inputs)
            elif self.initial_state == "random":
                self.state = copy.deepcopy(self.model.default_inputs)
                states_ = []
                for i, row in self.model.variables.iterrows():
                    if row.type == "state":
                        # Use more conservative random ranges to avoid numerical issues
                        curr_dof = self.model.variables.iloc[i].values[1]
                        if curr_dof.startswith("q_"):
                            val_range = abs(row["xmax"] - row["xmin"])
                            center = (row["xmax"] + row["xmin"]) / 2
                            val = center + np.random.uniform(-val_range / 2, val_range / 2)
                            val = np.clip(val, row["xmin"], row["xmax"])  # Ensure bounds
                            if curr_dof.endswith("tx"):
                                val = 0
                            elif curr_dof.endswith("ty"):
                                val = np.random.uniform(0, 1)  # Small positive height
                        else:
                            val = 0.0  # Do not pre-define speeds for now
                        states_.append(val)
                states_array = jnp.array(states_)
                # Ensure finite values
                states_array = jnp.where(jnp.isfinite(states_array), states_array, 0.0)
                self.state = self.state.replace_vector("states", "model", states_array)
            else:
                raise ValueError(f"Simulation: initial state '{self.initial_state}' is not supported.")
            # Reset contact / ground contact model
            if hasattr(self.model, "contact_model"):
                self.model.contact_model.reset(self.state, **kwargs)
            # Reset actuator model
            if hasattr(self.model, "actuator_model"):
                self.model.actuator_model.reset(self.state, **kwargs)
        elif type(mode) == dict:
            assert "states" in mode and "constants" in mode, (
                "Simulation: initial state must contain 'states' and 'constants' keys."
            )
            assert "model" in mode["states"] and "gc_model" in mode["states"] and "actuator_model" in mode["states"], (
                "Simulation: initial state must contain 'model', 'gc_model', and 'actuator_model' keys."
            )
            assert (
                "model" in mode["constants"]
                and "gc_model" in mode["constants"]
                and "actuator_model" in mode["constants"]
            ), "Simulation: initial state must contain 'model', 'gc_model', and 'actuator_model' keys."
            self.state = copy.deepcopy(mode)
        else:
            raise ValueError(f"Simulation: initial state '{mode}' is not supported.")

        return copy.deepcopy(self.state)

    def run(self):
        pass
