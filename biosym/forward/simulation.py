import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
import copy

class SimulationEnvironment:
    def __init__(self, model, dt=0.01, integrator="RK4", n_steps=1000, stopping_criteria=None, initial_state="neutral", **kwargs):
        """
        Initialize the forward simulation.

        Parameters:
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
        self.reset()

    def _create_simulation_ode(self):
        """
            Create the simulation ODE function.
            This function is used to compute the state derivatives for the simulation.
        """
        def simulation_ode(model_state_vector, t, others, model):
            constants, controls, all_states = others
            all_states['model'] = model_state_vector.copy()
            # Run actuator model
            if hasattr(model, "actuator_model"):
                actuator_signals = model.run['actuator_model'](all_states)
            else:
                actuator_signals = controls

            model_state_vector.at[model.forces['idx']:model.forces['idx'] + model.forces['n']].set(actuator_signals)

            # Run ground contact model
            if hasattr(model, "contact_model"):
                contact_signals = model.run['contact_model'](all_states)
            else:
                contact_signals = jnp.zeros(model.ext_forces['n']), jnp.zeros(model.ext_torques['n'])

            model_state_vector.at[model.ext_forces['idx']:model.ext_forces['idx'] + model.ext_forces['n']].set(contact_signals[0])
            model_state_vector.at[model.ext_torques['idx']:model.ext_torques['idx'] + model.ext_torques['n']].set(contact_signals[1])
            M = model.run['mass_matrix'](all_states, constants)
            F = model.run['forcing'](all_states, constants)
            acc = jnp.linalg.solve(M, F).squeeze()
            return_states = jnp.zeros_like(model_state_vector)
            return_states = return_states.at[model.coordinates['idx']:model.coordinates['idx'] + model.coordinates['n']].set(model_state_vector[model.speeds['idx']:model.speeds['idx'] + model.speeds['n']])
            return_states = return_states.at[model.speeds['idx']:model.speeds['idx'] + model.speeds['n']].set(acc)
            return return_states

        #states = self.model.default_inputs['states']
        #constants = self.model.default_inputs['constants']
        self.simulation_ode = lambda states, t, others: simulation_ode(states, t, others, self.model)

    def step(self, controls):
        """
            Perform a single step of the simulation.
            This function updates the state of the simulation by one time step.
            Parameters:
                controls: The control inputs for the simulation.
            Returns:
                state: The updated state of the simulation.
                reward: The reward for the current step.
                terminated: A boolean indicating if the simulation has terminated.
                truncated: A boolean indicating if the simulation has been truncated.
                info: Additional information about the simulation step.
        """
        def _step_internal(controls, state, dt, simulation_ode):
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
            new_state_ = odeint(simulation_ode, state['states']['model'], t, (state['constants'], controls, state['states']))
            # The current state is only the last value of the integration
            return new_state_[-1]
        
        if not hasattr(self, "_step"):
            _step_internal(controls, self.state, self.model, self.dt, self.simulation_ode)
            self._step = jax.jit(_step_internal, static_argnames=["model", "simulation_ode"])
        self.state['states']['model'] = self._step(controls, self.state, self.dt, self.simulation_ode)
        # Check for callbacks for rewards and stopping criteria
        reward = self.reward_callback(self.state, controls, self.model) if hasattr(self, "reward_callback") else 0.0
        terminated = self.termination_callback(self.state, self.model) if hasattr(self, "termination_callback") else False
        truncated = self.truncation_callback(self.state, self.model) if hasattr(self, "truncation_callback") else False
        info = self.info_callback(self.state, controls, self.model) if hasattr(self, "info_callback") else {}

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, initial_state=None, mode="neutral", **kwargs):
        """
            Reset the simulation environment to the initial state.
            This function initializes the state of the simulation and prepares it for a new run.
            Parameters:
                seed: The random seed for the simulation.
                initial_state: The initial state of the simulation.
                mode: The mode of the simulation (e.g. "neutral", "random").
        """
        if initial_state is not None:
            self.state = initial_state

        if seed is not None:
            np.random.seed(seed)
        
        if self.initial_state == "neutral":
            self.state = copy.deepcopy(self.model.default_inputs)
        elif self.initial_state == "random":
            self.state = copy.deepcopy(self.model.default_inputs)
            for i, row in self.model.variables.iterrows():
                if row.type == "state":
                    self.state['states']['model'][i] = np.random.uniform(row['xmin'], row['xmax'])
        else:
            raise ValueError(f"Simulation: initial state '{self.initial_state}' is not supported.")
        
        # Reset contact / ground contact model
        if hasattr(self.model, "contact_model"):
            self.model.contact_model.reset(self.state, **kwargs)
        # Reset actuator model
        if hasattr(self.model, "actuator_model"):
            self.model.actuator_model.reset(self.state, **kwargs)

        # Convert state to JAX arrays
        for key in ['states', 'constants']:
            for subkey in self.state[key].keys():
                if isinstance(self.state[key][subkey], jnp.ndarray):
                    self.state[key][subkey] = jnp.asarray(self.state[key][subkey])
                elif isinstance(self.state[key][subkey], np.ndarray):
                    self.state[key][subkey] = jnp.asarray(self.state[key][subkey])
                else:
                    raise ValueError(f"Simulation: state '{self.state[key][subkey]}' is not a valid type.")

        # States can only contain "states" and "constants"
        self.state = {
            'states': self.state['states'],
            'constants': self.state['constants']
        }

    def run(self):
        pass

