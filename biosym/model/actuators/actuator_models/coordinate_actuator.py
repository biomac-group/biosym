import jax.numpy as jnp

from biosym.model.actuators.base_actuator import BaseActuator


class CoordinateActuator(BaseActuator):
    """
    Direct coordinate (joint) torque actuator. Equivalent to OpenSim's
    CoordinateActuator and to MuJoCo's motor / the legacy "general" actuator:
    the commanded value is applied straight to the joint as a generalized torque.

    Receives a list of parsed actuator dicts from the actuator parser, each like:
        {"name": str, "joint": str, "min": float, "max": float, ...}
    -- source-agnostic, so the same class serves xml, mujoco, and osim inputs.

    Produces a "coordinate" load (the default): joint torques placed at the
    actuated joints' indices.
    """

    def __init__(self, actuator_dicts):
        super().__init__()
        # actuator_dicts: list of normalized dicts from the parser.
        self.actuators = {a["name"]: a for a in actuator_dicts}
        self.n_actuators = len(self.actuators)
        self.states = [f"torque_{name}" for name in self.actuators.keys()]
        self.state_vector = self.states
        self.bounds = {
            "states": {
                "min": jnp.array([float(a.get("min", -1e4)) for a in actuator_dicts]),
                "max": jnp.array([float(a.get("max",  1e4)) for a in actuator_dicts]),
            }
        }

    def get_actuators(self):
        return self.actuators

    def get_n_actuators(self):
        return self.n_actuators

    def get_actuated_joints(self):
        return [a["joint"] for a in self.actuators.values() if a.get("joint") is not None]

    def get_n_states(self):
        return self.get_n_actuators()

    def get_n_constants(self):
        return 0

    def is_torque_actuator(self):
        return True

    def reset(self):
        pass

    def forward(self, states, constants, model):
        # Identical to the old General.
        """
        Evaluate the actuator model to compute joint torques.

        This method maps actuator states (torque commands) to the appropriate
        joints in the biomechanical model.

        Parameters
        ----------
        states : object
            Current state values containing actuator_model attribute with
            torque values for each actuator.
        constants : object
            Current constant parameter values (unused for general actuators).
        model : biosym.model.model.BiosymModel
            The biomechanical model containing joint and force information.

        Returns
        -------
        jax.Array
            Array of joint torques with shape (n_coordinates,). Torques are
            placed at the indices specified by model.forces["active_idx"].

        Notes
        -----
        The method creates a zero array for all coordinates and fills in
        the actuator torques at the active joint indices. This ensures
        proper mapping between actuator outputs and joint inputs.
        """
        all_joints = jnp.zeros((len(states), model.coordinates["n"]))
        all_joints = all_joints.at[:, jnp.array(model.forces["active_idx"])].set(
            states.actuator_model
        )
        return all_joints if (states.model.ndim > 1) else all_joints[0]
