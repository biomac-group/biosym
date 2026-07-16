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

    forward() returns a joint-sized torque array with this actuator's commanded
    torque placed at its actuated joints, zero elsewhere. model.py sums these
    across actuators and applies each joint's net torque as +M on child /
    -M on parent.
    """

    def __init__(self, actuator_dicts):
        super().__init__()
        self.actuators = {a["name"]: a for a in actuator_dicts}
        self.n_actuators = len(self.actuators)
        self._defs = list(actuator_dicts)  # ordered, matches actuator_model order
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
        return [a["joint"] for a in self._defs if a.get("joint") is not None]

    def get_n_states(self):
        return self.get_n_actuators()

    def get_n_constants(self):
        return 0

    def reset(self):
        pass

    def forward(self, states, constants, model):
        """
        Return a joint-sized torque array: this actuator's commanded torques
        placed at their joints' coordinate indices, zero elsewhere.
        """
        # Position of each actuator's joint in the model's coordinate list.
        joint_names = [j["name"] for j in model.dicts["joints"]]
        joint_idx = jnp.array([joint_names.index(a["joint"]) for a in self._defs])

        n_dof = model.coordinates.n
        commanded = states.actuator_model  # one value per actuator, in _defs order

        if commanded.ndim == 1:
            out = jnp.zeros(n_dof)
            return out.at[joint_idx].set(commanded)
        else:
            # batched: (n_samples, n_actuators) -> (n_samples, n_dof)
            out = jnp.zeros((commanded.shape[0], n_dof))
            return out.at[:, joint_idx].set(commanded)