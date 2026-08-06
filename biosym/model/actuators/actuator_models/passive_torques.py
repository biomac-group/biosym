import jax.numpy as jnp
import numpy as np

from biosym.model.actuators.base_actuator import BaseActuator

JOINT_RANGE_TOL = np.deg2rad(2)  # 2 degrees transition zone for joint limits


class PassiveTorques(BaseActuator):
    """
    Passive joint torques (damping + range springs), built from the joints list
    rather than parsed from a file. forward() returns a joint-sized torque array
    (one entry per DOF): damping and range-limit torque where a joint has them,
    zero elsewhere. model.py sums this with the active actuators' arrays.
    """

    def __init__(self, joints_dict) -> None:
        self.joints_dict = joints_dict
        self.n_actuators = len(joints_dict)
        self.actuators = {}

        self.damping = jnp.array([ji.get("damping", 0.0) for ji in joints_dict])
        self.stiffness = jnp.array([ji.get("stiffness", 0.0) for ji in joints_dict])
        self.upper_limits = jnp.array([ji.get("range", [-np.inf, np.inf])[1] for ji in joints_dict])
        self.lower_limits = jnp.array([ji.get("range", [-np.inf, np.inf])[0] for ji in joints_dict])

        # Integer dtype so an EMPTY result (no damped joints is still a valid index array, not float64.
        self.idx_actuated_joints = jnp.array(
            [i for i, ji in enumerate(joints_dict)
             if ji.get("damping", 0.0) > 0.0 or ji.get("stiffness", 0.0) > 0.0],
            dtype=jnp.int32,
        )

    def get_n_actuators(self):
        return self.n_actuators

    def reset(self) -> None:
        """Resets the actuator behaviour."""

    def get_n_states(self) -> int:
        return 0

    def get_n_constants(self) -> int:
        return 0

    def get_actuated_joints(self):
        """Joints that actually have passive behaviour (damping or stiffness)."""
        return [
            ji["name"] for ji in self.joints_dict
            if ji.get("damping", 0.0) > 0.0 or ji.get("stiffness", 0.0) > 0.0
        ]

    def forward(self, states, constants, model, states_prev=None, h=None):
        # states_prev/h: unused, see CoordinateActuator.forward.
        def f_plus(x):
            return 0.5 * (x + jnp.sqrt(x**2 + JOINT_RANGE_TOL**2))

        speeds = states.qd
        coordinates = states.q

        damp_term = -self.damping * speeds
        upper_limit_term = f_plus(coordinates - self.upper_limits)
        lower_limit_term = f_plus(self.lower_limits - coordinates)

        # Full joint-sized array: zero where a joint has no damping/stiffness,
        # since those coefficients are zero there. The model sums it with the active actuators' arrays.
        passive_torque = damp_term - self.stiffness * (upper_limit_term - lower_limit_term)
        return passive_torque