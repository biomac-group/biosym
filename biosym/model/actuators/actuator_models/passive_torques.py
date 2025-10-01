import jax.numpy as jnp
import numpy as np

from biosym.model.actuators.base_actuator import BaseActuator

JOINT_RANGE_TOL = np.deg2rad(2)  # 2 degrees transition zone for joint limits


class PassiveTorques(BaseActuator):
    """
    Passive torque actuator model.
    """

    def __init__(self, joints_dict):
        self.joints_dict = joints_dict
        self.n_actuators = len(joints_dict)
        self.actuators = {}

        self.damping = jnp.array([ji.get("damping", 0.0) for ji in joints_dict])
        self.stiffness = jnp.array([ji.get("stiffness", 0.0) for ji in joints_dict])
        self.upper_limits = jnp.array(
            [ji.get("range", [-np.inf, np.inf])[1] for ji in joints_dict]
        )
        self.lower_limits = jnp.array(
            [ji.get("range", [-np.inf, np.inf])[0] for ji in joints_dict]
        )

        self.idx_actuated_joints = jnp.array(
            [
                i
                for i, ji in enumerate(joints_dict)
                if ji.get("damping", 0.0) > 0.0 or ji.get("stiffness", 0.0) > 0.0
            ]
        )  # or ji.get("armature", 0.0) > 0.0]

    def get_n_actuators(self):
        """
        Returns the number of actuators in the model.
        """
        return self.n_actuators

    def reset(self):
        """
        Resets the actuator behaviour.
        """

    def get_n_states(self):
        return 0

    def get_n_constants(self):
        return 0

    def get_actuated_joints(self):
        """
        Returns the list of actuated joints.
        """
        return [
            ji["name"]
            for ji in self.joints_dict
            if ji.get("damping", 0.0) > 0.0 or ji.get("stiffness", 0.0) > 0.0
        ]  # or ji.get("armature", 0.0) > 0.0]

    def forward(self, states, constants, model):
        def f_plus(x):
            return 0.5 * (x + jnp.sqrt(x**2 + JOINT_RANGE_TOL**2))

        if states.model.ndim < 2:
            speeds = states.model[
                model.speeds["idx"] : model.speeds["idx"] + model.speeds["n"]
            ]
            coordinates = states.model[
                model.coordinates["idx"] : model.coordinates["idx"]
                + model.coordinates["n"]
            ]
        else:
            speeds = states.model[
                :, model.speeds["idx"] : model.speeds["idx"] + model.speeds["n"]
            ]
            coordinates = states.model[
                :,
                model.coordinates["idx"] : model.coordinates["idx"]
                + model.coordinates["n"],
            ]

        damp_term = -self.damping * speeds
        upper_limit_term = f_plus(coordinates - self.upper_limits)
        lower_limit_term = f_plus(self.lower_limits - coordinates)

        passive_torque = damp_term - self.stiffness * (
            upper_limit_term - lower_limit_term
        )

        return passive_torque  # Always return full array, even if some joints are not actuated
