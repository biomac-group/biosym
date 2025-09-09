import jax.numpy as jnp
import numpy as np

from biosym.model.actuators.base_actuator import BaseActuator

JOINT_RANGE_TOL = np.deg2rad(2)  # 2 degrees transition zone for joint limits


class Hill2d(BaseActuator):
    """
    A reimplementation of the 2D Hill muscle model as in gait2d.
    """

    def __init__(self, joints_dict, muscles_dict):
        super().__init__(joints_dict)
        self.muscles_dict = muscles_dict

    def get_n_actuators(self):
        """
        Returns the number of actuators in the model.
        """
        return self.n_actuators

    def reset(self):
        """
        Resets the actuator behaviour.
        """

    def get_actuated_joints(self):
        """
        Returns the list of actuated joints.
        """
        return self.actuated_joints

    def process_eom(self, model):
        return super().process_eom(model)

    def forward(self, states, constants, model):
        moments = jnp.zeros((model.coordinates["n"],))
        # Todo: Hill's equations in here
        # What is the force at every joint?
        return moments
