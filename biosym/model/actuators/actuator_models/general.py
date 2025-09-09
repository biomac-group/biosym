import jax.numpy as jnp

from biosym.model.actuators.base_actuator import BaseActuator


class General(BaseActuator):
    """
    General actuator model, torque actuators only.
    """

    def __init__(self, xml_root):
        super().__init__(xml_root)
        actuators = {}
        for actuator in xml_root.findall("general"):
            actuator_name = actuator.get("name")
            if actuator_name is None:
                raise ValueError("Actuator name is not specified.")
            actuators[actuator_name] = {}
            for key, value in actuator.attrib.items():
                if key == "pos":
                    actuators[actuator_name][key] = [float(x) for x in value.split()]
                else:
                    actuators[actuator_name][key] = value
        self.actuators = actuators
        self.n_actuators = len(actuators)
        self.states = [f"torque_{i}" for i in range(self.n_actuators)]

    def get_n_actuators(self):
        """
        Returns the number of actuators in the model.
        """
        return self.n_actuators

    def get_actuators(self):
        """
        Returns the list of actuators in the model.
        """
        return self.actuators

    def get_n_states(self):
        """
        Returns the number of states required by the actuator model.
        """
        return self.get_n_actuators()

    def get_n_constants(self):
        """
        Returns the number of constants required by the actuator model.
        """
        return 0

    def is_torque_actuator(self):
        """
        Returns True if the actuator is a torque actuator.
        """
        return True

    def reset(self):
        """
        Resets the actuator behaviour.
        """

    def forward(self, states, constants, model):
        """
        Evaluate the actuator model.

        :param states: Current states.
        :param constants: Current constants.
        :param model: The biosym model.
        :return: The evaluated actuator model.
        """
        all_joints = jnp.zeros(model.coordinates["n"])
        all_joints = all_joints.at[jnp.array(model.forces["active_idx"])].set(states.actuator_model)
        return all_joints


class GeneralMujoco(General):
    """
    General actuator model for Mujoco.
    """

    def __init__(self, actuator_list):
        self.actuators = {}
        for actuator in actuator_list:
            actuator_name = actuator.get("name", "actuator has no name")
            if actuator_name is None:
                raise ValueError("Actuator name is not specified.")
            self.actuators[actuator_name] = {}
            for key, value in actuator.attrib.items():
                self.actuators[actuator_name][key] = value
        self.n_actuators = len(actuator_list)
        self.states = [f"torque_{i}" for i in range(self.n_actuators)]
