from biosym.model.actuators.base_actuator import BaseActuator

class General(BaseActuator):
    """
        General actuator model.
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
    
    def is_torque_actuator(self):
        """
            Returns True if the actuator is a torque actuator.
        """
        return True

    def reset(self):
        """
            Resets the actuator behaviour.
        """
        pass


    
class GeneralMujoco(General):
    """
        General actuator model for Mujoco.
    """
    def __init__(self, acutator_list):
        self.actuators = {}
        for actuator in acutator_list:
            actuator_name = actuator.get("name", "actuator has no name")
            if actuator_name is None:
                raise ValueError("Actuator name is not specified.")
            self.actuators[actuator_name] = {}
            for key, value in actuator.attrib.items():
                self.actuators[actuator_name][key] = value