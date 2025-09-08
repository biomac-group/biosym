from abc import ABC, abstractmethod


class BaseActuator(ABC):
    """
    Abstract base class for actuators.
    """

    def __init__(self, xml_root):
        self.xml_root = xml_root
        self.actuator = None

    @abstractmethod
    def get_n_actuators(self):
        """
        Returns the number of actuators in the model.
        """

    @abstractmethod
    def reset(self):
        """
        Resets the actuator behaviour.
        """

    def process_eom(self, model):
        """
        Processes the equations of motion for the actuator.
        """
