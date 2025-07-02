from abc import ABC, abstractmethod


class BaseContact(ABC):
    """
    Abstract base class for contact models.
    """

    def __init__(self, xml_root):
        pass

    @abstractmethod
    def get_bodies():
        """
        Returns the list of bodies that can be in contact.
        """

    @abstractmethod
    def get_n_bodies():
        """
        Returns the number of bodies that can be in contact.
        """

    @abstractmethod
    def get_n_states():
        """
        Returns the number of states in the contact model.
        The number returned is the additional states that are needed during optimization.
        """

    @abstractmethod
    def get_n_constants():
        """
        Returns the number of constants in the contact model.
        The number returned is the additional constants that are needed during optimization.
        """

    @abstractmethod
    def get_states():
        """
        Returns the list of states in the contact model (list of strings).
        The states are the additional states that are needed during optimization.
        """

    @abstractmethod
    def get_constants():
        """
        Returns the list of constants in the contact model (list of strings).
        The constants are the additional constants that are needed during optimization.
        """

    @abstractmethod
    def process_eom():
        """
        Builds the extra equations of motion for the contact model.
        This is called after the base equations of motion are built.
        Not all contact models need this, so it is optional.
        The default implementation does nothing.
        """

    @abstractmethod
    def forward():
        """
        Returns the contact forces for the given bodies in the global frame.
        """

    @abstractmethod
    def reset():
        """
        Resets the contact model.
        This is called at the beginning of each simulation.
        """

    @abstractmethod
    def plot(self, states, constants, model, mode):
        """
        Plots the contact model.
        2 modes:
            "init": initializes the plot, saving all line parameters
            "update": updates the plot with new data
        """
