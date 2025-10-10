from abc import ABC, abstractmethod


class BaseActuator(ABC):
    """
    Abstract base class for actuator models in biomechanical simulations.

    This class defines the interface that all actuator implementations must follow.
    Actuators represent force/torque generating elements in the biomechanical model,
    such as muscles, motors, or other active components.

    The base class handles common functionality like XML parsing and provides
    abstract methods that must be implemented by specific actuator types.

    Parameters
    ----------
    xml_root : xml.etree.ElementTree.Element
        Root element of the XML tree containing actuator definitions.

    Attributes
    ----------
    xml_root : xml.etree.ElementTree.Element
        The XML root element containing actuator configuration.
    actuator : object or None
        The actuator object instance (implementation-specific).

    Notes
    -----
    Subclasses must implement all abstract methods to define the specific
    behavior of different actuator types (e.g., general torque actuators,
    Hill-type muscle models, passive torques).

    See Also
    --------
    biosym.model.actuators.actuator_models.general.General : General torque actuators
    biosym.model.actuators.actuator_models.hill2d.Hill2D : Hill-type muscle model
    biosym.model.actuators.actuator_models.passive_torques.PassiveTorques : Passive torques
    """

    def __init__(self, xml_root):
        self.xml_root = xml_root
        self.actuator = None

    @abstractmethod
    def get_n_actuators(self):
        """
        Get the number of actuators in the model.

        Returns
        -------
        int
            Number of actuators defined in this actuator model.

        Notes
        -----
        This method must be implemented by all actuator subclasses to specify
        how many individual actuator elements are present in the model.
        """

    @abstractmethod
    def get_actuated_joints(self):
        """
        Get the list of joints actuated by this actuator model.

        Returns
        -------
        list of str
            Names of joints that are actuated by this actuator model.

        Notes
        -----
        This method must be implemented by all actuator subclasses to specify
        which joints in the biomechanical model are influenced by the actuators.
        """

    @abstractmethod
    def get_n_states(self):
        """
        Get the number of states associated with this actuator model.

        Returns
        -------
        int
            Number of states defined by this actuator model.

        Notes
        -----
        The default implementation returns 0. Actuator subclasses that define
        internal states (e.g., muscle activation, fiber length) should override
        this method to return the correct number of states.
        """
        return 0

    @abstractmethod
    def reset(self):
        """
        Reset the actuator model to its initial state.

        Notes
        -----
        This method is called at the beginning of simulations to ensure
        actuators start from a clean state. Implementations should reset
        any internal state variables, cached values, or dynamic properties.

        The exact reset behavior depends on the specific actuator type:
        - Muscle models may reset activation states
        - Motor models may reset control states
        - Passive elements may reset stored energy states
        """

    def process_eom(self, model):
        """
        Process the equations of motion for the actuator model.

        This method is called during the symbolic equation generation phase
        to integrate actuator dynamics into the overall system equations.

        Parameters
        ----------
        model : biosym.model.model.BiosymModel
            The biomechanical model containing the actuator.

        Notes
        -----
        The default implementation does nothing. Actuator subclasses should
        override this method if they need to add additional equations of motion,
        constraints, or symbolic relationships to the model.

        Examples of when this is needed:
        - Muscle activation dynamics
        - Force-length-velocity relationships
        - Internal state evolution equations
        """

    def get_n_constraints(self, *args, **kwargs):
        """
        Get the number of constraints defined by this actuator model.

        Returns
        -------
        int
            Number of constraints. Default is 0.

        Notes
        -----
        The default implementation returns 0. Actuator subclasses that define
        constraints (e.g., activation dynamics, force equilibrium) should override
        this method to return the correct number of constraints.
        """
        return 0

    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian of the actuator model.

        Returns
        -------
        int
            Number of non-zero entries in the Jacobian. Default is 0.

        Notes
        -----
        The default implementation returns 0. Actuator subclasses that define
        constraints or dynamics should override this method to return the correct
        number of non-zero entries in their Jacobian matrices.
        """
        return 0
