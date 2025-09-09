from abc import ABC, abstractmethod


class BaseContact(ABC):
    """
    Abstract base class for contact models in biomechanical simulations.
    
    This class defines the interface that all contact model implementations must follow.
    Contact models handle interactions between the biomechanical model and its environment,
    particularly ground contact forces during locomotion and other activities.
    
    The base class provides a common framework for different contact formulations,
    such as point contacts, distributed contacts, or analytical contact models.
    
    Parameters
    ----------
    xml_root : xml.etree.ElementTree.Element
        Root element of the XML tree containing contact model definitions.
        
    Notes
    -----
    Contact models are responsible for:
    - Calculating ground reaction forces
    - Managing contact state during simulation
    - Providing additional states/constants for optimization
    - Handling contact detection and penetration
    - Computing friction and normal forces
    
    Subclasses must implement all abstract methods to define specific contact
    behavior for different contact types (e.g., point contacts, distributed contacts).
    
    See Also
    --------
    biosym.model.contact.contact_models.contact_points.ContactPoints : Point contact implementation
    """

    def __init__(self, xml_root):
        pass

    @abstractmethod
    def get_bodies(self):
        """
        Get the list of bodies that can be in contact with the environment.
        
        Returns
        -------
        list of str
            Names of bodies that have contact points or can interact with
            the environment through this contact model.
            
        Notes
        -----
        This method identifies which rigid bodies in the model can experience
        contact forces. Typically includes feet, hands, or other body segments
        that interact with the ground or environment.
        """

    @abstractmethod
    def get_n_bodies(self):
        """
        Get the number of bodies that can be in contact with the environment.
        
        Returns
        -------
        int
            Number of bodies that can experience contact forces through
            this contact model.
            
        Notes
        -----
        This is typically the length of the list returned by get_bodies(),
        but may be computed differently for efficiency in some implementations.
        """

    @abstractmethod
    def get_n_states(self):
        """
        Get the number of additional states required by the contact model.
        
        Returns
        -------
        int
            Number of additional state variables needed during optimization
            or simulation that are specific to the contact model.
            
        Notes
        -----
        Contact models may require additional states beyond the basic
        mechanical states (positions, velocities) to properly represent
        contact dynamics. Examples include:
        - Contact penetration depths
        - Sliding velocities
        - Contact activation states
        - Internal contact model variables
        """

    @abstractmethod
    def get_n_constants(self):
        """
        Get the number of additional constants required by the contact model.
        
        Returns
        -------
        int
            Number of additional constant parameters needed during optimization
            or simulation that are specific to the contact model.
            
        Notes
        -----
        Contact models may require additional constants beyond the basic
        mechanical parameters. Examples include:
        - Contact stiffness parameters
        - Friction coefficients  
        - Contact geometry parameters
        - Material properties
        """

    @abstractmethod
    def get_states(self):
        """
        Get the list of additional state variable names for the contact model.
        
        Returns
        -------
        list of str
            Names of additional state variables needed during optimization
            that are specific to this contact model.
            
        Notes
        -----
        The returned list should have length equal to get_n_states().
        State names are used for identification in optimization and analysis.
        """

    @abstractmethod
    def get_constants(self):
        """
        Get the list of additional constant parameter names for the contact model.
        
        Returns
        -------
        list of str
            Names of additional constant parameters needed during optimization
            that are specific to this contact model.
            
        Notes
        -----
        The returned list should have length equal to get_n_constants().
        Constant names are used for identification in optimization and analysis.
        """

    @abstractmethod
    def process_eom(self):
        """
        Build additional equations of motion specific to the contact model.
        
        This method is called during the symbolic equation generation phase
        to add contact-specific dynamics to the overall system equations.
        
        Notes
        -----
        Not all contact models require additional equations of motion.
        Simple contact models may only provide forces without additional
        dynamic states. Complex models may add:
        - Contact penetration dynamics
        - Friction state evolution
        - Contact mode switching logic
        
        The default implementation should do nothing for simple contact models.
        """

    @abstractmethod
    def forward(self):
        """
        Calculate contact forces for the current model state.
        
        Returns
        -------
        jax.Array
            Contact forces for all bodies in the global coordinate frame.
            Forces are typically organized as [Fx, Fy, Fz] for each body.
            
        Notes
        -----
        This is the core computational method that evaluates the contact
        model given the current state of the system. The method should:
        - Compute contact penetrations
        - Calculate normal and friction forces
        - Transform forces to the global frame
        - Handle contact detection and activation
        """

    @abstractmethod
    def reset(self):
        """
        Reset the contact model to its initial state.
        
        Notes
        -----
        This method is called at the beginning of simulations to ensure
        the contact model starts from a clean state. Implementations should:
        - Reset any internal state variables
        - Clear contact history
        - Initialize contact detection flags
        - Reset accumulated contact energies or impulses
        """

    @abstractmethod
    def plot(self, states, constants, model, mode):
        """
        Visualize the contact model state and forces.
        
        Parameters
        ----------
        states : object
            Current state values for the model and contact system.
        constants : object  
            Current constant parameter values.
        model : biosym.model.model.BiosymModel
            The biomechanical model containing this contact model.
        mode : str
            Plotting mode, either "init" or "update".
            
        Notes
        -----
        Plotting modes:
        - "init": Initialize the plot, set up axes, create line objects,
          and save plotting parameters for efficient updates.
        - "update": Update existing plot elements with new data without
          recreating the entire visualization.
          
        Contact visualizations typically show:
        - Contact point locations
        - Ground reaction force vectors
        - Contact penetration indicators
        - Friction force directions
        """
