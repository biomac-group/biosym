import jax.numpy as jnp

from biosym.model.actuators.base_actuator import BaseActuator


class General(BaseActuator):
    """
    General torque actuator model for biomechanical simulations.
    
    This class implements a simple torque actuator model where actuators
    directly apply torques to joints without complex dynamics. It is suitable
    for models where actuator dynamics are simplified or when representing
    idealized motor inputs.
    
    Parameters
    ----------
    xml_root : xml.etree.ElementTree.Element
        Root element of the XML tree containing actuator definitions.
        Expected to contain 'general' subelements with actuator specifications.
        
    Attributes
    ----------
    actuators : dict
        Dictionary mapping actuator names to their properties parsed from XML.
        Each actuator entry contains attributes like name, position, etc.
    n_actuators : int
        Number of actuators defined in the model.
    states : list of str
        List of state variable names, formatted as "torque_i" where i is the
        actuator index.
        
    Notes
    -----
    The general actuator model assumes:
    - Direct torque application to joints
    - No actuator dynamics (instantaneous response)
    - Linear mapping from actuator states to joint torques
    
    XML Format
    ----------
    The expected XML format for actuator definition:
    
    .. code-block:: xml
    
        <actuators type="general">
            <general name="actuator_1" pos="0 0 1"/>
            <general name="actuator_2" pos="1 0 0"/>
        </actuators>
        
    Examples
    --------
    Create actuators from XML:
    
    >>> import xml.etree.ElementTree as ET
    >>> root = ET.fromstring(xml_string)
    >>> actuators = General(root)
    >>> n_act = actuators.get_n_actuators()
    
    See Also
    --------
    GeneralMujoco : MuJoCo-specific general actuator implementation
    biosym.model.actuators.actuator_models.hill2d.Hill2D : Hill-type muscle model
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
        Get the number of actuators in the model.
        
        Returns
        -------
        int
            Number of actuators defined in this actuator model.
        """
        return self.n_actuators

    def get_actuators(self):
        """
        Get the dictionary of actuator definitions.
        
        Returns
        -------
        dict
            Dictionary mapping actuator names to their property dictionaries.
            Each property dictionary contains attributes parsed from the XML.
        """
        return self.actuators

    def get_n_states(self):
        """
        Get the number of state variables required by the actuator model.
        
        Returns
        -------
        int
            Number of actuator state variables, equal to the number of actuators.
            Each actuator contributes one torque state variable.
        """
        return self.get_n_actuators()

    def get_n_constants(self):
        """
        Get the number of constant parameters required by the actuator model.
        
        Returns
        -------
        int
            Number of constant parameters. Always 0 for general actuators
            as they have no internal parameters.
        """
        return 0

    def is_torque_actuator(self):
        """
        Check if this is a torque-based actuator model.
        
        Returns
        -------
        bool
            True, as general actuators directly apply torques.
        """
        return True

    def reset(self):
        """
        Reset the actuator model to its initial state.
        
        Notes
        -----
        For general actuators, there is no internal state to reset.
        This method is provided for interface compatibility.
        """

    def forward(self, states, constants, model):
        """
        Evaluate the actuator model to compute joint torques.
        
        This method maps actuator states (torque commands) to the appropriate
        joints in the biomechanical model.
        
        Parameters
        ----------
        states : object
            Current state values containing actuator_model attribute with
            torque values for each actuator.
        constants : object
            Current constant parameter values (unused for general actuators).
        model : biosym.model.model.BiosymModel
            The biomechanical model containing joint and force information.
            
        Returns
        -------
        jax.Array
            Array of joint torques with shape (n_coordinates,). Torques are
            placed at the indices specified by model.forces["active_idx"].
            
        Notes
        -----
        The method creates a zero array for all coordinates and fills in
        the actuator torques at the active joint indices. This ensures
        proper mapping between actuator outputs and joint inputs.
        """
        all_joints = jnp.zeros(model.coordinates["n"])
        all_joints = all_joints.at[jnp.array(model.forces["active_idx"])].set(states.actuator_model)
        return all_joints


class GeneralMujoco(General):
    """
    General actuator model specifically for MuJoCo-format XML files.
    
    This class extends the General actuator class to handle actuator
    definitions in MuJoCo XML format, which uses 'motor' elements instead
    of 'general' elements.
    
    Parameters
    ----------
    actuator_list : list of xml.etree.ElementTree.Element
        List of XML elements representing motor actuators from MuJoCo format.
        
    Notes
    -----
    MuJoCo format differences:
    - Uses 'motor' elements instead of 'general'
    - May have different attribute naming conventions
    - Handles MuJoCo-specific actuator properties
    
    See Also
    --------
    General : Base general actuator implementation
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
