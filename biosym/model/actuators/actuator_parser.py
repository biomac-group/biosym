"""
Parser for actuator model definitions from XML files.

This module provides functionality to parse XML files containing actuator
definitions and instantiate the appropriate actuator model classes.
"""

import xml.etree.ElementTree as ET

from biosym.model.actuators.actuator_models import *


def get(file_path, body_weight=None):
    """
    Parse an actuator model file and return the appropriate actuator instance.
    
    This function reads XML files containing actuator definitions and creates
    the corresponding actuator model objects based on the specified type.
    
    Parameters
    ----------
    file_path : str or xml.etree.ElementTree.Element
        Path to the XML file containing actuator definitions, or an XML Element
        if the XML has already been parsed.
    body_weight : float, optional
        Body weight for scaling actuator forces. Currently not used but may
        be implemented for muscle force scaling in future versions.
        
    Returns
    -------
    BaseActuator
        An instance of the appropriate actuator model class based on the
        XML type specification.
        
    Raises
    ------
    ValueError
        If the actuator type specified in the XML is not recognized or supported.
        
    Notes
    -----
    Supported actuator types:
    - "actuator" or "general": General torque actuators
    - None (MuJoCo format): Motor elements parsed from MuJoCo XML
    
    The function automatically detects the format and actuator type from the
    XML structure and attributes.
    
    Examples
    --------
    Load actuators from an XML file:
    
    >>> actuators = get("path/to/actuators.xml")
    >>> n_actuators = actuators.get_n_actuators()
    
    Load from an already parsed XML element:
    
    >>> import xml.etree.ElementTree as ET
    >>> tree = ET.parse("actuators.xml")
    >>> root = tree.getroot()
    >>> actuators = get(root)
    
    See Also
    --------
    biosym.model.actuators.actuator_models.general.General : General actuator implementation
    biosym.model.actuators.actuator_models.general.GeneralMujoco : MuJoCo actuator implementation
    """
    if type(file_path) is str:
        tree = ET.parse(file_path)
        root = tree.getroot()
    else:
        root = file_path
    if root.get("type") in ["actuator", "general"]:
        return general.General(root)
    if root.get("type") is None:
        # This might not work for every model
        return general.GeneralMujoco(root.findall("motor"))
    raise ValueError(f"Unknown actuator model type: {root.get('type')}")
