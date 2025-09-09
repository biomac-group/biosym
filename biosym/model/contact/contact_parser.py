"""
Parser for contact model definitions from XML files.

This module provides functionality to parse XML files containing contact
model definitions and instantiate the appropriate contact model classes.
"""

import xml.etree.ElementTree as ET

from biosym.model.contact.contact_models import *


def get(file_path, body_weight=None):
    """
    Parse a contact model file and return the appropriate contact instance.
    
    This function reads XML files containing contact model definitions and creates
    the corresponding contact model objects based on the specified type.
    
    Parameters
    ----------
    file_path : str
        Path to the XML file containing contact model definitions.
    body_weight : float, optional
        Body weight in kilograms for scaling contact forces. This is used
        to scale contact parameters like stiffness and damping based on
        the body weight of the subject being modeled.
        
    Returns
    -------
    BaseContact
        An instance of the appropriate contact model class based on the
        XML type specification.
        
    Raises
    ------
    ValueError
        If the contact type specified in the XML is not recognized or supported.
        
    Notes
    -----
    Supported contact types:
    - "contact_points": Point-based contact model with specified contact locations
    
    The body_weight parameter is particularly important for contact models as
    it allows scaling of contact parameters to match different subject sizes.
    Typical usage involves providing the subject's body weight in kg.
    
    Examples
    --------
    Load contact model for a 70kg subject:
    
    >>> contact = get("path/to/contact.xml", body_weight=70.0)
    >>> n_contacts = contact.get_n_bodies()
    
    Load contact model with default scaling:
    
    >>> contact = get("path/to/contact.xml")
    >>> bodies = contact.get_bodies()
    
    See Also
    --------
    biosym.model.contact.contact_models.contact_points.ContactPoints : Point contact implementation
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    if root.get("type") == "contact_points":
        return contact_points.ContactPoints(root, body_weight=body_weight)
    raise ValueError(f"Unknown contact model type: {root.get('type')}")
