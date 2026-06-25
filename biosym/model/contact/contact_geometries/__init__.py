"""
Contact Geometry Models

This module provides functions for interpreting contact geometries.
Contact Geometry models inherit from the BaseGeometry class.
Contact Geometries are essential for simulating locomotion and other activities
where the model interacts with external surfaces.
"""

from biosym.model.contact.contact_geometries.contact_point import ContactPoint
from biosym.model.contact.contact_geometries.contact_sphere import ContactSphere
from biosym.model.contact.contact_geometries.contact_halfspace import ContactHalfSpace

__all__ = ["contact_point", "contact_sphere", "contact_halfspace"]
