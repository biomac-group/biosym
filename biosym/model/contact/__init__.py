"""
Contact Models for Ground and Environmental Interactions.

This module provides classes and functions for modeling contact between the
biomechanical model and its environment, particularly ground contact forces.
Contact models are essential for simulating locomotion and other activities
where the model interacts with external surfaces.

The module includes:
- Base contact classes defining the common interface for contact force laws
- Base geometry classes defining the common interface for contact geometry definitions 
- Parsers for loading contact definitions from XML/YAML files and OSIM-parsed structures

Contact models handle:
- Ground reaction force calculation
- Friction and normal force computation  

Contact geometries handle:
- Contact state management during simulation
- Contact detection and penetration

NOTE: If new contact geometries or contact models (force laws) are implemented, 
the contact_parser.py code needs to be adjusted, accordingly!
"""

__all__ = ["base_contact", "base_geometry", "multi_contact", "contact_parser"]
