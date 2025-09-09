"""
Contact Models for Ground and Environmental Interactions.

This module provides classes and functions for modeling contact between the
biomechanical model and its environment, particularly ground contact forces.
Contact models are essential for simulating locomotion and other activities
where the model interacts with external surfaces.

The module includes:
- Base contact classes defining the common interface
- Parsers for loading contact definitions from XML files
- Specific contact model implementations (contact points, ground reaction forces)

Contact models handle:
- Ground reaction force calculation
- Contact detection and penetration
- Friction and normal force computation  
- Contact state management during simulation

Examples
--------
Load contact model from an XML file:

>>> import biosym.model.contact.contact_parser as parser
>>> contact = parser.get("path/to/contact.xml", body_weight=70.0)

Access contact properties:

>>> n_bodies = contact.get_n_bodies()
>>> contact_states = contact.get_n_states()
>>> bodies = contact.get_bodies()
"""

__all__ = ["base_contact", "contact_parser"]
