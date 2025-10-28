"""
Actuator Models for Biomechanical Simulation.

This module provides classes and functions for modeling different types of actuators
in biomechanical systems. Actuators represent force/torque generating elements such as
muscles, motors, or other active components that can produce motion in the model.

The module includes:
- Base actuator classes defining the common interface
- Parsers for loading actuator definitions from XML files
- Specific actuator model implementations (general, Hill-type muscle, passive torques)

Examples
--------
Load actuators from an XML file:

>>> import biosym.model.actuators.actuator_parser as parser
>>> actuators = parser.get("path/to/actuators.xml")

Access actuator properties:

>>> n_actuators = actuators.get_n_actuators()
>>> actuator_states = actuators.get_n_states()
"""

__all__ = ["actuator_parser", "base_actuator"]
