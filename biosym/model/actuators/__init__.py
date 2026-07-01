"""
Actuator Models for Biomechanical Simulation.

This module provides classes and functions for modeling different types of internal forces
in biomechanical systems. Actuators/Internal forces represent force/torque generating elements
such as muscles, motors, or other active/passive components that can produce motion in the model.

The module includes:
- Base actuator classes defining the common interface
- Parsers for loading actuator definitions + multi_actuator for systems with different types of actuators
- Specific actuator model implementations (coordinate actuator, Hill-type muscle, passive torques, etc.)
"""

__all__ = ["actuator_parser", "base_actuator", "multi_actuator"]