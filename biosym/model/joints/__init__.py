"""
Joint Types for OpenSim Models
==============================

This module provides classes and functions for interpreting different joint types 
in OpenSim models. Joints are essential for defining the kinematic structure and 
movement constraints of biomechanical models.

The module includes:
- Base joint classes defining the common interface
- Specific joint type implementations (WeldJoint, PinJoint, etc.)

Examples
--------
Load joint from a parsed osim file (using the OsimParser) and access joint properties:

>>> type = joint.get_type()
>>> axis = joint.get_axis()
>>> range = joint.get_range()
>>> parent_body = joint.get_parent_body()
>>> child_body = joint.get_child_body()
>>> damping = joint.get_damping()
>>> stiffness = joint.get_stiffness()
"""

__all__ = ["base_joint"]
