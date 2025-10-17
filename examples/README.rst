.. _examples:

Examples
========

This directory contains examples demonstrating the biosym library functionality.

Available Examples
------------------

**gait2d.py**
    2D gait simulation
    * Loading 2D model
    * Standing and walking optimal control problems

**interacting_with_models.py**
    Basic model interface demonstration:
    
    * Model loading and structure inspection
    * State vector organization and manipulation
    * Model functions

Model Configuration Files
--------------------------

* **standing2d.yaml** - 2D standing optimal control configuration
* **walking2d.yaml** - 2D walking optimal control configuration

Running Examples
----------------

.. code-block:: bash

    python interacting_with_models.py
    python gait2d.py

Dependencies
------------

Required: numpy, matplotlib, jax, sympy, biosym

Model Files
-----------

Examples use models from ``../tests/models/gait2d_torque/gait2d_torque.yaml``
