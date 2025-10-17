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

**batching.py**
    Demonstration of batching capabilities in biosym:
    
    * How to use batched operations for model simulations
    * Performance comparison between batched and non-batched simulations


Running Examples
----------------

.. code-block:: bash

    python interacting_with_models.py
    python gait2d.py

Model Files
-----------

Examples use models from ``../tests/models/gait2d_torque/gait2d_torque.yaml``
