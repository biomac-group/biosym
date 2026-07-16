=============================
v0.1.6 Refactor Changelog & Migration
=============================

This document details the API updates, structure changes, and deprecations introduced in version **0.1.6**.

Summary of Key Changes
======================

* **Transition to Namedtuples**: All model symbols and constants are now exposed as namedtuples instead of dictionaries.
* **Dot Notation Access**: Namedtuple values must be accessed using dot notation (e.g., ``model.coordinates.names``) instead of dictionary bracket lookup (e.g., ``model.coordinates['names']``).
* **Updated Data Structures**: The ``States`` and ``Constants`` structures in :mod:`biosym.utils.states` have been updated with the new state/constant symbols.

---

Detailed Changes & Migration Instructions
=========================================

1. State Vectors & Symbols
--------------------------

All dynamic state variables and symbols are now organized into distinct namedtuples under the ``model`` class:

.. list-table:: State Vectors Namedtuples
   :widths: 30 70
   :header-rows: 1

   * - Namedtuple
     - Description
   * - ``model.coordinates``
     - Model degrees of freedom / coordinates.
   * - ``model.speeds``
     - Generalized speeds.
   * - ``model.accs``
     - Generalized accelerations.
   * - ``model.tau`` (formerly ``model.forces``)
     - Generalized forces corresponding to the coordinates.
   * - ``model.ext_forces``
     - External forces acting on the system.
   * - ``model.ext_torques``
     - External torques acting on the system.

.. admonition:: Migration Tip
   :class: note

   These structures were previously dictionaries. You must now use **dot notation** to access their contents. For example, change ``model.coordinates['names']`` to ``model.coordinates.names``.

2. Model Constants
------------------

Similarly, physical constants have been converted from dictionaries to namedtuples. They are still located under the following attributes:

- ``model.g``
- ``model.mass`` (previously ``model.masses``)
- ``model.inertia``
- ``model.com``
- ``model.offset``
- (and others...)

Like the state symbols, access these constants using **dot notation** instead of bracket notation.

3. States & Constants Data Structures
-------------------------------------

The data structures in :mod:`biosym.utils.states` (e.g., the :class:`~biosym.utils.states.States` and :class:`~biosym.utils.states.Constants` classes) have been updated to support the new state and constant symbols:

* Contains ``states.q``, ``states.qd``, etc.
* **Breaking Change**: ``states.model`` and ``constants.model``have been removed.
* **Breaking Change**: The ``utils.states.StatesDict`` class has been removed. Instead, please use ``States`` and ``Constants`` dataclasses directly.
* Older compatibility layers for accessing variables have been removed. Use the new JAX-compatible structures directly instead of accessing the legacy ``model.coordinates`` or ``model.speeds`` representations.
* States and Constants can be filtered by name using the ``filter`` method:

.. code-block:: python

    filtered_states = model.default_states.filter(["q", "qd"])
    filtered_constants = model.default_constants.filter(["g", "mass"])

    # For convinience, all states / constants that were in the "model" field, can be accessed via:
    model_states = model.default_states.filter("model").to_array()
    model_constants = model.default_constants.filter("model").to_array()

    # --> Can be concatenated to get the old model.states / model.constants

* **Deprecations**: ``stack_dataclasses``, ``reduce_dataclasses`` and ``dict_to_dataclass`` were removed:
  - ``concatenate`` (from ``dataclasses`` module) should be used instead of ``stack_dataclasses``.
  - ``resample`` is a new function for down/up sampling of states.
  - ``reduce_dataclasses`` is an open todo, lets see what this breaks.

4. Internal SymPy Cleanups
---------------------------

* The large internal SymPy matrix bases ``_v`` and ``_constants`` have been removed. ``_v`` is reintroduced as ``_symbols`` as a convinient export helper.
* The ``self._dynamic`` SymPy matrix is now split into ``self._dynamic_qd`` (speeds) and ``self._dynamic_qdd`` (accelerations).

5. Refactored parts of biosym
-----------------------------

I hope that I have found all breaking changes (and continous ones) in the code base.

Currently, these files have been updated to the new state & constants representation:
* ``model.model.py``
* ``utils.states.py``
* ``model.contact.contact_model.contact_points.py``
* ``model.actuators.actuator_model.hill2d.py``
* ``ocp``

This list is very non-exhaustive, there is a few lines that needed to be changed in every constraint and objective tbh :) (and in OCP etc.)

6. New features: ABA and RNEA
-----------------------------

Introduced RNEA and ABA, now with faster jacobians.

7. Model.run dictionary
-----------------------

You can now call model.run functions with n-dimensional inputs, as long as the model states/constant dimenstions are last.
* **Breaking Change** Renamed ``confun`` and ``jacobian`` to ``kane`` and ``kane_jacobian`` to better reflect what they actually do.
* Added ``rnea``, ``aba``, ``rnea_jacobian`` and ``aba_jacobian`` functions. 

8. OCP Refactoring
------------------

The ``biosym.ocp`` module has been reorganised and updated to the new ``States``/``Constants`` structures.

**Module restructuring**

* ``biosym/ocp/`` now contains dedicated sub-modules:

  - ``utils/initial_guess.py`` – initial-guess helpers (migrated out of ``collocation.py``)
  - ``utils/settings.py`` – settings processing (migrated out of ``collocation.py``)
  - ``utils/vectorize.py`` – vectorisation utilities (formerly ``ocp/utils.py``)
  - ``utils/problem.py`` – cyipopt interface (formerly inline in ``collocation.py``)

**Breaking changes in constraints & objectives**

* All old dict-style accesses on ``_ModelProperties`` / ``_ForceProperties`` namedtuples have been removed. Use dot notation:

  .. list-table::
     :widths: 50 50
     :header-rows: 1

     * - Old (broken)
       - New
     * - ``model.tau["n"]``
       - ``model.tau.n``
     * - ``model.tau["idx"]``
       - ``model.tau.combined_idx``
     * - ``model.coordinates["n"]``
       - ``model.coordinates.n``
     * - ``model.speeds["n"]``
       - ``model.speeds.n``

  Affected files: ``constraints/actuators.py``, ``objectives/effort_term.py``,
  ``model/actuators/actuator_models/hill2d.py``,
  ``model/contact/contact_models/gait2dc_contact.py``,
  ``model/contact/contact_models/contact_points.py``.

* Constraint functions in ``constraints/discretization.py``, ``constraints/adaptive_h.py``,
  and ``constraints/dynamics_unified.py`` no longer access ``states.h`` or ``states.states``.
  Step-size ``h`` is now read from ``globals_dict.h`` directly.

* ``Collocation.solve`` now returns a ``namedtuple`` for easier access and more intuitive result interactions.

**Conceptual change in periodicity** 

(only planned yet) - don't have an extra node, we can just verify against the first
It may be that there is a periodic_expand function that mimics the creation of the last node.

9. Misc updates
---------------

* Corrected plotting & framerate in stickfigure
* Performance optimizations in confun
* ``constraints.dynamics`` is set up to hold RNEA, ABA, Kane's method options for the future.

10. Contact & Actuator Model refactor
------------------------

* Now supports multiple contact point and multiple actuator models to be mixed.
* The base classes, ``ContactModel`` and ``ActuatorModel``, have been refactored (...)
* Contact is split into ``contact_geometries`` and ``contact_models``.
