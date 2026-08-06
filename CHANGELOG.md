# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] -

### Added

## [0.1.12]

* No changes made, release 0.1.11 was messed up.

## [0.1.11]

* Fixed a body-weight/units bug in the `dynamics` constraint and the ground-contact model handoff: `body_weight` was being computed and passed around as raw mass (kg) instead of a force in Newtons (`mass * g`). Since the dynamics residual is force-valued, normalizing it by mass instead of weight left a leftover factor of `g` for any unresolved force imbalance, causing standing/walking collocation problems to stall (IPOPT constraint violation pegged at exactly `9.81`, ending in local infeasibility). Fixed in `constraints/dynamics.py`, `ocp/confun.py`, `model/model.py`, `contact_springdamper.py`, and `contact_models/gait2dc_contact.py` (removed a fragile `bw_n < 200` kg-vs-Newtons guessing heuristic that had papered over the same bug).
* Deprecated the `"1/BW"` special-case constraint weight (`confun.py`, `utils/reporting.py`): constraints needing body-weight normalization (e.g. `dynamics`) now do so internally and correctly in Newtons, so their residuals are already O(1) — a plain numeric weight (typically `1.0`) should be used instead. Using `"1/BW"` now raises a clear `ValueError`.
* Updated `examples/gait2d.py` and its generated docs gallery pages, plus the `dorschky2019` project's `standing2d.yaml`, to use `weight: 1.0` instead of the now-unsupported `"1/BW"`.
* Added full 3D model support, verified against `rajogopal2016.osim` (Rajagopal et al. 2016; 22 bodies, 39 DOF, 80 muscles): `CustomJoint`/`UniversalJoint` parsing (`biosym/model/joints/joint_models/custom_joint.py`, `universal_joint.py`), decomposed into biosym's existing hinge/slide `flat_joints` convention.
* Added a `dynamics_backend="aba"|"sympy"` option on `SimulationEnvironment` (`biosym/forward/simulation.py`), defaulting to the pure-JAX ABA kernel (`biosym/utils/aba.py`) for forward dynamics; the SymPy/Kane pipeline remains available and is still the better choice for small/2D models, where its compile cost is negligible. Implemented `SimulationEnvironment.run()` (previously an empty stub) as a fixed-step trajectory rollout, and added a `SimState` dataclass to carry state snapshots.
* Added muscle-path polynomial fitting from live OpenSim geometry (`OsimParser._fit_muscle_paths`, opt-in via `fit_muscle_paths=True` on `OsimParser`/`BiosymModel`, off by default since fitting a full muscle set is expensive — see `performance_bottlenecks.md`).
* `model.py`'s `load_model` (hash-based model cache) now accepts and caches on `fit_muscle_paths` (previously it wasn't even a `load_model` parameter, so any caller wanting muscles had to call `BiosymModel(..., fit_muscle_paths=True)` directly, bypassing the cache and re-paying the multi-minute fit on every run). The cache filename now also encodes `compile_eom`/`fit_muscle_paths` (previously the cache key came only from `BiosymModel._get_hash()`, which hashes file contents and is agnostic to either flag) — this closes a latent correctness gap too: switching either flag for the same model file could previously silently return a cached model built with the other flag's value. `biosym.ocp.collocation.Collocation`'s YAML `settings.fit_muscle_paths` option threads this through for OCP configs, mirroring `settings.compile_eom`.
* Added the `Millard2012` muscle actuator model (`biosym/model/actuators/actuator_models/millard2012.py`), consuming the fitted path polynomials for state-dependent length/moment arms, with real per-muscle curve parameters (force-length, force-velocity, tendon compliance, pennation) read from the OSIM XML. Wired into `actuator_parser` (`Millard2012EquilibriumMuscle` OSIM reader/builder), gracefully skipping muscles without fitted path data instead of failing model load.
* `actuator_parser` now reads OpenSim's `optimal_force` on `CoordinateActuator`/`TorqueActuator` and applies it as a control-to-generalized-force scale (previously ignored); non-finite `min_control`/`max_control` now fall back to a nominal +/-1 activation range instead of leaving unusable infinite bounds.
* Added a YAML-configurable actuator torque scale override (`additional_parameters.actuators.scale` block in `model.py`, per-joint or `default`, with an optional `max_activation`), and made the YAML `additional_parameters.actuators` block itself optional (previously mandatory whenever `additional_parameters` was present at all, even for contact-only YAMLs).
* Fixed `MultiActuator` (`biosym/model/actuators/multi_actuator.py`) to support combining two or more state-bearing actuator types in one model (e.g. muscles + coordinate actuators, as in Rajagopal2016): `forward()`/`reset()` now slice each member's own segment of the combined `states.actuator_model`/`constants.actuator_model` vectors instead of handing every member the full combined vector; added a `.bounds` property needed by OCP variable-bounds setup. Previously this only worked by accident because no existing model combined two state-bearing actuator types.
* `effort_term` now supports a `weighting` option ("volumeweighted" or "equal"), matching BioMAC-Sim-Toolbox's `effortTermMuscles.m`/`effortTermMusclesAct.m`: "volumeweighted" weights each actuator by `fmax * lceopt` (a muscle volume/mass proxy), normalized to sum to 1, and is now the **default** for muscle actuator models (falls back to "equal", i.e. `1/n_actuators`, for non-muscle/torque-driven models). **Behavior change:** the objective value is now a properly normalized weighted mean over actuators (weights summing to 1) instead of an unweighted, unnormalized sum — existing configs relying on the old raw-sum magnitude (e.g. tuned `effort_term` weights) will see a different effective scale (roughly divided by the number of actuators) and may need re-tuning.
* `effort_term` now also works for pure torque/coordinate-actuator models (previously `n_actuators = model.actuators.idx["a"].shape[0]` and `range_actuators = model.actuators.idx["a"]` assumed a Hill2d/Millard2012-style actuator exposing an `.idx["a"]` activation slice; `CoordinateActuator`/`TorqueActuator` have no such split, since their whole `actuator_model` state block already is one commanded-force value per actuator). Now uses `model.actuators.get_n_actuators()` and falls back to `jnp.arange(n_actuators)` when `.idx` isn't present.
* Added `model.run["mass_matrix"]`/`["forcing"]` computed via RNEA (`BiosymModel._register_rnea_mass_matrix_forcing`, `model.py`), used automatically whenever a model is built with `compile_eom=False` (previously these simply didn't exist for such models, since only the SymPy/Kane path registered them) — a drop-in replacement for `SimulationEnvironment`'s `"sympy"` forward-dynamics backend on such models.
* `biosym/constraints/dynamics.py`'s OCP `dynamics` constraint now uses a **direct RNEA residual** (`confun_rnea_tau`: `model.run["rnea"](states, constants) - tau_applied`) instead of the mass_matrix/forcing formulation whenever `compile_eom=False`. RNEA already fuses `M(q) qdd + C(q,qd) + g(q)` into one O(n) pass without forming `M` or the bias term separately, so reconstructing them (`n+2` extra RNEA calls) to then recompose `M @ qdd - forcing` is pure overhead when `qdd` is already known (an OCP decision variable, not something to solve for). Benchmarked on `rajogopal2016.osim` (39 DOF, `scratch/benchmark_rnea_vs_rnea_massmatrix.py`): the direct residual is ~3x faster per call and its Jacobian ~4.3x faster (both numerically identical to the reconstructed version) — since IPOPT calls the Jacobian roughly as often as the constraint itself in practice, the Jacobian cost dominates, and a naive per-call-only comparison would have understated the win (~3.9x faster end to end, weighting 2 constraint evals + 1 Jacobian eval per the observed IPOPT call pattern). No custom `jax.custom_jvp` was needed for this path either — unlike the mass_matrix/forcing formulation (which needs one to avoid differentiating through the full O(n^2) mass matrix), RNEA's own forward pass has no such intermediate to avoid, so plain reverse-mode `jax.jacobian` (already what `get_jacobian()` uses generically) is already efficient here, consistent with the earlier RNEA-Jacobian investigation. Unblocks OCP-based problems (e.g. a standing/equilibrium solve) on large 3D models like Rajagopal2016, where the SymPy/Kane symbolic derivation does not finish in practical time (confirmed: still running after 18+ minutes, vs. ~7s to build the same model with `compile_eom=False`).
* `biosym/constraints/dynamics.py`'s and `biosym/ocp/confun.py`'s `Constraint.ncons_model`/`Constraints.ncons_model` (the per-node dynamics-equation count) no longer requires `model.fr` (SymPy/Kane's residual vector, `compile_eom=True` only) — falls back to `model.coordinates.n` (the same value) for `compile_eom=False` models, needed for the OCP `dynamics` constraint to work at all on RNEA-backed models.
* Added `biosym.ocp.collocation.Collocation`'s YAML `settings.compile_eom` option (defaults to `True`, matching `BiosymModel`'s default), threaded through to `load_model`, so a collocation config can opt a model into the RNEA-backed dynamics path above.
* `MultiActuator` (`biosym/model/actuators/multi_actuator.py`) now proxies `constraints()`/`jacobian()` to constraint-bearing members (e.g. Hill2d/Millard2012's force-equilibrium + activation-dynamics constraints) — previously entirely missing, so any model mixing a constraint-bearing muscle actuator with another actuator type (e.g. Rajagopal2016's muscles + coordinate actuators) crashed with `AttributeError` the instant an OCP was built. `constraints()` forwards to each member (same per-member `_member_slices()` offsets `forward()` already uses) and concatenates. `jacobian()` deliberately does *not* hand-derive its own sparse structure the way Hill2d does (per-node/predecessor-node/dur-column bookkeeping) — instead it's a plain `jax.jacobian` of `MultiActuator`'s own `constraints()`, declaring every structurally-possible entry present (a fully dense block per constraint row, mirroring the same tradeoff `dynamics.py`'s own `nnodes==1` block already makes) rather than inventing new per-actuator-type sparse-Jacobian math. `get_nnz()` is overridden to match exactly (cyipopt requires this). Verified against a single-member Hill2d wrap (bit-for-bit identical constraints/Jacobian to calling Hill2d directly) and the real 97-actuator Rajagopal2016 model (66,665 nonzeros for the whole `dynamics` block — small and tractable, not the ~66x-worse naive estimate first feared).


## [0.1.10]

* Added a new advanced example (`examples/advanced/sliding_contact_gait2d.py`) demonstrating a sliding ground-contact gait2d model, plus matching XML/YAML model files and generated docs gallery pages.
* Updated the `gait2d.py` example and `standing2d.yaml`/`walking2d.yaml` example configs.
* Added `register_contact_model` in `contact_parser.py`, letting a user plug in a custom stateful ground-contact model (selected via `<ground_contact_model type="...">`) without editing the parser module.
* Minor updates to `hill2d.py` actuator model, `effort_term.py`/`track_grf.py` objectives, and `stickfigure.py` visualization.

## [0.1.9]

* Added a missing `gait2d_huntcrossley` model (actuators, ground contact, and YAML config) to the downloadable models.
* Added a `docs/downloads.rst` page for downloadable models.

## [0.1.8]

* Restructured the docs changelog: removed `docs/changelog.rst` in favor of including the root `CHANGELOG.md` directly.
* Added a downloadable-models section to the docs guides.
* Fixed the `uv.lock` file.

## [0.1.7]

* Unified the `dynamics` constraint: removed the separate `actuators` and `ground_contact` constraint modules (and `dynamics_unified.py`), folding their logic into a single expanded `dynamics.py` constraint. Example YAMLs and the `gait2d.py` example script were updated to drop the now-redundant constraint entries.
* Added an `opt_states` template (q + qd + gc + actuators) for the collocation state vector, with matching updates to objectives, initial-guess, and settings utilities.
* Fixed contact model bias terms in `contact_huntcrossley.py` and `contact_springdamper.py` so the solver always has a gradient toward the ground.
* Corrected constraint bookkeeping in the iteration logger and implemented a random initial guess for the standing problem.

## [0.1.6] - Refactor Changelog & Migration

This section details the API updates, structure changes, and deprecations introduced in version **0.1.6**.

### Summary of Key Changes

* **Transition to Namedtuples**: All model symbols and constants are now exposed as namedtuples instead of dictionaries.
* **Dot Notation Access**: Namedtuple values must be accessed using dot notation (e.g., `model.coordinates.names`) instead of dictionary bracket lookup (e.g., `model.coordinates['names']`).
* **Updated Data Structures**: The `States` and `Constants` structures in `biosym.utils.states` have been updated with the new state/constant symbols.

### Detailed Changes & Migration Instructions

#### 1. State Vectors & Symbols

All dynamic state variables and symbols are now organized into distinct namedtuples under the `model` class:

| Namedtuple | Description |
| --- | --- |
| `model.coordinates` | Model degrees of freedom / coordinates. |
| `model.speeds` | Generalized speeds. |
| `model.accs` | Generalized accelerations. |
| `model.tau` (formerly `model.forces`) | Generalized forces corresponding to the coordinates. |
| `model.ext_forces` | External forces acting on the system. |
| `model.ext_torques` | External torques acting on the system. |

> **Migration Tip:** These structures were previously dictionaries. You must now use **dot notation** to access their contents. For example, change `model.coordinates['names']` to `model.coordinates.names`.

#### 2. Model Constants

Similarly, physical constants have been converted from dictionaries to namedtuples. They are still located under the following attributes:

- `model.g`
- `model.mass` (previously `model.masses`)
- `model.inertia`
- `model.com`
- `model.offset`
- (and others...)

Like the state symbols, access these constants using **dot notation** instead of bracket notation.

#### 3. States & Constants Data Structures

The data structures in `biosym.utils.states` (e.g., the `States` and `Constants` classes) have been updated to support the new state and constant symbols:

* Contains `states.q`, `states.qd`, etc.
* **Breaking Change**: `states.model` and `constants.model` have been removed.
* **Breaking Change**: The `utils.states.StatesDict` class has been removed. Instead, please use `States` and `Constants` dataclasses directly.
* Older compatibility layers for accessing variables have been removed. Use the new JAX-compatible structures directly instead of accessing the legacy `model.coordinates` or `model.speeds` representations.
* States and Constants can be filtered by name using the `filter` method:

```python
filtered_states = model.default_states.filter(["q", "qd"])
filtered_constants = model.default_constants.filter(["g", "mass"])

# For convinience, all states / constants that were in the "model" field, can be accessed via:
model_states = model.default_states.filter("model").to_array()
model_constants = model.default_constants.filter("model").to_array()

# --> Can be concatenated to get the old model.states / model.constants
```

* **Deprecations**: `stack_dataclasses`, `reduce_dataclasses` and `dict_to_dataclass` were removed:
  - `concatenate` (from `dataclasses` module) should be used instead of `stack_dataclasses`.
  - `resample` is a new function for down/up sampling of states.
  - `reduce_dataclasses` is an open todo, lets see what this breaks.

#### 4. Internal SymPy Cleanups

* The large internal SymPy matrix bases `_v` and `_constants` have been removed. `_v` is reintroduced as `_symbols` as a convinient export helper.
* The `self._dynamic` SymPy matrix is now split into `self._dynamic_qd` (speeds) and `self._dynamic_qdd` (accelerations).

#### 5. Refactored parts of biosym

I hope that I have found all breaking changes (and continous ones) in the code base.

Currently, these files have been updated to the new state & constants representation:
* `model.model.py`
* `utils.states.py`
* `model.contact.contact_model.contact_points.py`
* `model.actuators.actuator_model.hill2d.py`
* `ocp`

This list is very non-exhaustive, there is a few lines that needed to be changed in every constraint and objective tbh :) (and in OCP etc.)

#### 6. New features: ABA and RNEA

Introduced RNEA and ABA, now with faster jacobians.

#### 7. Model.run dictionary

You can now call model.run functions with n-dimensional inputs, as long as the model states/constant dimenstions are last.
* **Breaking Change** Renamed `confun` and `jacobian` to `kane` and `kane_jacobian` to better reflect what they actually do.
* Added `rnea`, `aba`, `rnea_jacobian` and `aba_jacobian` functions.

#### 8. OCP Refactoring

The `biosym.ocp` module has been reorganised and updated to the new `States`/`Constants` structures.

**Module restructuring**

* `biosym/ocp/` now contains dedicated sub-modules:

  - `utils/initial_guess.py` – initial-guess helpers (migrated out of `collocation.py`)
  - `utils/settings.py` – settings processing (migrated out of `collocation.py`)
  - `utils/vectorize.py` – vectorisation utilities (formerly `ocp/utils.py`)
  - `utils/problem.py` – cyipopt interface (formerly inline in `collocation.py`)

**Breaking changes in constraints & objectives**

* All old dict-style accesses on `_ModelProperties` / `_ForceProperties` namedtuples have been removed. Use dot notation:

  | Old (broken) | New |
  | --- | --- |
  | `model.tau["n"]` | `model.tau.n` |
  | `model.tau["idx"]` | `model.tau.combined_idx` |
  | `model.coordinates["n"]` | `model.coordinates.n` |
  | `model.speeds["n"]` | `model.speeds.n` |

  Affected files: `constraints/actuators.py`, `objectives/effort_term.py`,
  `model/actuators/actuator_models/hill2d.py`,
  `model/contact/contact_models/gait2dc_contact.py`,
  `model/contact/contact_models/contact_points.py`.

* Constraint functions in `constraints/discretization.py`, `constraints/adaptive_h.py`,
  and `constraints/dynamics_unified.py` no longer access `states.h` or `states.states`.
  Step-size `h` is now read from `globals_dict.h` directly.

* `Collocation.solve` now returns a `namedtuple` for easier access and more intuitive result interactions.

**Conceptual change in periodicity**

(only planned yet) - don't have an extra node, we can just verify against the first
It may be that there is a periodic_expand function that mimics the creation of the last node.

#### 9. Misc updates

* Corrected plotting & framerate in stickfigure
* Performance optimizations in confun
* `constraints.dynamics` is set up to hold RNEA, ABA, Kane's method options for the future.

#### 10. Contact & Actuator Model refactor

* Now supports multiple contact point and multiple actuator models to be mixed.
* The base classes, `ContactModel` and `ActuatorModel`, have been refactored (...)
* Contact is split into `contact_geometries` and `contact_models`.
