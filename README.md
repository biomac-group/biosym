# biosym
![Tests](https://github.com/yourusername/biosym/actions/workflows/test-and-lint.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-85%25-green)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Roadmap towards simulation toolbox
This is my personal opinion, everything can be discussed or changed. Please use a good template before starting. 

### Project Structure
```python
|- biosym # main folder
  |- model # 
    |- parser
      |- parsing.py # Base class(es)
      |- mujoco_parser.py # see below
      |- opensim_parser.py
    |- model.py # Base model class
    |- muscle # tbd what we need
      |- ...# muscle model classes
    |- ground contact
      |- ... # ground contact classes
  |- configs
    |- defaults
      |- model.yaml
      |- collocation.yaml
    |- models
    |- .yaml files
  |- ocp
    |- collocation.py
    |- confun.py
    |- objfun.py # Base class for objectives
    |- utils # I guess we need that folder at some point
  |- objectives
    |- base_objective.py # Base class
    |- ... # jax translations
  |- constraints
    |- base_constraint.py # Base class
    |- ... # jax translations
  |- forward sim / DL / other stuff
  |- utils # whatever is needed
  |- tests 
  |- ... # other stuff
|- examples
|- docs
```
### (Faster) parsing and lambdify

Based on current observations, using a single vector `v_` to store all variables before calling `lambdify` is best for generating usable `jax` output. To avoid the long duration of `xreplace` routines, we should start with this vector from the beginning and set all our variables, or as many as possible, as parts of this vector. Model should have dictionaries for all the necessary information, e.g. `run` (running jax functions), `n` (number of states, constraints, etc.), `states`, mappings, etc. 

Parser structure: The base model builder will get receive a `Parsing` class object, which itself has functions for getting info about e.g. a `dof`, `nDof`, inertial values, ... everthing we need. That way, it should be easy to expand to other modeling formats. Yipeng's project is a good start for this, it did both the model builing and mujoco parsing, so that will be split in two parts. 

### Old and new classes

Biomech-Sim-Toolbox.src structure
```python
src
  |- model
  |- problem
  |- trackingData
  |- result
  |- tests
  |- solver
```

Some things we should improve on:
1. `model`: We should have a common class that covers all models
2. `problem`: Rename to collocation, leave room for e.g. deep learning implementations or forward dynamics (unified interface for all yay!).  **Plotting** is important, and during the collocation process, e.g. dash app / plotly are way more interactive than static plots.
3. 'trackingData': I have no opinion on this, but I believe the implementation is MatLab-specific and doesn't translate super well to python. 
4. 'result': Specific to collocation, should be moved to the collocation class. I think it can be returned as a dict and we do some plotting helper functions instead. It should be implemented in a way that all objective functions (e.g. plotting) can still be called.
5. 'tests': Should be expanded.
6. 'solver': Specific to collocation, with `cyipopt` that should only be a couple lines of code anyways. 


### Collocation details
In our MatLab toolbox, we give `X` every objective, which is then sliced anyways. I would suggest to reshape `X` in the collocation class, so that all necessary information is available as a dict `{'coordinates':..., 'torques':..., 'model':..., ...etc}`. That should make the objectives more readable and easier to implement.

We will try to `jax.jit`-compile the completed objective functions, therefore, objectives need to be static. That means, that all `init` information should be stored centrally in the collocation class. When registering an objective, it will return it init data, name, etc. We only `jit`-compile objectives that are written in `jax`, others will be called iteratively. `jax` objectives do not need a gradient function, as `jax` will automatically differentiate them. Objectives will be class objects.

### Documentation
readthedocs.io; sphinx? I really don't think doxygen is very useful

### First steps
This project is really ambitious, and I think we should start with a small subset of the toolbox. I would suggest to start with the following:
1. From Yipeng Zhang's project [branch](https://mad-srv.informatik.uni-erlangen.de/MadLab/Biomech-Simu/student-code/p_zhang_yipeng/-/tree/optimal_control?ref_type=heads), refine the parser and optimal control functions to the above structure.
2. Objectives: translate from matlab to jax, where applicable.
3. Start with small subset, make it a minimum viable product (a bit cleaner than the previous implementation), then refine and add further features.

### User friendlyness: Yes
Config files (use OmegaConf!):
The default way of starting the introduction example should be as easy as: 
```python
import biosym
model = biosym.Model('defaults/model.yaml') # or 'gait3d.osim'...
problem = biosym.Collocation('defaults/collocation.yaml', model)
result = problem.solve()
```

[Check out the BT "OCP Radar Tracking" implementation for a current use of config files](https://mad-srv.informatik.uni-erlangen.de/MadLab/Biomech-Simu/radar-tracking/-/blob/main/data/benchmarks/T02/T02_periodic.yaml?ref_type=heads)
