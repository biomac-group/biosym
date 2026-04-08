# biosym
[![Tests](https://github.com/biomac-group/biosym/actions/workflows/test.yml/badge.svg)](https://github.com/biomac-group/biosym/actions/workflows/test.yml)
[![Lint](https://github.com/biomac-group/biosym/actions/workflows/lint.yml/badge.svg)](https://github.com/biomac-group/biosym/actions/workflows/lint.yml)
[![Docs](https://readthedocs.org/projects/biosym/badge/?version=latest)](https://biosym.readthedocs.io/en/latest/)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Installation:
1. https://cyipopt.readthedocs.io/en/stable/install.html
2. Run `uv sync`

## Usage

biosym is a Python toolbox for building biomechanical models, exploring their structure and dynamics, and solving movement optimization problems. The examples cover three core workflows: loading and inspecting models for forward computations, setting up predictive gait optimal control problems from configuration files, and batching model evaluations with JAX for high-throughput or learning-based applications.

Load and inspect a model:

```python
from biosym.model.model import load_model

# Load a simple pendulum model and rebuild the cached functions from source.
model = load_model("tests/models/pendulum.xml", force_rebuild=True)

# Inspect the generalized coordinates, speeds, and overall problem size.
print(model.coordinates["names"])
print(model.speeds["names"])
print(model.n_states, model.n_constants)
```

Set up and solve an optimal control problem from a YAML file:

```python
from biosym.ocp import collocation

# Read model, objectives, constraints, and solver settings from YAML.
ocp = collocation.Collocation("examples/standing2d.yaml")
# Solve the optimal control problem and open the default visualization.
solution = ocp.solve(visualize=True)
```

Batch model evaluations with JAX:

```python
import jax

from biosym.model.model import load_model
from biosym.utils import states

# Load a gait model with contact and actuator dynamics.
model = load_model("tests/models/gait2d_torque/gait2d_torque.yaml")
# Replicate the default input state into a batch for vectorized evaluation.
batched_inputs = states.stack_dataclasses([model.default_inputs] * 128)

# Vectorize the model constraint function across the batch dimension.
dynamics_fn = jax.vmap(model.run["confun"], in_axes=(0, None))
batched_output = dynamics_fn(batched_inputs.states, batched_inputs.constants)

# The result contains one dynamics evaluation per batch element.
print(batched_output.shape)
```

## Near-term roadmap
We are actively developing biosym and welcome contributions from the community. Currently, we work on making it easier to install and use, which also means that some structural changes should be expected. Contact us if you want to contribute or have suggestions for improvements.

Additionally, here is what to expect from biosym in the near future:

1. Deep learning integration.
2. Differentiability with respect to musculoskeletal model parameters.
3. 3D-ready optimal control simulations.
4. Continued documentation improvements.

Follow our group's [LinkedIn page](https://www.linkedin.com/company/fau-biomac-group/) for updates on new features and releases.

## Citation
If you use biosym in academic work, please cite it as software:

```bibtex
@software{biosym_2026,
  title = {biosym: A Python toolbox for biomechanical movement simulation and optimal control},
  author = {Markus Gambietz and Theodoros Balougias and Yipeng Zhang and Anne Koelewijn},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/biomac-group/biosym}
}
```
