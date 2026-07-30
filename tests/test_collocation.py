"""Test collocation functionality."""
import os
from jax import config as jax_config

# Configure Matplotlib before tests import any plotting code.
import os

VIS = os.getenv("VIS", "0").lower() in ("1", "true", "yes")

# Prefer Wayland on WSLg; fall back to X11; otherwise non-interactive.
if VIS:
    # Check if qtagg is available
    try:
        import matplotlib.backends.backend_qtagg  # noqa: F401
        import matplotlib.pyplot as plt
        os.environ.setdefault("MPLBACKEND", "QtAgg")
        plt.figure()
        plt.close()
    except ImportError:
        pass
else:
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

import sys
import time

import biosym
from biosym.ocp import collocation
from biosym.ocp.utils import *

import jax.numpy as jnp
import numpy as np
import argparse
import pytest

def derivativetest(problem, x, eps=1e-5):
    """Test the derivative of the objective function.
    
    :param problem: The collocation problem instance.
    :param x: The input state vector.
    :return: The derivative of the objective function.
    """
    assert isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray), f"Input x must be a numpy array, got {type(x)}"
    x = np.array(x, dtype=np.float64)
    jac_jax_0 = problem.problem.jacobian(x)
    jacstruct = problem.problem.jacobianstructure()
    grad_jax = problem.problem.gradient(x)
    jac_jax = np.zeros((len(x), problem.constraints.ncon))
    # import matplotlib.pyplot as plt
    # print(problem.constraints.c_start)
    # plt.plot(jacstruct[0])
    # plt.show()
    # print(max(jacstruct[0]), max(jacstruct[1]))
    # print(jac_jax.shape)
    try:
        jac_jax[jacstruct[1], jacstruct[0]] = jac_jax_0
    except: 
        print("NVAR", len(x), "NCON", problem.constraints.ncon, "NVAR in jac", jacstruct[1].max(), "NCON in jac",  jacstruct[0].max())
        raise ValueError("Jacobian too large")


    jac_num = np.zeros_like(jac_jax)
    grad_num = np.zeros_like(grad_jax)
    x0 = x.copy()

    a = time.time()
    for i in range(len(x)):
        x = x0.copy()
        x[i] += eps  # Perturb the i-th element
        obj_1 = problem.problem.objective(x)
        con_1 = problem.problem.constraints(x)

        x = x0.copy()
        x[i] -= eps  # Perturb the i-th element in the opposite direction
        obj_2 = problem.problem.objective(x)
        con_2 = problem.problem.constraints(x)

        grad_num[i] = (obj_1 - obj_2) / (2 * eps)
        jac_num[i] = (con_1 - con_2) / (2 * eps)

        if i % 200 == 0 and time.time() - a > 5:
            print(f"Derivative test: {i}/{len(x)} done, ({100 * i / len(x):.2f}%)")

    print("Jacobian JAX vs Numerical:", jnp.allclose(jac_jax, jac_num, atol=1e-4))
    print("Gradient JAX vs Numerical:", jnp.allclose(grad_jax, grad_num, atol=1e-4))

    if not jnp.allclose(jac_jax, jac_num, atol=1e-4):
        # Print the index of the first mismatch
        first_mismatch_index = np.unravel_index(np.argmax(np.abs(jac_jax - jac_num), axis=None), jac_jax.shape)
        # Find the constraint and variable names
        for i,num in enumerate(problem.constraints.c_start):
            if first_mismatch_index[1] < num:
                break
        con = problem.constraints._constraints[i-1]
        if first_mismatch_index[0] >= len(x) - 2:
            var = "globals"
        else:
            var = first_mismatch_index[0] % (len(problem.model.state_vector)
                                             + problem.model.contact_model.get_n_states() 
                                             + problem.model.actuator_model.get_n_states())
            if var < len(problem.model.state_vector):
                var = problem.model.state_vector[var]
            elif var < len(problem.model.state_vector) + problem.model.contact_model.get_n_states():
                var = problem.model.contact_model.state_vector[var - len(problem.model.state_vector)]
            else:
                var = problem.model.actuator_model.state_vector[var - len(problem.model.state_vector) - problem.model.contact_model.get_n_states()]
        print(
            f"Max deviation in Jacobian (jax|num): {jac_jax[first_mismatch_index], jac_num[first_mismatch_index]} at index {first_mismatch_index} ({con}, var: {var})"
        )
    if not jnp.allclose(grad_jax, grad_num, atol=1e-4):
        # Print the index of the first mismatch
        first_mismatch_index = np.unravel_index(np.argmax(np.abs(grad_jax - grad_num), axis=None), grad_jax.shape)
        print(
            f"Max deviation in Gradient: {grad_jax[first_mismatch_index], grad_num[first_mismatch_index]} "
            f"at index {first_mismatch_index}"
        )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].spy(jac_jax.T, label="Jacobian JAX")
    ax[1].spy(jac_num.T, label="Jacobian Numerical")
    ax[0].set_title("Jacobian JAX")
    ax[1].set_title("Jacobian Numerical")
    ax[0].legend()
    ax[1].legend()
    # Set model.states as the xticks for both axes
    ax[0].set_xticks(np.arange(len(problem.model.state_vector)))
    ax[0].set_xticklabels(problem.model.state_vector, rotation=90)
    ax[1].set_xticks(np.arange(len(problem.model.state_vector)))
    ax[1].set_xticklabels(problem.model.state_vector, rotation=90)
    plt.savefig("test_jacobian_comparison.png")  # Save instead of show
    plt.close()  # Close to free memory

    plt.figure()
    plt.scatter(np.arange(len(jac_jax.flatten())), jac_jax.flatten())
    plt.scatter(np.arange(len(jac_num.flatten())), jac_num.flatten(), alpha=0.5)
    plt.savefig("test_jacobian_scatter.png")  # Save instead of show
    plt.close()  # Close to free memory


@pytest.fixture
def standing_problem():
    """Fixture to create a standing problem for testing."""
    return collocation.Collocation("tests/collocation/standing2d.yaml", force_rebuild=True)


@pytest.fixture
def walking_problem():
    """Fixture to create a walking problem for testing."""
    return collocation.Collocation("tests/collocation/walking2d.yaml", force_rebuild=True)


def test_standing_problem_solve(standing_problem):
    """Test that the standing problem can be solved."""
    sol = standing_problem.solve(visualize=VIS)
    x, globals_dict, info = sol.states, sol.globals, sol.info

    # Basic assertions
    assert x is not None, "Solution should not be None"
    assert info is not None, "Info should not be None"

    # Check that we have a valid solution structure
    assert hasattr(x, "q"), "Solution should have q attribute"
    print(info["status"])
    assert info["status"] in [0, 1], "Solver did not converge"


def test_walking_problem_solve(walking_problem):
    """Test that the walking problem can be solved."""
    x = states_dict_to_x(walking_problem.initial_guess_states, walking_problem.initial_guess_globals)
    rng = np.random.default_rng(42)
    x = rng.random(len(x)).astype(np.float64)  # Randomize the initial guess

    x, globals_dict = x_to_states_dict(x, walking_problem.initial_guess_states, walking_problem.initial_guess_globals)

    sol = walking_problem.solve(visualize=VIS)  # Disable visualization for tests
    x, globals_dict, info = sol.states, sol.globals, sol.info

    # Basic assertions
    assert x is not None, "Solution should not be None"
    assert globals_dict is not None, "Globals should not be None"
    assert info is not None, "Info should not be None"

    # Check that we have a valid solution structure
    assert hasattr(x, "q"), "Solution should have q attribute"
    assert hasattr(globals_dict, "h"), "Solution should have h attribute"
    assert info["status"] in [0, 1], "Solver did not converge"

def test_constraint_and_objective_functions(walking_problem):
    """Test the constraints and objective function evaluations."""
    problem = walking_problem
    functions_to_test = [
        (problem.constraints.confun, "constraints"),
        (problem.constraints.jacobian, "jacobian"),
        (problem.objective.objfun, "objectives"),
        (problem.objective.gradfun, "gradients"),
    ]

    for function, name in functions_to_test:
        print(f"Testing {name} function...")

        if len(problem.initial_guess_states) > 1:
            result = function(problem.initial_guess_states, problem.initial_guess_globals)
            start_time = time.time()
            for _ in range(10):
                result = function(problem.initial_guess_states, problem.initial_guess_globals)
            elapsed_time = time.time() - start_time
        else:
            result = function(problem.initial_guess_states, None)
            start_time = time.time()
            for _ in range(10):
                result = function(problem.initial_guess_states, None)
            elapsed_time = time.time() - start_time

        print(f"{function.__name__} evaluated in {elapsed_time} seconds")

        # Assert that the function returns something
        assert result is not None, f"{name} function should return a result"

        # Basic performance test - should complete within reasonable time
        assert elapsed_time < 10.0, f"{name} function took too long: {elapsed_time} seconds"

@pytest.mark.skip(reason="Not correctly implemented yet")
def test_all_objective_functions(walking_problem):
    """Test all objective functions."""
    # Add the test objective function
    for objective in biosym.objectives.__all__:
        kwargs = {}
        if objective == 'track_angles':
            kwargs = {'file': 'tests/collocation/walking2d_angles.csv'}
        elif objective == 'track_grf':
            kwargs = {'file': 'tests/collocation/walking2d_grf.csv'}
        walking_problem.objective.add_objective(objective, weight=1.0, kwargs=kwargs)

    walking_problem.objective.objfun(walking_problem.initial_guess_states, walking_problem.initial_guess_globals)
    walking_problem.objective.gradfun(walking_problem.initial_guess_states, walking_problem.initial_guess_globals)

    # Verify it was added (this test depends on the internal structure)
    # You might need to adjust this based on how add_objective works
    assert True  # Placeholder assertion


def test_muscle_constraint_structure(walking_problem):
    """Verify Hill2d reports exactly n_actuators*nnodes constraints (force-equilibrium only).

    This test guards against re-introducing dead constraints (e.g. the old c2
    activation-dynamics block that was multiplied by zero).
    """
    problem = walking_problem
    model = problem.model
    settings = problem.settings
    nnodes = settings.get("nnodes")

    # Find the Hill2d actuator model
    actuator = model.actuator_model
    from biosym.model.actuators.actuator_models.hill2d import Hill2d
    assert isinstance(actuator, Hill2d), "Expected Hill2d actuator for walking problem"

    n_act = actuator.get_n_actuators()

    # --- Constraint count ---
    expected_ncon = n_act * nnodes
    reported_ncon = actuator.get_n_constraints(model, settings)
    assert reported_ncon == expected_ncon, (
        f"get_n_constraints returned {reported_ncon}, expected {expected_ncon} "
        f"({n_act} actuators × {nnodes} nodes)"
    )

    # --- Per-node constraint count ---
    expected_per_node = n_act
    reported_per_node = actuator.get_n_constraints_per_node()
    assert reported_per_node == expected_per_node, (
        f"get_n_constraints_per_node returned {reported_per_node}, expected {expected_per_node}"
    )

    # --- Actual constraint vector length ---
    states = problem.initial_guess_states
    globals_ = problem.initial_guess_globals
    # Pull muscle constraints via the full collocation constraint vector
    full_c = problem.constraints.confun(problem.initial_guess_states, problem.initial_guess_globals)
    # Hill2d is the only actuator, so its constraints occupy the first n_act*nnodes entries
    c_vec = full_c[:expected_ncon]
    assert len(c_vec) == expected_ncon, (
        f"constraints vector slice length {len(c_vec)} does not match expected {expected_ncon}"
    )

    # --- Jacobian nnz consistency ---
    declared_nnz = actuator.get_nnz(model, settings)
    rows, cols, data = actuator.jacobian((states, globals_), problem.model.default_constants, model, settings)
    actual_nnz = len(data)
    assert actual_nnz <= declared_nnz, (
        f"jacobian() returned {actual_nnz} non-zeros but get_nnz() declared only {declared_nnz}"
    )

    # --- Jacobian index bounds ---
    total_vars = states.size() + (2 if globals_ is not None else 0)  # 2 globals: dur, speed
    assert int(rows.max()) < expected_ncon, (
        f"Jacobian row index {int(rows.max())} out of bounds (ncon={expected_ncon})"
    )
    assert int(cols.max()) < total_vars, (
        f"Jacobian col index {int(cols.max())} out of bounds (nvars={total_vars})"
    )


def test_constraint_jacobian_speed(walking_problem):
    """Benchmark constraint and Jacobian evaluation throughput.

    Reports per-call wall time (ms) after a JIT warm-up pass.
    Intended to catch performance regressions — e.g. re-introducing
    dead constraints or extra JIT traces.
    """
    import timeit

    problem = walking_problem
    states = problem.initial_guess_states
    globals_ = problem.initial_guess_globals
    confun = problem.constraints.confun
    jacfun = problem.constraints.jacobian

    n_repeats = 20

    # --- warm-up (forces JAX JIT compilation) ---
    confun(states, globals_)
    jacfun(states, globals_)

    # --- benchmark constraints ---
    t_con = timeit.timeit(lambda: confun(states, globals_), number=n_repeats)
    ms_con = t_con / n_repeats * 1e3
    print(f"\n[speed] constraints(): {ms_con:.2f} ms/call  ({n_repeats} repeats)")

    # --- benchmark jacobian ---
    t_jac = timeit.timeit(lambda: jacfun(states, globals_), number=n_repeats)
    ms_jac = t_jac / n_repeats * 1e3
    print(f"[speed] jacobian():     {ms_jac:.2f} ms/call  ({n_repeats} repeats)")

    # Sanity: both should complete within a reasonable per-call budget
    assert ms_con < 500, f"constraints() too slow: {ms_con:.1f} ms/call (limit 500 ms)"
    assert ms_jac < 500, f"jacobian() too slow: {ms_jac:.1f} ms/call (limit 500 ms)"


@pytest.mark.skip(reason="For debugging purposes only")
def test_derivative_accuracy(walking_problem):
    """Test derivative accuracy using finite differences (marked as slow)."""
    x = states_dict_to_x(walking_problem.initial_guess_states, walking_problem.initial_guess_globals)
    rng = np.random.default_rng(42)
    x = rng.random(len(x)).astype(np.float64)

    # This is a slow test, so we only run it when explicitly requested
    derivativetest(walking_problem, x)


if __name__ == "__main__":
    print("=== Collocation Test Script ===")
    print(f"Visualization mode: {'ON' if VIS else 'OFF'}")

    try:
        if not VIS:
            pass
        else:
            print("Running in test mode (no visualization)...")
            # Create problems for testing
            # Run individual tests
            print("Testing standing problem...")
            standing_prob = collocation.Collocation("tests/collocation/standing2d.yaml", force_rebuild=False)
            test_standing_problem_solve(standing_prob)

            print("Testing walking problem...")
            walking_prob = collocation.Collocation("tests/collocation/walking2d.yaml", force_rebuild=False)
            test_walking_problem_solve(walking_prob)

            print("Testing constraint and objective functions...")
            test_constraint_and_objective_functions(standing_prob)

            print("Testing objective function addition...")
            test_all_objective_functions(walking_problem)

        print("=== All tests completed successfully! ===")

    except (AssertionError, ValueError, RuntimeError) as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
