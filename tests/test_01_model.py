import time
import timeit
import unittest

import numpy as np

from biosym.model import model

testmodellist = [
    "tests/models/pendulum.xml",
    "tests/models/pendulum_3d.xml",
    "tests/models/gait2d_torque/gait2d_torque.yaml",
]

class TestModel(unittest.TestCase):
    """Test building and executing model functions."""

    def test_01_model_build(self) -> None:
        """Test the model building process."""
        print("========= Test Model Build =========")
        for testmodel in testmodellist:
            m = model.load_model(testmodel, force_rebuild=True)
            assert m is not None, f"Model {testmodel} failed to build."
        print("========= Test Model Build Done =========")
        print()

    def test_02_sympy_functions(self) -> None:
        """Test the sympy functions in the model."""
        print("========= Test Sympy Functions =========")
        for testmodel in testmodellist:
            m = model.load_model(testmodel, force_rebuild=False)
            mock_input = m.default_inputs
            for func in m.run:
                if func.endswith("uncompiled"):
                    print(f"Skipping uncompiled function: {func}.")
                    continue
                start_time = time.time()
                m.run[func](mock_input.states, mock_input.constants)
                end_time = time.time()
                print(f"Function {func}: jit took {end_time - start_time:.3f} seconds.")
        print("========= Test Sympy Functions Done =========")
        print()

    def test_03_dynamics(self) -> None:
        """Test the dynamics of the model."""
        print("========= Test Dynamics =========")
        print("Dynamics testing is skipped for now.")
        print("========= Test Dynamics Done =========")
        print()

    def test_04_speed_of_simulation(self) -> None:
        """Test the speed of the model functions."""
        print("========= Test Speed of Simulation =========")
        for testmodel in testmodellist:
            m = model.load_model(testmodel, force_rebuild=False)

            default_inputs = m.default_inputs   
            for func in m.run:
                if func.endswith("uncompiled"):
                    print(f"Skipping uncompiled function: {func}.")
                    continue
                # Measure the time taken to run the function

                start_time = time.time()
                m.run[func](default_inputs.states, default_inputs.constants)
                end_time = time.time()
                print(
                    f"JIT/Caching of {func} took {end_time - start_time:.6f} seconds."
                )
                time_ = timeit.timeit(
                    lambda m=m, f=func, s=default_inputs.states, c=default_inputs.constants: m.run[f](s, c),
                    number=1000,
                )
                print(f"Running {func} runs in {time_/1000:.6f} seconds.")
        print("========= Test Speed of Simulation Done =========")
        print()


if __name__ == "__main__":
    unittest.main()
