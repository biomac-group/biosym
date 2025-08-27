import time
import timeit
import unittest

import numpy as np

from biosym.forward import simulation as sim
from biosym.model import model

testmodellist = [
    #"tests/models/pendulum.xml",
    #"tests/models/pendulum_3d.xml",
    "tests/models/gait2d_torque/gait2d_torque.yaml",
]


class TestForwardSim(unittest.TestCase):
    """Test biosym.forward.simulation."""

    def test_freefall_simulation(self) -> None:
        """Test the freefall simulation of the model."""
        print("========= Test Forward Simulation =========")
        for testmodel in testmodellist:
            print(f"Testing model: {testmodel}")
            m = model.load_model(testmodel, force_rebuild=False)
            env = sim.SimulationEnvironment(m, dt=0.0001, initial_state="random")
            if m.actuators.is_torque_actuator():
                t = np.zeros(m.forces["n"])
            else:
                raise NotImplementedError("Non-torque actuators are not supported yet.")
            func = env.step
            start_time = time.time()
            func(t)
            end_time = time.time()
            print(
                f"JIT/Caching of step function took {end_time - start_time:.6f} seconds."
            )
            print(
                f"Step function runs in {timeit.timeit(lambda func=func,t=t: func(t), number=1000)/1000:.6f} seconds."
            )
            env_0001 = sim.SimulationEnvironment(m, dt=0.0001, initial_state="random")
            env_0001.reset(seed=0)
            for _ in range(10000):
                env_0001.step(t)
            env_1 = sim.SimulationEnvironment(m, dt=0.01, initial_state="random")
            env_1.reset(seed=0)
            for _ in range(100):
                env_1.step(t)
            print("Testing step function with dt=0.0001 and dt=0.01")

            assert np.allclose(
                env_0001.state.states.model,
                env_1.state.states.model,
                atol=5e-3,
            )
        print("========= Test Forward Simulation Done =========")

        del env_0001, env_1

        print()


if __name__ == "__main__":
    unittest.main()
