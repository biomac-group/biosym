import unittest
from biosym.model import model
from biosym.forward import simulation as sim
import numpy as np
import timeit, time

testmodellist = ['tests/test_models/pendulum.xml']

class TestForwardSim(unittest.TestCase):
    def test_freefall_simulation(self):
        """
        Test the forward simulation of the model.
        """
        print("========= Test Forward Simulation =========")
        m = model.load_model(testmodellist[0], force_rebuild=False)
        env = sim.SimulationEnvironment(m, dt=0.01, initial_state="random")
        if m.actuators.is_torque_actuator():
            t = np.zeros((m.forces['n']))
        else:
            raise NotImplementedError("Non-torque actuators are not supported yet.")
        func = env.step
        start_time = time.time()
        func(t)
        end_time = time.time()
        print(f"JIT/Caching of step function took {end_time - start_time:.6f} seconds.")
        print(f"Step function runs in {timeit.timeit(lambda: func(t), number=1000)/1000:.6f} seconds.")
        env_0001 = sim.SimulationEnvironment(m, dt=0.0001, initial_state="random")
        env_0001.reset(seed=0)
        for i in range(10000):
            env_0001.step(t)
        env_001 = sim.SimulationEnvironment(m, dt=1, initial_state="random")
        env_001.reset(seed=0)
        for i in range(1):
            env_001.step(t)
        print("Testing step function with dt=0.0001 and dt=1")
        assert np.allclose(env_0001.state['states']['model'], env_001.state['states']['model'], atol=1e-5), "The state vectors are not close enough!"
        print("========= Test Forward Simulation Done =========")
        print()

if __name__ == "__main__":
    unittest.main()