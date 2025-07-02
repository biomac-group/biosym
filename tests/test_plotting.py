import time
import unittest

import matplotlib
import numpy as np

from biosym.forward import simulation as sim
from biosym.model import model
from biosym.visualization import stickfigure

test_modellist = [
    "tests/test_models/pendulum.xml",
    "tests/test_models/pendulum_3d.xml",
    "tests/test_models/gait2d_torque/gait2d_torque.yaml",
]
test_modellist = ["tests/test_models/gait2d_torque/gait2d_torque.yaml"]


class TestPlotting(unittest.TestCase):
    """
    Test biosym.model.model.

    This includes testing the stick figure plotting functionality.
    """

    def test_stick_figure(self) -> None:
        """
        Test plotting a stick figure of the model.

        Two stage test that produces a standing stick figure and an animated stick figure.
        """

        def test_(modelfile: str) -> None:
            print("Testing single state stick figure plotting.")
            m = model.load_model(modelfile, force_rebuild=False)
            print("Please close the stick figure window to continue.")
            stickfigure.plot_stick_figure(m, m.default_inputs, 0.01)
            x = input("Was this the correct stick figure? [y]")
            assert x in [
                "y",
                "Y",
                "yes",
                "Yes",
                "YES",
            ], "The stick figure is not correct!"

            print("Testing list of states stick figure plotting.")
            env = sim.SimulationEnvironment(m, dt=0.01, initial_state="random")
            env.step()
            # print(f"JIT/Caching of step function took {timeit.timeit(f, number=10000)/10000:.6f} seconds.")
            t = np.zeros(m.forces["n"])

            states = [env.reset()]  # seed=0)]
            a = time.time()
            for _ in range(1000):
                s_, _, _, _, _ = env.step(t)
                states.append(s_)
            b = time.time()
            print(f"Simulation took {b-a:.2f} seconds.")
            print("Please close the stick figure window to continue.")
            stickfigure.plot_stick_figure(m, states, 0.01)
            x = input("Was this the correct stick figure animation? [y]")
            assert x in [
                "y",
                "Y",
                "yes",
                "Yes",
                "YES",
            ], "The stick figure animation is not correct!"

        print("========= Test Show Stick Neutral =========")
        # This test should be run with a GUI backend, e.g. TkAgg or Qt5Agg
        # Check if the GUI backend is available
        try:
            matplotlib.use("TkAgg")
        except ImportError:
            print("GUI backend not available. Skipping test.")
            return

        for testmodel in test_modellist:
            print(f"Testing model: {testmodel}")
            test_(testmodel)
            print("Testing model: Done")
        print("========= Test Show Stick Neutral Done =========")
        print()


if __name__ == "__main__":
    unittest.main()
