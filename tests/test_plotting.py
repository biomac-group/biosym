import unittest
from biosym.model import model
import numpy as np
from biosym.forward import simulation as sim
from biosym.visualization import stickfigure

test_modellist = ['tests/test_models/pendulum.xml', 'tests/test_models/pendulum_3d.xml','tests/test_models/gait2d_torque/gait2d_torque.yaml']
test_modellist = ['tests/test_models/gait2d_torque/gait2d_torque.yaml']

class TestPlotting(unittest.TestCase):
    def test_stick_figure(self):
        """
        Test plotting a stick figure of the model.
        """
        def test_(modelfile):
            print("Testing single state stick figure plotting.")
            m = model.load_model(modelfile, force_rebuild=False)
            print("Please close the stick figure window to continue.")
            stickfigure.plot_stick_figure(m, m.default_inputs, 0.01)
            x = input("Was this the correct stick figure? [y]")
            assert x in ['y', 'Y', 'yes', 'Yes', 'YES'], "The stick figure is not correct!"

            print("Testing list of states stick figure plotting.")
            env = sim.SimulationEnvironment(m, dt=0.01, initial_state="random")
            t = np.zeros((m.forces['n']))
            import time
            states = [env.reset()]#seed=0)]
            a = time.time()
            for i in range(1000):
                s_, _, _, _, _ = env.step(t)
                states.append(s_)
            b = time.time()
            print(f"Simulation took {b-a:.2f} seconds.")
            print("Please close the stick figure window to continue.")
            stickfigure.plot_stick_figure(m, states, 0.01)
            x = input("Was this the correct stick figure animation? [y]")
            assert x in ['y', 'Y', 'yes', 'Yes', 'YES'], "The stick figure animation is not correct!"


        print("========= Test Show Stick Neutral =========")
        # This test should be run with a GUI backend, e.g. TkAgg or Qt5Agg
        # Check if the GUI backend is available
        try:
            import matplotlib
            matplotlib.use('TkAgg')
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
    #xunittest.main()
    def test_(modelfile):
        print("Testing single state stick figure plotting.")
        m = model.load_model(modelfile, force_rebuild=True)
        print("Please close the stick figure window to continue.")
        #stickfigure.plot_stick_figure(m, m.default_inputs, 0.01)
        #x = input("Was this the correct stick figure? [y]")
        #assert x in ['y', 'Y', 'yes', 'Yes', 'YES'], "The stick figure is not correct!"
        print("Testing list of states stick figure plotting.")
        env = sim.SimulationEnvironment(m, dt=0.001, initial_state="neutral")
        t = np.zeros((m.forces['n']))
        import time
        states = [env.reset()]#seed=0)]
        a = time.time()
        for i in range(1000):
            s_, _, _, _, _ = env.step(t)
            states.append(s_)
        b = time.time()
        print(f"Simulation took {b-a:.2f} seconds.")
        print("Please close the stick figure window to continue.")
        stickfigure.plot_stick_figure(m, states, 0.001)
        x = input("Was this the correct stick figure animation? [y]")
        assert x in ['y', 'Y', 'yes', 'Yes', 'YES'], "The stick figure animation is not correct!"

    
    print("========= Test Show Stick Neutral =========")
    # This test should be run with a GUI backend, e.g. TkAgg or Qt5Agg
    # Check if the GUI backend is available
    try:
        import matplotlib
        matplotlib.use('TkAgg')
    except ImportError:
        print("GUI backend not available. Skipping test.")
        

    for testmodel in test_modellist:
        print(f"Testing model: {testmodel}")
        test_(testmodel)
        print("Testing model: Done")
    print("========= Test Show Stick Neutral Done =========")
    print()
