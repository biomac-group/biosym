import unittest
from biosym.model import model
import numpy as np

testmodellist = ['tests/test_models/pendulum.xml', 'tests/test_models/pendulum_3d.xml','tests/test_models/gait2d_torque/gait2d_torque.yaml']

class TestModel(unittest.TestCase):
    def test_01_model_build(self):
        """
        Test the model building process.
        """
        print("========= Test Model Build =========")
        for testmodel in testmodellist:
            m = model.load_model(testmodel, force_rebuild=True)
            self.assertIsNotNone(m, f"Model {testmodel} failed to build.")
        print("========= Test Model Build Done =========")
        print()

    def test_02_sympy_functions(self):
        """
        Test the sympy functions in the model.
        """
        print("========= Test Sympy Functions =========")
        for testmodel in testmodellist:
            m = model.load_model(testmodel, force_rebuild=False)
            mock_input = m.default_inputs
            for func in m.run:
                if func.endswith("uncompiled"):
                    print(f"Skipping uncompiled function: {func}.")
                    continue
                m.run[func](mock_input['states'], mock_input['constants'])
        print("========= Test Sympy Functions Done =========")
        print()



    def test_03_dynamics(self):
        """
        Test the dynamics of the model.
        """
        print("========= Test Dynamics =========")
        print('Dynamics testing is skipped for now.')
        print("========= Test Dynamics Done =========")
        print()
    
    def test_04_speed_of_simulation(self):
        """
        Test the speed of the model functions.
        """
        print("========= Test Speed of Simulation =========")
        for testmodel in testmodellist:
            m = model.load_model(testmodel, force_rebuild=False)

            states = {
                'model': np.zeros(m.n_states),
                'gc_model': np.zeros(0),
                'actuator_model': np.zeros(0),
            }
            constants = {
                'model': np.zeros(m.n_constants),
                'gc_model': np.zeros(0),
                'actuator_model': np.zeros(0),
            }
            for func in m.run:
                if func.endswith("uncompiled"):
                    print(f"Skipping uncompiled function: {func}.")
                    continue
                # Measure the time taken to run the function
                import timeit, time
                start_time = time.time()
                m.run[func](states, constants) # Caching
                end_time = time.time()
                print(f"JIT/Caching of {func} took {end_time - start_time:.6f} seconds.")
                print(f"Running {func} runs in {timeit.timeit(lambda: m.run[func](states, constants), number=1000)/1000:.6f} seconds.")
        print("========= Test Speed of Simulation Done =========")
        print()

"""
Matlab model tests:
%> | derivativetest_Dynamics         | x          |          |           |        |       |         | x           |         |        |                 |                 |             |           |
%> | derivativetest_SimuAccGyro_Acc  | x          |          |           |        |       |         |             |         |        |                 |                 | x           |           |
%> | derivativetest_SimuAccGyro_Gyro | x          |          |           |        |       |         |             |         |        |                 |                 | x           |           |
%> | derivativetest_Moments          | x          |          |           |        |       |         |             |         |        | x               |                 |             |           |
%> | derivativetest_GRF              | x          |          |           |        |       |         |             |         | x      |                 |                 |             |           |
%> | derivativetest_Fkin             | x          |          |           |        |       |         |             | x       |        |                 |                 |             |           |
%> | test_simulateFreefall           |            | x        |           |        |       |         | x           |         |        |                 |                 |             |           |
%> | test_showStickNeutral           |            |          |           |        |       | x       |             |         |        |                 |                 |             | x         |
%> | test_dynamics                   |            |          |           |        |       |         | x           |         | x      | x               |                 |             |           |
%> | test_memory                     |            |          |           | x      |       |         | x           |         |        |                 |                 |             |           |
%> | test_speedOfSimuAccGyro         |            |          |           |        | x     |         |             |         |        |                 |                 | x           |           |
%> | test_speedOfMex                 |            |          |           |        | x     |         | x           |         |        |                 |                 |             |           |
%> | test_s_hip_flexion_gyro         |            |          |           |        |       |         |             |         |        |                 |                 | x           |           |
%> | test_s_pelvis_rotation_gyro     |            |          |           |        |       |         |             |         |        |                 |                 | x           |           |
%> | test_s_hip_flexion_acc          |            |          |           |        |       |         |             |         | x      |                 |                 | x           |           |
%> | test_s_pelvis_rotation_acc      |            |          |           |        |       |         |             |         | x      |                 |                 | x           |           |
%> | test_s_static_upright_acc       |            |          |           |        |       | x       |             |         |        |                 |                 | x           |           |
%> | test_s_static_upright_gyro      |            |          |           |        |       | x       |             |         |        |                 |                 | x           |           |
%> | test_isometric                  |            |          | x         |        |       |         | x           |         |        | x               | x               |             |           |
%> | test_Fkin                       |            |          |           |        |       |         |             | x       |        |                 |                 |             | x         |
%> | test_grf 
"""
    
if __name__ == '__main__':
    unittest.main()