import biosym.ocp.collocation as collocation

class TestObjectiveFunction: 
    """
    Test objective function for unit testing.
    We only use it to test the integration of user-defined objective functions in the collocation framework.
    """
    def __init__(self, model, settings):
        """
        Initialize the TestObjectiveFunction class with a model and settings.
        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the objective function.
        """
        self.model = model
        self.settings = settings

    def _get_info(self):
        return {
            'name': self.__class__.__name__,
            'description': 'Test objective function for unit testing.',
            'required_variables': None
        }
    
    def get_objfun(self):
        """ :return: The objective function. """
        return lambda x,y: 0
    def get_gradient(self):
        """ :return: The gradient of the objective function. """
        return lambda x,y: 0


standing_problem = collocation.Collocation("tests/collocation/standing2d.yaml")

# Testing the constraints and objective function
import time 
import timeit

for function, name in zip([standing_problem.constraints.evaluate_constraints, 
                standing_problem.constraints.evaluate_jacobian, 
                standing_problem.objective.evaluate_objectives, 
                standing_problem.objective.evaluate_gradients],
                ['constraints', 'jacobian', 'objectives', 'gradients']):
    print(f"Testing {name} function...")

    start_time = time.time()
    function(standing_problem.initial_guess_states, None)
    print(f"{function.__name__} evaluated in {time.time() - start_time} seconds")

    # Test 1k compiled evaluations
    a = timeit.timeit(lambda: function(standing_problem.initial_guess_states, None), number=1000)
    print(f"1k evaluations of {name} took {a/1000} seconds")


# Show this at the end
standing_problem.objective.add_objective(TestObjectiveFunction, weight=1.0)
