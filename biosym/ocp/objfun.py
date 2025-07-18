from biosym.objectives import *
from biosym.ocp import utils

class ObjectiveFunction:
    """
    Base class for objective functions in the biosym optimization framework.
    """
    def __init__(self, model, settings):
        """
        Initialize the ObjectiveFunction class with a model and settings.
        
        :param model: biosym model object representing the system to be controlled.
        :param settings: Dictionary containing settings for the objective function.
        """
        self.model = model
        self.settings = settings
        self.objective_functions = []
        self.objective_gradients = []
        self.required_variables = {'states': [], 'constants': [], 'calculated': []}
        self.weights = settings.get('weights', [])
        for objective in settings.get('objectives', []):
            self.add_objective(objective.get('name'), objective.get('weight'), objective.get('args', None))

    def add_objective(self, name, weight, kwargs=None):
        """
        Add an objective function to the optimization problem.
        
        :param name: Name of the objective function class or instance.
        :param weight: Weight for the objective function.
        :param args: Additional arguments for the objective function.
        """
        if isinstance(name, str):
            objective_class = globals().get(name)
            if objective_class:
                objective = objective_class.Objective(self.model, self.settings, **kwargs if kwargs else {})
            else:
                raise ValueError(f"Objective class '{name}' not found.")
        elif hasattr(name, '__init__'):
            objective = name(self.model, self.settings, **kwargs if kwargs else {})
        else:
            raise ValueError("Invalid objective object provided.")
        
        info = objective._get_info()
        print(f"Adding objective: {info.get('name')} with weight {weight}")

        self.objective_functions.append(objective.get_objfun())
        self.objective_gradients.append(objective.get_gradient())
        self.weights.append(weight)

        # Update required variables
        if info['required_variables']:
            for var_type, vars in info['required_variables'].items():
                if var_type not in self.required_variables:
                    self.required_variables[var_type] = []
                self.required_variables[var_type].extend(vars)

    def evaluate_objectives(self, states_list, globals_dict=None):
        """
        Evaluate the objective functions.
        
        :param states_list: Dictionary containing the current states.
        :param globals_dict: Dictionary containing global variables (optional).
        :return: The evaluated values of the objective functions.
        """
        results = 0
        for i, obj_fun in enumerate(self.objective_functions):
            result = obj_fun(states_list, globals_dict)
            results += result * self.weights[i]  # Apply the corresponding weight
        return results
    
    def evaluate_gradients(self, states_list, globals_dict=None):
        """
        Evaluate the gradients of the objective functions.
        
        :param states_list: Dictionary containing the current states.
        :param globals_dict: Dictionary containing global variables (optional).
        :return: The evaluated gradients of the objective functions.
        """
        gradients = []
        for i, grad_fun in enumerate(self.objective_gradients):
            gradient = grad_fun(states_list, globals_dict)
            gradients.append(gradient)
        # Add all gradients together
        utils.sum_states_dicts(gradients, self.weights)
        return gradients

