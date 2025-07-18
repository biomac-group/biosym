from biosym.model.model import * 
from biosym.ocp import confun, objfun, utils
from biosym.constraints import *
import yaml


class Collocation:
    """
    A class to handle collocation methods for optimal control problems.
    This class is a placeholder and should be implemented with specific methods
    for collocation techniques.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Collocation class with a model and settings.
        :param args: Positional arguments, can be a model and settings or a YAML file.
        :param kwargs: Keyword arguments, can include 'model' and 'settings'.
        """
        if ("model" in kwargs) and ("settings" in kwargs):
            self.model = kwargs['model']
            self.settings = process_collocation_settings(self.model, kwargs['settings'])
        elif type(args[0]) == str:
            if args[0].endswith('.yaml'):
                self._process_yaml(args[0], **kwargs)
            else:
                raise ValueError("Invalid file format. Expected a YAML file.")
        else:
            self.model = args[0]
            self.settings = process_collocation_settings(self.model, args[1])

        
        # Initialize objectives and constraints
        self.constraints = confun.Constraints(self.model, self.settings)
        self.objective = objfun.ObjectiveFunction(self.model, self.settings)
        self.make_initial_guess(kwargs.get('initial_guess', None))
        self.build_index()

    def _process_yaml(self, yaml_data, **kwargs):
        """
        Process a YAML file to extract model and settings.
        :param yaml_data: Path to the YAML file.
        """
        with open(yaml_data, 'r') as file:
            self.settings = yaml.safe_load(file)['collocation']
            if "force_rebuild" in kwargs:
                if kwargs['force_rebuild']:
                    self.model = load_model(self.settings.get('settings').get('model'), force_rebuild=True)
            else:
                self.model = load_model(self.settings.get('settings').get('model'))
        self.settings = process_collocation_settings(self.model, self.settings)

    def make_initial_guess(self, initial_guess):
        """
        Set up the collocation problem.
        This method should be overridden in subclasses.

        This method needs to be implemented for the most part. It should always return a list with nNodes entries, each entry being a dict with the initial guess for the optimization problem.

        :param initial_guess: Initial guess for the optimization problem.
        :type initial_guess: str, dict, list, or None
        :raises ValueError: If the initial guess is not valid.

        """
        info_in_initial_guess = {
            'dur': False,
            'h': False,
            'speed': False,
        }

        if initial_guess is None:
            self.initial_guess_states = utils.states_list_to_dict([self.model.default_inputs] * self.settings['nnodes'])

        elif isinstance(initial_guess, dict):
            self.initial_guess_states = initial_guess

        elif isinstance(initial_guess, list):
            self.initial_guess_states = utils.states_list_to_dict(initial_guess)
        elif isinstance(initial_guess, str):
            if initial_guess == 'random':
                pass
            elif initial_guess == 'default':
                self.initial_guess_states = utils.states_list_to_dict([self.model.default_inputs] * self.settings['nnodes'])
            elif initial_guess == 'mid':
                pass
        else:
            raise ValueError("Invalid initial guess type. Must be dict, list, str, or None.")
        
        if self.settings['nnodes'] > 1000:
            # We need length, dur and speed information for the initial guess, if these variables are required by the collocation method.
            # When bounds exist, we go middle of the bounds.
            if not info_in_initial_guess['dur']:
                if 'dur' in self.settings['bounds']:
                    self.initial_guess_states['globals']['dur'] = (self.settings['bounds']['values'][0]['globals']['dur'] + self.settings['bounds']['values'][-1]['globals']['dur']) / 2
                else:
                    self.initial_guess_states['globals']['dur'] = self.model.default_inputs['globals']['dur']
                raise ValueError("Initial guess must contain 'dur' information for collocation with multiple nodes.")
            if not info_in_initial_guess['h']:
                raise ValueError("Initial guess must contain 'h' information for collocation with multiple nodes.")
            if not info_in_initial_guess['speed']:
                if 'speed' in self.settings['bounds']:
                    raise ValueError("Initial guess must contain 'speed' information for collocation with multiple nodes.")
    
        # Convert states to dataclass
        self.initial_guess_states = utils.dict_to_dataclass(self.initial_guess_states)

    def build_index(self):
        """
        Build the index for the collocation problem.
        This method should be overridden in subclasses.
        """
        self.index = {
            'states': {},
            'globals': {},
        }
        pass
        

    def solve(self):
        """
        Solve the collocation problem.
        This method should be overridden in subclasses.
        """
        raise NotImplementedError("This method should be overridden in subclasses.")
    
    


def process_collocation_settings(model, settings):
    """
    Process and validate collocation settings.
    :param settings: Dictionary containing settings for the collocation method.
    :return: Processed settings dictionary.
    """
    # Assert that required settings are present
    required_keys = {"settings": ['model', 'nnodes'], "constraints": [], "objectives": [], "initial_guess": [], "bounds": []}
    for key, value in required_keys.items():
        if key not in settings:
            raise ValueError(f"Missing required key: {key}")
        for sub_key in value:
            if sub_key not in settings[key]:
                raise ValueError(f"Missing required sub-key: {sub_key} in {key}")
            
    # Check nnodes is a positive integer
    settings['bounds']['values'] = [model.default_inputs] * settings['settings']['nnodes']

    settings['nnodes'] = settings['settings']['nnodes']
    if not isinstance(settings['settings']['nnodes'], int) or settings['settings']['nnodes'] <= 0:
        raise ValueError("nnodes must be a positive integer.")
    if settings['settings']['nnodes'] > 1: # Discretization is needed
        if 'discretization' not in settings['settings']:
            raise ValueError("Discretization settings are required when nnodes > 1.")
        if 'discretization' in settings['settings']:
            required_keys = ['type', 'mode']
            for key in required_keys:
                if key not in settings['settings']['discretization']:
                    raise ValueError(f"Missing required discretization setting: {key}")
        if "periodicity" in settings['constraints']:
            settings['nnodes_dur'] = settings['nnodes'] + 1
        else:
            settings['nnodes_dur'] = settings['nnodes']
    else: # Bounds on ddot and dot values set to zero
        settings['bounds']['values'][0]['states']['model'][model.speeds['idx']:model.speeds['idx'] + model.speeds['n']] = 0.0
        settings['bounds']['values'][0]['states']['model'][model.accs['idx']:model.accs['idx'] + model.accs['n']] = 0.0
    return settings



    