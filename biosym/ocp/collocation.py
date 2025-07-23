from biosym.model.model import * 
from biosym.ocp import confun, objfun, utils
from biosym.constraints import *
from biosym.utils import states
from biosym.visualization import stickfigure
import yaml
import cyipopt
import jax.numpy as jnp
import pickle

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
        self.setup()
        self._solved = False

    def _process_yaml(self, yaml_data, **kwargs):
        """
        Process a YAML file to extract model and settings.
        :param yaml_data: Path to the YAML file.
        """
        with open(yaml_data, 'r') as file:
            self.settings = yaml.safe_load(file)['collocation']
            self.model = load_model(self.settings.get('settings').get('model'), force_rebuild=kwargs.get('force_rebuild', False))
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
            self.initial_guess_states = states.stack_dataclasses([self.model.default_inputs] * self.settings['nnodes'])

        elif isinstance(initial_guess, dict):
            self.initial_guess_states = initial_guess

        elif isinstance(initial_guess, list):
            self.initial_guess_states = states.stack_dataclasses(initial_guess)
        elif isinstance(initial_guess, str):
            if initial_guess == 'random':
                pass
            elif initial_guess == 'default':
                self.initial_guess_states = states.stack_dataclasses([self.model.default_inputs] * self.settings['nnodes'])
            elif initial_guess == 'mid':
                pass
        else:
            raise ValueError("Invalid initial guess type. Must be dict, list, str, or None.")
        
        if self.settings['nnodes'] > 1:
            raise NotImplementedError("Initial guess for collocation with multiple nodes is not implemented yet.")
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

    def setup(self):
        """
        Solve the collocation problem.
        This method should be overridden in subclasses.
        """
        # Create a cyipopt.problem
        self.x0 = utils.states_dict_to_x(self.initial_guess_states)
        n = len(self.x0)
        m = self.constraints.ncon
        lb = utils.states_dict_to_x(self.settings['bounds']['min'])
        ub = utils.states_dict_to_x(self.settings['bounds']['max'])
        self.problem = CyIpoptProblem(self.model, self.objective, self.constraints, self.initial_guess_states, lb, ub)
        cl = np.zeros(m)
        cu = np.zeros(m)

        self.nlp = cyipopt.problem(
            n=n,
            m=m,
            problem_obj=self.problem,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )
        self.nlp.add_option('mu_strategy', 'adaptive')
        self.nlp.add_option('tol', float(self.settings["settings"].get('tol', 1e-5)))
        self.nlp.add_option('print_level', 5)
        self.nlp.add_option('max_iter', self.settings["settings"].get('max_iter', 1000))
        self.nlp.add_option('hessian_approximation', 'limited-memory')
        self.nlp.add_option('print_timing_statistics', 'yes')

    def solve(self, visualize=False, **kwargs):
        if self._solved:
            if input("Collocation problem has already been solved. Press y to repeat, x to leave") in ['y','Y']:
                return
        x, info = self.nlp.solve(self.x0)

        self.x = utils.x_to_states_dict(x, self.initial_guess_states)
        if visualize:
            stickfigure.plot_stick_figure(self.model, self.x, **kwargs)
        if "output" in self.settings["settings"]:
            output = (self.x, info, self.settings)
            with open(self.settings["settings"]['output']['file'], 'wb') as f:
                pickle.dump(output, f)
        # Todo: Cleanup the result a bit nicer
        return self.x, info
       

class CyIpoptProblem:
    """
        The actual class object that handles ipopt communications.
        Following functions must be implemented:
        objectve
        gradient
        constraints
        jacobian
        jacobianstructure
    """

    def __init__(self, model, objective, constraints, template, upper_bound, lower_bound):
        """ 
            Helper class as a cyipopt adapter
        """
        self.model = model
        self.objs = objective
        self.cons = constraints
        self.template = template # For reconstructing something better looking
        self.ub, self.lb = upper_bound, lower_bound
        self._init_jac = False
        self.jacobianstructure()

    def objective(self, x):
        x, globals = utils.x_to_states_dict(x, self.template)
        return self.objs.objfun(x, globals)
    
    def gradient(self, x):
        x, globals = utils.x_to_states_dict(x, self.template)
        return utils.states_dict_to_x(self.objs.gradfun(x, globals))

    def constraints(self, x):
        x, globals = utils.x_to_states_dict(x, self.template)
        return self.cons.confun(x, globals)

    def jacobian(self, x):
        x, globals = utils.x_to_states_dict(x, self.template)
        _, _, jac = self.cons.jacobian(x, globals)
        return jac[self.jac_indices]

    def jacobianstructure(self):
        def jac_0(x):
            x, globals = utils.x_to_states_dict(x, self.template)
            return self.cons.jacobian(x, globals)

        if self._init_jac:
            return self.jacstruct
        else: 
            rows, cols, j0 = jac_0(self.lb)
            curr_nonzeros = np.nonzero(j0)
            nnz = len(curr_nonzeros[0])
            no_new_nonzero_found = 0

            while no_new_nonzero_found < 20:
                # Create a random vector between lb and ub
                x_random = np.random.uniform(self.lb, self.ub)
                _, _, j0_ = jac_0(x_random)

                j0 += j0_
                curr_nonzeros = np.nonzero(j0)
                if nnz < len(curr_nonzeros):
                    nnz = len(curr_nonzeros[0])
                else:
                    no_new_nonzero_found += 1

            print(f"Found {nnz} nonzeros in jacobian structure")
            print(f"{100*nnz/len(j0):.2f}% of the (sparse) jacobian is nonzero")
            print(f"{100*nnz/(len(x_random)*self.cons.ncon):.2f}% of the full jacobian is nonzero")
            self.jac_indices = np.nonzero(j0)
            self._init_jac = True
            self.jacstruct = rows[self.jac_indices[0]], cols[self.jac_indices[0]]
            return self.jacstruct


    

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
    print('Collocation warning in process collocation settings: Bounds are not correctly handled')
    settings['bounds']['max'] = states.stack_dataclasses([model.default_inputs] * settings['settings']['nnodes']).add(1e3)
    settings['bounds']['min'] = states.stack_dataclasses([model.default_inputs] * settings['settings']['nnodes']).add(-1e3)

    settings['nnodes'] = settings['settings']['nnodes']
    if not isinstance(settings['settings']['nnodes'], int) or settings['settings']['nnodes'] <= 0:
        raise ValueError("nnodes must be a positive integer.")
    elif settings['settings']['nnodes'] > 1: # Discretization is needed
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
        #return settings

        for section in ['min', 'max']:
            settings['bounds'][section] = settings['bounds'][section].replace_vector("states", "model", settings['bounds'][section].states.model.at[0,model.speeds['idx']:model.speeds['idx'] + model.speeds['n']].set(jnp.zeros(model.speeds['n'])))
            settings['bounds'][section] = settings['bounds'][section].replace_vector("states", "model", settings['bounds'][section].states.model.at[0,model.accs['idx']:model.accs['idx'] + model.accs['n']].set(jnp.zeros(model.accs['n'])))
    return settings



    