"""
Collocation methods for optimal control problems in biosym.

This module provides collocation-based optimal control problem (OCP) solvers
using direct transcription methods. It includes interfaces to IPOPT and other
nonlinear optimization solvers for biomechanical motion optimization.
"""

import os

from biosym.model.model import *

_cachedir = os.path.expanduser("~/.biosym/jax_cache")
_model_cache = os.path.expanduser("~/.biosym/")
os.environ["JAX_COMPILATION_CACHE_DIR"] = (
    _cachedir  # This needs to happen before importing jax
)
os.makedirs((_cachedir), exist_ok=True)


# jax.config.update("jax_enable_x64", True)

import cyipopt
import jax
jax.config.update('jax_enable_x64', False)
import jax.numpy as jnp
import yaml

from biosym.constraints import *
from biosym.ocp import confun, objfun, utils
from biosym.utils import states
from biosym.visualization import stickfigure


class Collocation:
    """A class to handle collocation methods for optimal control problems.

    This class provides direct transcription methods for solving optimal control
    problems using collocation techniques. It interfaces with nonlinear solvers
    like IPOPT for biomechanical motion optimization.

    Attributes
    ----------
    model : BiosymModel
        The biomechanical model to optimize
    settings : dict
        Processed collocation settings and parameters
    """

    def __init__(self, *args, **kwargs):
        """Initialize the Collocation class.

        Parameters
        ----------
        *args : tuple
            Positional arguments, can be a YAML file path
        **kwargs : dict
            Keyword arguments, can include 'model' and 'settings'

        Raises
        ------
        ValueError
            If invalid file format provided (expected YAML)
        """
        if ("model" in kwargs) and ("settings" in kwargs):
            self.model = kwargs["model"]
            self.settings = process_collocation_settings(self.model, kwargs["settings"])
        elif type(args[0]) == str:
            if args[0].endswith(".yaml"):
                self._process_yaml(args[0], **kwargs)
            else:
                raise ValueError("Invalid file format. Expected a YAML file.")
        else:
            self.model = args[0]
            self.settings = process_collocation_settings(self.model, args[1])

        # Initialize objectives and constraints
        self.constraints = confun.Constraints(self.model, self.settings)
        self.objective = objfun.ObjectiveFunction(self.model, self.settings)
        self.make_initial_guess(self.settings.get("initial_guess", None))
        self.setup()
        
        # Auto-enable iteration logging if specified in settings after defining self.objective
        if self.settings.get("settings", {}).get("enable_logging", False):
            log_interval = self.settings.get("settings", {}).get("iteration_log_interval", 100)
            launch_dashboard = self.settings.get("settings", {}).get("launch_dashboard", True)
            dashboard_port = self.settings.get("settings", {}).get("dashboard_port", 8050)
            self.enable_logging(
                log_interval=log_interval,
                launch_dashboard=launch_dashboard,
                dashboard_port=dashboard_port
            )
        
        self._solved = False

    def _process_yaml(self, yaml_data, **kwargs):
        """Process a YAML file to extract model and settings.

        Parameters
        ----------
        yaml_data : str
            Path to the YAML file containing collocation configuration
        **kwargs : dict
            Additional keyword arguments, including 'force_rebuild'
        """
        with open(yaml_data) as file:
            self.settings = yaml.safe_load(file)["collocation"]
            self.model = load_model(
                self.settings.get("settings").get("model"),
                force_rebuild=kwargs.get("force_rebuild", False),
            )
        self.settings = process_collocation_settings(self.model, self.settings)

    def make_initial_guess(self, initial_guess):
        """Set up the initial guess for the collocation problem.

        This method needs to be implemented for the most part. It should always
        return a list with nNodes entries, each entry being a dict with the
        initial guess for the optimization problem.

        Parameters
        ----------
        initial_guess : str, dict, list, or None
            Initial guess for the optimization problem

        Raises
        ------
        ValueError
            If the initial guess is not valid
        """

        info_in_initial_guess = {
            "dur": False,
            "h": False,
            "speed": False,
        }

        if initial_guess is None:
            self.initial_guess_states = states.stack_dataclasses(
                [self.model.default_inputs] * self.settings["nnodes_dur"]
            )

        elif isinstance(initial_guess, dict):
            if initial_guess["type"] == "random":
                pass
            elif initial_guess["type"] == "default":
                self.initial_guess_states = states.stack_dataclasses(
                    [self.model.default_inputs] * self.settings["nnodes_dur"]
                )
            elif initial_guess["type"] == "mid":
                pass
            elif initial_guess["type"] == "from_file":
                print("Loading initial guess from file")
                ig_file = os.path.expanduser(initial_guess["file"])
                with open(ig_file, "rb") as f:
                    (x, globals), info, ig_settings = cloudpickle.load(f)
                if ig_settings["nnodes"] == 1:
                    x = x[0].replace_vector("states", "h", jnp.ones((1,)))
                    self.initial_guess_states = states.stack_dataclasses(
                        [x] * self.settings["nnodes_dur"]
                    )
                else:
                    raise NotImplementedError(
                        "Initial guess from file with resampling is not implemented yet."
                    )
            else:
                raise ValueError(
                    f"Invalid initial guess type: {initial_guess['type']}. Allowed types are 'random', 'default', 'mid', 'from_file'."
                )
        elif isinstance(initial_guess, list):
            self.initial_guess_states = states.stack_dataclasses(initial_guess)
        elif isinstance(initial_guess, str):
            if initial_guess == "random":
                pass
            elif initial_guess == "default":
                self.initial_guess_states = states.stack_dataclasses(
                    [self.model.default_inputs] * self.settings["nnodes_dur"]
                )
            elif initial_guess == "mid":
                pass
        else:
            raise ValueError(
                "Invalid initial guess type. Must be dict, list, str, or None."
            )

        if self.settings["nnodes"] > 1:
            if "dur" in self.settings["bounds"]:
                dur_ = np.array(self.settings["bounds"]["dur"])
                if len(dur_) == 1:
                    dur_ = np.array([dur_, dur_])
                dur_mean = jnp.mean(dur_)
            if "speed" in self.settings["bounds"]:
                speed_ = np.array(self.settings["bounds"]["speed"])
                if len(speed_) == 1:
                    speed_ = np.array([speed_, speed_])
                speed_mean = jnp.mean(speed_)
            if (
                self.settings["discretization"]["args"]["adaptive_h"]
                and len(self.initial_guess_states.states.h) == 0
            ):
                self.initial_guess_states = self.initial_guess_states.replace_vector(
                    "states",
                    "h",
                    jnp.ones((self.settings["nnodes_dur"], 1))
                    * dur_mean
                    / (self.settings["nnodes_dur"] - 1),
                )
            if not self.settings["discretization"]["args"]["adaptive_h"]:
                self.initial_guess_states = self.initial_guess_states.replace_vector(
                    "states", "h", jnp.zeros((self.settings["nnodes_dur"], 0))
                )

            self.initial_guess_globals = states.Globals(
                dur=dur_mean,
                speed=speed_mean,
            )

    def setup(self):
        """
        Set up the nonlinear programming problem for IPOPT optimization.

        This method configures the optimization problem by setting up bounds,
        initial conditions, and solver options. It creates the CyIpopt problem
        interface and configures IPOPT-specific parameters for the collocation
        optimization.

        Notes
        -----
        - Creates the optimization variable vector from initial guess
        - Sets up lower and upper bounds for variables and constraints
        - Configures IPOPT solver options (tolerance, iteration limits, etc.)
        - Prepares the problem for optimization with zero constraint bounds
        - Uses limited-memory Hessian approximation for efficiency
        """
        # Create a cyipopt.problem
        ig_globals = self.initial_guess_globals if self.settings["nnodes"] > 1 else None
        self.x0 = utils.states_dict_to_x(self.initial_guess_states, ig_globals)
        n = len(self.x0)
        m = int(self.constraints.ncon)

        lb = utils.states_dict_to_x(
            self.settings["bounds"]["min"],
            self.settings["bounds"]["global_min"]
            if self.settings["nnodes"] > 1
            else None,
        )
        ub = utils.states_dict_to_x(
            self.settings["bounds"]["max"],
            self.settings["bounds"]["global_max"]
            if self.settings["nnodes"] > 1
            else None,
        )
        self.problem = CyIpoptProblem(
            self.model,
            self.objective,
            self.constraints,
            self.initial_guess_states,
            ub,
            lb,
            globals=ig_globals,
        )
        cl = np.zeros(m)
        cu = np.zeros(m)

        self.nlp = cyipopt.Problem(
            n=n,
            m=m,
            problem_obj=self.problem,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )

        self.nlp.add_option("mu_strategy", "adaptive")
        self.nlp.add_option("tol", float(self.settings["settings"].get("tol", 1e-5)))
        for option in ["acceptable_tol", 
                       "acceptable_constr_viol_tol",
                       "acceptable_dual_inf_tol"]:
            if option in self.settings["settings"]:
                self.nlp.add_option(
                    option,
                    float(self.settings["settings"].get(option, 1e-5)),
                )
        self.nlp.add_option(
            "constr_viol_tol",
            float(self.settings["settings"].get("constr_viol_tol", 1)),
        )
        self.nlp.add_option(
            "dual_inf_tol",
            float(self.settings["settings"].get("dual_inf_tol", 1)),
        )
        self.nlp.add_option("print_level", 5)
        self.nlp.add_option("max_iter", self.settings["settings"].get("max_iter", 1000))
        self.nlp.add_option("hessian_approximation", "limited-memory")
        self.nlp.add_option("print_timing_statistics", "yes")
        # Initialize iteration logger (will be None unless enabled)
        self.iteration_logger = None
    
    def enable_logging(self, log_interval: int = 100, launch_dashboard: bool = True, dashboard_port: int = 8050):
        """
        Enable logging of objective function values during optimization.
        
        This sets up an IterationLogger that will capture individual objective
        values every `log_interval` iterations. Optionally launches a real-time
        Dash visualization dashboard.
        
        Parameters
        ----------
        log_interval : int, optional
            How frequently to log (e.g., 100 = every 100th iteration), by default 100
        launch_dashboard : bool, optional
            Whether to automatically launch the Dash visualization app, by default True
        dashboard_port : int, optional
            Port for the Dash server, by default 8050
            
        Examples
        --------
        >>> problem.enable_logging(log_interval=100)
        >>> x, info = problem.solve()
        >>> df = problem.iteration_logger.get_dataframe()
        >>> print(df[['iteration', 'total_objective', 'track_markers']])
        """
        from biosym.ocp.iteration_logger import IterationLogger
        
        # Create the iteration logger
        self.iteration_logger = IterationLogger(
            objective_manager=self.objective,
            problem=self.problem,
            initial_guess_states=self.initial_guess_states,
            iteration_interval=log_interval
        )
        
        # Set it as the callback for the CyIpoptProblem
        self.problem._iteration_callback = self.iteration_logger
        
        # Launch dashboard if requested
        if launch_dashboard:
            import socket
            import subprocess
            import threading
            import time
            from biosym.ocp.dash_logger import create_dashboard_app

            # Helper to check if port is in use
            def is_port_in_use(port: int) -> bool:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    return s.connect_ex(('localhost', port)) == 0

            # Check port before launching, and kill if occupied
            if is_port_in_use(dashboard_port):
                print(f"\n Warning: Dashboard port {dashboard_port} is already in use.")
                print(f"   Attempting to kill the existing process to free the port...")
                # This command is for Linux/macOS. It finds the PID using the port and kills it.
                kill_command = f"kill -9 $(lsof -t -i:{dashboard_port})"
                subprocess.run(
                    kill_command, shell=True, check=True, capture_output=True
                )
                print(f"   Successfully killed process on port {dashboard_port}.")
                # Give the OS a moment to release the port before the new server binds to it
                time.sleep(0.5)
            
            # Create the Dash app
            self._dash_app = create_dashboard_app(self.iteration_logger, port=dashboard_port)
            
            # Run it in a separate thread so it doesn't block optimization
            def run_dash():
                print(f"\n Starting real-time visualization dashboard at http://localhost:{dashboard_port}")
                print("   Open this URL in your browser to see objective convergence")
                self._dash_app.run(debug=False, port=dashboard_port, use_reloader=False)
            
            self._dash_thread = threading.Thread(target=run_dash, daemon=True)
            self._dash_thread.start()
            
            # Give the server a moment to start
            time.sleep(1)

    def solve(self, visualize=False, **kwargs):
        """
        Solve the optimal control problem using IPOPT.

        This method executes the nonlinear optimization using the IPOPT solver
        to find the optimal trajectory that minimizes the objective function
        while satisfying all constraints.

        Parameters
        ----------
        visualize : bool, optional
            Whether to create a stick figure visualization of the solution.
            Default is False.
        **kwargs : dict
            Additional keyword arguments passed to the visualization function.

        Returns
        -------
        tuple
            Tuple containing:
            - x: Optimized states dictionary with solution trajectory
            - info: IPOPT solver information and statistics

        Notes
        -----
        - Prevents re-solving unless explicitly confirmed by user
        - Automatically saves results if output file is specified in settings
        - Solution is converted back to structured state format for analysis
        - Can generate stick figure animations of the optimized motion
        """
        if self._solved:
            if input(
                "Collocation problem has already been solved. Press y to repeat, x to leave"
            ) in ["y", "Y"]:
                return None
        x, info = self.nlp.solve(self.x0)

        self.x = utils.x_to_states_dict(
            x,
            self.initial_guess_states,
            self.initial_guess_globals if self.settings["nnodes"] > 1 else None,
        )
        if visualize:
            stickfigure.plot_stick_figure(self.model, self.x, **kwargs)
        if "output" in self.settings["settings"]:
            output = (self.x, info, self.settings)
            out_file = os.path.expanduser(self.settings["settings"]["output"]["file"])
            print(f"Saving results to {out_file}")
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            with open(out_file, "wb") as f:
                cloudpickle.dump(output, f)
        # Todo: Cleanup the result a bit nicer
        return self.x, info


class CyIpoptProblem:
    """
    IPOPT problem interface for optimal control problems.

    This class provides the required interface for the CyIpopt Python wrapper,
    implementing all necessary methods for objective function evaluation,
    constraint evaluation, and their respective gradients/Jacobians.

    The class serves as a bridge between the biosym optimal control formulation
    and the IPOPT nonlinear programming solver, handling data conversion and
    function evaluations efficiently.

    Attributes
    ----------
    model : BiosymModel
        The biomechanical model being optimized.
    objective : ObjectiveFunction
        Objective function manager for the optimization problem.
    constraints : Constraints
        Constraint manager for the optimization problem.
    template : StatesDict
        Template for state variable structure and dimensions.
    lower_bound : jnp.ndarray
        Lower bounds for optimization variables.
    upper_bound : jnp.ndarray
        Upper bounds for optimization variables.
    globals : dict, optional
        Global variables and parameters for the problem.

    Notes
    -----
    Required methods for IPOPT interface:
    - objective: Evaluate objective function
    - gradient: Compute objective gradient
    - constraints: Evaluate constraint functions
    - jacobian: Compute constraint Jacobian
    - jacobianstructure: Return Jacobian sparsity pattern
    """

    def __init__(
        self,
        model,
        objective,
        constraints,
        template,
        upper_bound,
        lower_bound,
        globals=None,
    ):
        """
        Initialize the IPOPT problem interface.

        Parameters
        ----------
        model : BiosymModel
            The biomechanical model being optimized.
        objective : ObjectiveFunction
            Objective function manager for cost evaluation.
        constraints : Constraints
            Constraint manager for constraint evaluation.
        template : StatesDict
            Template for reconstructing state structures from flat vectors.
        upper_bound : jnp.ndarray
            Upper bounds for optimization variables.
        lower_bound : jnp.ndarray
            Lower bounds for optimization variables.
        globals : dict, optional
            Global variables and parameters for the problem.
        """
        self.model = model
        self.objs = objective
        self.cons = constraints
        self.template = template  # For reconstructing something better looking
        self.ub, self.lb = upper_bound, lower_bound
        self.globals = globals
        self._init_jac = False
        self.jacobianstructure()
        # Store current x for iteration callback access
        self._current_x = None
        # Iteration callback (will be set by enable_logging)
        self._iteration_callback = None
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                    d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """
        IPOPT intermediate callback - called once per iteration.
        
        This method is called by IPOPT if present. It delegates to the
        iteration_callback if one has been set via enable_logging().
        """
        if self._iteration_callback is not None:
            return self._iteration_callback(
                alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                d_norm, regularization_size, alpha_du, alpha_pr, ls_trials
            )
        return True  # Continue optimization

    def objective(self, x):
        """
        Evaluate the objective function for IPOPT.

        Parameters
        ----------
        x : jnp.ndarray
            Flat optimization vector containing states and global parameters.

        Returns
        -------
        float
            Scalar objective function value to minimize.
        """
        # Store current x for callback access
        self._current_x = x
        x, globals = utils.x_to_states_dict(x, self.template, self.globals)
        return self.objs.objfun(x, globals)

    def gradient(self, x):
        """
        Compute the gradient of the objective function for IPOPT.

        Parameters
        ----------
        x : jnp.ndarray
            Flat optimization vector.

        Returns
        -------
        jnp.ndarray
            Gradient vector with respect to optimization variables.
        """
        x, globals = utils.x_to_states_dict(x, self.template, self.globals)
        return utils.states_dict_to_x(*self.objs.gradfun(x, globals))

    def constraints(self, x):
        """
        Evaluate all constraint functions for IPOPT.

        Parameters
        ----------
        x : jnp.ndarray
            Flat optimization vector.

        Returns
        -------
        jnp.ndarray
            Constraint violation vector (should be zero at optimum).
        """
        x, globals = utils.x_to_states_dict(x, self.template, self.globals)
        return self.cons.confun(x, globals)

    def jacobian(self, x):
        """
        Compute the constraint Jacobian for IPOPT.

        Parameters
        ----------
        x : jnp.ndarray
            Flat optimization vector.

        Returns
        -------
        jnp.ndarray
            Sparse Jacobian matrix values at current point.
        """
        x, globals = utils.x_to_states_dict(x, self.template, self.globals)
        _, _, jac = self.cons.jacobian(x, globals)
        return jac[self.jac_indices]

    def jacobianstructure(self):
        """
        Determine the sparsity structure of the constraint Jacobian.

        This method analyzes the Jacobian sparsity pattern by evaluating
        the Jacobian at multiple random points to identify consistently
        non-zero entries. This structure is used by IPOPT for efficient
        sparse matrix computations.

        Returns
        -------
        tuple
            Tuple of (row_indices, col_indices) indicating non-zero locations
            in the sparse Jacobian matrix.

        Notes
        -----
        - Uses multiple random evaluations to capture all possible non-zeros
        - Caches the structure to avoid repeated computation
        - Reports sparsity statistics for optimization insights
        """
        if self._init_jac:
            return self.jacstruct

        def jac_0(x):
            x, globals = utils.x_to_states_dict(x, self.template, self.globals)
            return self.cons.jacobian(x, globals)

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
        print(f"{100 * nnz / len(j0):.2f}% of the (sparse) jacobian is nonzero")
        print(
            f"{100 * nnz / (len(x_random) * self.cons.ncon):.2f}% of the full jacobian is nonzero"
        )
        self.jac_indices = np.nonzero(j0)
        self._init_jac = True
        self.jacstruct = rows[self.jac_indices[0]], cols[self.jac_indices[0]]
        return self.jacstruct


def process_collocation_settings(model, settings):
    """
    Process and validate collocation settings for optimal control problems.

    This function takes raw settings from configuration files or dictionaries
    and processes them into a standardized format suitable for collocation-based
    optimal control. It validates required keys, sets up discretization parameters,
    and handles data type conversions.

    Parameters
    ----------
    model : BiosymModel
        The biomechanical model for which settings are being processed.
    settings : dict
        Dictionary containing collocation method configuration including:
        - settings: solver parameters, nnodes, discretization
        - constraints: constraint specifications
        - objectives: objective function specifications
        - bounds: variable bounds for optimization

    Returns
    -------
    dict
        Processed and validated settings dictionary with additional computed
        fields like dtype conversions, node handling, and discretization setup.

    Raises
    ------
    ValueError
        If required settings keys are missing or invalid values are provided.

    Notes
    -----
    - Validates required configuration structure for collocation methods
    - Handles both single-node and multi-node (discretized) problems
    - Converts data types appropriately for JAX compatibility
    - Sets up adaptive time stepping if specified
    """
    # Assert that required settings are present
    required_keys = {
        "settings": ["model", "nnodes"],
        "constraints": [],
        "objectives": [],
        "bounds": [],
    }
    for key, value in required_keys.items():
        if key not in settings:
            raise ValueError(f"Missing required key: {key}")
        for sub_key in value:
            if sub_key not in settings[key]:
                raise ValueError(f"Missing required sub-key: {sub_key} in {key}")

    # Check nnodes is a positive integer
    print(
        "Collocation warning in process collocation settings: Bounds are not correctly handled"
    )
    if settings["settings"]["nnodes"] > 1:
        if settings["settings"]["discretization"]["args"]["adaptive_h"]:
            model.default_inputs = model.default_inputs.replace_vector(
                "states", "h", jnp.ones((1,))
            )

    settings["nnodes"] = settings["settings"]["nnodes"]


    # Nnodes handling
    if (
        not isinstance(settings["settings"]["nnodes"], int)
        or settings["settings"]["nnodes"] <= 0
    ):
        raise ValueError("nnodes must be a positive integer.")
    if settings["settings"]["nnodes"] > 1:  # Discretization is needed
        if "discretization" not in settings["settings"]:
            raise ValueError("Discretization settings are required when nnodes > 1.")

        settings["discretization"] = settings["settings"]["discretization"]
        required_keys = ["type"]
        for key in required_keys:
            if key not in settings["settings"]["discretization"]:
                raise ValueError(f"Missing required discretization setting: {key}")
        # Allowed types:
        if settings["discretization"]["type"] not in ["euler", "rk4"]:
            raise ValueError(
                f"Invalid discretization type: {settings['discretization']['type']}. Allowed types are 'euler', 'rk4'."
            )
        if settings["discretization"]["type"] not in ["euler"]:
            raise NotImplementedError(
                f"Discretization type {settings['discretization']['type']} is not implemented yet."
            )

        # Add the discretization to the constraints:
        settings["constraints"].append(
            {
                "name": "discretization",
                "weight": settings["settings"]["discretization"]["args"].get(
                    "weight", 1
                ),
                "args": settings["settings"]["discretization"].get("args", {}),
            }
        )
        settings["constraints"].append(
            {
                "name": "speed",
                "weight": settings["settings"]["discretization"]["args"].get(
                    "weight", 1
                ),
                "args": settings["settings"]["discretization"].get("args", {}),
            }
        )

        if settings["discretization"]["args"]["adaptive_h"]:
            settings["constraints"].append(
                {
                    "name": "adaptive_h",
                    "weight": settings["discretization"].get("weight", 1),
                    "args": None,
                }
            )
            # The initial guess needs to include h in this case, otherwise we leave it
            Warning(
                "Initial guess handling for adaptive step size is not implemented yet."
            )

        if any(
            constraint["name"] == "periodicity"
            for constraint in settings["constraints"]
        ):
            settings["nnodes_dur"] = settings["nnodes"] + 1
        else:
            settings["nnodes_dur"] = settings["nnodes"]

        # set globals and bounds for dur and speed, and verify inputs being set correctly
        if "speed" not in settings["bounds"]:
            raise ValueError(
                "Speed bounds are required for collocation with multiple nodes."
            )
        if "dur" not in settings["bounds"]:
            raise ValueError(
                "Duration bounds are required for collocation with multiple nodes."
            )
        if type(settings["bounds"]["speed"]) == float:
            settings["bounds"]["speed"] = [
                settings["bounds"]["speed"],
                settings["bounds"]["speed"],
            ]
        if type(settings["bounds"]["dur"]) == float:
            settings["bounds"]["dur"] = [
                settings["bounds"]["dur"],
                settings["bounds"]["dur"],
            ]
        if len(settings["bounds"]["speed"]) != 2 or len(settings["bounds"]["dur"]) != 2:
            raise ValueError(
                "Speed and duration bounds must be a list of two values [min, max]."
            )
        settings["bounds"]["global_min"] = states.Globals(
            dur=settings["bounds"]["dur"][0], speed=settings["bounds"]["speed"][0]
        )
        settings["bounds"]["global_max"] = states.Globals(
            dur=settings["bounds"]["dur"][1], speed=settings["bounds"]["speed"][1]
        )
    else:
        settings["nnodes_dur"] = settings["nnodes"]

    states_variables = model.variables[model.variables.type == "state"]
    min_generic = model.default_inputs.replace_vector(
        "states", "model", jnp.array(states_variables.xmin)
    )
    max_generic = model.default_inputs.replace_vector(
        "states", "model", jnp.array(states_variables.xmax)
    )

    # Placeholders for gc and actuator model
    min_generic = min_generic.replace_vector(
        "states", "actuator_model", min_generic.states.actuator_model - 1e3
    )
    max_generic = max_generic.replace_vector(
        "states", "actuator_model", max_generic.states.actuator_model + 1e3
    )
    min_generic = min_generic.replace_vector(
        "states", "gc_model", min_generic.states.gc_model - 1e3
    )
    max_generic = max_generic.replace_vector(
        "states", "gc_model", max_generic.states.gc_model + 1e3
    )

    settings["bounds"]["min"] = states.stack_dataclasses(
        [min_generic] * settings["nnodes_dur"]
    )
    settings["bounds"]["max"] = states.stack_dataclasses(
        [max_generic] * settings["nnodes_dur"]
    )

    if settings["settings"]["nnodes"] == 1:  # Bounds on ddot and dot values set to zero
        # return settings
        for section in ["min", "max"]:
            settings["bounds"][section] = settings["bounds"][section].replace_vector(
                "states",
                "model",
                settings["bounds"][section]
                .states.model.at[
                    0, model.speeds["idx"] : model.speeds["idx"] + model.speeds["n"]
                ]
                .set(jnp.zeros(model.speeds["n"], dtype=float)+(1e-8 if section=="max" else -1e-8)),
            )
            settings["bounds"][section] = settings["bounds"][section].replace_vector(
                "states",
                "model",
                settings["bounds"][section]
                .states.model.at[
                    0, model.accs["idx"] : model.accs["idx"] + model.accs["n"]
                ]
                .set(jnp.zeros(model.accs["n"], dtype=float)),
            )
    elif not settings["discretization"]["args"]["adaptive_h"]:
        settings["bounds"]["min"] = settings["bounds"]["min"].replace_vector(
            "states", "h", jnp.ones((settings["nnodes_dur"], 0))
        )
        settings["bounds"]["max"] = settings["bounds"]["max"].replace_vector(
            "states", "h", jnp.ones((settings["nnodes_dur"], 0))
        )
    else:
        settings["bounds"]["min"] = settings["bounds"]["min"].replace_vector(
            "states", "h", jnp.zeros(settings["nnodes_dur"])
        )

    if settings["bounds"]["start_at_origin"]:
        settings["bounds"]["min"] = settings["bounds"]["min"].replace_vector(
            "states", "model", settings["bounds"]["min"].states.model.at[0, 0].set(0)
        )
        settings["bounds"]["max"] = settings["bounds"]["max"].replace_vector(
            "states", "model", settings["bounds"]["max"].states.model.at[0, 0].set(0)
        )

    if model.actuator_model.get_n_states() > 0:
        for bound in ["min", "max"]:
            settings["bounds"][bound] = settings["bounds"][bound].replace_vector(
                "states",
                "actuator_model",
                jnp.tile(
                    model.actuator_model.bounds["states"][bound][jnp.newaxis, :],
                    (settings["nnodes_dur"], 1)
                ),
            )
    return settings
