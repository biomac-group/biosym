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
from biosym.ocp import confun, objfun
from biosym.ocp import utils
from biosym.ocp.utils import process_collocation_settings, CyIpoptProblem
from biosym.ocp.logging import IterationLogger, create_dashboard_app
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
            if type(args[1]) == str:
                if args[1].endswith(".yaml"):
                    self._process_yaml(args[1], **kwargs, skip_model_loading=True)
                else:
                    raise ValueError("Invalid file format. Expected a YAML file.")
            else:
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
            if not kwargs.get("skip_model_loading", False):
                self.model = load_model(
                    self.settings.get("settings").get("model"),
                    force_rebuild=kwargs.get("force_rebuild", False),
                )
        self.settings = process_collocation_settings(self.model, self.settings)

    def make_initial_guess(self, initial_guess):
        """Set up the initial guess for the collocation problem.

        This method delegates to the initial_guess utility module.

        Parameters
        ----------
        initial_guess : str, dict, list, or None
            Initial guess for the optimization problem
        """
        self.initial_guess_states, self.initial_guess_globals = utils.initial_guess.make_initial_guess(
            self.model, self.settings, initial_guess
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


