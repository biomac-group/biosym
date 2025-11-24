"""
Iteration callback for logging objective function values during IPOPT optimization.

This module provides a callback function that IPOPT calls at each iteration,
allowing us to track the evolution of individual objective terms during solving.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from biosym.ocp import utils


class IterationLogger:
    """
    Logger that captures objective function values at specified iteration intervals.
    
    This callback is invoked by IPOPT during optimization and stores objective
    values every N iterations for later analysis and visualization.
    
    Attributes
    ----------
    objective_manager : ObjectiveFunction
        The objective function manager containing all objective terms
    problem : CyIpoptProblem
        The IPOPT problem interface that has x_to_states method
    initial_guess_states : StatesDict
        Initial guess structure for converting x vectors to StatesDict
    iteration_interval : int
        How often to log (e.g., 100 means log every 100th iteration)
    log_data : List[Dict]
        Accumulated log entries with iteration number and objective values
    """
    
    def __init__(self, objective_manager, problem, initial_guess_states, iteration_interval: int = 100):
        """
        Initialize the iteration logger.
        
        Parameters
        ----------
        objective_manager : ObjectiveFunction
            The objective function manager from Collocation problem
        problem : CyIpoptProblem
            The IPOPT problem interface
        initial_guess_states : StatesDict
            Initial guess structure for converting x vectors
        iteration_interval : int, optional
            Log every N iterations (default: 100)
        """
        self.objective_manager = objective_manager
        self.problem = problem
        self.initial_guess_states = initial_guess_states
        self.iteration_interval = iteration_interval
        self.log_data = []
        
        # Store objective names for column headers
        self.objective_names = []
        for obj_instance in objective_manager._objectives:
            info = obj_instance._get_info()
            name = info.get("name", f"Objective_{len(self.objective_names)}")
            self.objective_names.append(name)
    
    def __call__(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                 d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """
        Callback function invoked by IPOPT at each iteration.
        
        Parameters
        ----------
        alg_mod : int
            Algorithm mode (0=regular, 1=restoration)
        iter_count : int
            Current iteration number
        obj_value : float
            Current objective function value
        inf_pr : float
            Primal infeasibility
        inf_du : float
            Dual infeasibility
        mu : float
            Barrier parameter
        d_norm : float
            Norm of step
        regularization_size : float
            Regularization value
        alpha_du : float
            Dual step size
        alpha_pr : float
            Primal step size
        ls_trials : int
            Number of line search trials
        
        Returns
        -------
        bool
            True to continue optimization, False to abort
        """
        # Only log at specified intervals
        if iter_count % self.iteration_interval == 0 and iter_count != 0:
            # Get current x from problem object (stored during objective evaluation)
            if hasattr(self.problem, '_current_x') and self.problem._current_x is not None:
                self._log_current_iteration(iter_count, self.problem._current_x)
        
        # Return True to continue optimization
        return True
    
    def _log_current_iteration(self, iter_count: int, x: np.ndarray):
        """
        Log the current objective values.
        
        Parameters
        ----------
        iter_count : int
            Current iteration number
        x : np.ndarray
            Current optimization vector
        """
        # Convert x to StatesDict
        states, globals_dict = utils.x_to_states_dict(
            x,
            self.initial_guess_states,
            self.problem.globals if hasattr(self.problem, 'globals') else None
        )
        
        # Evaluate each objective at current x
        log_entry = {"iteration": iter_count}
        
        for i, (obj_func, weight, name) in enumerate(
            zip(
                self.objective_manager.objective_functions,
                self.objective_manager.weights,
                self.objective_names
            )
        ):
            # Evaluate unweighted objective
            try:
                obj_value = obj_func(states, globals_dict)
                # Store weighted value
                log_entry[name] = weight * obj_value
                print(f"Objective {name} has {log_entry[name]} value at iteration {iter_count}")
            except Exception as e:
                # If evaluation fails, store NaN
                log_entry[name] = np.nan
                print(f"Warning: Failed to evaluate {name} at iteration {iter_count}: {e}")
        
        self.log_data.append(log_entry)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get logged data as a DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: iteration, objective_1, objective_2, ...
        """
        if not self.log_data:
            return pd.DataFrame()
        
        return pd.DataFrame(self.log_data)
    
    def reset(self):
        """Clear all logged data."""
        self.log_data = []
        self._last_logged_iteration = -1
