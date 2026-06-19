"""
CyIpoptProblem interface for optimal control problems in biosym.
"""

import numpy as np
import jax
import jax.numpy as jnp
from biosym.ocp.utils.vectorize import x_to_states_dict, states_dict_to_x


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
        x, globals = x_to_states_dict(x, self.template, self.globals)
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
        x, globals = x_to_states_dict(x, self.template, self.globals)
        return states_dict_to_x(*self.objs.gradfun(x, globals))

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
        x, globals = x_to_states_dict(x, self.template, self.globals)
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
        x, globals = x_to_states_dict(x, self.template, self.globals)
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
            x, globals = x_to_states_dict(x, self.template, self.globals)
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
