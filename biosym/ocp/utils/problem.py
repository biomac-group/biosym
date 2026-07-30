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
    globals_ : dict, optional
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
        globals_=None,
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
        globals_ : dict, optional
            Global variables and parameters for the problem.
        """
        self.model = model
        self.objs = objective
        self.cons = constraints
        self.template = template  # For reconstructing something better looking
        self.ub, self.lb = upper_bound, lower_bound
        self.globals_ = globals_
        self._init_jac = False
        self.jacobianstructure()
        # Store current x for iteration callback access
        self._current_x = None
        # Iteration callback (will be set by enable_logging)
        self._iteration_callback = None
        # Caching de-vectorized states to avoid redundant x_to_states_dict calls
        self._cached_x = None
        self._cached_states = None
        self._cached_globals = None

    def _get_states(self, x):
        """Helper to retrieve structured states, utilizing a fast cache if x is identical."""
        if self._cached_x is not None and (x is self._cached_x or np.array_equal(x, self._cached_x)):
            return self._cached_states, self._cached_globals
        
        states, globals_ = x_to_states_dict(x, self.template, self.globals_)
        self._cached_x = x
        self._cached_states = states
        self._cached_globals = globals_
        return states, globals_
    
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
        states, globals_ = self._get_states(x)
        return self.objs.objfun(states, globals_)

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
        states, globals_ = self._get_states(x)
        return states_dict_to_x(*self.objs.gradfun(states, globals_))

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
        states, globals_ = self._get_states(x)
        return self.cons.confun(states, globals_)

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
        states, globals_ = self._get_states(x)
        return self._compiled_jacobian(states, globals_)

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
            x, globals_ = x_to_states_dict(x, self.template, self.globals_)
            return self.cons.jacobian(x, globals_)

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
        
        # Compile a JAX JIT wrapper function that slices the Jacobian array on the backend
        static_indices = jnp.array(self.jac_indices[0])
        self._compiled_jacobian = jax.jit(
            lambda states, globals_: self.cons.jacobian_data(states, globals_)[static_indices]
        )
        
        return self.jacstruct

    def check_derivatives(self, x: jnp.ndarray = None, eps=1e-5, rtol=1e-4, atol=1e-4):
        """
        Validate analytical objective gradients and constraint Jacobians
        against numerical finite differences (central differences).
        
        This test always runs in double-precision (x64) mode to avoid
        numerical precision limitations in single-precision float32.

        Parameters
        ----------
        x : jnp.ndarray, optional
            Flat optimization vector. If None, a random value between the bounds of the
            optimization problem is used.
        eps : float, optional
            Perturbation size for finite differences, by default 1e-5.
        rtol : float, optional
            Relative tolerance for matching, by default 1e-4.
        atol : float, optional
            Absolute tolerance for matching, by default 1e-4.

        Returns
        -------
        dict
            A dictionary containing validation results and detailed info
            on any mismatches.
        """
        import time
        from jax.flatten_util import ravel_pytree
        from dataclasses import fields

        ## Save and set x64
        #original_x64 = jax.config.read("jax_enable_x64")
        #jax.config.update("jax_enable_x64", True)

        if x is None:
            x = np.random.uniform(self.lb, self.ub)

        x_64 = np.array(x)
        n_vars = len(x_64)

        # Evaluate analytical value
        grad_jax = np.array(self.gradient(x_64))
        
        jac_jax_0 = np.array(self.jacobian(x_64))
        jacstruct = self.jacobianstructure()
        n_cons = self.cons.ncon
        
        jac_jax = np.zeros((n_vars, n_cons))
        # jacstruct[1] corresponds to cols (variables), jacstruct[0] to rows (constraints)
        jac_jax[jacstruct[1], jacstruct[0]] = jac_jax_0

        # Numerical evaluations using central differences

        grad_num = np.zeros_like(grad_jax)
        jac_num = np.zeros_like(jac_jax)
        
        x0 = x_64.copy()
        start_time = time.time()

        for i in range(n_vars):
            # Positive perturbation
            x_perturbed = x0.copy()
            x_perturbed[i] += eps
            obj_plus = float(self.objective(x_perturbed))
            con_plus = np.array(self.constraints(x_perturbed))

            # Negative perturbation
            x_perturbed = x0.copy()
            x_perturbed[i] -= eps
            obj_minus = float(self.objective(x_perturbed))
            con_minus = np.array(self.constraints(x_perturbed))

            grad_num[i] = (obj_plus - obj_minus) / (2.0 * eps)
            jac_num[i] = (con_plus - con_minus) / (2.0 * eps)

            if i % 200 == 0 and time.time() - start_time > 5.0:
                print(f"Derivative test: {i}/{n_vars} variables evaluated ({100.0 * i / n_vars:.1f}%)")

        # Determine mismatches
        grad_diff = np.abs(grad_jax - grad_num)
        grad_ok = bool(np.allclose(grad_jax, grad_num, rtol=rtol, atol=atol))
        
        jac_diff = np.abs(jac_jax - jac_num)
        jac_ok = bool(np.allclose(jac_jax, jac_num, rtol=rtol, atol=atol))

        # Set up the index mapping logic to translate indices to names
        # Build example state and unraveling functions
        example_state = jax.tree_util.tree_map(
            lambda arr: arr[0] if isinstance(arr, jnp.ndarray) else arr, self.template
        )
        flat_ex, unravel_state = ravel_pytree(example_state)
        d = flat_ex.shape[0]
        N = len(self.template)
        indices_tree = unravel_state(jnp.arange(d))

        def map_var_index_to_name(idx):
            if idx < N * d:
                node_idx = idx // d
                var_idx = idx % d
                for field_name in ["q", "qd", "qdd", "tau", "ext_forces", "ext_torques", "gc_model", "actuator_model"]:
                    indices = getattr(indices_tree, field_name, None)
                    if indices is not None:
                        matching_indices = np.where(indices == var_idx)[0]
                        if len(matching_indices) > 0:
                            local_idx = matching_indices[0]
                            if field_name == "q" and hasattr(self.model, "coordinates") and "names" in self.model.coordinates:
                                coord_name = self.model.coordinates["names"][local_idx]
                                return f"node {node_idx}, q[{coord_name}]"
                            elif field_name == "qd" and hasattr(self.model, "coordinates") and "names" in self.model.coordinates:
                                coord_name = self.model.coordinates["names"][local_idx]
                                return f"node {node_idx}, qd[{coord_name}]"
                            elif field_name == "qdd" and hasattr(self.model, "coordinates") and "names" in self.model.coordinates:
                                coord_name = self.model.coordinates["names"][local_idx]
                                return f"node {node_idx}, qdd[{coord_name}]"
                            elif field_name == "tau" and hasattr(self.model, "coordinates") and "names" in self.model.coordinates:
                                coord_name = self.model.coordinates["names"][local_idx]
                                return f"node {node_idx}, tau[{coord_name}]"
                            elif field_name == "gc_model" and hasattr(self.model, "contact_model") and hasattr(self.model.contact_model, "state_vector"):
                                gc_name = self.model.contact_model.state_vector[local_idx]
                                return f"node {node_idx}, gc_model[{gc_name}]"
                            elif field_name == "actuator_model" and hasattr(self.model, "actuator_model") and hasattr(self.model.actuator_model, "state_vector"):
                                act_name = self.model.actuator_model.state_vector[local_idx]
                                return f"node {node_idx}, actuator_model[{act_name}]"
                            else:
                                return f"node {node_idx}, {field_name}[{local_idx}]"
                return f"node {node_idx}, unknown_var[{var_idx}]"
            else:
                global_idx = idx - (N * d)
                if self.globals_ is not None:
                    current = 0
                    for field_desc in fields(self.globals_):
                        val = getattr(self.globals_, field_desc.name)
                        if val is not None:
                            size = val.size
                            if global_idx < current + size:
                                local_idx = global_idx - current
                                return f"globals_.{field_desc.name}[{local_idx}]"
                            current += size
                return f"global_var[{global_idx}]"

        def map_con_index_to_name(idx):
            for k, start_idx in enumerate(self.cons.c_start[:-1]):
                end_idx = self.cons.c_start[k+1]
                if start_idx <= idx < end_idx:
                    constraint_obj = self.cons._constraints[k]
                    local_idx = idx - start_idx
                    constraint_name = constraint_obj._get_info().get("name", constraint_obj.__class__.__name__)
                    return f"{constraint_name} (local index {local_idx})"
            return f"unknown_constraint[{idx}]"

        grad_mismatches = []
        if not grad_ok:
            mismatch_indices = np.where(grad_diff > (atol + rtol * np.abs(grad_num)))[0]
            for idx in mismatch_indices:
                grad_mismatches.append({
                    "index": int(idx),
                    "var": map_var_index_to_name(idx),
                    "jax_val": float(grad_jax[idx]),
                    "num_val": float(grad_num[idx]),
                    "diff": float(grad_diff[idx])
                })

        jac_mismatches = []
        if not jac_ok:
            mismatches_v, mismatches_c = np.where(jac_diff > (atol + rtol * np.abs(jac_num)))
            for v, c in zip(mismatches_v, mismatches_c):
                jac_mismatches.append({
                    "var_index": int(v),
                    "con_index": int(c),
                    "var": map_var_index_to_name(v),
                    "con": map_con_index_to_name(c),
                    "jax_val": float(jac_jax[v, c]),
                    "num_val": float(jac_num[v, c]),
                    "diff": float(jac_diff[v, c])
                })

        res = {
            "gradient_ok": grad_ok,
            "jacobian_ok": jac_ok,
            "gradient_error": {
                "max_diff": float(np.max(grad_diff)) if grad_diff.size > 0 else 0.0,
                "max_diff_idx": int(np.argmax(grad_diff)) if grad_diff.size > 0 else 0,
                "max_diff_var": map_var_index_to_name(np.argmax(grad_diff)) if grad_diff.size > 0 else "",
                "mismatches": grad_mismatches
            } if not grad_ok else None,
            "jacobian_error": {
                "max_diff": float(np.max(jac_diff)) if jac_diff.size > 0 else 0.0,
                "max_diff_idx": [int(idx) for idx in np.unravel_index(np.argmax(jac_diff), jac_diff.shape)] if jac_diff.size > 0 else [0, 0],
                "max_diff_var": map_var_index_to_name(np.unravel_index(np.argmax(jac_diff), jac_diff.shape)[0]) if jac_diff.size > 0 else "",
                "max_diff_con": map_con_index_to_name(np.unravel_index(np.argmax(jac_diff), jac_diff.shape)[1]) if jac_diff.size > 0 else "",
                "mismatches": jac_mismatches
            } if not jac_ok else None,
        }


        return res
