from biosym.constraints.base_constraint import BaseConstraint
from biosym.ocp import utils
import jax.numpy as jnp
import jax
import os
from functools import partial

# any constraint needs to be named Constraint, otherwise it will not be found by the OCP class
class Constraint(BaseConstraint):
    """
    Base class for periodicity constraints in the biosym package.
    """
    def __init__(self, model, settings, args):
        """
        Initialize the PeriodicityConstraint class with a model and settings.
        """
        self.model = model
        self.settings = settings.copy()
        self.args = args
        self.settings['nvpn'] = len(model.state_vector)
        self.nvar = settings.get('nvar')

        # We exclude certain dimensions from the periodicity constraint
        self.exclude_dims = self.args.get('exclude', [])
        self.dims = jnp.array([i for i in range(self.model.n_states) if i not in self.exclude_dims])

        if args.get('symmetry', False):
            self.id_symmetry = jnp.array(get_symmetry_indices(self.model.state_vector))
        else: 
            self.id_symmetry = jnp.arange(model.n_states)

        self.adaptive_h = settings.get('adaptive_h', False)

    def _get_info(self):
        """
        Get information about the periodicity constraint.
        
        This method can be overridden in subclasses to provide specific information.
        """
        return {
            'name': os.path.splitext(os.path.basename(__file__))[0],
            'description': 'Periodicity constraint class for biosym constraints.',
            'required_variables': {'states': ["model"], "constants": ["model"]},
            'nnz': self.get_nnz(),
            'ncons': self.get_n_constraints(),
        }
    
    def get_confun(self):
        """
        Evaluate the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The dynamics constraint function.
        """
        return jax.jit(partial(confun, dims=self.dims, id_symmetry=self.id_symmetry))

    def get_jacobian(self):
        """
        Get the Jacobian of the dynamics constraint function.

        :param states_list: Dictionary containing the current states.
        :return: The Jacobian of the dynamics constraint function.
        """
        return jax.jit(partial(jacobian_per,dims=self.dims, id_symmetry=self.id_symmetry, nnodes_dur=self.settings['nnodes_dur'], settings=self.settings))

    def get_n_constraints(self):
        """
        Get the number of constraints defined by this dynamics constraint.
        
        :return: The number of constraints.
        """
        return len(self.dims)  # One constraint for each dimension that is not excluded
    
    def get_nnz(self):
        """
        Get the number of non-zero entries in the Jacobian of the dynamics constraint.
        
        :return: The number of non-zero entries.
        """
        return 2 * self.get_n_constraints()  # Each constraint has two non-zero entries (start and end of the periodicity)
    

def confun(states_list, _, dims, id_symmetry):
    """
    Evaluate the constraint function for adaptive step sizes.
    :param states_list: Dictionary containing the current states.
    :param globals_dict: Dictionary containing global variables (e.g., duration).
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The evaluated constraint function.
    """
    return states_list[0].states.flatten()[dims] - states_list[-1].states.flatten()[id_symmetry][dims]  # Example: periodicity constraint between first and last state

def jacobian_per(states_list, _, dims, id_symmetry, nnodes_dur, settings):
    """
    Placeholder for the Jacobian of the constraint function.
    
    This function should be implemented in subclasses to compute the Jacobian of the dynamics constraints.
    
    :param states_list: List containing the current states.
    :param settings: Dictionary containing settings for the dynamics constraint.
    :param info: Information about the constraint function.
    :return: The Jacobian of the constraint function.
    """
    r = jnp.tile(jnp.arange(len(dims), dtype=settings['int_dtype']), 2)
    c = jnp.concatenate((jnp.array(dims, dtype=settings['int_dtype']), id_symmetry[dims]+states_list[0].states.size()*(nnodes_dur-1)))
    d = jnp.concatenate((jnp.ones(len(dims), dtype=settings['dtype']), -jnp.ones(len(dims), dtype=settings['dtype'])))
    return r, c, d
    
    
    
def get_symmetry_indices(names):
    index_map = {}
    used = set()
    
    for i, name in enumerate(names):
        if i in used:
            continue
        if '_l' in name:
            mirror_name = name.replace('_l', '_r')
            if mirror_name in names:
                j = names.index(mirror_name)
                index_map[i] = j
                index_map[j] = i
                used.add(i)
                used.add(j)
            else:
                index_map[i] = i
        elif '_r' in name:
            mirror_name = name.replace('_r', '_l')
            if mirror_name in names:
                j = names.index(mirror_name)
                index_map[i] = j
                index_map[j] = i
                used.add(i)
                used.add(j)
            else:
                index_map[i] = i
        else:
            index_map[i] = i

    return [index_map[i] for i in range(len(names))]

