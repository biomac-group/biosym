import biosym.ocp.collocation as collocation
from biosym.ocp.utils import states_dict_to_x, x_to_states_dict
from biosym.utils.states import StatesDict
import jax.numpy as jnp
import matplotlib.pyplot as plt
# Import sparse for sparse matrix operations
import numpy as np
import jax
import time


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

def derivativetest(problem, x, eps = 1e-3):
    """
    Test the derivative of the objective function.
    :param problem: The collocation problem instance.
    :param x: The input state vector.
    :return: The derivative of the objective function.
    """
    assert isinstance(x, np.ndarray), "Input x must be a numpy array."  
    x = np.array(x, dtype=np.float64)
    jac_jax_0 = problem.problem.jacobian(x)
    grad_jax = problem.problem.gradient(x)
    jacstruct = problem.problem.jacobianstructure()
    jac_jax = np.zeros((len(x), problem.constraints.ncon))
    #import matplotlib.pyplot as plt
    #print(problem.constraints.c_start)
    #plt.plot(jacstruct[0])
    #plt.show()
    #print(max(jacstruct[0]), max(jacstruct[1]))
    #print(jac_jax.shape)
    jac_jax[jacstruct[1], jacstruct[0]] = jac_jax_0
    jac_num = np.zeros_like(jac_jax)
    grad_num = np.zeros_like(grad_jax)
    x0 = x.copy()
    
    a = time.time()
    for i in range(len(x)):
        x = x0.copy()
        x[i] += eps  # Perturb the i-th element
        obj_1 = problem.problem.objective(x)
        con_1 = problem.problem.constraints(x)

        x = x0.copy()
        x[i] -= eps  # Perturb the i-th element in the opposite direction
        obj_2 = problem.problem.objective(x)
        con_2 = problem.problem.constraints(x)

        grad_num[i] = (obj_1 - obj_2) / (2 * eps)
        jac_num[i] = (con_1 - con_2) / (2 * eps)

        if (i % 200 == 0):
            if time.time() - a > 5:
                print(f"Derivative test: {i}/{len(x)} done, ({100* i/len(x):.2f}%)")
        
    print("Jacobian JAX vs Numerical:", jnp.allclose(jac_jax, jac_num, atol=1e-4))
    print("Gradient JAX vs Numerical:", jnp.allclose(grad_jax, grad_num, atol=1e-4))

    if not jnp.allclose(jac_jax, jac_num, atol=1e-4):
        # Print the max deviation and index of the first mismatch
        max_deviation = jnp.max(jnp.abs(jac_jax - jac_num))
        first_mismatch_index = np.unravel_index(np.argmax(np.abs(jac_jax - jac_num), axis=None), jac_jax.shape)
        print(f"Max deviation in Jacobian: {jac_jax[first_mismatch_index], jac_num[first_mismatch_index]} at index {first_mismatch_index}")
    if not jnp.allclose(grad_jax, grad_num, atol=1e-4):
        # Print the max deviation and index of the first mismatch
        max_deviation = jnp.max(jnp.abs(grad_jax - grad_num))
        first_mismatch_index = np.unravel_index(np.argmax(np.abs(grad_jax - grad_num), axis=None), grad_jax.shape)
        print(f"Max deviation in Gradient: {grad_jax[first_mismatch_index], grad_num[first_mismatch_index]} at index {first_mismatch_index}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].spy(jac_jax.T, label='Jacobian JAX')
    ax[1].spy(jac_num.T, label='Jacobian Numerical')
    ax[0].set_title('Jacobian JAX')
    ax[1].set_title('Jacobian Numerical')
    ax[0].legend()
    ax[1].legend()
    # Set model.states as the xticks for both axes
    ax[0].set_xticks(np.arange(len(problem.model.state_vector)))
    ax[0].set_xticklabels(problem.model.state_vector, rotation=90)
    ax[1].set_xticks(np.arange(len(problem.model.state_vector)))
    ax[1].set_xticklabels(problem.model.state_vector, rotation=90)
    plt.show()
    plt.scatter(np.arange(len(jac_jax.flatten())), jac_jax.flatten())
    plt.scatter(np.arange(len(jac_num.flatten())), jac_num.flatten(), alpha=0.5)    
    plt.show()


standing_problem = collocation.Collocation("tests/collocation/standing2d.yaml", force_rebuild=True)
(x, globals), info = standing_problem.solve(visualize=False)

walking_problem = collocation.Collocation("tests/collocation/walking2d.yaml", force_rebuild=False)

x = states_dict_to_x(walking_problem.initial_guess_states, walking_problem.initial_guess_globals)
x = np.random.rand(len(x)).astype(np.float64)  # Randomize the initial guess
#derivativetest(walking_problem, x)
x, globals = x_to_states_dict(x, walking_problem.initial_guess_states, walking_problem.initial_guess_globals
                            )
#asd =asd 
(x, globals), info = walking_problem.solve(visualize=True)
print(x.states.h)

# Testing the constraints and objective function
import time 
import timeit
i = 0

#for i, state in enumerate(standing_problem.model.state_vector):
#    print(standing_problem.model.state_vector[i], x.states.model[0,i])
print(globals)

for function, name in zip([standing_problem.constraints.confun, 
                standing_problem.constraints.jacobian, 
                standing_problem.objective.objfun, 
                standing_problem.objective.gradfun],
                ['constraints', 'jacobian', 'objectives', 'gradients']):

    continue
    print(f"Testing {name} function...")
    
    
    start_time = time.time()
    function(standing_problem.initial_guess_states, None)
    print(f"{function.__name__} evaluated in {time.time() - start_time} seconds")

    # Test 1k compiled evaluations
    n_evals = 10000
    a = timeit.timeit(lambda: function(standing_problem.initial_guess_states, None), number=n_evals)
    print(f"{n_evals} evaluations of {name} took {a/n_evals} seconds on average")


# Show this at the end
standing_problem.objective.add_objective(TestObjectiveFunction, weight=1.0)
