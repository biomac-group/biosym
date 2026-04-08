"""
Model Comparison and Optimization Integration
==============================================

This example demonstrates how to compare different model configurations
and integrate BiosymModel with optimization frameworks. We'll compare
simple and complex models and show how to set up optimization problems.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from typing import Dict, Any

# Add parent directory to path for importing biosym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biosym.model.model import load_model

###############################################################################
# Load Multiple Models for Comparison
# ------------------------------------
# 
# We'll load different models to compare their complexity and performance
# characteristics.

models = {}
model_files = {
    "pendulum": "../tests/models/pendulum.xml",
    "gait2d": "../tests/models/gait2d_torque/gait2d_torque.yaml"
}

print("Loading multiple models for comparison...")
for name, file_path in model_files.items():
    try:
        start_time = time.time()
        models[name] = load_model(file_path)
        load_time = time.time() - start_time
        print(f"  {name}: loaded in {load_time:.3f}s ({models[name].n_states} states)")
    except Exception as e:
        print(f"  {name}: failed to load - {e}")

###############################################################################
# Model Comparison Analysis
# -------------------------
# 
# Compare the structural and computational characteristics of different models.

def analyze_model(model, name: str) -> Dict[str, Any]:
    """Analyze a model and return its characteristics."""
    
    # Set up basic states and constants
    states = {
        "model": np.zeros(model.n_states),
        "gc_model": np.zeros(0) if not hasattr(model, 'gc_model') else np.zeros(model.gc_model.n_states),
        "actuator_model": np.zeros(0) if not hasattr(model, 'actuators') else np.zeros(model.actuators.n_states),
    }
    
    constants = {
        "model": model.default_values[model.n_states:],
        "gc_model": np.zeros(0) if not hasattr(model, 'gc_model') else model.gc_model.default_values,
        "actuator_model": np.zeros(0) if not hasattr(model, 'actuators') else model.actuators.default_values,
    }
    
    # Benchmark key functions
    import timeit
    
    def benchmark_function(func_name: str, n_runs: int = 100) -> float:
        """Benchmark a model function."""
        if func_name not in model.run:
            return np.nan
        
        try:
            # Warm up
            model.run[func_name](states, constants)
            
            # Benchmark
            elapsed = timeit.timeit(
                lambda: model.run[func_name](states, constants),
                number=n_runs
            )
            return elapsed / n_runs * 1000  # ms per call
        except Exception:
            return np.nan
    
    # Collect model characteristics
    analysis = {
        "name": name,
        "n_states": model.n_states,
        "n_constants": model.n_constants,
        "n_bodies": len(model.dicts['bodies']),
        "n_joints": len(model.dicts['joints']),
        "n_dof": model.coordinates['n'],
        "has_contact": hasattr(model, 'gc_model'),
        "has_actuators": hasattr(model, 'actuators'),
        
        # Performance metrics
        "fk_time": benchmark_function("FK"),
        "dynamics_time": benchmark_function("confun"),
        "jacobian_time": benchmark_function("jacobian"),
        "mass_matrix_time": benchmark_function("mass_matrix"),
    }
    
    return analysis

print("\n--- Model Analysis ---")
model_analyses = {}
for name, model in models.items():
    print(f"Analyzing {name} model...")
    model_analyses[name] = analyze_model(model, name)

# Display comparison table
print("\nModel Comparison:")
print("=" * 80)
header = f"{'Model':<12} {'DOF':<4} {'Bodies':<7} {'Joints':<7} {'States':<7} {'FK(ms)':<8} {'Dyn(ms)':<8}"
print(header)
print("-" * 80)

for analysis in model_analyses.values():
    row = (f"{analysis['name']:<12} "
           f"{analysis['n_dof']:<4} "
           f"{analysis['n_bodies']:<7} "
           f"{analysis['n_joints']:<7} "
           f"{analysis['n_states']:<7} "
           f"{analysis['fk_time']:<8.3f} "
           f"{analysis['dynamics_time']:<8.3f}")
    print(row)

###############################################################################
# Optimization Problem Setup
# ---------------------------
# 
# Demonstrate how to set up optimization problems using BiosymModel.
# We'll create a simple trajectory optimization example.

def setup_optimization_problem(model, target_position: np.ndarray):
    """Set up a simple trajectory optimization problem.
    
    Objective: minimize the difference between end-effector position and target.
    """
    
    def objective_function(x: np.ndarray) -> float:
        """Objective function for optimization."""
        
        # Set up states from optimization variables
        states = {
            "model": x,
            "gc_model": np.zeros(0) if not hasattr(model, 'gc_model') else np.zeros(model.gc_model.n_states),
            "actuator_model": np.zeros(0) if not hasattr(model, 'actuators') else np.zeros(model.actuators.n_states),
        }
        
        constants = {
            "model": model.default_values[model.n_states:],
            "gc_model": np.zeros(0) if not hasattr(model, 'gc_model') else model.gc_model.default_values,
            "actuator_model": np.zeros(0) if not hasattr(model, 'actuators') else model.actuators.default_values,
        }
        
        # Compute forward kinematics
        positions = model.run["FK"](states, constants)
        
        # Extract end-effector position (first body, first 2 coordinates)
        end_effector_pos = positions.flatten()[:2]
        
        # Compute distance to target
        error = np.linalg.norm(end_effector_pos - target_position)
        
        return error
    
    def jacobian_function(x: np.ndarray) -> np.ndarray:
        """Jacobian of objective function."""
        
        states = {
            "model": x,
            "gc_model": np.zeros(0) if not hasattr(model, 'gc_model') else np.zeros(model.gc_model.n_states),
            "actuator_model": np.zeros(0) if not hasattr(model, 'actuators') else np.zeros(model.actuators.n_states),
        }
        
        constants = {
            "model": model.default_values[model.n_states:],
            "gc_model": np.zeros(0) if not hasattr(model, 'gc_model') else model.gc_model.default_values,
            "actuator_model": np.zeros(0) if not hasattr(model, 'actuators') else model.actuators.default_values,
        }
        
        # Compute Jacobian of forward kinematics
        jac = model.run["jacobian"](states, constants)
        
        # Extract relevant part of Jacobian (simplified)
        if isinstance(jac, dict) and "model" in jac:
            return jac["model"][:2, :len(x)]  # First 2 rows (x,y), relevant columns
        else:
            return np.zeros((2, len(x)))
    
    return objective_function, jacobian_function

# Set up optimization for pendulum model
if "pendulum" in models:
    print("\n--- Optimization Setup Example ---")
    pendulum_model = models["pendulum"]
    
    # Define target position
    target_pos = np.array([0.5, -0.5])  # Target x,y position
    
    # Set up optimization problem
    obj_func, jac_func = setup_optimization_problem(pendulum_model, target_pos)
    
    # Initial guess
    x0 = np.zeros(pendulum_model.n_states)
    x0[0] = np.pi/6  # Initial angle guess
    
    # Evaluate objective and jacobian
    initial_error = obj_func(x0)
    print(f"Initial end-effector error: {initial_error:.4f} m")
    
    # Simple gradient descent optimization (educational purposes)
    learning_rate = 0.01
    max_iterations = 50
    tolerance = 1e-4
    
    x = x0.copy()
    errors = [initial_error]
    
    print("Running simple gradient descent optimization...")
    for i in range(max_iterations):
        # Compute objective and gradient
        error = obj_func(x)
        
        # Simple numerical gradient (for robustness)
        grad = np.zeros_like(x)
        eps = 1e-6
        for j in range(len(x)):
            x_plus = x.copy()
            x_plus[j] += eps
            x_minus = x.copy() 
            x_minus[j] -= eps
            grad[j] = (obj_func(x_plus) - obj_func(x_minus)) / (2 * eps)
        
        # Update
        x = x - learning_rate * grad
        errors.append(error)
        
        if error < tolerance:
            print(f"Converged after {i+1} iterations")
            break
        
        if i % 10 == 0:
            print(f"  Iteration {i:2d}: error = {error:.6f}")
    
    final_error = obj_func(x)
    print(f"Final end-effector error: {final_error:.6f} m")
    print(f"Optimal angle: {x[0]:.3f} rad ({x[0]*180/np.pi:.1f}°)")

###############################################################################
# Model Scaling Analysis
# -----------------------
# 
# Analyze how computational cost scales with model complexity.

def scaling_analysis(models: Dict[str, Any]):
    """Analyze computational scaling with model complexity."""
    
    scaling_data = []
    
    for name, model in models.items():
        # Get model complexity metrics
        n_dof = model.coordinates['n']
        n_states = model.n_states
        
        # Get performance metrics from previous analysis
        if name in model_analyses:
            fk_time = model_analyses[name]['fk_time']
            dyn_time = model_analyses[name]['dynamics_time']
            
            scaling_data.append({
                'name': name,
                'n_dof': n_dof,
                'n_states': n_states,
                'fk_time': fk_time,
                'dyn_time': dyn_time
            })
    
    return scaling_data

scaling_data = scaling_analysis(models)

print("\n--- Computational Scaling Analysis ---")
print("DOF vs Performance:")
for data in scaling_data:
    print(f"  {data['name']}: {data['n_dof']} DOF -> FK: {data['fk_time']:.3f}ms, Dynamics: {data['dyn_time']:.3f}ms")

###############################################################################
# Advanced Optimization Integration
# ----------------------------------
# 
# Show how to integrate with scipy.optimize or other optimization libraries.

def demonstrate_scipy_integration(model):
    """Demonstrate integration with scipy.optimize."""
    
    try:
        from scipy.optimize import minimize
        
        def objective(x):
            """Scipy-compatible objective function."""
            states = {
                "model": np.zeros(model.n_states),
                "gc_model": np.zeros(0),
                "actuator_model": np.zeros(0),
            }
            constants = {
                "model": model.default_values[model.n_states:],
                "gc_model": np.zeros(0),
                "actuator_model": np.zeros(0),
            }
            
            # Set coordinates from optimization variables
            n_coords = model.coordinates['n']
            states["model"][:n_coords] = x
            
            # Compute some objective (e.g., energy)
            M = model.run["mass_matrix"](states, constants)
            kinetic_energy = 0.5 * np.sum(np.diag(M))  # Simplified
            
            return kinetic_energy
        
        # Set up bounds (simple joint limits)
        n_coords = model.coordinates['n']
        bounds = [(-np.pi, np.pi) for _ in range(n_coords)]
        
        # Initial guess
        x0 = np.zeros(n_coords)
        
        print(f"\nScipy optimization setup for {model.coordinates['n']}-DOF model:")
        print(f"  Variables: {n_coords} joint angles")
        print(f"  Bounds: ±π rad for all joints")
        print(f"  Objective: simplified kinetic energy")
        
        # Note: Actual optimization would be run here
        print("  (Optimization would be executed with scipy.minimize)")
        
    except ImportError:
        print("\nScipy not available - install with: pip install scipy")

# Demonstrate for available models
for name, model in models.items():
    demonstrate_scipy_integration(model)
    break  # Just show one example

###############################################################################
# Visualization of Results
# -------------------------
# 
# Create comprehensive visualizations of the model comparison and optimization.

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Model complexity comparison
ax1 = axes[0, 0]
if model_analyses:
    names = list(model_analyses.keys())
    n_dofs = [model_analyses[name]['n_dof'] for name in names]
    n_states = [model_analyses[name]['n_states'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, n_dofs, width, label='DOF', alpha=0.8)
    ax1.bar(x + width/2, n_states, width, label='States', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Count')
    ax1.set_title('Model Complexity Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.legend()

# Plot 2: Performance comparison
ax2 = axes[0, 1]
if model_analyses:
    fk_times = [model_analyses[name]['fk_time'] for name in names if not np.isnan(model_analyses[name]['fk_time'])]
    dyn_times = [model_analyses[name]['dynamics_time'] for name in names if not np.isnan(model_analyses[name]['dynamics_time'])]
    
    if fk_times and dyn_times:
        x = np.arange(len(names))
        ax2.bar(x - width/2, fk_times, width, label='FK Time', alpha=0.8)
        ax2.bar(x + width/2, dyn_times, width, label='Dynamics Time', alpha=0.8)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title('Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.legend()

# Plot 3: Optimization convergence (if available)
ax3 = axes[0, 2]
if 'errors' in locals():
    ax3.plot(errors, 'b-', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Error (m)')
    ax3.set_title('Optimization Convergence')
    ax3.grid(True)
    ax3.set_yscale('log')

# Plot 4: Scaling analysis
ax4 = axes[1, 0]
if scaling_data:
    dofs = [data['n_dof'] for data in scaling_data]
    times = [data['dyn_time'] for data in scaling_data if not np.isnan(data['dyn_time'])]
    
    if len(dofs) == len(times):
        ax4.scatter(dofs, times, s=100, alpha=0.7)
        for i, data in enumerate(scaling_data):
            if not np.isnan(data['dyn_time']):
                ax4.annotate(data['name'], (data['n_dof'], data['dyn_time']), 
                           xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('Degrees of Freedom')
        ax4.set_ylabel('Dynamics Time (ms)')
        ax4.set_title('Computational Scaling')
        ax4.grid(True)

# Plot 5: Model feature comparison
ax5 = axes[1, 1]
if model_analyses:
    features = ['has_contact', 'has_actuators']
    feature_counts = {}
    
    for feature in features:
        feature_counts[feature] = sum(1 for analysis in model_analyses.values() 
                                    if analysis.get(feature, False))
    
    if feature_counts:
        bars = ax5.bar(feature_counts.keys(), feature_counts.values(), 
                      color=['orange', 'green'], alpha=0.7)
        ax5.set_title('Model Features')
        ax5.set_ylabel('Number of Models')
        
        # Add value labels
        for bar, value in zip(bars, feature_counts.values()):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value}', ha='center', va='bottom')

# Plot 6: Performance vs complexity
ax6 = axes[1, 2]
if model_analyses:
    complexities = [analysis['n_states'] for analysis in model_analyses.values()]
    perf_times = [analysis['dynamics_time'] for analysis in model_analyses.values() 
                  if not np.isnan(analysis['dynamics_time'])]
    
    if len(complexities) == len(perf_times):
        ax6.scatter(complexities, perf_times, s=100, alpha=0.7, c='red')
        ax6.set_xlabel('Number of States')
        ax6.set_ylabel('Dynamics Time (ms)')
        ax6.set_title('Performance vs Complexity')
        ax6.grid(True)
        
        # Add trend line if we have enough points
        if len(complexities) > 1:
            z = np.polyfit(complexities, perf_times, 1)
            p = np.poly1d(z)
            ax6.plot(complexities, p(complexities), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

###############################################################################
# Summary and Best Practices
# ---------------------------
# 
# Provide recommendations for model selection and optimization integration.

print("\n--- Summary and Best Practices ---")
print("Model Selection Guidelines:")
print("  • Use simpler models (pendulum) for algorithm development and testing")
print("  • Use complex models (gait2d) for realistic biomechanical analysis")
print("  • Consider computational budget when choosing model complexity")

print("\nOptimization Integration:")
print("  • BiosymModel functions are JAX-compiled for fast gradients")
print("  • Use model.run['jacobian'] for gradient-based optimization")
print("  • Cache model compilation for repeated optimization calls")

print("\nPerformance Considerations:")
if model_analyses:
    fastest_model = min(model_analyses.values(), key=lambda x: x.get('dynamics_time', float('inf')))
    print(f"  • Fastest model: {fastest_model['name']} ({fastest_model.get('dynamics_time', 'N/A'):.3f} ms)")
    most_complex = max(model_analyses.values(), key=lambda x: x['n_states'])
    print(f"  • Most complex: {most_complex['name']} ({most_complex['n_states']} states)")

print("\nNext Steps:")
print("  • Integrate with optimization libraries (scipy, JAX, PyTorch)")
print("  • Use in trajectory optimization and optimal control")
print("  • Apply to biomechanical analysis and simulation studies")