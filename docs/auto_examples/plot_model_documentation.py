"""
BiosymModel Documentation Example
=================================

This example provides comprehensive documentation for the BiosymModel class,
covering model loading, structure inspection, and the key concepts needed
to understand and use the biomechanical modeling framework.

Note: This example focuses on documentation and conceptual understanding.
For working code examples, please refer to the test files and constraint
implementations in the biosym codebase.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for importing biosym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biosym.model.model import load_model

###############################################################################
# Model Loading and Caching
# --------------------------
# 
# BiosymModel provides efficient model loading with automatic caching based
# on model file hashes. This prevents recompilation of identical models.

print("="*60)
print("BIOSYM MODEL DOCUMENTATION EXAMPLE")
print("="*60)

print("\n1. MODEL LOADING")
print("-" * 30)

# Load a simple pendulum model
model_file = "../tests/models/pendulum.xml"
print(f"Loading model from: {model_file}")

# The load_model function handles caching automatically
model = load_model(model_file)

print(f"✓ Model loaded successfully!")
print(f"  • Model type: {type(model).__name__}")
print(f"  • Model file format: XML")
print(f"  • Caching: Automatic (hash-based)")

###############################################################################
# Model Structure and Organization
# ---------------------------------
# 
# BiosymModel organizes multibody systems into a structured hierarchy
# with clear indexing for efficient computation.

print(f"\n2. MODEL STRUCTURE")
print("-" * 30)

print(f"State Vector Organization:")
print(f"  • Total states: {model.n_states}")
print(f"  • Total constants: {model.n_constants}")

print(f"\nDegrees of Freedom:")
print(f"  • Coordinates: {model.coordinates['n']} (indices {model.coordinates['idx']}-{model.coordinates['idx'] + model.coordinates['n'] - 1})")
print(f"    Names: {model.coordinates['names']}")
print(f"  • Speeds: {model.speeds['n']} (indices {model.speeds['idx']}-{model.speeds['idx'] + model.speeds['n'] - 1})")
print(f"    Names: {model.speeds['names']}")

print(f"\nDynamic Variables:")
print(f"  • Accelerations: {model.accs['n']} (indices {model.accs['idx']}-{model.accs['idx'] + model.accs['n'] - 1})")
print(f"  • Forces: {model.forces['n']} (indices {model.forces['idx']}-{model.forces['idx'] + model.forces['n'] - 1})")
print(f"    Names: {model.forces['names']}")

if hasattr(model, 'ext_forces') and model.ext_forces['n'] > 0:
    print(f"  • External forces: {model.ext_forces['n']}")

print(f"\nMultibody Structure:")
print(f"  • Bodies: {len(model.dicts['bodies'])}")
for i, body in enumerate(model.dicts['bodies']):
    mass = body['mass'][0] if isinstance(body['mass'], list) else body['mass']
    print(f"    {i+1}. {body['name']} (mass: {mass:.3f} kg)")

print(f"  • Joints: {len(model.dicts['joints'])}")
for i, joint in enumerate(model.dicts['joints']):
    print(f"    {i+1}. {joint['name']} ({joint['type']})")

###############################################################################
# Compiled Functions and Performance
# -----------------------------------
# 
# BiosymModel uses JAX for high-performance computations with automatic
# differentiation and just-in-time compilation.

print(f"\n3. COMPILED FUNCTIONS")
print("-" * 30)

print("Available model functions:")
available_functions = sorted(model.run.keys())
core_functions = ["FK", "FK_dot", "confun", "jacobian", "mass_matrix", "forcing"]
advanced_functions = [f for f in available_functions if f not in core_functions]

print("\nCore Functions:")
for func in core_functions:
    if func in available_functions:
        description = {
            "FK": "Forward kinematics (body positions)",
            "FK_dot": "Velocity kinematics (body velocities)", 
            "confun": "Equations of motion residual",
            "jacobian": "Jacobian of equations of motion",
            "mass_matrix": "Mass/inertia matrix",
            "forcing": "Forcing terms (gravity, Coriolis, etc.)"
        }
        print(f"  • {func:12} - {description.get(func, 'Available')}")

if advanced_functions:
    print("\nAdvanced Functions:")
    for func in advanced_functions:
        if "actuator" in func:
            print(f"  • {func:20} - Actuator model functions")
        elif "gc_model" in func:
            print(f"  • {func:20} - Ground contact functions")
        elif "marker" in func:
            print(f"  • {func:20} - Marker/sensor functions")
        else:
            print(f"  • {func:20} - Specialized function")

###############################################################################
# State and Constant Vectors
# ---------------------------
# 
# Understanding how to structure input data for model functions.

print(f"\n4. DATA STRUCTURES")
print("-" * 30)

print("State Vector Structure:")
print("  The state vector contains all time-varying quantities:")
print("  • Generalized coordinates (joint angles, positions)")
print("  • Generalized speeds (joint velocities)")  
print("  • Accelerations (computed or prescribed)")
print("  • Control forces/torques")
print("  • External force variables (if contact models present)")

print(f"\nConstant Vector Structure:")
print("  The constant vector contains all time-invariant parameters:")
print("  • Body masses and inertias")
print("  • Joint limits and properties")
print("  • Geometric parameters")
print("  • Model-specific constants")

print(f"\nInput Format:")
print("  Model functions expect structured inputs:")
print("  • states: Contains model states organized by component")
print("  • constants: Contains model parameters organized by component")
print("  • Components: 'model', 'gc_model', 'actuator_model'")

###############################################################################
# Symbolic Framework
# ------------------
# 
# BiosymModel builds on symbolic mechanics for exact representations.

print(f"\n5. SYMBOLIC FRAMEWORK")
print("-" * 30)

print("Symbolic Mechanics Foundation:")
print("  • Built on SymPy's mechanics module")
print("  • Uses Kane's method for equation derivation")
print("  • Automatic generation of equations of motion")
print("  • Exact symbolic differentiation")

print(f"\nJAX Compilation Pipeline:")
print("  • Symbolic expressions → lambdified functions")
print("  • JAX JIT compilation for performance")
print("  • Automatic differentiation for gradients")
print("  • GPU/TPU acceleration support")

print(f"\nCaching Strategy:")
print("  • Model hash-based caching")
print("  • JAX compilation caching")
print("  • Persistent disk storage")
print("  • Automatic cache management")

###############################################################################
# Model Components and Extensions
# --------------------------------
# 
# Understanding the modular architecture.

print(f"\n6. MODEL COMPONENTS")
print("-" * 30)

print("Core Model:")
print("  • Rigid body dynamics")
print("  • Joint constraints")
print("  • Conservative and non-conservative forces")

if hasattr(model, 'gc_model'):
    print("\nGround Contact:")
    print("  • Contact detection and forces")
    print("  • Friction models")
    print("  • Constraint enforcement")

if hasattr(model, 'actuators'):
    print("\nActuator Models:")
    print("  • Muscle models")
    print("  • Motor models") 
    print("  • Passive elements")

print(f"\nConfiguration Options:")
print("  • XML model files (basic structure)")
print("  • YAML configuration files (advanced features)")
print("  • Programmatic model construction")

###############################################################################
# Usage Patterns and Applications
# --------------------------------
# 
# Common use cases and integration patterns.

print(f"\n7. USAGE PATTERNS")
print("-" * 30)

print("Simulation:")
print("  • Forward dynamics integration")
print("  • Inverse dynamics computation") 
print("  • Kinematic analysis")

print(f"\nOptimization:")
print("  • Trajectory optimization")
print("  • Parameter estimation")
print("  • Optimal control")

print(f"\nAnalysis:")
print("  • Sensitivity analysis")
print("  • Stability assessment")
print("  • Performance evaluation")

###############################################################################
# Integration Examples
# --------------------
# 
# Show conceptual integration with other libraries.

print(f"\n8. INTEGRATION EXAMPLES")
print("-" * 30)

print("Optimization Libraries:")
print("  • scipy.optimize - General optimization")
print("  • JAX optimizers - Gradient-based methods")
print("  • PyTorch - Deep learning integration")

print(f"\nSimulation Frameworks:")
print("  • SciPy ODE solvers - Time integration")
print("  • JAX integrators - Differentiable simulation")
print("  • Custom solvers - Specialized applications")

print(f"\nVisualization:")
print("  • Matplotlib - 2D plotting and analysis")
print("  • Mayavi/VTK - 3D visualization")
print("  • Custom renderers - Application-specific display")

###############################################################################
# Performance Characteristics
# ----------------------------
# 
# Understanding computational complexity and scaling.

print(f"\n9. PERFORMANCE CHARACTERISTICS")
print("-" * 30)

print("Computational Complexity:")
print(f"  • Forward kinematics: O(n) where n = number of bodies")
print(f"  • Mass matrix: O(n³) for dense systems")
print(f"  • Equations of motion: O(n²) typically")

print(f"\nJAX Compilation Benefits:")
print("  • First call: Compilation overhead (slower)")
print("  • Subsequent calls: Near C++ performance") 
print("  • Batch processing: Vectorized operations")
print("  • Hardware acceleration: GPU/TPU support")

print(f"\nMemory Considerations:")
print("  • Model caching: Disk storage for compiled models")
print("  • JAX arrays: Immutable data structures")
print("  • Large models: Consider memory-efficient representations")

###############################################################################
# Example Model Analysis
# ----------------------
# 
# Demonstrate model inspection for the loaded pendulum.

print(f"\n10. EXAMPLE MODEL ANALYSIS")
print("-" * 30)

print(f"Pendulum Model Details:")
print(f"  • Configuration: Double pendulum ({model.coordinates['n']} DOF)")
print(f"  • Bodies: {', '.join([body['name'] for body in model.dicts['bodies']])}")
print(f"  • Joints: {', '.join([joint['name'] for joint in model.dicts['joints']])}")

# Create a simple visualization showing the model structure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: State vector organization
state_components = [
    ('Coordinates', model.coordinates['n']),
    ('Speeds', model.speeds['n']),
    ('Accelerations', model.accs['n']),
    ('Forces', model.forces['n'])
]

if hasattr(model, 'ext_forces') and model.ext_forces['n'] > 0:
    state_components.append(('Ext Forces', model.ext_forces['n']))

labels, sizes = zip(*state_components)
colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))

ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
ax1.set_title('State Vector Composition\n(Number of Variables)')

# Plot 2: Model complexity comparison
models_comparison = {
    'Current\nPendulum': model.n_states,
    'Typical\nHumanoid': 50,  # Example
    'Simple\nRobot': 12,      # Example
    'Complex\nBiomech': 100   # Example
}

bars = ax2.bar(models_comparison.keys(), models_comparison.values(), 
               color=['blue', 'orange', 'green', 'red'], alpha=0.7)
ax2.set_ylabel('Number of States')
ax2.set_title('Model Complexity Comparison')
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, models_comparison.values()):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{value}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

###############################################################################
# Summary and Next Steps
# ----------------------
# 
# Provide guidance for getting started with BiosymModel.

print(f"\n" + "="*60)
print("SUMMARY AND NEXT STEPS")
print("="*60)

print("Key Takeaways:")
print("  ✓ BiosymModel provides high-performance biomechanical modeling")
print("  ✓ JAX compilation enables fast, differentiable computations")
print("  ✓ Modular architecture supports various model types")
print("  ✓ Automatic caching optimizes development workflows")

print(f"\nGetting Started:")
print("  1. Start with simple models (pendulum, simple robot)")
print("  2. Understand state vector organization")
print("  3. Explore available model functions")
print("  4. Integrate with optimization/simulation workflows")

print(f"\nAdvanced Usage:")
print("  • Add custom contact models")
print("  • Implement actuator dynamics")
print("  • Create model variants with YAML configs")
print("  • Optimize performance for large-scale problems")

print(f"\nResources:")
print("  • Model files: tests/models/ directory")
print("  • Implementation: biosym/model/ module")
print("  • Examples: biosym/constraints/ for usage patterns")
print("  • Tests: tests/ directory for validation")

print("="*60)