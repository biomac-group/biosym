"""
Working BiosymModel Interface Example
=====================================

This example demonstrates the actual working interface for BiosymModel
based on the patterns used in the biosym codebase. It shows model loading,
structure inspection, and the concepts behind the biomechanical framework.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for importing biosym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biosym.model.model import load_model

###############################################################################
# Model Loading and Basic Inspection
# -----------------------------------
# 
# This section demonstrates model loading and how to inspect the model
# structure to understand its organization.

print("Loading BiosymModel...")
model = load_model("../tests/models/pendulum.xml")
print(f"✓ Model loaded: {model.coordinates['n']}-DOF system")

###############################################################################
# Understanding Model Organization
# --------------------------------
# 
# BiosymModel organizes multibody dynamics into structured components.

print(f"\n--- Model Structure Analysis ---")
print(f"Coordinates ({model.coordinates['n']}): {model.coordinates['names']}")
print(f"Bodies ({len(model.dicts['bodies'])}): {[b['name'] for b in model.dicts['bodies']]}")
print(f"Joints ({len(model.dicts['joints'])}): {[j['name'] for j in model.dicts['joints']]}")

print(f"\nState Vector Layout (total: {model.n_states}):")
print(f"  Coordinates:   indices {model.coordinates['idx']:2} - {model.coordinates['idx'] + model.coordinates['n'] - 1:2}")
print(f"  Speeds:        indices {model.speeds['idx']:2} - {model.speeds['idx'] + model.speeds['n'] - 1:2}")
print(f"  Accelerations: indices {model.accs['idx']:2} - {model.accs['idx'] + model.accs['n'] - 1:2}")
print(f"  Forces:        indices {model.forces['idx']:2} - {model.forces['idx'] + model.forces['n'] - 1:2}")

###############################################################################
# Available Computational Functions
# ----------------------------------
# 
# BiosymModel provides JAX-compiled functions for efficient computation.

print(f"\n--- Available Functions ---")
functions = sorted(model.run.keys())
print(f"Total functions available: {len(functions)}")

# Categorize functions
core_funcs = [f for f in functions if not any(x in f for x in ['uncompiled', 'marker', 'actuator', 'gc_model'])]
marker_funcs = [f for f in functions if 'marker' in f]
actuator_funcs = [f for f in functions if 'actuator' in f]
contact_funcs = [f for f in functions if 'gc_model' in f]

print(f"\nCore dynamics functions ({len(core_funcs)}):")
for func in core_funcs:
    print(f"  • {func}")

if marker_funcs:
    print(f"\nMarker/sensor functions ({len(marker_funcs)}):")
    for func in marker_funcs:
        print(f"  • {func}")

if actuator_funcs:
    print(f"\nActuator functions ({len(actuator_funcs)}):")
    for func in actuator_funcs:
        print(f"  • {func}")

###############################################################################
# Model Constants and Default Values
# -----------------------------------
# 
# Understanding the parameter structure and default values.

print(f"\n--- Model Parameters ---")
print(f"Total constants: {model.n_constants}")
print(f"Default values available: {len(model.default_values)} parameters")

# Show a sample of the first few constants
if hasattr(model, 'default_values') and len(model.default_values) > model.n_states:
    constants_sample = model.default_values[model.n_states:model.n_states+5]
    print(f"Sample constants (first 5): {constants_sample}")

###############################################################################
# Symbolic Framework Insight
# ---------------------------
# 
# Understanding the symbolic mechanics foundation.

print(f"\n--- Symbolic Framework ---")
print("BiosymModel uses symbolic mechanics for exact formulations:")
print("  • SymPy mechanics module for symbolic representation")
print("  • Kane's method for equation of motion derivation")  
print("  • JAX compilation for high-performance evaluation")
print("  • Automatic differentiation for gradients")

print(f"\nEquations of motion structure:")
print(f"  • Independent coordinates: {model.coordinates['n']}")
print(f"  • Constraint equations: Automatically handled")
print(f"  • Mass matrix size: {model.coordinates['n']} × {model.coordinates['n']}")

###############################################################################
# Data Flow and Interface Patterns
# ---------------------------------
# 
# Understanding how data flows through the model functions.

print(f"\n--- Data Interface Patterns ---")
print("BiosymModel functions expect structured data:")
print("  • Input: (states, constants) - organized by model component")
print("  • Output: JAX arrays with computed quantities")
print("  • Components: model, gc_model, actuator_model")

print(f"\nTypical workflow:")
print("  1. Create state and constant dictionaries")
print("  2. Convert to appropriate data structures") 
print("  3. Call model functions with structured inputs")
print("  4. Process JAX array outputs")

###############################################################################
# Performance and Compilation
# ----------------------------
# 
# Understanding JAX compilation and performance characteristics.

print(f"\n--- Performance Characteristics ---")
print("JAX compilation process:")
print("  • First call: JIT compilation (slower)")
print("  • Subsequent calls: Compiled performance (fast)")
print("  • Batch operations: Vectorized execution")
print("  • Gradients: Automatic differentiation")

print(f"\nCaching benefits:")
print("  • Model compilation cached by hash")
print("  • JAX function compilation cached")
print("  • Persistent storage for reuse")

###############################################################################
# Visualization of Model Structure
# ---------------------------------
# 
# Create visualizations to understand model organization.

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: State vector composition
ax1 = axes[0, 0]
components = ['Coords', 'Speeds', 'Accs', 'Forces']
sizes = [model.coordinates['n'], model.speeds['n'], model.accs['n'], model.forces['n']]
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']

wedges, texts, autotexts = ax1.pie(sizes, labels=components, colors=colors, 
                                   autopct='%1.0f', startangle=90)
ax1.set_title('State Vector Composition')

# Plot 2: Model complexity metrics
ax2 = axes[0, 1]
metrics = ['States', 'Constants', 'Bodies', 'Joints', 'DOF']
values = [model.n_states, model.n_constants, len(model.dicts['bodies']), 
          len(model.dicts['joints']), model.coordinates['n']]

bars = ax2.bar(metrics, values, color=['blue', 'green', 'red', 'orange', 'purple'], alpha=0.7)
ax2.set_title('Model Complexity Metrics')
ax2.set_ylabel('Count')

# Add value labels
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value}', ha='center', va='bottom')

# Plot 3: Function categories
ax3 = axes[1, 0]
func_categories = ['Core', 'Markers', 'Actuators', 'Contact']
func_counts = [len(core_funcs), len(marker_funcs), len(actuator_funcs), len(contact_funcs)]

bars = ax3.bar(func_categories, func_counts, color=['navy', 'teal', 'maroon', 'darkgreen'], alpha=0.7)
ax3.set_title('Available Function Categories')
ax3.set_ylabel('Number of Functions')

for bar, value in zip(bars, func_counts):
    if value > 0:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{value}', ha='center', va='bottom')

# Plot 4: Performance scaling illustration
ax4 = axes[1, 1]
# Theoretical performance scaling
dofs = np.array([1, 2, 5, 10, 20, 50])
fk_times = 0.1 * dofs  # O(n)
dynamics_times = 0.01 * dofs**2  # O(n²)
mass_matrix_times = 0.001 * dofs**3  # O(n³)

ax4.loglog(dofs, fk_times, 'b-o', label='Forward Kinematics O(n)', linewidth=2)
ax4.loglog(dofs, dynamics_times, 'r-s', label='Dynamics O(n²)', linewidth=2)
ax4.loglog(dofs, mass_matrix_times, 'g-^', label='Mass Matrix O(n³)', linewidth=2)

# Highlight current model
current_dof = model.coordinates['n']
ax4.axvline(current_dof, color='black', linestyle='--', alpha=0.7, label=f'Current Model ({current_dof} DOF)')

ax4.set_xlabel('Degrees of Freedom')
ax4.set_ylabel('Computation Time (ms)')
ax4.set_title('Theoretical Performance Scaling')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

###############################################################################
# Integration Guidance
# ---------------------
# 
# Practical guidance for using BiosymModel in applications.

print(f"\n--- Integration Guidance ---")
print("For successful BiosymModel integration:")

print(f"\n1. Model Development:")
print("   • Start with simple models (pendulum, simple robots)")
print("   • Validate against known solutions")
print("   • Gradually increase complexity")

print(f"\n2. Performance Optimization:")
print("   • Use caching for repeated model loads")
print("   • Batch computations when possible")
print("   • Profile critical code paths")

print(f"\n3. Data Handling:")
print("   • Understand state vector organization")
print("   • Use appropriate data structures")
print("   • Handle JAX array immutability")

print(f"\n4. Common Applications:")
print("   • Trajectory optimization: Use jacobian functions")
print("   • Forward simulation: Use confun for dynamics")
print("   • Analysis: Use FK and mass_matrix functions")
print("   • Control: Integrate with optimization libraries")

###############################################################################
# Summary
# -------

print(f"\n" + "="*60)
print("BIOSYM MODEL INTERFACE SUMMARY")
print("="*60)

print(f"Successfully demonstrated BiosymModel interface:")
print(f"  ✓ Model loading and structure inspection")
print(f"  ✓ Function inventory and categorization")
print(f"  ✓ Data organization understanding")
print(f"  ✓ Performance characteristics")
print(f"  ✓ Integration guidance")

print(f"\nThis {model.coordinates['n']}-DOF pendulum model demonstrates:")
print(f"  • {len(functions)} compiled functions available")
print(f"  • {model.n_states} state variables organized efficiently")
print(f"  • JAX compilation for high performance")
print(f"  • Modular architecture for extensibility")

print(f"\nNext steps:")
print(f"  • Explore constraint implementations in biosym/constraints/")
print(f"  • Review test files for validated API usage")
print(f"  • Integrate with optimization frameworks")
print(f"  • Scale to more complex biomechanical models")

print("="*60)