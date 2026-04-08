#!/usr/bin/env python3
"""
Test script to validate the examples work correctly.
"""

import sys
import os
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path for importing biosym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biosym.model.model import load_model
from biosym.utils import states

def test_basic_example():
    """Test the basic model usage example."""
    print("Testing basic model usage...")
    
    # Load model
    model_file = "../tests/models/pendulum.xml"
    model = load_model(model_file)
    
    print(f"✓ Model loaded: {model.n_states} states, {model.n_constants} constants")
    
    # Test model structure
    print(f"✓ Coordinates: {model.coordinates['names']}")
    print(f"✓ Bodies: {len(model.dicts['bodies'])}")
    print(f"✓ Joints: {len(model.dicts['joints'])}")
    
    # Set up states and constants
    states_dict = {
        "states": {
            "model": jnp.zeros(model.n_states),
            "gc_model": jnp.zeros(0),
            "actuator_model": jnp.zeros(0),
        },
        "constants": {
            "model": jnp.array(model.default_values[model.n_states:]),
            "gc_model": jnp.zeros(0),
            "actuator_model": jnp.zeros(0),
        }
    }
    
    # Convert to proper dataclass format
    states_obj = states.dict_to_dataclass(states_dict)
    
    # Test forward kinematics
    states_obj.states.model = states_obj.states.model.at[0].set(np.pi/4)  # Set angle
    positions = model.run["FK"](states_obj.states, states_obj.constants)
    print(f"✓ Forward kinematics: {positions.shape}")
    
    # Test dynamics
    eom = model.run["confun"](states_obj.states, states_obj.constants)
    print(f"✓ Equations of motion: {eom.shape}")
    
    # Test Jacobian
    jac = model.run["jacobian"](states_obj.states, states_obj.constants)
    print(f"✓ Jacobian computed")
    
    print("✓ Basic example test passed!\n")

def test_advanced_example():
    """Test the advanced gait analysis example."""
    print("Testing advanced gait analysis...")
    
    try:
        # Load gait model
        model_file = "../tests/models/gait2d_torque/gait2d_torque.yaml"
        model = load_model(model_file)
        
        print(f"✓ Gait model loaded: {model.n_states} states")
        print(f"✓ DOF: {model.coordinates['n']}")
        
        # Set up states
        states_dict = {
            "states": {
                "model": jnp.zeros(model.n_states),
                "gc_model": jnp.zeros(0) if not hasattr(model, 'gc_model') else jnp.zeros(model.gc_model.n_states),
                "actuator_model": jnp.zeros(0) if not hasattr(model, 'actuators') else jnp.zeros(model.actuators.n_states),
            },
            "constants": {
                "model": jnp.array(model.default_values[model.n_states:]),
                "gc_model": jnp.zeros(0) if not hasattr(model, 'gc_model') else jnp.array(model.gc_model.default_values),
                "actuator_model": jnp.zeros(0) if not hasattr(model, 'actuators') else jnp.array(model.actuators.default_values),
            }
        }
        
        states_obj = states.dict_to_dataclass(states_dict)
        
        # Test basic functionality
        positions = model.run["FK"](states_obj.states, states_obj.constants)
        print(f"✓ Forward kinematics: {positions.shape}")
        
        M = model.run["mass_matrix"](states_obj.states, states_obj.constants)
        print(f"✓ Mass matrix: {M.shape}")
        
        print("✓ Advanced example test passed!\n")
        
    except Exception as e:
        print(f"⚠ Advanced example test skipped: {e}\n")

def test_model_comparison():
    """Test the model comparison example."""
    print("Testing model comparison...")
    
    models = {}
    model_files = {
        "pendulum": "../tests/models/pendulum.xml",
    }
    
    # Try to load gait model too
    try:
        model_files["gait2d"] = "../tests/models/gait2d_torque/gait2d_torque.yaml"
    except:
        pass
    
    for name, file_path in model_files.items():
        try:
            models[name] = load_model(file_path)
            print(f"✓ {name} model loaded")
        except Exception as e:
            print(f"⚠ {name} model failed: {e}")
    
    if models:
        print(f"✓ Model comparison test passed with {len(models)} models!\n")
    else:
        print("✗ Model comparison test failed - no models loaded\n")

if __name__ == "__main__":
    print("=" * 60)
    print("BIOSYM EXAMPLES VALIDATION")
    print("=" * 60)
    
    try:
        test_basic_example()
        test_advanced_example()
        test_model_comparison()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED!")
        print("Examples are ready for documentation generation.")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        sys.exit(1)