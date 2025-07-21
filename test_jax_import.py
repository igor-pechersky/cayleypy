#!/usr/bin/env python3
"""Test JAX imports to debug the issue."""

try:
    import jax
    print(f"JAX version: {jax.__version__}")
    
    import jax.numpy as jnp
    print("JAX numpy imported successfully")
    
    from jax import jit, vmap, lax
    print("JAX core functions imported successfully")
    
    try:
        from jax.experimental.pjit import PartitionSpec as P
        print("PartitionSpec imported successfully")
        print(f"P type: {type(P)}")
        
        # Test creating a PartitionSpec
        test_spec = P('batch', None)
        print(f"Test PartitionSpec created: {test_spec}")
        
    except ImportError as e:
        print(f"Failed to import PartitionSpec: {e}")
        
        # Try alternative import
        try:
            from jax.sharding import PartitionSpec as P
            print("PartitionSpec imported from jax.sharding")
            test_spec = P('batch', None)
            print(f"Test PartitionSpec created: {test_spec}")
        except ImportError as e2:
            print(f"Failed to import from jax.sharding: {e2}")
    
    try:
        from jax.experimental import pjit
        print("pjit imported successfully")
    except ImportError as e:
        print(f"Failed to import pjit: {e}")
        
except ImportError as e:
    print(f"Failed to import JAX: {e}")