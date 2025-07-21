#!/usr/bin/env python
"""Test script for JAX hasher optimizations.

This script tests the basic functionality of the JAX hasher classes
without using advanced TPU features that might cause compatibility issues.
"""

import time
import numpy as np

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    print(f"JAX is available, version: {jax.__version__}")
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    print("JAX not available - skipping test")
    exit(0)

# Import the hasher classes
try:
    from cayleypy.jax_hasher import JAXStateHasher
    print("Successfully imported JAXStateHasher")
except Exception as e:
    print(f"Error importing JAXStateHasher: {e}")
    exit(1)


def test_basic_hashing():
    """Test basic hashing functionality."""
    print("\n=== Testing basic hashing ===")
    
    # Create a hasher
    state_size = 10
    hasher = JAXStateHasher(state_size=state_size, random_seed=42)
    
    # Create test data
    states = jnp.ones((100, state_size), dtype=jnp.int32)
    states = states.at[:, 0].set(jnp.arange(100))  # Make states unique
    
    # Test hash_states
    try:
        start_time = time.time()
        hashes = hasher.hash_states(states)
        end_time = time.time()
        
        print(f"hash_states completed in {end_time - start_time:.6f} seconds")
        print(f"Result shape: {hashes.shape}")
        print(f"Result dtype: {hashes.dtype}")
        
        # Check that hashes are unique
        unique_hashes = jnp.unique(hashes)
        print(f"Number of unique hashes: {len(unique_hashes)}")
        print(f"All hashes are unique: {len(unique_hashes) == len(states)}")
        
        if len(unique_hashes) == len(states):
            print("TEST PASSED: All states have unique hashes")
        else:
            print("TEST WARNING: Some states have duplicate hashes")
    except Exception as e:
        print(f"TEST FAILED: hash_states raised an exception: {e}")


def test_hash_single_state():
    """Test hashing a single state."""
    print("\n=== Testing hash_single_state ===")
    
    # Create a hasher
    state_size = 10
    hasher = JAXStateHasher(state_size=state_size, random_seed=42)
    
    # Create test data
    state = jnp.ones(state_size, dtype=jnp.int32)
    
    # Test hash_single_state
    try:
        start_time = time.time()
        hash_value = hasher.hash_single_state(state)
        end_time = time.time()
        
        print(f"hash_single_state completed in {end_time - start_time:.6f} seconds")
        print(f"Result: {hash_value}")
        print(f"Result type: {type(hash_value)}")
        
        # Check that the result is an integer
        if isinstance(hash_value, (int, np.integer)):
            print("TEST PASSED: hash_single_state returns an integer")
        else:
            print(f"TEST FAILED: hash_single_state returns {type(hash_value)}, not an integer")
    except Exception as e:
        print(f"TEST FAILED: hash_single_state raised an exception: {e}")


def test_batch_hashing():
    """Test batch hashing functionality."""
    print("\n=== Testing batch hashing ===")
    
    # Create a hasher
    state_size = 10
    hasher = JAXStateHasher(state_size=state_size, random_seed=42)
    
    # Create test data
    batch_size = 1000
    num_batches = 5
    batches = []
    for i in range(num_batches):
        batch = jnp.ones((batch_size, state_size), dtype=jnp.int32)
        batch = batch.at[:, 0].set(jnp.arange(i * batch_size, (i + 1) * batch_size))
        batches.append(batch)
    
    # Test hashing each batch separately
    try:
        start_time = time.time()
        separate_hashes = []
        for batch in batches:
            separate_hashes.append(hasher.hash_states(batch))
        separate_hashes = jnp.concatenate(separate_hashes)
        end_time = time.time()
        
        separate_time = end_time - start_time
        print(f"Separate batch hashing completed in {separate_time:.6f} seconds")
        print(f"Result shape: {separate_hashes.shape}")
        
        # Test hashing all batches at once
        combined_batch = jnp.concatenate(batches)
        
        start_time = time.time()
        combined_hashes = hasher.hash_states(combined_batch)
        end_time = time.time()
        
        combined_time = end_time - start_time
        print(f"Combined batch hashing completed in {combined_time:.6f} seconds")
        print(f"Result shape: {combined_hashes.shape}")
        
        # Check that results match
        match = jnp.array_equal(separate_hashes, combined_hashes)
        print(f"Results match: {match}")
        
        # Calculate speedup
        speedup = separate_time / combined_time
        print(f"Speedup from combining batches: {speedup:.2f}x")
        
        if match:
            print("TEST PASSED: Batch hashing produces consistent results")
        else:
            print("TEST FAILED: Batch hashing produces inconsistent results")
    except Exception as e:
        print(f"TEST FAILED: Batch hashing raised an exception: {e}")


def test_different_state_sizes():
    """Test hashing with different state sizes."""
    print("\n=== Testing different state sizes ===")
    
    # Test different state sizes
    state_sizes = [1, 10, 100]
    batch_size = 100
    
    for state_size in state_sizes:
        print(f"\nTesting state size: {state_size}")
        
        # Create a hasher
        hasher = JAXStateHasher(state_size=state_size, random_seed=42)
        
        # Create test data
        states = jnp.ones((batch_size, state_size), dtype=jnp.int32)
        states = states.at[:, 0].set(jnp.arange(batch_size))  # Make states unique
        
        # Test hash_states
        try:
            start_time = time.time()
            hashes = hasher.hash_states(states)
            end_time = time.time()
            
            print(f"hash_states completed in {end_time - start_time:.6f} seconds")
            print(f"Result shape: {hashes.shape}")
            
            # Check that hashes are unique
            unique_hashes = jnp.unique(hashes)
            print(f"Number of unique hashes: {len(unique_hashes)}")
            print(f"All hashes are unique: {len(unique_hashes) == len(states)}")
        except Exception as e:
            print(f"TEST FAILED: hash_states raised an exception: {e}")


if __name__ == "__main__":
    test_basic_hashing()
    test_hash_single_state()
    test_batch_hashing()
    test_different_state_sizes()