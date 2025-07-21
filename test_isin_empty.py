#!/usr/bin/env python
"""Test script for the isin_via_searchsorted function with empty arrays.

This script specifically tests the fix for the empty array case in the
isin_via_searchsorted function.
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

# Import the function to test
try:
    from cayleypy.jax_tensor_ops import isin_via_searchsorted
    print("Successfully imported isin_via_searchsorted")
except Exception as e:
    print(f"Error importing isin_via_searchsorted: {e}")
    exit(1)


def test_empty_array():
    """Test isin_via_searchsorted with empty test_elements."""
    print("\n=== Testing isin_via_searchsorted with empty array ===")
    
    # Create test data
    elements = jnp.array([1, 2, 3])
    test_elements = jnp.array([])
    
    # Run the function
    try:
        start_time = time.time()
        result = isin_via_searchsorted(elements, test_elements)
        end_time = time.time()
        
        print(f"Function completed in {end_time - start_time:.6f} seconds")
        print(f"Result: {result}")
        print(f"Result shape: {result.shape}")
        print(f"Result dtype: {result.dtype}")
        
        # Verify the result is all False
        expected = jnp.zeros_like(elements, dtype=bool)
        match = jnp.array_equal(result, expected)
        print(f"Result matches expected (all False): {match}")
        
        if match:
            print("TEST PASSED: isin_via_searchsorted correctly handles empty arrays")
        else:
            print("TEST FAILED: isin_via_searchsorted did not return all False values")
    except Exception as e:
        print(f"TEST FAILED: isin_via_searchsorted raised an exception: {e}")


def test_normal_array():
    """Test isin_via_searchsorted with normal test_elements."""
    print("\n=== Testing isin_via_searchsorted with normal array ===")
    
    # Create test data
    elements = jnp.array([1, 2, 3, 4, 5])
    test_elements = jnp.array([2, 4, 6])
    
    # Run the function
    try:
        start_time = time.time()
        result = isin_via_searchsorted(elements, test_elements)
        end_time = time.time()
        
        print(f"Function completed in {end_time - start_time:.6f} seconds")
        print(f"Result: {result}")
        
        # Verify the result matches expected
        expected = jnp.array([False, True, False, True, False])
        match = jnp.array_equal(result, expected)
        print(f"Result matches expected: {match}")
        
        if match:
            print("TEST PASSED: isin_via_searchsorted correctly handles normal arrays")
        else:
            print("TEST FAILED: isin_via_searchsorted returned incorrect results")
    except Exception as e:
        print(f"TEST FAILED: isin_via_searchsorted raised an exception: {e}")


def test_performance():
    """Test performance of isin_via_searchsorted."""
    print("\n=== Testing isin_via_searchsorted performance ===")
    
    # Create test data
    elements = jnp.arange(10000)
    test_elements = jnp.arange(0, 10000, 10)  # Every 10th element
    
    # Define a naive implementation for comparison
    def naive_isin(elements, test_elements):
        return jnp.array([e in test_elements for e in elements])
    
    # Test naive implementation (only for small arrays)
    small_elements = elements[:1000]
    start_time = time.time()
    _ = naive_isin(small_elements, test_elements)
    end_time = time.time()
    naive_time = end_time - start_time
    print(f"Naive implementation (small array): {naive_time:.6f} seconds")
    
    # Test optimized implementation on small array
    start_time = time.time()
    _ = isin_via_searchsorted(small_elements, test_elements)
    end_time = time.time()
    optimized_small_time = end_time - start_time
    print(f"Optimized implementation (small array): {optimized_small_time:.6f} seconds")
    
    # Calculate speedup for small array
    speedup_small = naive_time / optimized_small_time
    print(f"Speedup (small array): {speedup_small:.2f}x")
    
    # Test optimized implementation on large array
    start_time = time.time()
    _ = isin_via_searchsorted(elements, test_elements)
    end_time = time.time()
    optimized_large_time = end_time - start_time
    print(f"Optimized implementation (large array): {optimized_large_time:.6f} seconds")
    
    # Calculate throughput
    throughput = len(elements) / optimized_large_time
    print(f"Throughput: {throughput:.2f} elements/second")


if __name__ == "__main__":
    test_empty_array()
    test_normal_array()
    test_performance()