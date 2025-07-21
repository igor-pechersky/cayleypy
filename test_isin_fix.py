#!/usr/bin/env python
"""Test script for the isin_via_searchsorted function with empty arrays.

This script specifically tests the fix for handling empty arrays in the
isin_via_searchsorted function.
"""

import time
import numpy as np

print("Testing JAX availability...")

# Check if JAX is available
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax, jit
    JAX_AVAILABLE = True
    print(f"JAX is available, version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    lax = None
    print("JAX not available")
    exit(0)

print("\n=== Testing isin_via_searchsorted implementations ===")

# Original implementation with direct conditional
def isin_direct(elements, test_elements_sorted):
    """Implementation using direct conditional."""
    # Handle empty test_elements case directly
    if len(test_elements_sorted) == 0:
        return jnp.zeros_like(elements, dtype=bool)
    
    # Find insertion points
    indices = jnp.searchsorted(test_elements_sorted, elements)
    # Clamp indices to valid range
    indices = jnp.clip(indices, 0, len(test_elements_sorted) - 1)
    # Check if elements match at insertion points
    return test_elements_sorted[indices] == elements

# Implementation using lax.cond
@jit
def isin_lax_cond(elements, test_elements_sorted):
    """Implementation using lax.cond."""
    def empty_case():
        return jnp.zeros_like(elements, dtype=bool)
    
    def non_empty_case():
        # Find insertion points
        indices = jnp.searchsorted(test_elements_sorted, elements)
        # Clamp indices to valid range
        indices = jnp.clip(indices, 0, len(test_elements_sorted) - 1)
        # Check if elements match at insertion points
        return test_elements_sorted[indices] == elements
    
    return lax.cond(
        len(test_elements_sorted) == 0,
        empty_case,
        non_empty_case
    )

# Test with normal array
print("\n=== Testing with normal array ===")
elements = jnp.array([1, 2, 3, 4, 5])
test_elements = jnp.array([2, 4, 6])
expected = jnp.array([False, True, False, True, False])

# Test direct implementation
start_time = time.time()
result_direct = isin_direct(elements, test_elements)
end_time = time.time()
direct_time = end_time - start_time
print(f"Direct implementation: {direct_time:.6f} seconds")
print(f"Result: {result_direct}")
print(f"Matches expected: {jnp.array_equal(result_direct, expected)}")

# Test lax.cond implementation
start_time = time.time()
result_lax = isin_lax_cond(elements, test_elements)
end_time = time.time()
lax_time = end_time - start_time
print(f"lax.cond implementation: {lax_time:.6f} seconds")
print(f"Result: {result_lax}")
print(f"Matches expected: {jnp.array_equal(result_lax, expected)}")
print(f"Matches direct implementation: {jnp.array_equal(result_direct, result_lax)}")

# Test with empty array
print("\n=== Testing with empty array ===")
empty_test_elements = jnp.array([])
expected_empty = jnp.array([False, False, False, False, False])

# Test direct implementation
start_time = time.time()
result_direct_empty = isin_direct(elements, empty_test_elements)
end_time = time.time()
direct_time_empty = end_time - start_time
print(f"Direct implementation: {direct_time_empty:.6f} seconds")
print(f"Result: {result_direct_empty}")
print(f"Matches expected: {jnp.array_equal(result_direct_empty, expected_empty)}")

# Test lax.cond implementation
try:
    start_time = time.time()
    result_lax_empty = isin_lax_cond(elements, empty_test_elements)
    end_time = time.time()
    lax_time_empty = end_time - start_time
    print(f"lax.cond implementation: {lax_time_empty:.6f} seconds")
    print(f"Result: {result_lax_empty}")
    print(f"Matches expected: {jnp.array_equal(result_lax_empty, expected_empty)}")
    print(f"Matches direct implementation: {jnp.array_equal(result_direct_empty, result_lax_empty)}")
    lax_success = True
except Exception as e:
    print(f"lax.cond implementation failed: {e}")
    lax_success = False

# Summary
print("\n=== Summary ===")
print("Direct implementation:")
print(f"  - Normal array: {direct_time:.6f} seconds")
print(f"  - Empty array: {direct_time_empty:.6f} seconds")

if lax_success:
    print("lax.cond implementation:")
    print(f"  - Normal array: {lax_time:.6f} seconds")
    print(f"  - Empty array: {lax_time_empty:.6f} seconds")
    
    # Compare implementations
    print("\nComparison:")
    print(f"  - Normal array speedup: {direct_time / lax_time:.2f}x")
    print(f"  - Empty array speedup: {direct_time_empty / lax_time_empty:.2f}x")
else:
    print("lax.cond implementation failed with empty array")
    print("This confirms that the direct implementation is more robust")