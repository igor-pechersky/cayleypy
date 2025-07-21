#!/usr/bin/env python
"""Minimal test script for JAX functionality.

This script tests basic JAX functionality without importing from cayleypy.
"""

import time
import numpy as np

print("Testing JAX availability...")

# Check if JAX is available
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    print(f"JAX is available, version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    print("JAX not available")

if not JAX_AVAILABLE:
    print("JAX is not available. Exiting.")
    exit(0)

# Test basic JAX operations
print("\n=== Testing basic JAX operations ===")

# Create test data
array = jnp.arange(1000)

# Test JIT compilation
print("\n=== Testing JIT compilation ===")

def standard_fn(x):
    return jnp.sum(x ** 2)

jitted_fn = jax.jit(standard_fn)

# Warm-up
_ = standard_fn(array)
_ = jitted_fn(array)

# Test standard function
start_time = time.time()
result1 = standard_fn(array)
end_time = time.time()
standard_time = end_time - start_time
print(f"Standard function: {standard_time:.6f} seconds")

# Test JIT-compiled function
start_time = time.time()
result2 = jitted_fn(array)
end_time = time.time()
jit_time = end_time - start_time
print(f"JIT-compiled function: {jit_time:.6f} seconds")

# Calculate speedup
speedup = standard_time / jit_time
print(f"JIT speedup: {speedup:.2f}x")

# Test vmap
print("\n=== Testing vmap ===")

def scalar_fn(x):
    return jnp.sum(x)

vectorized_fn = jax.vmap(scalar_fn)

# Create test data
batch_size = 100
data = jnp.ones((batch_size, 10))

# Test loop-based approach
start_time = time.time()
loop_results = []
for i in range(batch_size):
    loop_results.append(scalar_fn(data[i]))
loop_results = jnp.array(loop_results)
end_time = time.time()
loop_time = end_time - start_time
print(f"Loop-based approach: {loop_time:.6f} seconds")

# Test vectorized approach
start_time = time.time()
vmap_results = vectorized_fn(data)
end_time = time.time()
vmap_time = end_time - start_time
print(f"Vectorized approach: {vmap_time:.6f} seconds")

# Calculate speedup
speedup = loop_time / vmap_time
print(f"vmap speedup: {speedup:.2f}x")

# Test our own implementation of isin_via_searchsorted
print("\n=== Testing isin_via_searchsorted ===")

def isin_via_searchsorted(elements, test_elements_sorted):
    """JAX equivalent of the optimized isin function using searchsorted."""
    # Handle empty test_elements case directly
    if len(test_elements_sorted) == 0:
        return jnp.zeros_like(elements, dtype=bool)
    
    # Find insertion points
    indices = jnp.searchsorted(test_elements_sorted, elements)
    # Clamp indices to valid range
    indices = jnp.clip(indices, 0, len(test_elements_sorted) - 1)
    # Check if elements match at insertion points
    return test_elements_sorted[indices] == elements

# Create test data
elements = jnp.array([1, 2, 3, 4, 5])
test_elements = jnp.array([2, 4, 6])

# Test with normal array
result = isin_via_searchsorted(elements, test_elements)
print(f"Result with normal array: {result}")
expected = jnp.array([False, True, False, True, False])
print(f"Results match expected: {jnp.array_equal(result, expected)}")

# Test with empty array
empty_result = isin_via_searchsorted(elements, jnp.array([]))
print(f"Result with empty array: {empty_result}")
expected_empty = jnp.array([False, False, False, False, False])
print(f"Results match expected: {jnp.array_equal(empty_result, expected_empty)}")

print("\nAll tests completed successfully!")