#!/usr/bin/env python
"""Simple test script for JAX/TPU optimizations in CayleyPy.

This script runs quick performance tests on the optimized JAX implementations.
"""

import time
import numpy as np
from typing import Dict, List, Callable

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    print("JAX not available - using numpy for tests")

# Import CayleyPy components
if JAX_AVAILABLE:
    from cayleypy.jax_tensor_ops import (
        unique_with_indices, isin_via_searchsorted, sort_with_indices,
        batch_matmul, vectorized_element_wise_equal, batch_isin_via_searchsorted
    )
    
    from cayleypy.jax_hasher import (
        JAXStateHasher, OptimizedJAXStateHasher, vectorized_hash_states
    )


def time_function(func, *args, **kwargs) -> float:
    """Time a function execution.
    
    Args:
        func: Function to time
        *args: Arguments to pass to function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Execution time in seconds
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def run_comparison(name: str, funcs: Dict[str, Callable], *args, **kwargs):
    """Run and compare multiple implementations of the same function.
    
    Args:
        name: Name of the operation
        funcs: Dictionary of implementation name -> function
        *args: Arguments to pass to functions
        **kwargs: Keyword arguments to pass to functions
    """
    print(f"\n=== Testing {name} ===")
    results = {}
    times = {}
    
    baseline_impl = next(iter(funcs.keys()))
    
    for impl_name, func in funcs.items():
        time_taken, result = time_function(func, *args, **kwargs)
        results[impl_name] = result
        times[impl_name] = time_taken
        print(f"{impl_name}: {time_taken:.6f} seconds")
    
    # Calculate speedups
    baseline_time = times[baseline_impl]
    for impl_name, time_taken in times.items():
        if impl_name != baseline_impl:
            speedup = baseline_time / time_taken
            print(f"{impl_name} speedup: {speedup:.2f}x")
    
    # Verify results match
    baseline_result = results[baseline_impl]
    for impl_name, result in results.items():
        if impl_name != baseline_impl:
            if JAX_AVAILABLE:
                try:
                    match = jnp.array_equal(result, baseline_result)
                    print(f"{impl_name} results match baseline: {match}")
                except:
                    print(f"{impl_name} results could not be compared")
            else:
                try:
                    match = np.array_equal(result, baseline_result)
                    print(f"{impl_name} results match baseline: {match}")
                except:
                    print(f"{impl_name} results could not be compared")


def test_tensor_operations():
    """Test tensor operations performance."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping tensor operation tests")
        return
    
    print("\n=== Testing Tensor Operations ===")
    
    # Test unique_with_indices
    array = jnp.array([3, 1, 2, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10])
    run_comparison("unique_with_indices", {
        "jnp.unique": lambda x: jnp.unique(x, return_inverse=True, return_counts=True),
        "optimized": lambda x: unique_with_indices(x, True, True)
    }, array)
    
    # Test isin
    elements = jnp.arange(1000)
    test_elements = jnp.arange(0, 1000, 10)  # Every 10th element
    run_comparison("isin", {
        "manual": lambda x, y: jnp.array([e in y for e in x]),
        "optimized": lambda x, y: isin_via_searchsorted(x, y)
    }, elements[:100], test_elements)  # Use subset for manual method
    
    # Test batch operations
    batch_size = 100
    state_size = 10
    batch_count = 5
    batches = [jnp.ones((batch_size, state_size)) * i for i in range(batch_count)]
    
    # Test vectorized operations
    a = jnp.array([[1, 2, 3], [4, 5, 6]])
    b = jnp.array([[1, 2, 3], [4, 5, 7]])
    run_comparison("element_wise_equal", {
        "standard": lambda x, y: x == y,
        "vectorized": lambda x, y: vectorized_element_wise_equal(x, y)
    }, a, b)
    
    # Test batch isin
    elements_batch = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    test_elements = jnp.array([2, 5, 8, 11])
    run_comparison("batch_isin", {
        "loop": lambda x, y: jnp.stack([isin_via_searchsorted(row, y) for row in x]),
        "vectorized": lambda x, y: batch_isin_via_searchsorted(x, y)
    }, elements_batch, test_elements)
    
    # Test matrix multiplication
    matrix_size = 100
    a = jnp.ones((matrix_size, matrix_size))
    b = jnp.ones((matrix_size, matrix_size))
    run_comparison("matmul", {
        "jnp.matmul": jnp.matmul,
        "batch_matmul": batch_matmul
    }, a, b)


def test_hash_functions():
    """Test hash function performance."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping hash function tests")
        return
    
    print("\n=== Testing Hash Functions ===")
    
    # Create test data
    batch_size = 10000
    state_size = 10
    states = jnp.ones((batch_size, state_size), dtype=jnp.int32)
    states = states.at[:, 0].set(jnp.arange(batch_size))  # Make states unique
    
    # Create hashers
    standard_hasher = JAXStateHasher(state_size=state_size, random_seed=42)
    optimized_hasher = OptimizedJAXStateHasher(state_size=state_size, random_seed=42)
    
    # Test hash_states
    run_comparison("hash_states", {
        "standard": standard_hasher.hash_states,
        "vectorized": lambda x: vectorized_hash_states(x, standard_hasher),
        "optimized": optimized_hasher.hash_states_optimized
    }, states)
    
    # Test hash_single_state
    single_state = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # We can't easily compare these since they return Python int vs JAX array
    print("\n=== Testing single state hashing ===")
    time_taken, result = time_function(standard_hasher.hash_single_state, single_state)
    print(f"standard: {time_taken:.6f} seconds, result: {result}")
    
    # Test batch processing
    small_batches = [states[i:i+1000] for i in range(0, batch_size, 1000)]
    
    def process_batches_standard(batches):
        results = []
        for batch in batches:
            results.append(standard_hasher.hash_states(batch))
        return jnp.concatenate(results)
    
    def process_batches_optimized(batches):
        return optimized_hasher.batch_hash_with_vectorization(batches)
    
    run_comparison("batch_processing", {
        "standard": process_batches_standard,
        "optimized": lambda x: optimized_hasher.batch_hash_with_vectorization(x)
    }, small_batches)


def main():
    """Run all tests."""
    print("=== JAX/TPU Optimization Tests ===")
    print(f"JAX available: {JAX_AVAILABLE}")
    
    if JAX_AVAILABLE:
        print(f"JAX version: {jax.__version__}")
        print(f"Available devices: {jax.devices()}")
        print(f"Default backend: {jax.default_backend()}")
    
    test_tensor_operations()
    test_hash_functions()
    
    print("\nTests completed!")


if __name__ == "__main__":
    main()