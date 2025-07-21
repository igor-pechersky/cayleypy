#!/usr/bin/env python
"""Simple test script for JAX optimizations in CayleyPy.

This script runs basic performance tests on the core JAX implementations
without using advanced TPU features that might cause compatibility issues.
"""

import time
import numpy as np
from typing import Dict, Callable

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    print(f"JAX is available, version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    print("JAX not available - using numpy for tests")


def time_function(func, *args, **kwargs):
    """Time a function execution."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result


def run_comparison(name, funcs, *args, **kwargs):
    """Run and compare multiple implementations of the same function."""
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
    
    return times


def test_numpy_operations():
    """Test basic numpy operations as a fallback."""
    print("\n=== Testing NumPy Operations ===")
    
    # Test array operations
    array_size = 10000
    array = np.arange(array_size)
    
    # Test sort
    start_time = time.time()
    _ = np.sort(array)
    end_time = time.time()
    print(f"np.sort: {end_time - start_time:.6f} seconds")
    
    # Test unique
    start_time = time.time()
    _ = np.unique(array)
    end_time = time.time()
    print(f"np.unique: {end_time - start_time:.6f} seconds")
    
    # Test matrix multiplication
    matrix_size = 1000
    a = np.ones((matrix_size, matrix_size))
    b = np.ones((matrix_size, matrix_size))
    
    start_time = time.time()
    _ = np.matmul(a, b)
    end_time = time.time()
    print(f"np.matmul: {end_time - start_time:.6f} seconds")


def test_jax_basic_operations():
    """Test basic JAX operations."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping JAX basic operations test")
        return
    
    print("\n=== Testing Basic JAX Operations ===")
    
    # Test array operations
    array_size = 10000
    array = jnp.arange(array_size)
    
    # Test sort
    start_time = time.time()
    _ = jnp.sort(array)
    end_time = time.time()
    print(f"jnp.sort: {end_time - start_time:.6f} seconds")
    
    # Test unique
    start_time = time.time()
    _ = jnp.unique(array)
    end_time = time.time()
    print(f"jnp.unique: {end_time - start_time:.6f} seconds")
    
    # Test matrix multiplication
    matrix_size = 1000
    a = jnp.ones((matrix_size, matrix_size))
    b = jnp.ones((matrix_size, matrix_size))
    
    start_time = time.time()
    _ = jnp.matmul(a, b)
    end_time = time.time()
    print(f"jnp.matmul: {end_time - start_time:.6f} seconds")


def test_jit_compilation():
    """Test JIT compilation benefits."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping JIT compilation test")
        return
    
    print("\n=== Testing JIT Compilation ===")
    
    # Define a function to JIT compile
    def standard_fn(x):
        return jnp.sum(x ** 2)
    
    # JIT-compiled version
    jitted_fn = jax.jit(standard_fn)
    
    # Test data
    data = jnp.arange(1000000)
    
    # Warm-up
    _ = standard_fn(data)
    _ = jitted_fn(data)
    
    # Test standard function
    start_time = time.time()
    result1 = standard_fn(data)
    end_time = time.time()
    standard_time = end_time - start_time
    print(f"Standard function: {standard_time:.6f} seconds")
    
    # Test JIT-compiled function
    start_time = time.time()
    result2 = jitted_fn(data)
    end_time = time.time()
    jit_time = end_time - start_time
    print(f"JIT-compiled function: {jit_time:.6f} seconds")
    
    # Calculate speedup
    speedup = standard_time / jit_time
    print(f"JIT speedup: {speedup:.2f}x")
    
    # Verify results match
    print(f"Results match: {jnp.array_equal(result1, result2)}")


def test_vmap():
    """Test vectorization with vmap."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping vmap test")
        return
    
    print("\n=== Testing vmap Vectorization ===")
    
    # Define a function to vectorize
    def scalar_fn(x):
        return jnp.sum(x ** 2)
    
    # Vectorized version
    vectorized_fn = jax.vmap(scalar_fn)
    
    # Test data
    batch_size = 1000
    data = jnp.ones((batch_size, 1000))
    
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
    
    # Verify results match
    print(f"Results match: {jnp.allclose(loop_results, vmap_results)}")


def test_chunked_operations():
    """Test chunked operations for memory efficiency."""
    if not JAX_AVAILABLE:
        print("JAX not available - skipping chunked operations test")
        return
    
    print("\n=== Testing Chunked Operations ===")
    
    # Define a memory-intensive operation
    def memory_intensive_op(x):
        return jnp.sum(x ** 2, axis=1)
    
    # Define a chunked version
    def chunked_op(x, chunk_size):
        results = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i+chunk_size]
            results.append(memory_intensive_op(chunk))
        return jnp.concatenate(results)
    
    # Test data
    array_size = 100000
    feature_size = 100
    data = jnp.ones((array_size, feature_size))
    
    # Test standard operation (may fail on large arrays)
    try:
        start_time = time.time()
        standard_result = memory_intensive_op(data)
        end_time = time.time()
        standard_time = end_time - start_time
        print(f"Standard operation: {standard_time:.6f} seconds")
        standard_success = True
    except Exception as e:
        print(f"Standard operation failed: {e}")
        standard_success = False
    
    # Test chunked operation
    chunk_size = 10000
    start_time = time.time()
    chunked_result = chunked_op(data, chunk_size)
    end_time = time.time()
    chunked_time = end_time - start_time
    print(f"Chunked operation: {chunked_time:.6f} seconds")
    
    # Compare results if standard operation succeeded
    if standard_success:
        print(f"Results match: {jnp.allclose(standard_result, chunked_result)}")
        speedup = standard_time / chunked_time
        print(f"Chunked speedup: {speedup:.2f}x")


def main():
    """Run all tests."""
    print("=== JAX Optimization Tests ===")
    
    # Run tests
    test_numpy_operations()
    test_jax_basic_operations()
    test_jit_compilation()
    test_vmap()
    test_chunked_operations()
    
    print("\nTests completed!")


if __name__ == "__main__":
    main()