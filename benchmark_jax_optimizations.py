#!/usr/bin/env python
"""Benchmark script for JAX/TPU optimizations in CayleyPy.

This script compares the performance of different implementations:
1. Original implementation
2. JIT-optimized implementation
3. Vectorized implementation
4. Sharded implementation (if TPU available)

It measures performance across different batch sizes and operations.
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    print("JAX not available - using numpy for benchmarks")

# Import CayleyPy components
from cayleypy.jax_tensor_ops import (
    unique_with_indices, isin_via_searchsorted, sort_with_indices,
    batch_matmul, chunked_operation, optimized_chunked_operation,
    batch_isin_via_searchsorted, distributed_batch_matmul,
    memory_efficient_unique
)

from cayleypy.jax_hasher import (
    JAXStateHasher, OptimizedJAXStateHasher, vectorized_hash_states,
    distributed_hash_states, memory_efficient_hash_large_batch,
    benchmark_hash_performance_advanced
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
    _ = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time


def benchmark_tensor_ops(batch_sizes: List[int], 
                        num_iterations: int = 5) -> Dict[str, Dict[int, float]]:
    """Benchmark tensor operations.
    
    Args:
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations for each test
        
    Returns:
        Dictionary of operation -> batch_size -> time
    """
    results = {
        "unique": {},
        "isin": {},
        "sort": {},
        "matmul": {},
        "chunked": {}
    }
    
    for batch_size in batch_sizes:
        print(f"Benchmarking tensor operations with batch size {batch_size}...")
        
        # Generate test data
        if JAX_AVAILABLE:
            array = jnp.arange(batch_size)
            matrix_size = min(1000, batch_size)
            matrix_a = jnp.ones((matrix_size, matrix_size))
            matrix_b = jnp.ones((matrix_size, matrix_size))
        else:
            array = np.arange(batch_size)
            matrix_size = min(1000, batch_size)
            matrix_a = np.ones((matrix_size, matrix_size))
            matrix_b = np.ones((matrix_size, matrix_size))
        
        # Benchmark unique
        times = []
        for _ in range(num_iterations):
            if JAX_AVAILABLE:
                times.append(time_function(unique_with_indices, array, True, True))
            else:
                times.append(time_function(np.unique, array, return_inverse=True, return_counts=True))
        results["unique"][batch_size] = sum(times) / len(times)
        
        # Benchmark isin
        if batch_size > 1:
            test_elements = array[::10]  # Take every 10th element
            times = []
            for _ in range(num_iterations):
                if JAX_AVAILABLE:
                    times.append(time_function(isin_via_searchsorted, array, test_elements))
                else:
                    times.append(time_function(np.isin, array, test_elements))
            results["isin"][batch_size] = sum(times) / len(times)
        
        # Benchmark sort
        times = []
        for _ in range(num_iterations):
            if JAX_AVAILABLE:
                times.append(time_function(sort_with_indices, array))
            else:
                times.append(time_function(lambda x: (np.sort(x), np.argsort(x)), array))
        results["sort"][batch_size] = sum(times) / len(times)
        
        # Benchmark matmul
        times = []
        for _ in range(num_iterations):
            if JAX_AVAILABLE:
                times.append(time_function(batch_matmul, matrix_a, matrix_b))
            else:
                times.append(time_function(np.matmul, matrix_a, matrix_b))
        results["matmul"][batch_size] = sum(times) / len(times)
        
        # Benchmark chunked operation
        def sum_op(x):
            return jnp.sum(x, axis=1) if JAX_AVAILABLE else np.sum(x, axis=1)
        
        if batch_size >= 1000:
            chunk_size = batch_size // 10
            array_2d = array.reshape(-1, 10)
            times = []
            for _ in range(num_iterations):
                if JAX_AVAILABLE:
                    times.append(time_function(chunked_operation, array_2d, sum_op, chunk_size))
                else:
                    times.append(time_function(sum_op, array_2d))
            results["chunked"][batch_size] = sum(times) / len(times)
    
    return results


def benchmark_hash_functions(batch_sizes: List[int], state_size: int = 10,
                           num_iterations: int = 5) -> Dict[str, Dict[int, float]]:
    """Benchmark hash functions.
    
    Args:
        batch_sizes: List of batch sizes to test
        state_size: Size of state vectors
        num_iterations: Number of iterations for each test
        
    Returns:
        Dictionary of method -> batch_size -> time
    """
    if not JAX_AVAILABLE:
        print("JAX not available - skipping hash function benchmarks")
        return {}
    
    results = {
        "standard": {},
        "vectorized": {},
        "optimized": {},
        "distributed": {},
        "memory_efficient": {}
    }
    
    # Create hashers
    standard_hasher = JAXStateHasher(state_size=state_size, random_seed=42)
    optimized_hasher = OptimizedJAXStateHasher(state_size=state_size, random_seed=42)
    
    for batch_size in batch_sizes:
        print(f"Benchmarking hash functions with batch size {batch_size}...")
        
        # Generate test data
        states = jnp.ones((batch_size, state_size), dtype=jnp.int32)
        states = states.at[:, 0].set(jnp.arange(batch_size))  # Make states unique
        
        # Benchmark standard hashing
        times = []
        for _ in range(num_iterations):
            times.append(time_function(standard_hasher.hash_states, states))
        results["standard"][batch_size] = sum(times) / len(times)
        
        # Benchmark vectorized hashing
        times = []
        for _ in range(num_iterations):
            times.append(time_function(vectorized_hash_states, states, standard_hasher))
        results["vectorized"][batch_size] = sum(times) / len(times)
        
        # Benchmark optimized hashing
        times = []
        for _ in range(num_iterations):
            times.append(time_function(optimized_hasher.hash_states_optimized, states))
        results["optimized"][batch_size] = sum(times) / len(times)
        
        # Benchmark distributed hashing (if available)
        try:
            times = []
            for _ in range(num_iterations):
                times.append(time_function(distributed_hash_states, states, standard_hasher))
            results["distributed"][batch_size] = sum(times) / len(times)
        except Exception as e:
            print(f"Distributed hashing not available: {e}")
        
        # Benchmark memory-efficient hashing
        if batch_size >= 1000:
            times = []
            for _ in range(num_iterations):
                times.append(time_function(
                    memory_efficient_hash_large_batch, states, standard_hasher, 0.1))
            results["memory_efficient"][batch_size] = sum(times) / len(times)
    
    return results


def benchmark_optimized_vs_original(batch_sizes: List[int], 
                                  num_iterations: int = 5) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Compare optimized vs original implementations.
    
    Args:
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations for each test
        
    Returns:
        Dictionary of operation -> implementation -> batch_size -> time
    """
    if not JAX_AVAILABLE:
        print("JAX not available - skipping optimization comparison")
        return {}
    
    results = {
        "isin": {"original": {}, "optimized": {}, "batch": {}, "distributed": {}},
        "matmul": {"original": {}, "optimized": {}, "distributed": {}},
        "unique": {"original": {}, "optimized": {}, "memory_efficient": {}}
    }
    
    for batch_size in batch_sizes:
        print(f"Comparing implementations with batch size {batch_size}...")
        
        # Generate test data
        array = jnp.arange(batch_size)
        test_elements = array[::10]  # Take every 10th element
        
        # For batch operations
        if batch_size >= 100:
            batch_count = min(10, batch_size // 10)
            batch_arrays = [jnp.arange(i, i + batch_size // batch_count) for i in range(0, batch_size, batch_size // batch_count)]
        
        # Matrix operations
        matrix_size = min(1000, batch_size)
        matrix_a = jnp.ones((matrix_size, matrix_size))
        matrix_b = jnp.ones((matrix_size, matrix_size))
        
        # isin comparisons
        if batch_size > 1:
            # Original
            times = []
            for _ in range(num_iterations):
                times.append(time_function(
                    lambda x, y: jnp.array([x_i in y for x_i in x]), array, test_elements))
            results["isin"]["original"][batch_size] = sum(times) / len(times)
            
            # Optimized
            times = []
            for _ in range(num_iterations):
                times.append(time_function(isin_via_searchsorted, array, test_elements))
            results["isin"]["optimized"][batch_size] = sum(times) / len(times)
            
            # Batch
            if batch_size >= 100:
                batch_arrays_2d = [arr.reshape(-1, 1) for arr in batch_arrays]
                times = []
                for _ in range(num_iterations):
                    times.append(time_function(batch_isin_via_searchsorted, 
                                             jnp.concatenate(batch_arrays_2d, axis=1), 
                                             test_elements))
                results["isin"]["batch"][batch_size] = sum(times) / len(times)
            
            # Distributed
            try:
                times = []
                for _ in range(num_iterations):
                    times.append(time_function(
                        lambda x, y: distributed_batch_matmul(
                            x.reshape(-1, 1), 
                            jnp.array([(z in y) for z in range(x.shape[0])]).reshape(1, -1)), 
                        array, test_elements))
                results["isin"]["distributed"][batch_size] = sum(times) / len(times)
            except Exception as e:
                print(f"Distributed isin not available: {e}")
        
        # matmul comparisons
        # Original
        times = []
        for _ in range(num_iterations):
            times.append(time_function(jnp.matmul, matrix_a, matrix_b))
        results["matmul"]["original"][batch_size] = sum(times) / len(times)
        
        # Optimized
        times = []
        for _ in range(num_iterations):
            times.append(time_function(batch_matmul, matrix_a, matrix_b))
        results["matmul"]["optimized"][batch_size] = sum(times) / len(times)
        
        # Distributed
        try:
            times = []
            for _ in range(num_iterations):
                times.append(time_function(distributed_batch_matmul, matrix_a, matrix_b))
            results["matmul"]["distributed"][batch_size] = sum(times) / len(times)
        except Exception as e:
            print(f"Distributed matmul not available: {e}")
        
        # unique comparisons
        # Original
        times = []
        for _ in range(num_iterations):
            times.append(time_function(jnp.unique, array))
        results["unique"]["original"][batch_size] = sum(times) / len(times)
        
        # Optimized
        times = []
        for _ in range(num_iterations):
            times.append(time_function(unique_with_indices, array))
        results["unique"]["optimized"][batch_size] = sum(times) / len(times)
        
        # Memory efficient
        if batch_size >= 1000:
            times = []
            for _ in range(num_iterations):
                times.append(time_function(memory_efficient_unique, array, 0.1))
            results["unique"]["memory_efficient"][batch_size] = sum(times) / len(times)
    
    return results


def plot_results(results: Dict[str, Dict[int, float]], 
                title: str, 
                ylabel: str = "Time (seconds)",
                log_scale: bool = True,
                save_path: Optional[str] = None):
    """Plot benchmark results.
    
    Args:
        results: Dictionary of method -> batch_size -> time
        title: Plot title
        ylabel: Y-axis label
        log_scale: Whether to use log scale for y-axis
        save_path: Path to save plot (if None, display plot)
    """
    plt.figure(figsize=(12, 8))
    
    for method, batch_times in results.items():
        batch_sizes = sorted(batch_times.keys())
        times = [batch_times[size] for size in batch_sizes]
        plt.plot(batch_sizes, times, marker='o', label=method)
    
    plt.title(title)
    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_comparison(results: Dict[str, Dict[str, Dict[int, float]]],
                   title: str,
                   save_path: Optional[str] = None):
    """Plot comparison of implementations.
    
    Args:
        results: Dictionary of operation -> implementation -> batch_size -> time
        title: Plot title
        save_path: Path to save plot (if None, display plot)
    """
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 6 * len(results)))
    
    for i, (operation, implementations) in enumerate(results.items()):
        ax = axes[i] if len(results) > 1 else axes
        
        for impl, batch_times in implementations.items():
            batch_sizes = sorted(batch_times.keys())
            times = [batch_times[size] for size in batch_sizes]
            ax.plot(batch_sizes, times, marker='o', label=impl)
        
        ax.set_title(f"{title} - {operation}")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Time (seconds)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def calculate_speedups(results: Dict[str, Dict[int, float]], 
                      baseline: str = "standard") -> Dict[str, Dict[int, float]]:
    """Calculate speedups relative to baseline.
    
    Args:
        results: Dictionary of method -> batch_size -> time
        baseline: Baseline method to compare against
        
    Returns:
        Dictionary of method -> batch_size -> speedup
    """
    speedups = {}
    
    for method, batch_times in results.items():
        if method != baseline:
            speedups[method] = {}
            for batch_size, time in batch_times.items():
                if batch_size in results[baseline]:
                    baseline_time = results[baseline][batch_size]
                    speedups[method][batch_size] = baseline_time / time
    
    return speedups


def print_speedup_table(speedups: Dict[str, Dict[int, float]]):
    """Print speedup table.
    
    Args:
        speedups: Dictionary of method -> batch_size -> speedup
    """
    # Get all batch sizes
    all_batch_sizes = set()
    for method_speedups in speedups.values():
        all_batch_sizes.update(method_speedups.keys())
    batch_sizes = sorted(all_batch_sizes)
    
    # Print header
    print("\nSpeedup Table (higher is better):")
    header = "Method"
    for size in batch_sizes:
        header += f" | {size:,}"
    print(header)
    print("-" * len(header))
    
    # Print rows
    for method, method_speedups in speedups.items():
        row = method
        for size in batch_sizes:
            if size in method_speedups:
                row += f" | {method_speedups[size]:.2f}x"
            else:
                row += " | -"
        print(row)


def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark JAX/TPU optimizations")
    parser.add_argument("--batch-sizes", type=str, default="100,1000,10000,100000",
                      help="Comma-separated list of batch sizes to test")
    parser.add_argument("--iterations", type=int, default=3,
                      help="Number of iterations for each test")
    parser.add_argument("--save-plots", action="store_true",
                      help="Save plots instead of displaying them")
    parser.add_argument("--state-size", type=int, default=10,
                      help="Size of state vectors for hash benchmarks")
    parser.add_argument("--skip-tensor-ops", action="store_true",
                      help="Skip tensor operation benchmarks")
    parser.add_argument("--skip-hash-funcs", action="store_true",
                      help="Skip hash function benchmarks")
    parser.add_argument("--skip-comparisons", action="store_true",
                      help="Skip implementation comparisons")
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    
    # Run benchmarks
    if not args.skip_tensor_ops:
        tensor_results = benchmark_tensor_ops(batch_sizes, args.iterations)
        if args.save_plots:
            plot_results(tensor_results, "Tensor Operations Benchmark", 
                       save_path="tensor_ops_benchmark.png")
        else:
            plot_results(tensor_results, "Tensor Operations Benchmark")
    
    if not args.skip_hash_funcs and JAX_AVAILABLE:
        hash_results = benchmark_hash_functions(batch_sizes, args.state_size, args.iterations)
        if hash_results:
            hash_speedups = calculate_speedups(hash_results)
            print_speedup_table(hash_speedups)
            
            if args.save_plots:
                plot_results(hash_results, "Hash Functions Benchmark",
                           save_path="hash_funcs_benchmark.png")
            else:
                plot_results(hash_results, "Hash Functions Benchmark")
    
    if not args.skip_comparisons and JAX_AVAILABLE:
        comparison_results = benchmark_optimized_vs_original(batch_sizes, args.iterations)
        if comparison_results:
            if args.save_plots:
                plot_comparison(comparison_results, "Implementation Comparison",
                              save_path="implementation_comparison.png")
            else:
                plot_comparison(comparison_results, "Implementation Comparison")
            
            # Calculate and print speedups for each operation
            for operation, implementations in comparison_results.items():
                speedups = calculate_speedups(implementations, "original")
                print(f"\nSpeedups for {operation}:")
                print_speedup_table(speedups)


if __name__ == "__main__":
    main()