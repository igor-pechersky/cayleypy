#!/usr/bin/env python3
"""
Performance demonstration for TPU BFS on large graphs.

This script demonstrates that TPU BFS provides significant performance improvements
on medium-to-large graphs where the computational workload justifies the compilation overhead.
"""

import time
import logging
from typing import Dict, Any, List, Tuple

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore

from .tpu_backend import get_tpu_backend
from .tpu_bfs import tpu_bfs
from .tpu_bfs_bitmask import tpu_bfs_bitmask
from .bfs_numpy import bfs_numpy
from .bfs_bitmask import bfs_bitmask
from .graphs_lib import PermutationGroups
from .cayley_graph import CayleyGraph


def setup_logging():
    """Set up logging for performance testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def benchmark_graph(graph: CayleyGraph, name: str, max_diameter: int, use_bitmask: bool = False) -> Dict[str, Any]:
    """Benchmark a single graph with both CPU and TPU implementations."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {name}")
    print(f"Graph size: {len(graph.central_state)} elements")
    print(f"Generators: {len(graph.definition.generators_permutations)}")
    print(f"Max diameter: {max_diameter}")
    print(f"Use bitmask: {use_bitmask}")
    print(f"{'='*60}")
    
    results = {
        'graph_name': name,
        'state_size': len(graph.central_state),
        'generators_count': len(graph.definition.generators_permutations),
        'max_diameter': max_diameter,
        'use_bitmask': use_bitmask,
        'cpu_time': 0.0,
        'tpu_time': 0.0,
        'speedup': 0.0,
        'cpu_result': [],
        'tpu_result': [],
        'results_identical': False,
        'tpu_compilation_time': 0.0,
        'tpu_execution_time': 0.0
    }
    
    # Choose the appropriate algorithms
    if use_bitmask and graph.definition.state_size > 8 and graph.definition.is_permutation_group():
        cpu_func = bfs_bitmask
        tpu_func = tpu_bfs_bitmask
        algorithm_type = "Bitmask BFS"
    else:
        cpu_func = bfs_numpy
        tpu_func = tpu_bfs
        algorithm_type = "Regular BFS"
    
    print(f"Algorithm: {algorithm_type}")
    
    try:
        # CPU Benchmark
        print(f"\n🖥️  Running CPU {algorithm_type}...")
        start_time = time.time()
        cpu_result = cpu_func(graph, max_diameter)
        cpu_time = time.time() - start_time
        
        results['cpu_time'] = cpu_time
        results['cpu_result'] = cpu_result
        
        print(f"   ✅ CPU completed in {cpu_time:.3f}s")
        print(f"   📊 Growth function: {cpu_result}")
        
        # TPU Benchmark (including compilation time)
        print(f"\n🚀 Running TPU {algorithm_type}...")
        
        # First run includes compilation time
        total_start_time = time.time()
        tpu_result = tpu_func(graph, max_diameter)
        total_tpu_time = time.time() - total_start_time
        
        # Second run to measure execution time without compilation
        print(f"   🔄 Running second iteration to measure execution time...")
        exec_start_time = time.time()
        tpu_result_2 = tpu_func(graph, max_diameter)
        exec_time = time.time() - exec_start_time
        
        # Verify both runs are identical
        if tpu_result != tpu_result_2:
            print(f"   ⚠️  Warning: TPU results differ between runs!")
        
        results['tpu_time'] = total_tpu_time
        results['tpu_result'] = tpu_result
        results['tpu_compilation_time'] = total_tpu_time - exec_time
        results['tpu_execution_time'] = exec_time
        
        print(f"   ✅ TPU completed in {total_tpu_time:.3f}s (total)")
        print(f"   🔧 Compilation time: {results['tpu_compilation_time']:.3f}s")
        print(f"   ⚡ Execution time: {exec_time:.3f}s")
        print(f"   📊 Growth function: {tpu_result}")
        
        # Verify correctness
        results['results_identical'] = (cpu_result == tpu_result)
        
        if results['results_identical']:
            print(f"   ✅ Results are IDENTICAL")
        else:
            print(f"   ❌ Results DIFFER!")
            print(f"      CPU: {cpu_result}")
            print(f"      TPU: {tpu_result}")
        
        # Calculate speedups
        total_speedup = cpu_time / total_tpu_time if total_tpu_time > 0 else 0
        exec_speedup = cpu_time / exec_time if exec_time > 0 else 0
        
        results['speedup'] = total_speedup
        results['execution_speedup'] = exec_speedup
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"   Total speedup: {total_speedup:.2f}x")
        print(f"   Execution speedup: {exec_speedup:.2f}x")
        print(f"   Compilation overhead: {results['tpu_compilation_time']:.3f}s")
        
        if exec_speedup > 1.0:
            print(f"   🎉 TPU execution is {exec_speedup:.2f}x FASTER!")
        elif total_speedup > 0.8:
            print(f"   ✅ TPU total performance is competitive ({total_speedup:.2f}x)")
        else:
            print(f"   ⚠️  TPU performance needs optimization")
        
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")
        results['error'] = str(e)
    
    return results


def run_large_graph_performance_demo():
    """Run performance demonstration on progressively larger graphs."""
    print("TPU BFS LARGE GRAPH PERFORMANCE DEMONSTRATION")
    print("=" * 80)
    
    if not JAX_AVAILABLE:
        print("❌ JAX not available - cannot run TPU performance demo")
        return False
    
    try:
        backend = get_tpu_backend()
        if not backend.is_available:
            print("❌ TPU not available - cannot run performance demo")
            return False
        
        print(f"✅ TPU Backend: {backend.get_device_info()}")
        
    except Exception as e:
        print(f"❌ Failed to initialize TPU backend: {e}")
        return False
    
    # Define test graphs of increasing size
    test_graphs = [
        # Medium graphs where TPU should start showing benefits
        (PermutationGroups.coxeter(6), "S6_Coxeter", 20, False),
        (PermutationGroups.pancake(7), "Pancake_7", 25, False),
        (PermutationGroups.all_transpositions(6), "S6_AllTrans", 15, False),
        
        # Large graphs where TPU should excel
        (PermutationGroups.coxeter(7), "S7_Coxeter", 25, False),
        (PermutationGroups.pancake(8), "Pancake_8", 30, False),
        
        # Very large graphs suitable for bitmask
        (PermutationGroups.coxeter(9), "S9_Coxeter_Bitmask", 20, True),
        (PermutationGroups.pancake(9), "Pancake_9_Bitmask", 25, True),
    ]
    
    all_results = []
    successful_benchmarks = 0
    performance_improvements = 0
    
    for graph_def, name, max_diameter, use_bitmask in test_graphs:
        try:
            graph = CayleyGraph(graph_def)
            
            # Skip if graph is too large for available memory
            estimated_states = 1
            for i in range(1, graph.definition.state_size + 1):
                estimated_states *= i
                if estimated_states > 10**6:  # Limit to ~1M states for demo
                    print(f"\n⚠️  Skipping {name} - estimated {estimated_states} states too large for demo")
                    break
            else:
                result = benchmark_graph(graph, name, max_diameter, use_bitmask)
                all_results.append(result)
                successful_benchmarks += 1
                
                if result.get('execution_speedup', 0) > 1.0:
                    performance_improvements += 1
                
        except Exception as e:
            print(f"\n❌ Failed to benchmark {name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("PERFORMANCE DEMONSTRATION SUMMARY")
    print(f"{'='*80}")
    
    print(f"📊 Benchmarks completed: {successful_benchmarks}")
    print(f"🚀 Performance improvements: {performance_improvements}")
    
    if successful_benchmarks > 0:
        print(f"\n📈 DETAILED RESULTS:")
        print(f"{'Graph':<20} {'CPU Time':<10} {'TPU Exec':<10} {'Speedup':<10} {'Correct':<8}")
        print(f"{'-'*60}")
        
        for result in all_results:
            if 'error' not in result:
                name = result['graph_name'][:19]
                cpu_time = f"{result['cpu_time']:.3f}s"
                tpu_exec = f"{result.get('tpu_execution_time', 0):.3f}s"
                speedup = f"{result.get('execution_speedup', 0):.2f}x"
                correct = "✅" if result['results_identical'] else "❌"
                
                print(f"{name:<20} {cpu_time:<10} {tpu_exec:<10} {speedup:<10} {correct:<8}")
        
        # Calculate averages
        valid_results = [r for r in all_results if 'error' not in r and r.get('execution_speedup', 0) > 0]
        if valid_results:
            avg_speedup = sum(r.get('execution_speedup', 0) for r in valid_results) / len(valid_results)
            max_speedup = max(r.get('execution_speedup', 0) for r in valid_results)
            
            print(f"\n🎯 PERFORMANCE SUMMARY:")
            print(f"   Average execution speedup: {avg_speedup:.2f}x")
            print(f"   Maximum execution speedup: {max_speedup:.2f}x")
            print(f"   Graphs with speedup > 1.0x: {performance_improvements}/{successful_benchmarks}")
            
            if avg_speedup > 1.0:
                print(f"   🎉 TPU shows clear performance benefits on larger graphs!")
            else:
                print(f"   ⚠️  Performance benefits not yet evident - may need even larger graphs")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • TPU performance improves with graph size due to better parallelization")
    print(f"   • Compilation overhead is amortized over larger computations")
    print(f"   • For production use, consider batch processing multiple graphs")
    print(f"   • Bitmask approach is most beneficial for very large permutation groups")
    
    success = performance_improvements > 0
    if success:
        print(f"\n✅ DEMONSTRATION SUCCESSFUL: TPU shows performance benefits on larger graphs")
    else:
        print(f"\n⚠️  DEMONSTRATION INCONCLUSIVE: May need even larger graphs to show benefits")
    
    return success


if __name__ == "__main__":
    setup_logging()
    success = run_large_graph_performance_demo()
    exit(0 if success else 1)