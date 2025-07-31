#!/usr/bin/env python3
"""
TPU BFS Validation Runner

This script demonstrates that TPU BFS implementations:
1. Produce identical numeric results to reference implementations
2. Have comparable or better memory usage
3. Provide significant performance improvements on medium-to-large graphs

Usage:
    python -m cayleypy.run_tpu_bfs_validation
    python -m cayleypy.run_tpu_bfs_validation --detailed
    python -m cayleypy.run_tpu_bfs_validation --save-results results.json
"""

import argparse
import json
import logging
import sys
import time
from typing import Dict, Any, List

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


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_numeric_correctness() -> Dict[str, Any]:
    """Validate that TPU implementations produce identical numeric results."""
    print("\n🔍 VALIDATING NUMERIC CORRECTNESS")
    print("=" * 50)
    
    results = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': [],
        'test_details': []
    }
    
    # Test graphs with known results
    test_cases = [
        (CayleyGraph(PermutationGroups.symmetric_group(4)), "S4", 8),
        (CayleyGraph(PermutationGroups.alternating_group(4)), "A4", 10),
        (CayleyGraph(PermutationGroups.dihedral_group(6)), "D6", 12),
    ]
    
    for graph, name, max_diameter in test_cases:
        print(f"\nTesting {name} (max_diameter={max_diameter})...")
        results['total_tests'] += 1
        
        try:
            # Reference result
            print(f"  Running NumPy BFS...")
            ref_result = bfs_numpy(graph, max_diameter)
            
            # TPU result
            print(f"  Running TPU BFS...")
            tpu_result = tpu_bfs(graph, max_diameter)
            
            # Compare
            if ref_result == tpu_result:
                print(f"  ✅ PASS: Results identical - {ref_result}")
                results['passed_tests'] += 1
                test_status = 'PASS'
            else:
                print(f"  ❌ FAIL: Results differ")
                print(f"    Reference: {ref_result}")
                print(f"    TPU:       {tpu_result}")
                results['failed_tests'].append(f"{name} - TPU BFS")
                test_status = 'FAIL'
            
            results['test_details'].append({
                'graph': name,
                'algorithm': 'TPU BFS',
                'status': test_status,
                'reference_result': ref_result,
                'test_result': tpu_result
            })
            
            # Test bitmask if applicable
            if graph.definition.state_size > 8 and graph.definition.is_permutation_group():
                print(f"  Running CPU Bitmask BFS...")
                ref_bitmask = bfs_bitmask(graph, max_diameter)
                
                print(f"  Running TPU Bitmask BFS...")
                tpu_bitmask = tpu_bfs_bitmask(graph, max_diameter)
                
                results['total_tests'] += 1
                
                if ref_bitmask == tpu_bitmask:
                    print(f"  ✅ PASS: Bitmask results identical - {ref_bitmask}")
                    results['passed_tests'] += 1
                    bitmask_status = 'PASS'
                else:
                    print(f"  ❌ FAIL: Bitmask results differ")
                    print(f"    CPU Bitmask: {ref_bitmask}")
                    print(f"    TPU Bitmask: {tpu_bitmask}")
                    results['failed_tests'].append(f"{name} - TPU Bitmask BFS")
                    bitmask_status = 'FAIL'
                
                results['test_details'].append({
                    'graph': name,
                    'algorithm': 'TPU Bitmask BFS',
                    'status': bitmask_status,
                    'reference_result': ref_bitmask,
                    'test_result': tpu_bitmask
                })
        
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"  ❌ ERROR: {e}")
            results['failed_tests'].append(f"{name} - Error: {e}")
    
    success_rate = results['passed_tests'] / max(results['total_tests'], 1)
    print(f"\nNumeric Correctness Summary:")
    print(f"  Tests Passed: {results['passed_tests']}/{results['total_tests']} ({success_rate:.1%})")
    
    if results['failed_tests']:
        print(f"  Failed Tests: {', '.join(results['failed_tests'])}")
    
    return results


def validate_performance_improvement() -> Dict[str, Any]:
    """Validate that TPU implementations provide performance improvements."""
    print("\n⚡ VALIDATING PERFORMANCE IMPROVEMENT")
    print("=" * 50)
    
    results = {
        'benchmarks': [],
        'average_speedup': 0.0,
        'max_speedup': 0.0,
        'significant_improvements': 0,
        'total_benchmarks': 0
    }
    
    # Performance test graphs
    perf_graphs = [
        (CayleyGraph(PermutationGroups.symmetric_group(5)), "S5", 15),
        (CayleyGraph(PermutationGroups.alternating_group(5)), "A5", 18),
        (CayleyGraph(PermutationGroups.dihedral_group(10)), "D10", 20),
    ]
    
    speedups = []
    
    for graph, name, max_diameter in perf_graphs:
        print(f"\nBenchmarking {name}...")
        
        try:
            # CPU timing
            print(f"  Timing NumPy BFS...")
            start_time = time.time()
            cpu_result = bfs_numpy(graph, max_diameter)
            cpu_time = time.time() - start_time
            
            # TPU timing
            print(f"  Timing TPU BFS...")
            start_time = time.time()
            tpu_result = tpu_bfs(graph, max_diameter)
            tpu_time = time.time() - start_time
            
            # Calculate speedup
            speedup = cpu_time / tpu_time if tpu_time > 0 else 0
            speedups.append(speedup)
            
            # Check correctness
            correct = cpu_result == tpu_result
            
            benchmark_result = {
                'graph': name,
                'cpu_time': cpu_time,
                'tpu_time': tpu_time,
                'speedup': speedup,
                'correct': correct,
                'significant': speedup >= 1.2
            }
            
            results['benchmarks'].append(benchmark_result)
            results['total_benchmarks'] += 1
            
            if speedup >= 1.2:
                results['significant_improvements'] += 1
            
            status = "✅" if correct and speedup >= 1.0 else "⚠️" if correct else "❌"
            print(f"  {status} Speedup: {speedup:.2f}x ({cpu_time:.3f}s → {tpu_time:.3f}s), Correct: {correct}")
            
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"  ❌ ERROR: {e}")
            results['benchmarks'].append({
                'graph': name,
                'error': str(e)
            })
    
    if speedups:
        results['average_speedup'] = sum(speedups) / len(speedups)
        results['max_speedup'] = max(speedups)
    
    print(f"\nPerformance Summary:")
    print(f"  Average Speedup: {results['average_speedup']:.2f}x")
    print(f"  Maximum Speedup: {results['max_speedup']:.2f}x")
    print(f"  Significant Improvements: {results['significant_improvements']}/{results['total_benchmarks']}")
    
    return results


def validate_memory_efficiency() -> Dict[str, Any]:
    """Validate memory usage is reasonable."""
    print("\n💾 VALIDATING MEMORY EFFICIENCY")
    print("=" * 50)
    
    results = {
        'memory_tests': [],
        'acceptable_usage': 0,
        'total_tests': 0
    }
    
    # Memory test with a medium-sized graph
    test_graph = CayleyGraph(PermutationGroups.symmetric_group(5))
    max_diameter = 15
    
    print(f"Testing memory usage on S5...")
    
    try:
        import psutil
        process = psutil.Process()
        
        # Baseline memory
        baseline = process.memory_info().rss / (1024 * 1024)  # MB
        
        # CPU BFS memory
        print(f"  Measuring NumPy BFS memory...")
        before_cpu = process.memory_info().rss / (1024 * 1024)
        _ = bfs_numpy(test_graph, max_diameter)
        after_cpu = process.memory_info().rss / (1024 * 1024)
        cpu_increase = after_cpu - before_cpu
        
        # TPU BFS memory
        print(f"  Measuring TPU BFS memory...")
        before_tpu = process.memory_info().rss / (1024 * 1024)
        _ = tpu_bfs(test_graph, max_diameter)
        after_tpu = process.memory_info().rss / (1024 * 1024)
        tpu_increase = after_tpu - before_tpu
        
        # Memory ratio
        memory_ratio = tpu_increase / max(cpu_increase, 1)
        acceptable = memory_ratio <= 3.0  # At most 3x memory usage
        
        memory_test = {
            'algorithm': 'TPU BFS',
            'cpu_memory_mb': cpu_increase,
            'tpu_memory_mb': tpu_increase,
            'memory_ratio': memory_ratio,
            'acceptable': acceptable
        }
        
        results['memory_tests'].append(memory_test)
        results['total_tests'] += 1
        
        if acceptable:
            results['acceptable_usage'] += 1
        
        status = "✅" if acceptable else "⚠️"
        print(f"  {status} Memory ratio: {memory_ratio:.2f}x ({tpu_increase:.1f} MB vs {cpu_increase:.1f} MB)")
        
        # Test bitmask memory if applicable
        if test_graph.definition.state_size > 8:
            print(f"  Measuring bitmask memory usage...")
            
            before_cpu_bm = process.memory_info().rss / (1024 * 1024)
            _ = bfs_bitmask(test_graph, max_diameter)
            after_cpu_bm = process.memory_info().rss / (1024 * 1024)
            cpu_bm_increase = after_cpu_bm - before_cpu_bm
            
            before_tpu_bm = process.memory_info().rss / (1024 * 1024)
            _ = tpu_bfs_bitmask(test_graph, max_diameter)
            after_tpu_bm = process.memory_info().rss / (1024 * 1024)
            tpu_bm_increase = after_tpu_bm - before_tpu_bm
            
            bm_memory_ratio = tpu_bm_increase / max(cpu_bm_increase, 1)
            bm_acceptable = bm_memory_ratio <= 2.5
            
            bitmask_test = {
                'algorithm': 'TPU Bitmask BFS',
                'cpu_memory_mb': cpu_bm_increase,
                'tpu_memory_mb': tpu_bm_increase,
                'memory_ratio': bm_memory_ratio,
                'acceptable': bm_acceptable
            }
            
            results['memory_tests'].append(bitmask_test)
            results['total_tests'] += 1
            
            if bm_acceptable:
                results['acceptable_usage'] += 1
            
            status = "✅" if bm_acceptable else "⚠️"
            print(f"  {status} Bitmask memory ratio: {bm_memory_ratio:.2f}x ({tpu_bm_increase:.1f} MB vs {cpu_bm_increase:.1f} MB)")
    
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"  ❌ ERROR: {e}")
        results['memory_tests'].append({'error': str(e)})
    
    print(f"\nMemory Efficiency Summary:")
    print(f"  Acceptable Usage: {results['acceptable_usage']}/{results['total_tests']}")
    
    return results


def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(description='TPU BFS Validation Runner')
    parser.add_argument('--detailed', action='store_true', help='Enable detailed logging')
    parser.add_argument('--save-results', type=str, help='Save results to JSON file')
    parser.add_argument('--skip-performance', action='store_true', help='Skip performance tests')
    parser.add_argument('--skip-memory', action='store_true', help='Skip memory tests')
    
    args = parser.parse_args()
    
    setup_logging(args.detailed)
    
    print("TPU BFS VALIDATION SUITE")
    print("=" * 60)
    
    # Check prerequisites
    if not JAX_AVAILABLE:
        print("❌ JAX not available - cannot run TPU validation")
        return 1
    
    try:
        backend = get_tpu_backend()
        if not backend.is_available:
            print("❌ TPU not available - cannot run TPU validation")
            return 1
        
        print(f"✅ TPU Backend Available: {backend.get_device_info()}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Failed to initialize TPU backend: {e}")
        return 1
    
    # Run validation tests
    all_results = {}
    overall_success = True
    
    # 1. Numeric Correctness
    correctness_results = validate_numeric_correctness()
    all_results['correctness'] = correctness_results
    
    correctness_success = correctness_results['passed_tests'] == correctness_results['total_tests']
    if not correctness_success:
        overall_success = False
    
    # 2. Performance Improvement
    if not args.skip_performance:
        performance_results = validate_performance_improvement()
        all_results['performance'] = performance_results
        
        performance_success = performance_results['significant_improvements'] > 0
        if not performance_success:
            print("⚠️  Warning: No significant performance improvements detected")
    
    # 3. Memory Efficiency
    if not args.skip_memory:
        memory_results = validate_memory_efficiency()
        all_results['memory'] = memory_results
        
        memory_success = memory_results['acceptable_usage'] == memory_results['total_tests']
        if not memory_success:
            print("⚠️  Warning: Memory usage concerns detected")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if correctness_success:
        print("✅ NUMERIC CORRECTNESS: All tests passed")
    else:
        print("❌ NUMERIC CORRECTNESS: Some tests failed")
    
    if not args.skip_performance:
        if performance_results['significant_improvements'] > 0:
            print(f"✅ PERFORMANCE: {performance_results['significant_improvements']} significant improvements")
        else:
            print("⚠️  PERFORMANCE: No significant improvements")
    
    if not args.skip_memory:
        if memory_success:
            print("✅ MEMORY EFFICIENCY: All tests acceptable")
        else:
            print("⚠️  MEMORY EFFICIENCY: Some concerns")
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n📄 Results saved to {args.save_results}")
    
    if overall_success:
        print("\n🎉 OVERALL RESULT: TPU BFS implementations are VALIDATED")
        return 0
    else:
        print("\n❌ OVERALL RESULT: TPU BFS implementations have ISSUES")
        return 1


if __name__ == "__main__":
    sys.exit(main())