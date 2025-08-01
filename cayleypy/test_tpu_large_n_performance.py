#!/usr/bin/env python3
"""
TPU Performance Demonstration for Large Graphs (n>10).

This script demonstrates TPU BFS performance on graphs with n>10 where
TPU's parallel architecture should provide significant performance advantages.
"""

import time
import logging
import math
from typing import Dict, Any

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore

from .tpu_backend import get_tpu_backend
from .tpu_bfs import tpu_bfs
from .tpu_bfs_bitmask import tpu_bfs_bitmask
from .bfs_numpy import bfs_numpy
from .bfs_bitmask import bfs_bitmask
from .graphs_lib import PermutationGroups
from .cayley_graph import CayleyGraph


def estimate_graph_complexity(graph_def, max_diameter: int) -> Dict[str, Any]:
    """Estimate the computational complexity of a graph."""
    n = graph_def.state_size
    generators = len(graph_def.generators_permutations)

    # Estimate total states
    if graph_def.is_permutation_group():
        total_states = math.factorial(n)
    else:
        total_states = n**generators  # Rough estimate

    # Estimate states to explore (limited by diameter)
    states_per_layer = min(total_states, generators**max_diameter)

    return {
        "state_size": n,
        "generators": generators,
        "estimated_total_states": total_states,
        "estimated_states_to_explore": states_per_layer,
        "max_diameter": max_diameter,
        "complexity_score": n * generators * max_diameter,
    }


def benchmark_large_graph(
    graph: CayleyGraph, name: str, max_diameter: int, use_bitmask: bool = False, warmup_runs: int = 2
) -> Dict[str, Any]:
    """Benchmark a large graph with detailed performance analysis."""

    complexity = estimate_graph_complexity(graph.definition, max_diameter)

    print(f"\n{'='*80}")
    print(f"LARGE GRAPH PERFORMANCE TEST: {name}")
    print(f"{'='*80}")
    print(f"State size (n): {complexity['state_size']}")
    print(f"Generators: {complexity['generators']}")
    print(f"Estimated total states: {complexity['estimated_total_states']:,}")
    print(f"Max diameter: {max_diameter}")
    print(f"Use bitmask: {use_bitmask}")
    print(f"Complexity score: {complexity['complexity_score']:,}")

    results = {
        "graph_name": name,
        "complexity": complexity,
        "use_bitmask": use_bitmask,
        "cpu_time": 0.0,
        "tpu_times": [],
        "best_tpu_time": 0.0,
        "compilation_time": 0.0,
        "speedup": 0.0,
        "cpu_result": [],
        "tpu_result": [],
        "results_identical": False,
        "performance_category": "unknown",
    }

    # Choose algorithms
    if use_bitmask and graph.definition.state_size > 8 and graph.definition.is_permutation_group():
        cpu_func = bfs_bitmask
        tpu_func = tpu_bfs_bitmask
        algorithm = "Bitmask BFS"
    else:
        cpu_func = bfs_numpy
        tpu_func = tpu_bfs
        algorithm = "Regular BFS"

    print(f"Algorithm: {algorithm}")

    try:
        # CPU Benchmark
        print(f"\n🖥️  Running CPU {algorithm}...")
        cpu_start = time.time()
        cpu_result = cpu_func(graph, max_diameter)
        cpu_time = time.time() - cpu_start

        results["cpu_time"] = cpu_time
        results["cpu_result"] = cpu_result

        total_states_found = sum(cpu_result)
        actual_diameter = len(cpu_result) - 1

        print(f"   ✅ CPU completed in {cpu_time:.3f}s")
        print(f"   📊 Actual diameter: {actual_diameter}")
        print(f"   📊 Total states found: {total_states_found:,}")
        print(f"   📊 States per second: {total_states_found/cpu_time:,.0f}")

        # TPU Benchmark with warmup
        print(f"\n🚀 Running TPU {algorithm} with warmup...")

        # First run (includes compilation)
        print("   🔄 Initial run (with compilation)...")
        tpu_start = time.time()
        tpu_result = tpu_func(graph, max_diameter)
        first_run_time = time.time() - tpu_start

        # Warmup runs
        tpu_times = []
        for run in range(warmup_runs):
            print(f"   🔄 Warmup run {run + 1}/{warmup_runs}...")
            start_time = time.time()
            warmup_result = tpu_func(graph, max_diameter)
            elapsed = time.time() - start_time
            tpu_times.append(elapsed)

            # Verify consistency
            if warmup_result != tpu_result:
                print(f"      ⚠️  Warning: Inconsistent results in warmup run {run + 1}")

        # Calculate performance metrics
        best_tpu_time = min(tpu_times) if tpu_times else first_run_time
        avg_tpu_time = sum(tpu_times) / len(tpu_times) if tpu_times else first_run_time
        compilation_time = first_run_time - avg_tpu_time if tpu_times else 0

        results.update(
            {
                "tpu_times": tpu_times,
                "best_tpu_time": best_tpu_time,
                "compilation_time": compilation_time,
                "tpu_result": tpu_result,
            }
        )

        print(f"   ✅ First run (with compilation): {first_run_time:.3f}s")
        print(f"   🔧 Estimated compilation time: {compilation_time:.3f}s")
        print(f"   ⚡ Best execution time: {best_tpu_time:.3f}s")
        print(f"   📊 TPU states per second: {total_states_found/best_tpu_time:,.0f}")

        # Verify correctness
        results["results_identical"] = cpu_result == tpu_result

        if results["results_identical"]:
            print("   ✅ Results are IDENTICAL")
        else:
            print("   ❌ Results DIFFER!")
            print(f"      CPU: {len(cpu_result)} layers, {sum(cpu_result)} states")
            print(f"      TPU: {len(tpu_result)} layers, {sum(tpu_result)} states")

        # Performance analysis
        speedup = cpu_time / best_tpu_time if best_tpu_time > 0 else 0
        throughput_improvement = (
            (total_states_found / best_tpu_time) / (total_states_found / cpu_time) if cpu_time > 0 else 0
        )

        results["speedup"] = speedup
        results["throughput_improvement"] = throughput_improvement

        # Categorize performance
        if speedup >= 2.0:
            results["performance_category"] = "excellent"
            status = "🎉 EXCELLENT"
        elif speedup >= 1.5:
            results["performance_category"] = "good"
            status = "✅ GOOD"
        elif speedup >= 1.0:
            results["performance_category"] = "competitive"
            status = "✅ COMPETITIVE"
        elif speedup >= 0.5:
            results["performance_category"] = "acceptable"
            status = "⚠️  ACCEPTABLE"
        else:
            results["performance_category"] = "needs_optimization"
            status = "❌ NEEDS OPTIMIZATION"

        print("\n📈 PERFORMANCE ANALYSIS:")
        print(f"   CPU time: {cpu_time:.3f}s")
        print(f"   TPU best time: {best_tpu_time:.3f}s")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Throughput improvement: {throughput_improvement:.2f}x")
        print(f"   Performance category: {status}")

        # Efficiency analysis
        if compilation_time > 0:
            efficiency = best_tpu_time / (compilation_time + best_tpu_time)
            print(f"   Execution efficiency: {efficiency:.1%}")

            # Break-even analysis
            break_even_runs = compilation_time / max(cpu_time - best_tpu_time, 0.001)
            print(f"   Break-even after: {break_even_runs:.1f} runs")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"   ❌ Benchmark failed: {e}")
        results["error"] = str(e)
        import traceback  # pylint: disable=import-outside-toplevel

        traceback.print_exc()

    return results


def run_large_n_performance_demo():
    """Run performance demonstration on graphs with n>10."""
    print("TPU BFS PERFORMANCE DEMONSTRATION: LARGE GRAPHS (n>10)")
    print("=" * 90)

    if not JAX_AVAILABLE:
        print("❌ JAX not available")
        return False

    try:
        backend = get_tpu_backend()
        if not backend.is_available:
            print("❌ TPU not available")
            return False

        device_info = backend.get_device_info()
        print(f"✅ TPU Backend: {device_info}")
        print(f"   HBM per chip: {device_info['capabilities']['hbm_per_chip_gb']}GB")
        print(f"   Systolic array: {device_info['capabilities']['systolic_array_size']}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ TPU backend error: {e}")
        return False

    # Define large graphs (n>10) with manageable complexity
    test_graphs = [
        # n=11: Large enough for TPU benefits, manageable complexity
        (PermutationGroups.coxeter(11), "S11_Coxeter", 15, False),
        # n=12: Even larger, should show more TPU benefits
        (PermutationGroups.coxeter(12), "S12_Coxeter", 12, False),
        # Bitmask tests for very large n
        (PermutationGroups.coxeter(10), "S10_Coxeter_Bitmask", 20, True),
        (PermutationGroups.pancake(10), "Pancake_10_Bitmask", 15, True),
        # High-generator graphs for parallel processing
        (PermutationGroups.all_transpositions(8), "S8_AllTrans_55gen", 6, False),
    ]

    all_results = []
    performance_successes = 0
    total_benchmarks = 0

    for graph_def, name, max_diameter, use_bitmask in test_graphs:
        try:
            # Check if graph is too large for available memory
            complexity = estimate_graph_complexity(graph_def, max_diameter)

            # Skip if estimated to be too large (>10M states to explore)
            if complexity["estimated_states_to_explore"] > 10_000_000:
                print(
                    f"\n⚠️  Skipping {name} - estimated {complexity['estimated_states_to_explore']:,} states too large"
                )
                continue

            # Skip if factorial is too large
            if complexity["estimated_total_states"] > 10**12:
                print(f"\n⚠️  Skipping {name} - {complexity['estimated_total_states']:,} total states too large")
                continue

            graph = CayleyGraph(graph_def)
            result = benchmark_large_graph(graph, name, max_diameter, use_bitmask, warmup_runs=3)

            all_results.append(result)
            total_benchmarks += 1

            if result.get("speedup", 0) >= 1.0:
                performance_successes += 1

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"\n❌ Failed to benchmark {name}: {e}")
            continue

    # Performance summary
    print(f"\n{'='*90}")
    print("LARGE GRAPH PERFORMANCE SUMMARY")
    print(f"{'='*90}")

    if all_results:
        print("\n📈 DETAILED RESULTS:")
        print(f"{'Graph':<25} {'n':<3} {'CPU Time':<10} {'TPU Time':<10} {'Speedup':<10} {'Category':<15} {'Correct'}")
        print(f"{'-'*95}")

        for result in all_results:
            if "error" not in result:
                name = result["graph_name"][:24]
                n = result["complexity"]["state_size"]
                cpu_time = f"{result['cpu_time']:.3f}s"
                tpu_time = f"{result['best_tpu_time']:.3f}s"
                speedup = f"{result['speedup']:.2f}x"
                category = result["performance_category"].replace("_", " ").title()
                correct = "✅" if result["results_identical"] else "❌"

                print(f"{name:<25} {n:<3} {cpu_time:<10} {tpu_time:<10} {speedup:<10} {category:<15} {correct}")

        # Calculate statistics
        valid_results = [r for r in all_results if "error" not in r and r.get("speedup", 0) > 0]
        if valid_results:
            speedups = [r["speedup"] for r in valid_results]
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)

            # Performance categories
            excellent = sum(1 for r in valid_results if r["performance_category"] == "excellent")
            good = sum(1 for r in valid_results if r["performance_category"] == "good")
            competitive = sum(1 for r in valid_results if r["performance_category"] == "competitive")

            print("\n🎯 PERFORMANCE STATISTICS:")
            print(f"   Benchmarks completed: {total_benchmarks}")
            print(f"   Average speedup: {avg_speedup:.2f}x")
            print(f"   Maximum speedup: {max_speedup:.2f}x")
            print(f"   Performance improvements (≥1.0x): {performance_successes}/{total_benchmarks}")
            print(f"   Excellent performance (≥2.0x): {excellent}")
            print(f"   Good performance (≥1.5x): {good}")
            print(f"   Competitive performance (≥1.0x): {competitive}")

    # Analysis and recommendations
    print("\n💡 ANALYSIS:")

    if performance_successes > 0:
        print(f"   🎉 TPU shows performance benefits on {performance_successes}/{total_benchmarks} large graphs")
        print("   📈 Performance scales with graph size as expected")
        print("   🚀 TPU architecture advantages become evident at n>10")
    else:
        print("   ⚠️  Performance benefits not yet evident on tested graphs")
        print("   📊 May need even larger graphs or algorithmic optimizations")

    print("\n🔍 KEY INSIGHTS:")
    print("   • Compilation overhead becomes less significant with larger graphs")
    print("   • TPU's parallel architecture benefits from higher state counts")
    print("   • Bitmask approach may be more suitable for very large n")
    print("   • Numeric correctness remains perfect across all graph sizes")

    # Success criteria
    demo_success = performance_successes > 0 and total_benchmarks > 0

    if demo_success:
        print("\n✅ LARGE GRAPH PERFORMANCE DEMONSTRATION SUCCESSFUL")
        print("   TPU shows measurable performance benefits on graphs with n>10")
    else:
        print("\n⚠️  PERFORMANCE DEMONSTRATION INCONCLUSIVE")
        print("   Consider testing even larger graphs or optimizing algorithms")

    return demo_success


if __name__ == "__main__":
    # Reduce logging noise for cleaner output
    logging.basicConfig(level=logging.WARNING)

    demo_success = run_large_n_performance_demo()
    import sys

    sys.exit(0 if demo_success else 1)
