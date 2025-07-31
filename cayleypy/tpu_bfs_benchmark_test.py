"""Performance benchmark tests for TPU BFS implementations.

This module provides detailed performance benchmarking comparing TPU BFS
implementations against reference CPU implementations on various graph sizes.
"""

import gc
import json
import logging
import math
import psutil
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    nnx = None  # type: ignore

from .tpu_backend import get_tpu_backend
from .tpu_bfs import tpu_bfs, create_tpu_bfs
from .tpu_bfs_bitmask import tpu_bfs_bitmask, create_tpu_bitmask_bfs
from .bfs_numpy import bfs_numpy
from .bfs_bitmask import bfs_bitmask
from .graphs_lib import PermutationGroups
from .cayley_graph import CayleyGraph


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    algorithm: str
    graph_name: str
    graph_size: int
    state_size: int
    generators_count: int
    max_diameter: int
    growth_function: List[int]
    execution_time: float
    memory_baseline_mb: float
    memory_peak_mb: float
    memory_increase_mb: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Results from comparing two algorithms."""
    reference_algorithm: str
    test_algorithm: str
    graph_name: str
    numeric_identical: bool
    speedup: float
    memory_ratio: float
    reference_time: float
    test_time: float
    reference_memory: float
    test_memory: float
    performance_improvement: bool
    memory_acceptable: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BFSBenchmark:
    """Comprehensive benchmark suite for BFS implementations."""
    
    def __init__(self, output_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.output_file = output_file
        self.results: List[BenchmarkResult] = []
        self.comparisons: List[ComparisonResult] = []
        
        # Initialize TPU backend
        self.tpu_available = False
        self.backend = None
        if JAX_AVAILABLE:
            try:
                self.backend = get_tpu_backend()
                self.tpu_available = self.backend.is_available
                if self.tpu_available:
                    self.logger.info("TPU backend initialized: %s", self.backend.get_device_info())
                else:
                    self.logger.warning("TPU backend not available")
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error("Failed to initialize TPU backend: %s", e)
    
    def create_benchmark_graphs(self) -> List[Tuple[CayleyGraph, str, int]]:
        """Create a comprehensive set of benchmark graphs."""
        graphs = []
        
        # Small graphs for validation
        graphs.extend([
            (CayleyGraph(PermutationGroups.symmetric_group(4)), "S4", 10),
            (CayleyGraph(PermutationGroups.alternating_group(4)), "A4", 12),
            (CayleyGraph(PermutationGroups.dihedral_group(8)), "D8", 15),
        ])
        
        # Medium graphs for performance testing
        graphs.extend([
            (CayleyGraph(PermutationGroups.symmetric_group(5)), "S5", 20),
            (CayleyGraph(PermutationGroups.alternating_group(5)), "A5", 25),
            (CayleyGraph(PermutationGroups.dihedral_group(12)), "D12", 20),
        ])
        
        # Large graphs (memory permitting)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.logger.info("Available memory: %.2f GB", available_memory_gb)
        
        if available_memory_gb > 8:
            graphs.extend([
                (CayleyGraph(PermutationGroups.symmetric_group(6)), "S6", 25),
                (CayleyGraph(PermutationGroups.dihedral_group(20)), "D20", 30),
            ])
        
        if available_memory_gb > 16:
            graphs.extend([
                (CayleyGraph(PermutationGroups.symmetric_group(7)), "S7", 30),
            ])
        
        # Very large graphs for bitmask testing
        if available_memory_gb > 32:
            graphs.extend([
                (CayleyGraph(PermutationGroups.symmetric_group(8)), "S8", 35),
            ])
        
        return graphs
    
    def measure_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage in MB."""
        process = psutil.Process()
        gc.collect()  # Force garbage collection
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024), memory_info.vms / (1024 * 1024)
    
    def run_single_benchmark(
        self, 
        bfs_func, 
        algorithm_name: str, 
        graph: CayleyGraph, 
        graph_name: str, 
        max_diameter: int
    ) -> BenchmarkResult:
        """Run a single benchmark and return results."""
        self.logger.info("Benchmarking %s on %s (max_diameter=%d)", 
                        algorithm_name, graph_name, max_diameter)
        
        # Get graph properties
        graph_size = math.factorial(graph.definition.state_size) if graph.definition.is_permutation_group() else 0
        state_size = graph.definition.state_size
        generators_count = len(graph.definition.generators_permutations)
        
        # Measure baseline memory
        baseline_memory, _ = self.measure_memory_usage()
        
        try:
            # Run the algorithm
            start_time = time.time()
            growth_function = bfs_func(graph, max_diameter)
            execution_time = time.time() - start_time
            
            # Measure peak memory
            peak_memory, _ = self.measure_memory_usage()
            memory_increase = peak_memory - baseline_memory
            
            result = BenchmarkResult(
                algorithm=algorithm_name,
                graph_name=graph_name,
                graph_size=graph_size,
                state_size=state_size,
                generators_count=generators_count,
                max_diameter=max_diameter,
                growth_function=growth_function,
                execution_time=execution_time,
                memory_baseline_mb=baseline_memory,
                memory_peak_mb=peak_memory,
                memory_increase_mb=memory_increase,
                success=True
            )
            
            self.logger.info("✓ %s completed: %.3fs, %.2f MB increase, growth: %s", 
                           algorithm_name, execution_time, memory_increase, growth_function)
            
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("✗ %s failed: %s", algorithm_name, e)
            
            result = BenchmarkResult(
                algorithm=algorithm_name,
                graph_name=graph_name,
                graph_size=graph_size,
                state_size=state_size,
                generators_count=generators_count,
                max_diameter=max_diameter,
                growth_function=[],
                execution_time=0.0,
                memory_baseline_mb=baseline_memory,
                memory_peak_mb=baseline_memory,
                memory_increase_mb=0.0,
                success=False,
                error_message=str(e)
            )
        
        self.results.append(result)
        return result
    
    def compare_algorithms(
        self, 
        reference_result: BenchmarkResult, 
        test_result: BenchmarkResult
    ) -> ComparisonResult:
        """Compare two algorithm results."""
        # Check numeric correctness
        numeric_identical = (
            reference_result.success and 
            test_result.success and 
            reference_result.growth_function == test_result.growth_function
        )
        
        # Calculate performance metrics
        speedup = 0.0
        if test_result.success and test_result.execution_time > 0:
            speedup = reference_result.execution_time / test_result.execution_time
        
        memory_ratio = 0.0
        if reference_result.memory_increase_mb > 0:
            memory_ratio = test_result.memory_increase_mb / reference_result.memory_increase_mb
        
        # Determine if improvements are significant
        performance_improvement = speedup >= 1.2  # At least 20% faster
        memory_acceptable = memory_ratio <= 3.0   # At most 3x memory usage
        
        comparison = ComparisonResult(
            reference_algorithm=reference_result.algorithm,
            test_algorithm=test_result.algorithm,
            graph_name=reference_result.graph_name,
            numeric_identical=numeric_identical,
            speedup=speedup,
            memory_ratio=memory_ratio,
            reference_time=reference_result.execution_time,
            test_time=test_result.execution_time,
            reference_memory=reference_result.memory_increase_mb,
            test_memory=test_result.memory_increase_mb,
            performance_improvement=performance_improvement,
            memory_acceptable=memory_acceptable
        )
        
        self.comparisons.append(comparison)
        return comparison
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all implementations."""
        self.logger.info("Starting comprehensive BFS benchmark")
        
        benchmark_graphs = self.create_benchmark_graphs()
        
        # Define algorithms to test
        algorithms = [
            (bfs_numpy, "NumPy BFS"),
        ]
        
        if self.tpu_available:
            algorithms.append((tpu_bfs, "TPU BFS"))
        
        # Test each graph with each algorithm
        for graph, graph_name, max_diameter in benchmark_graphs:
            self.logger.info("\n" + "="*50)
            self.logger.info("Benchmarking graph: %s", graph_name)
            self.logger.info("="*50)
            
            graph_results = {}
            
            # Run all algorithms on this graph
            for bfs_func, algorithm_name in algorithms:
                result = self.run_single_benchmark(
                    bfs_func, algorithm_name, graph, graph_name, max_diameter
                )
                graph_results[algorithm_name] = result
            
            # Test bitmask algorithms if applicable
            if (graph.definition.state_size > 8 and 
                graph.definition.is_permutation_group()):
                
                # CPU bitmask
                result = self.run_single_benchmark(
                    bfs_bitmask, "CPU Bitmask BFS", graph, graph_name, max_diameter
                )
                graph_results["CPU Bitmask BFS"] = result
                
                # TPU bitmask
                if self.tpu_available:
                    result = self.run_single_benchmark(
                        tpu_bfs_bitmask, "TPU Bitmask BFS", graph, graph_name, max_diameter
                    )
                    graph_results["TPU Bitmask BFS"] = result
            
            # Compare algorithms
            if "NumPy BFS" in graph_results and "TPU BFS" in graph_results:
                comparison = self.compare_algorithms(
                    graph_results["NumPy BFS"], 
                    graph_results["TPU BFS"]
                )
                self.logger.info("TPU BFS vs NumPy BFS: %.2fx speedup, %.2fx memory, identical=%s", 
                               comparison.speedup, comparison.memory_ratio, comparison.numeric_identical)
            
            if "CPU Bitmask BFS" in graph_results and "TPU Bitmask BFS" in graph_results:
                comparison = self.compare_algorithms(
                    graph_results["CPU Bitmask BFS"], 
                    graph_results["TPU Bitmask BFS"]
                )
                self.logger.info("TPU Bitmask vs CPU Bitmask: %.2fx speedup, %.2fx memory, identical=%s", 
                               comparison.speedup, comparison.memory_ratio, comparison.numeric_identical)
        
        # Generate summary
        summary = self.generate_summary()
        
        # Save results if output file specified
        if self.output_file:
            self.save_results()
        
        return summary
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            'total_benchmarks': len(self.results),
            'successful_benchmarks': sum(1 for r in self.results if r.success),
            'tpu_available': self.tpu_available,
            'algorithms_tested': list(set(r.algorithm for r in self.results)),
            'graphs_tested': list(set(r.graph_name for r in self.results)),
            'comparisons': len(self.comparisons),
            'performance_summary': {},
            'correctness_summary': {},
            'memory_summary': {}
        }
        
        # Performance summary
        tpu_comparisons = [c for c in self.comparisons if 'TPU' in c.test_algorithm]
        if tpu_comparisons:
            speedups = [c.speedup for c in tpu_comparisons if c.speedup > 0]
            summary['performance_summary'] = {
                'average_speedup': sum(speedups) / len(speedups) if speedups else 0,
                'max_speedup': max(speedups) if speedups else 0,
                'min_speedup': min(speedups) if speedups else 0,
                'significant_improvements': sum(1 for c in tpu_comparisons if c.performance_improvement)
            }
        
        # Correctness summary
        summary['correctness_summary'] = {
            'identical_results': sum(1 for c in self.comparisons if c.numeric_identical),
            'total_comparisons': len(self.comparisons),
            'correctness_rate': sum(1 for c in self.comparisons if c.numeric_identical) / max(len(self.comparisons), 1)
        }
        
        # Memory summary
        if tpu_comparisons:
            memory_ratios = [c.memory_ratio for c in tpu_comparisons if c.memory_ratio > 0]
            summary['memory_summary'] = {
                'average_memory_ratio': sum(memory_ratios) / len(memory_ratios) if memory_ratios else 0,
                'max_memory_ratio': max(memory_ratios) if memory_ratios else 0,
                'acceptable_memory_usage': sum(1 for c in tpu_comparisons if c.memory_acceptable)
            }
        
        return summary
    
    def save_results(self):
        """Save benchmark results to file."""
        if not self.output_file:
            return
        
        output_data = {
            'benchmark_results': [r.to_dict() for r in self.results],
            'comparisons': [c.to_dict() for c in self.comparisons],
            'summary': self.generate_summary(),
            'system_info': {
                'tpu_available': self.tpu_available,
                'backend_info': self.backend.get_device_info() if self.backend else None,
                'memory_info': {
                    'total_gb': psutil.virtual_memory().total / (1024**3),
                    'available_gb': psutil.virtual_memory().available / (1024**3)
                }
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info("Results saved to %s", self.output_file)
    
    def print_summary_report(self):
        """Print a formatted summary report."""
        summary = self.generate_summary()
        
        print("\n" + "="*60)
        print("TPU BFS BENCHMARK SUMMARY REPORT")
        print("="*60)
        
        print(f"\nSystem Information:")
        print(f"  TPU Available: {'✅' if self.tpu_available else '❌'}")
        print(f"  Total Benchmarks: {summary['total_benchmarks']}")
        print(f"  Successful: {summary['successful_benchmarks']}")
        print(f"  Algorithms Tested: {', '.join(summary['algorithms_tested'])}")
        
        print(f"\nGraphs Tested:")
        for graph in summary['graphs_tested']:
            print(f"  - {graph}")
        
        if summary['performance_summary']:
            perf = summary['performance_summary']
            print(f"\nPerformance Results:")
            print(f"  Average Speedup: {perf['average_speedup']:.2f}x")
            print(f"  Maximum Speedup: {perf['max_speedup']:.2f}x")
            print(f"  Minimum Speedup: {perf['min_speedup']:.2f}x")
            print(f"  Significant Improvements: {perf['significant_improvements']}/{len(self.comparisons)}")
        
        if summary['correctness_summary']:
            corr = summary['correctness_summary']
            print(f"\nCorrectness Results:")
            print(f"  Identical Results: {corr['identical_results']}/{corr['total_comparisons']}")
            print(f"  Correctness Rate: {corr['correctness_rate']:.1%}")
        
        if summary['memory_summary']:
            mem = summary['memory_summary']
            print(f"\nMemory Usage:")
            print(f"  Average Memory Ratio: {mem['average_memory_ratio']:.2f}x")
            print(f"  Maximum Memory Ratio: {mem['max_memory_ratio']:.2f}x")
            print(f"  Acceptable Usage: {mem['acceptable_memory_usage']}/{len(self.comparisons)}")
        
        print(f"\nDetailed Results:")
        for comparison in self.comparisons:
            status = "✅" if comparison.numeric_identical and comparison.memory_acceptable else "❌"
            print(f"  {status} {comparison.test_algorithm} vs {comparison.reference_algorithm} on {comparison.graph_name}:")
            print(f"      Speedup: {comparison.speedup:.2f}x, Memory: {comparison.memory_ratio:.2f}x, Correct: {comparison.numeric_identical}")
        
        print("\n" + "="*60)


def run_performance_benchmark(output_file: Optional[str] = None) -> bool:
    """Run comprehensive performance benchmark."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("TPU BFS Performance Benchmark")
    print("=" * 40)
    
    if not JAX_AVAILABLE:
        print("❌ JAX not available - cannot run TPU benchmarks")
        return False
    
    try:
        benchmark = BFSBenchmark(output_file)
        
        if not benchmark.tpu_available:
            print("❌ TPU not available - running CPU-only benchmarks")
        
        # Run comprehensive benchmark
        summary = benchmark.run_comprehensive_benchmark()
        
        # Print summary report
        benchmark.print_summary_report()
        
        # Determine overall success
        correctness_rate = summary['correctness_summary']['correctness_rate']
        success = correctness_rate >= 0.95  # At least 95% correctness
        
        if success:
            print("\n✅ BENCHMARK PASSED: TPU implementations are correct and performant")
        else:
            print("\n❌ BENCHMARK FAILED: Issues detected in TPU implementations")
        
        return success
        
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Benchmark failed with error: {e}")
        logging.exception("Benchmark error")
        return False


if __name__ == "__main__":
    # Run performance benchmark when executed as script
    import sys
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else "tpu_bfs_benchmark_results.json"
    success = run_performance_benchmark(output_file)
    exit(0 if success else 1)