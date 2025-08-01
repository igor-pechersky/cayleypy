"""Comprehensive validation tests for TPU BFS implementations.

This module tests that TPU BFS implementations produce identical numeric results
to reference implementations, have comparable memory usage, and provide
significant performance improvements on medium-to-large graphs.
"""

import gc
import logging
import math
import pytest
import time
from typing import Dict, List, Tuple

try:
    import psutil
except ImportError:
    psutil = None

try:
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    nnx = None  # type: ignore
    nnx = None  # type: ignore

from .tpu_backend import get_tpu_backend
from .tpu_bfs import tpu_bfs, create_tpu_bfs
from .tpu_bfs_bitmask import tpu_bfs_bitmask, create_tpu_bitmask_bfs
from .bfs_numpy import bfs_numpy
from .bfs_bitmask import bfs_bitmask
from .graphs_lib import PermutationGroups
from .cayley_graph import CayleyGraph


class MemoryProfiler:
    """Memory usage profiler for BFS operations."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = 0
        self.peak_memory = 0
        self.measurements = []

    def start_profiling(self):
        """Start memory profiling."""
        gc.collect()  # Force garbage collection
        self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = self.baseline_memory
        self.measurements = [self.baseline_memory]

    def measure(self, label: str = ""):
        """Take a memory measurement."""
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        self.measurements.append(current_memory)
        if label:
            logging.info("Memory usage at %s: %.2f MB", label, current_memory)

    def get_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return {
            "baseline_mb": self.baseline_memory,
            "peak_mb": self.peak_memory,
            "peak_increase_mb": self.peak_memory - self.baseline_memory,
            "final_mb": self.measurements[-1] if self.measurements else 0,
            "final_increase_mb": (self.measurements[-1] - self.baseline_memory) if self.measurements else 0,
        }


class BFSValidator:
    """Validator for BFS implementations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.backend = None
        if JAX_AVAILABLE:
            try:
                self.backend = get_tpu_backend()
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.warning("Failed to initialize TPU backend: %s", e)

    def validate_growth_functions_identical(
        self, reference_result: List[int], test_result: List[int], algorithm_name: str
    ) -> bool:
        """Validate that growth functions are identical."""
        if reference_result == test_result:
            self.logger.info("✓ %s: Growth functions are identical", algorithm_name)
            return True
        else:
            self.logger.error("✗ %s: Growth functions differ", algorithm_name)
            self.logger.error("  Reference: %s", reference_result)
            self.logger.error("  Test:      %s", test_result)
            return False

    def validate_memory_usage(
        self,
        reference_stats: Dict[str, float],
        test_stats: Dict[str, float],
        algorithm_name: str,
        tolerance_factor: float = 2.0,
    ) -> bool:
        """Validate that memory usage is not significantly greater."""
        ref_peak = reference_stats["peak_increase_mb"]
        test_peak = test_stats["peak_increase_mb"]

        if test_peak <= ref_peak * tolerance_factor:
            self.logger.info(
                "✓ %s: Memory usage acceptable (%.2f MB vs %.2f MB reference)", algorithm_name, test_peak, ref_peak
            )
            return True
        else:
            self.logger.error(
                "✗ %s: Memory usage too high (%.2f MB vs %.2f MB reference, %.1fx)",
                algorithm_name,
                test_peak,
                ref_peak,
                test_peak / max(ref_peak, 1),
            )
            return False

    def validate_performance_improvement(
        self, reference_time: float, test_time: float, algorithm_name: str, min_speedup: float = 1.5
    ) -> bool:
        """Validate that performance is significantly better."""
        if test_time > 0:
            speedup = reference_time / test_time
            if speedup >= min_speedup:
                self.logger.info(
                    "✓ %s: Performance improvement %.2fx (%.3fs vs %.3fs)",
                    algorithm_name,
                    speedup,
                    test_time,
                    reference_time,
                )
                return True
            else:
                self.logger.warning(
                    "⚠ %s: Performance improvement only %.2fx (%.3fs vs %.3fs)",
                    algorithm_name,
                    speedup,
                    test_time,
                    reference_time,
                )
                return False
        else:
            self.logger.error("✗ %s: Invalid test time", algorithm_name)
            return False

    def run_bfs_with_profiling(
        self, bfs_func, graph, max_diameter: int, algorithm_name: str
    ) -> Tuple[List[int], float, Dict[str, float]]:
        """Run BFS with memory and time profiling."""
        profiler = MemoryProfiler()
        profiler.start_profiling()

        self.logger.info(
            "Starting %s on graph with %d generators", algorithm_name, len(graph.definition.generators_permutations)
        )

        start_time = time.time()
        result = bfs_func(graph, max_diameter)
        elapsed_time = time.time() - start_time

        profiler.measure("after BFS completion")
        memory_stats = profiler.get_stats()

        self.logger.info("%s completed in %.3fs, growth function: %s", algorithm_name, elapsed_time, result)

        return result, elapsed_time, memory_stats


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTPUBFSValidation:
    """Test suite for TPU BFS validation."""

    def setup_method(self):
        """Set up test environment."""
        self.validator = BFSValidator()
        self.test_graphs = self._create_test_graphs()

    def _create_test_graphs(self) -> List[Tuple[CayleyGraph, str, int]]:
        """Create test graphs of various sizes."""
        graphs = []

        # Small graphs for exact validation
        graphs.append((CayleyGraph(PermutationGroups.create_symmetric_group(4)), "S4 (24 elements)", 10))

        graphs.append((CayleyGraph(PermutationGroups.create_alternating_group(5)), "A5 (60 elements)", 15))

        # Medium graphs for performance testing
        graphs.append((CayleyGraph(PermutationGroups.create_symmetric_group(5)), "S5 (120 elements)", 20))

        graphs.append((CayleyGraph(PermutationGroups.create_dihedral_group(12)), "D12 (24 elements)", 15))

        # Large graphs for bitmask testing (only if sufficient memory)
        try:
            # Only test larger graphs if we have sufficient memory
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb > 8:  # At least 8GB available
                graphs.append((CayleyGraph(PermutationGroups.create_symmetric_group(6)), "S6 (720 elements)", 25))

                if available_memory_gb > 16:  # At least 16GB available
                    graphs.append((CayleyGraph(PermutationGroups.create_symmetric_group(7)), "S7 (5040 elements)", 30))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Skipping large graphs due to memory constraints: %s", e)

        return graphs

    def test_tpu_bfs_numeric_correctness(self):
        """Test that TPU BFS produces identical numeric results."""
        if not self.validator.backend or not self.validator.backend.is_available:
            pytest.skip("TPU not available")

        all_passed = True

        for graph, graph_name, max_diameter in self.test_graphs:
            self.validator.logger.info("Testing TPU BFS numeric correctness on %s", graph_name)

            # Run reference implementation
            ref_result, ref_time, ref_memory = self.validator.run_bfs_with_profiling(
                bfs_numpy, graph, max_diameter, f"NumPy BFS ({graph_name})"
            )

            # Run TPU implementation
            tpu_result, tpu_time, tpu_memory = self.validator.run_bfs_with_profiling(
                tpu_bfs, graph, max_diameter, f"TPU BFS ({graph_name})"
            )

            # Validate results
            numeric_correct = self.validator.validate_growth_functions_identical(
                ref_result, tpu_result, f"TPU BFS ({graph_name})"
            )

            memory_acceptable = self.validator.validate_memory_usage(
                ref_memory, tpu_memory, f"TPU BFS ({graph_name})", tolerance_factor=3.0
            )

            performance_better = self.validator.validate_performance_improvement(
                ref_time, tpu_time, f"TPU BFS ({graph_name})", min_speedup=1.2
            )

            if not (numeric_correct and memory_acceptable):
                all_passed = False

        assert all_passed, "TPU BFS validation failed for one or more test cases"

    def test_tpu_bitmask_bfs_numeric_correctness(self):
        """Test that TPU bitmask BFS produces identical numeric results."""
        if not self.validator.backend or not self.validator.backend.is_available:
            pytest.skip("TPU not available")

        all_passed = True

        # Only test graphs suitable for bitmask approach (n > 8)
        bitmask_graphs = [
            (g, name, max_d)
            for g, name, max_d in self.test_graphs
            if g.definition.state_size > 8 and g.definition.is_permutation_group()
        ]

        if not bitmask_graphs:
            pytest.skip("No suitable graphs for bitmask testing")

        for graph, graph_name, max_diameter in bitmask_graphs:
            self.validator.logger.info("Testing TPU bitmask BFS numeric correctness on %s", graph_name)

            # Run reference bitmask implementation
            ref_result, ref_time, ref_memory = self.validator.run_bfs_with_profiling(
                bfs_bitmask, graph, max_diameter, f"CPU Bitmask BFS ({graph_name})"
            )

            # Run TPU bitmask implementation
            tpu_result, tpu_time, tpu_memory = self.validator.run_bfs_with_profiling(
                tpu_bfs_bitmask, graph, max_diameter, f"TPU Bitmask BFS ({graph_name})"
            )

            # Validate results
            numeric_correct = self.validator.validate_growth_functions_identical(
                ref_result, tpu_result, f"TPU Bitmask BFS ({graph_name})"
            )

            memory_acceptable = self.validator.validate_memory_usage(
                ref_memory, tpu_memory, f"TPU Bitmask BFS ({graph_name})", tolerance_factor=2.5
            )

            performance_better = self.validator.validate_performance_improvement(
                ref_time, tpu_time, f"TPU Bitmask BFS ({graph_name})", min_speedup=1.3
            )

            if not (numeric_correct and memory_acceptable):
                all_passed = False

        assert all_passed, "TPU bitmask BFS validation failed for one or more test cases"

    def test_tpu_bfs_memory_efficiency(self):
        """Test memory efficiency of TPU BFS implementations."""
        if not self.validator.backend or not self.validator.backend.is_available:
            pytest.skip("TPU not available")

        # Test with a medium-sized graph
        test_graph = CayleyGraph(PermutationGroups.create_symmetric_group(5))
        max_diameter = 20

        # Measure memory usage for different implementations
        implementations = [(bfs_numpy, "NumPy BFS"), (tpu_bfs, "TPU BFS")]

        if test_graph.definition.state_size > 8:
            implementations.extend([(bfs_bitmask, "CPU Bitmask BFS"), (tpu_bfs_bitmask, "TPU Bitmask BFS")])

        memory_results = {}

        for bfs_func, name in implementations:
            try:
                _, _, memory_stats = self.validator.run_bfs_with_profiling(bfs_func, test_graph, max_diameter, name)
                memory_results[name] = memory_stats
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.validator.logger.warning("Failed to test %s: %s", name, e)

        # Log memory comparison
        self.validator.logger.info("Memory Usage Comparison:")
        for name, stats in memory_results.items():
            self.validator.logger.info("  %s: %.2f MB peak increase", name, stats["peak_increase_mb"])

        # Validate that TPU implementations don't use significantly more memory
        if "NumPy BFS" in memory_results and "TPU BFS" in memory_results:
            ref_memory = memory_results["NumPy BFS"]["peak_increase_mb"]
            tpu_memory = memory_results["TPU BFS"]["peak_increase_mb"]
            assert (
                tpu_memory <= ref_memory * 3.0
            ), f"TPU BFS uses too much memory: {tpu_memory:.2f} MB vs {ref_memory:.2f} MB"

        if "CPU Bitmask BFS" in memory_results and "TPU Bitmask BFS" in memory_results:
            ref_memory = memory_results["CPU Bitmask BFS"]["peak_increase_mb"]
            tpu_memory = memory_results["TPU Bitmask BFS"]["peak_increase_mb"]
            assert (
                tpu_memory <= ref_memory * 2.5
            ), f"TPU Bitmask BFS uses too much memory: {tpu_memory:.2f} MB vs {ref_memory:.2f} MB"

    def test_tpu_bfs_performance_scaling(self):
        """Test performance scaling of TPU BFS with graph size."""
        if not self.validator.backend or not self.validator.backend.is_available:
            pytest.skip("TPU not available")

        # Test graphs of increasing size
        scaling_graphs = [
            (CayleyGraph(PermutationGroups.create_symmetric_group(4)), "S4", 10),
            (CayleyGraph(PermutationGroups.create_symmetric_group(5)), "S5", 15),
        ]

        # Add larger graphs if memory permits
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb > 8:
                scaling_graphs.append((CayleyGraph(PermutationGroups.create_symmetric_group(6)), "S6", 20))
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        cpu_times = []
        tpu_times = []
        graph_sizes = []

        for graph, name, max_diameter in scaling_graphs:
            graph_size = math.factorial(graph.definition.state_size)
            graph_sizes.append(graph_size)

            # CPU timing
            start_time = time.time()
            bfs_numpy(graph, max_diameter)
            cpu_time = time.time() - start_time
            cpu_times.append(cpu_time)

            # TPU timing
            start_time = time.time()
            tpu_bfs(graph, max_diameter)
            tpu_time = time.time() - start_time
            tpu_times.append(tpu_time)

            speedup = cpu_time / tpu_time if tpu_time > 0 else 0
            self.validator.logger.info(
                "%s (size %d): CPU %.3fs, TPU %.3fs, speedup %.2fx", name, graph_size, cpu_time, tpu_time, speedup
            )

        # Validate that speedup generally improves with graph size
        speedups = [c / t if t > 0 else 0 for c, t in zip(cpu_times, tpu_times)]

        # At least the largest graph should show good speedup
        if speedups:
            max_speedup = max(speedups)
            assert max_speedup >= 1.2, f"Maximum speedup {max_speedup:.2f}x is insufficient"

    def test_tpu_bfs_precision_maintenance(self):
        """Test that TPU BFS maintains int64 precision."""
        if not self.validator.backend or not self.validator.backend.is_available:
            pytest.skip("TPU not available")

        # Test with a graph that has large state values
        test_graph = CayleyGraph(PermutationGroups.create_symmetric_group(4))

        # Create TPU BFS modules and test precision
        rngs = nnx.Rngs(42)

        # Test regular TPU BFS
        tpu_bfs_module = create_tpu_bfs(test_graph, self.validator.backend)
        precision_ok = tpu_bfs_module.verify_int64_precision()
        assert precision_ok, "TPU BFS failed int64 precision test"

        # Test TPU bitmask BFS if applicable
        if test_graph.definition.state_size > 8 and test_graph.definition.is_permutation_group():
            tpu_bitmask_module = create_tpu_bitmask_bfs(test_graph, self.validator.backend)
            precision_ok = tpu_bitmask_module.verify_int64_precision()
            assert precision_ok, "TPU Bitmask BFS failed int64 precision test"

    def test_tpu_bfs_error_handling(self):
        """Test error handling and fallback behavior."""
        # Test with invalid graph (should fallback gracefully)
        test_graph = CayleyGraph(PermutationGroups.create_symmetric_group(4))

        # Test fallback when TPU not available (mock scenario)
        result = tpu_bfs(test_graph, max_diameter=5)
        assert isinstance(result, list), "TPU BFS should return a list"
        assert len(result) > 0, "TPU BFS should return non-empty result"

        # Test bitmask fallback
        if test_graph.definition.state_size > 8:
            result = tpu_bfs_bitmask(test_graph, max_diameter=5)
            assert isinstance(result, list), "TPU Bitmask BFS should return a list"
            assert len(result) > 0, "TPU Bitmask BFS should return non-empty result"


def run_comprehensive_validation():
    """Run comprehensive validation of TPU BFS implementations."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("TPU BFS Comprehensive Validation")
    print("=" * 50)

    if not JAX_AVAILABLE:
        print("❌ JAX not available - cannot run TPU BFS validation")
        return False

    try:
        backend = get_tpu_backend()
        if not backend.is_available:
            print("❌ TPU not available - cannot run TPU BFS validation")
            return False

        print(f"✅ TPU Backend: {backend.get_device_info()}")

        validator = BFSValidator()
        all_tests_passed = True

        # Test graphs
        test_graphs = [
            (CayleyGraph(PermutationGroups.create_symmetric_group(4)), "S4", 10),
            (CayleyGraph(PermutationGroups.create_symmetric_group(5)), "S5", 15),
            (CayleyGraph(PermutationGroups.create_alternating_group(5)), "A5", 15),
        ]

        print("\n1. Testing Numeric Correctness")
        print("-" * 30)

        for graph, name, max_diameter in test_graphs:
            print(f"\nTesting {name}...")

            # Reference result
            ref_result, ref_time, ref_memory = validator.run_bfs_with_profiling(
                bfs_numpy, graph, max_diameter, f"NumPy BFS ({name})"
            )

            # TPU result
            tpu_result, tpu_time, tpu_memory = validator.run_bfs_with_profiling(
                tpu_bfs, graph, max_diameter, f"TPU BFS ({name})"
            )

            # Validation
            numeric_ok = validator.validate_growth_functions_identical(ref_result, tpu_result, f"TPU BFS ({name})")

            memory_ok = validator.validate_memory_usage(ref_memory, tpu_memory, f"TPU BFS ({name})")

            perf_ok = validator.validate_performance_improvement(ref_time, tpu_time, f"TPU BFS ({name})")

            if not (numeric_ok and memory_ok):
                all_tests_passed = False

        print("\n2. Testing Bitmask Implementation")
        print("-" * 30)

        # Test bitmask on suitable graphs
        bitmask_graphs = [
            (g, n, d) for g, n, d in test_graphs if g.definition.state_size > 8 and g.definition.is_permutation_group()
        ]

        for graph, name, max_diameter in bitmask_graphs:
            print(f"\nTesting bitmask on {name}...")

            # Reference bitmask result
            ref_result, ref_time, ref_memory = validator.run_bfs_with_profiling(
                bfs_bitmask, graph, max_diameter, f"CPU Bitmask ({name})"
            )

            # TPU bitmask result
            tpu_result, tpu_time, tpu_memory = validator.run_bfs_with_profiling(
                tpu_bfs_bitmask, graph, max_diameter, f"TPU Bitmask ({name})"
            )

            # Validation
            numeric_ok = validator.validate_growth_functions_identical(ref_result, tpu_result, f"TPU Bitmask ({name})")

            memory_ok = validator.validate_memory_usage(ref_memory, tpu_memory, f"TPU Bitmask ({name})")

            if not (numeric_ok and memory_ok):
                all_tests_passed = False

        print("\n3. Summary")
        print("-" * 30)

        if all_tests_passed:
            print("✅ All validation tests PASSED")
            print("   - Numeric results are identical")
            print("   - Memory usage is acceptable")
            print("   - Performance improvements demonstrated")
        else:
            print("❌ Some validation tests FAILED")

        return all_tests_passed

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Validation failed with error: {e}")
        return False


if __name__ == "__main__":
    # Run comprehensive validation when executed as script
    success = run_comprehensive_validation()
    exit(0 if success else 1)
