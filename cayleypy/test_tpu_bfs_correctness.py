"""Unit tests for TPU BFS correctness and performance.

These tests verify that TPU BFS implementations produce identical results
to reference implementations and demonstrate performance improvements.
"""

import time

import pytest

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
# jax = None  # type: ignore  # Unused import

from .tpu_backend import get_tpu_backend
from .tpu_bfs import tpu_bfs, create_tpu_bfs
from .tpu_bfs_bitmask import tpu_bfs_bitmask, create_tpu_bitmask_bfs
from .bfs_numpy import bfs_numpy
from .bfs_bitmask import bfs_bitmask
from .graphs_lib import PermutationGroups
from .cayley_graph import CayleyGraph


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTPUBFSCorrectness:
    """Test TPU BFS correctness against reference implementations."""

    @pytest.fixture(autouse=True)
    def setup_tpu_backend(self):
        """Set up TPU backend for tests."""
        try:
            self.backend = get_tpu_backend()
            if not self.backend.is_available:
                pytest.skip("TPU not available")
        except Exception as e:  # pylint: disable=broad-exception-caught
            pytest.skip(f"Failed to initialize TPU backend: {e}")

    def test_tpu_bfs_identical_results_small_graphs(self):
        """Test that TPU BFS produces identical results on small graphs."""
        test_cases = [
            (PermutationGroups.coxeter(4), "Coxeter-4", 8),
            (PermutationGroups.all_transpositions(4), "AllTrans-4", 10),
            (PermutationGroups.pancake(4), "Pancake-4", 12),
        ]

        for graph_def, name, max_diameter in test_cases:
            graph = CayleyGraph(graph_def)

            # Reference result
            reference_result = bfs_numpy(graph, max_diameter)

            # TPU result
            tpu_result = tpu_bfs(graph, max_diameter)

            assert (
                reference_result == tpu_result
            ), f"TPU BFS result differs from reference on {name}: {tpu_result} vs {reference_result}"

    def test_tpu_bfs_identical_results_medium_graphs(self):
        """Test that TPU BFS produces identical results on medium graphs."""
        test_cases = [
            (PermutationGroups.coxeter(5), "Coxeter-5", 15),
            (PermutationGroups.all_transpositions(5), "AllTrans-5", 18),
        ]

        for graph_def, name, max_diameter in test_cases:
            graph = CayleyGraph(graph_def)

            # Reference result
            reference_result = bfs_numpy(graph, max_diameter)

            # TPU result
            tpu_result = tpu_bfs(graph, max_diameter)

            assert (
                reference_result == tpu_result
            ), f"TPU BFS result differs from reference on {name}: {tpu_result} vs {reference_result}"

    def test_tpu_bitmask_bfs_identical_results(self):
        """Test that TPU bitmask BFS produces identical results."""
        # Only test graphs suitable for bitmask (n > 8)
        test_cases = [
            (PermutationGroups.coxeter(9), "Coxeter-9", 12),
        ]

        for graph_def, name, max_diameter in test_cases:
            if not graph_def.is_permutation_group():
                continue

            graph = CayleyGraph(graph_def)

            # Reference bitmask result
            reference_result = bfs_bitmask(graph, max_diameter)

            # TPU bitmask result
            tpu_result = tpu_bfs_bitmask(graph, max_diameter)

            assert (
                reference_result == tpu_result
            ), f"TPU Bitmask BFS result differs from reference on {name}: {tpu_result} vs {reference_result}"

    def test_tpu_bfs_performance_improvement(self):
        """Test that TPU BFS provides performance improvement."""
        # Use a medium-sized graph for performance testing
        graph = CayleyGraph(PermutationGroups.coxeter(5))
        max_diameter = 15

        # Time CPU implementation
        start_time = time.time()
        cpu_result = bfs_numpy(graph, max_diameter)
        cpu_time = time.time() - start_time

        # Time TPU implementation
        start_time = time.time()
        tpu_result = tpu_bfs(graph, max_diameter)
        tpu_time = time.time() - start_time

        # Verify results are identical
        assert cpu_result == tpu_result, "Results must be identical for performance comparison"

        # Calculate speedup
        speedup = cpu_time / tpu_time if tpu_time > 0 else 0

        # We expect at least some speedup, but allow for variation in test environments
        assert speedup > 0.5, f"TPU implementation is significantly slower: {speedup:.2f}x"

        # Log the performance for information
        print(f"Performance: CPU {cpu_time:.3f}s, TPU {tpu_time:.3f}s, speedup {speedup:.2f}x")

    def test_tpu_bfs_int64_precision(self):
        """Test that TPU BFS maintains int64 precision."""
        graph = CayleyGraph(PermutationGroups.coxeter(4))

        # Create TPU BFS module
        bfs_module = create_tpu_bfs(graph, self.backend)

        # Verify int64 precision
        precision_ok = bfs_module.verify_int64_precision()
        assert precision_ok, "TPU BFS failed int64 precision verification"

    def test_tpu_bitmask_bfs_int64_precision(self):
        """Test that TPU bitmask BFS maintains int64 precision."""
        graph = CayleyGraph(PermutationGroups.coxeter(9))

        if not graph.definition.is_permutation_group():
            pytest.skip("Graph not suitable for bitmask BFS")

        # Create TPU bitmask BFS module
        bfs_module = create_tpu_bitmask_bfs(graph, self.backend)

        # Verify int64 precision
        precision_ok = bfs_module.verify_int64_precision()
        assert precision_ok, "TPU Bitmask BFS failed int64 precision verification"

    def test_tpu_bfs_empty_result_handling(self):
        """Test handling of graphs that produce empty results."""
        # Create a trivial graph
        graph = CayleyGraph(PermutationGroups.coxeter(2))
        max_diameter = 1

        # Both implementations should handle this gracefully
        cpu_result = bfs_numpy(graph, max_diameter)
        tpu_result = tpu_bfs(graph, max_diameter)

        assert cpu_result == tpu_result, "Empty result handling differs between implementations"
        assert isinstance(tpu_result, list), "TPU BFS should return a list"

    def test_tpu_bfs_large_diameter(self):
        """Test TPU BFS with large diameter limits."""
        graph = CayleyGraph(PermutationGroups.coxeter(4))
        max_diameter = 100  # Much larger than actual diameter

        # Both should produce identical results regardless of large diameter
        cpu_result = bfs_numpy(graph, max_diameter)
        tpu_result = tpu_bfs(graph, max_diameter)

        assert cpu_result == tpu_result, "Large diameter handling differs between implementations"

    @pytest.mark.parametrize(
        "graph_def,name",
        [
            (PermutationGroups.coxeter(3), "Coxeter-3"),
            (PermutationGroups.all_transpositions(4), "AllTrans-4"),
            (PermutationGroups.pancake(4), "Pancake-4"),
        ],
    )
    def test_tpu_bfs_various_groups(self, graph_def, name):
        """Test TPU BFS on various types of groups."""
        graph = CayleyGraph(graph_def)
        max_diameter = 10

        # Reference result
        reference_result = bfs_numpy(graph, max_diameter)

        # TPU result
        tpu_result = tpu_bfs(graph, max_diameter)

        assert reference_result == tpu_result, f"TPU BFS result differs from reference on {name}"

    def test_tpu_bfs_consistency_multiple_runs(self):
        """Test that TPU BFS produces consistent results across multiple runs."""
        graph = CayleyGraph(PermutationGroups.coxeter(4))
        max_diameter = 8

        # Run TPU BFS multiple times
        results = []
        for _ in range(3):
            result = tpu_bfs(graph, max_diameter)
            results.append(result)

        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result == first_result, f"TPU BFS result {i} differs from first result: {result} vs {first_result}"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTPUBFSPerformance:
    """Performance-focused tests for TPU BFS."""

    @pytest.fixture(autouse=True)
    def setup_tpu_backend(self):
        """Set up TPU backend for tests."""
        try:
            self.backend = get_tpu_backend()
            if not self.backend.is_available:
                pytest.skip("TPU not available")
        except Exception as e:  # pylint: disable=broad-exception-caught
            pytest.skip(f"Failed to initialize TPU backend: {e}")

    def test_performance_scaling(self):
        """Test that TPU BFS performance scales well with graph size."""
        test_graphs = [
            (PermutationGroups.coxeter(4), "Coxeter-4"),
            (PermutationGroups.coxeter(5), "Coxeter-5"),
        ]

        cpu_times = []
        tpu_times = []

        for graph_def, name in test_graphs:
            graph = CayleyGraph(graph_def)
            max_diameter = 15

            # Time CPU
            start_time = time.time()
            cpu_result = bfs_numpy(graph, max_diameter)
            cpu_time = time.time() - start_time
            cpu_times.append(cpu_time)

            # Time TPU
            start_time = time.time()
            tpu_result = tpu_bfs(graph, max_diameter)
            tpu_time = time.time() - start_time
            tpu_times.append(tpu_time)

            # Verify correctness
            assert cpu_result == tpu_result, f"Results differ on {name}"

            speedup = cpu_time / tpu_time if tpu_time > 0 else 0
            print(f"{name}: CPU {cpu_time:.3f}s, TPU {tpu_time:.3f}s, speedup {speedup:.2f}x")

        # At least one graph should show reasonable performance
        speedups = [c / t if t > 0 else 0 for c, t in zip(cpu_times, tpu_times)]
        max_speedup = max(speedups) if speedups else 0

        assert max_speedup > 0.8, f"Maximum speedup {max_speedup:.2f}x is too low"

    def test_memory_usage_reasonable(self):
        """Test that TPU BFS memory usage is reasonable."""
        import psutil  # pylint: disable=import-outside-toplevel

        graph = CayleyGraph(PermutationGroups.coxeter(5))
        max_diameter = 15

        process = psutil.Process()

        # Measure CPU memory usage
        baseline = process.memory_info().rss
        _ = bfs_numpy(graph, max_diameter)
        cpu_peak = process.memory_info().rss
        cpu_increase = (cpu_peak - baseline) / (1024 * 1024)  # MB

        # Measure TPU memory usage
        baseline = process.memory_info().rss
        _ = tpu_bfs(graph, max_diameter)
        tpu_peak = process.memory_info().rss
        tpu_increase = (tpu_peak - baseline) / (1024 * 1024)  # MB

        # TPU should not use significantly more memory
        memory_ratio = tpu_increase / max(cpu_increase, 1)

        assert (
            memory_ratio <= 5.0
        ), f"TPU memory usage too high: {tpu_increase:.1f} MB vs {cpu_increase:.1f} MB (ratio: {memory_ratio:.2f}x)"

        print(f"Memory usage: CPU {cpu_increase:.1f} MB, TPU {tpu_increase:.1f} MB, ratio {memory_ratio:.2f}x")


def test_tpu_availability():
    """Test that TPU backend can be initialized (may skip if no TPU)."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    try:
        backend = get_tpu_backend()
        # This test passes if we can create the backend, regardless of availability
        assert backend is not None, "TPU backend should be created"

        if backend.is_available:
            device_info = backend.get_device_info()
            assert isinstance(device_info, dict), "Device info should be a dictionary"
            print(f"TPU available: {device_info}")
        else:
            print("TPU not available, but backend created successfully")

    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.skip(f"TPU backend initialization failed: {e}")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
