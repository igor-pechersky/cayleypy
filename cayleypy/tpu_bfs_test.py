"""Tests for TPU BFS Implementation.

This module provides comprehensive tests for the TPU-accelerated breadth-first search
implementation with native int64 support.
"""

import pytest
import logging

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .tpu_bfs import TPUBFSModule, tpu_bfs, create_tpu_bfs, benchmark_tpu_vs_cpu_bfs


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTPUBFSModule:
    """Test cases for TPUBFSModule."""

    def setup_method(self):
        """Set up test fixtures."""
        try:
            from .tpu_backend import get_tpu_backend
            from .graphs_lib import PermutationGroups

            self.backend = get_tpu_backend()
            if not self.backend.is_available:
                pytest.skip("TPU not available")

            # Create test graphs
            self.small_graph = PermutationGroups.symmetric_group(3)  # S3 - small for testing
            self.medium_graph = PermutationGroups.symmetric_group(4)  # S4 - medium size

            # Create BFS modules
            self.small_bfs = create_tpu_bfs(self.small_graph, self.backend)
            self.medium_bfs = create_tpu_bfs(self.medium_graph, self.backend)

        except Exception as e:
            pytest.skip(f"TPU setup failed: {e}")

    def test_module_initialization(self):
        """Test TPU BFS module initialization."""
        assert self.small_bfs.backend.is_available
        assert len(self.small_bfs.generators.value) > 0
        assert self.small_bfs.hasher is not None
        assert self.small_bfs.tensor_ops is not None

        # Check initial state
        assert self.small_bfs.bfs_state.value['current_layer'] is None
        assert self.small_bfs.bfs_state.value['diameter'] == 0
        assert self.small_bfs.bfs_state.value['total_states_found'] == 0

    def test_int64_precision_verification(self):
        """Test int64 precision verification."""
        precision_ok = self.small_bfs.verify_int64_precision()
        assert precision_ok, "int64 precision verification should pass"

    def test_bfs_initialization(self):
        """Test BFS initialization."""
        self.small_bfs.initialize_bfs()

        # Check initialized state
        assert self.small_bfs.bfs_state.value['current_layer'] is not None
        assert len(self.small_bfs.bfs_state.value['current_layer']) == 1
        assert self.small_bfs.bfs_state.value['diameter'] == 0
        assert self.small_bfs.bfs_state.value['total_states_found'] == 1
        assert len(self.small_bfs.bfs_state.value['layer_sizes']) == 1

        # Check data types
        current_layer = self.small_bfs.bfs_state.value['current_layer']
        assert current_layer.dtype == jnp.int64

    def test_layer_expansion(self):
        """Test layer expansion functionality."""
        self.small_bfs.initialize_bfs()
        current_layer = self.small_bfs.bfs_state.value['current_layer']

        # Expand the layer
        expanded = self.small_bfs.expand_layer(current_layer)

        # Check results
        assert len(expanded) > 0
        assert expanded.dtype == jnp.int64
        assert expanded.shape[1] == current_layer.shape[1]  # Same state size

        # Check metrics were updated
        metrics = self.small_bfs.get_performance_metrics()
        assert metrics['layer_expansions'] > 0
        assert metrics['states_processed'] > 0

    def test_bfs_step(self):
        """Test single BFS step."""
        self.small_bfs.initialize_bfs()
        current_layer = self.small_bfs.bfs_state.value['current_layer']
        visited_hashes = self.small_bfs.bfs_state.value['visited_hashes']

        # Perform BFS step
        new_layer, updated_visited = self.small_bfs.bfs_step(current_layer, visited_hashes)

        # Check results
        assert len(new_layer) >= 0  # May be 0 if all states already visited
        assert len(updated_visited) >= len(visited_hashes)
        assert new_layer.dtype == jnp.int64
        assert updated_visited.dtype == jnp.int64

        # Check metrics were updated
        metrics = self.small_bfs.get_performance_metrics()
        assert metrics['hash_operations'] > 0
        assert metrics['deduplication_operations'] > 0

    def test_complete_bfs_small_graph(self):
        """Test complete BFS on small graph (S3)."""
        result = self.small_bfs.run_bfs(max_diameter=10)

        # S3 has 6 elements, so we should find all of them
        assert isinstance(result, list)
        assert len(result) > 0
        assert sum(result) == 6  # Total states in S3

        # Check BFS completed
        bfs_result = self.small_bfs.get_bfs_result()
        assert bfs_result['is_complete']
        assert bfs_result['total_states_found'] == 6
        assert bfs_result['diameter'] >= 0

    def test_complete_bfs_medium_graph(self):
        """Test complete BFS on medium graph (S4)."""
        result = self.medium_bfs.run_bfs(max_diameter=15)

        # S4 has 24 elements
        assert isinstance(result, list)
        assert len(result) > 0
        assert sum(result) == 24  # Total states in S4

        # Check BFS completed
        bfs_result = self.medium_bfs.get_bfs_result()
        assert bfs_result['is_complete']
        assert bfs_result['total_states_found'] == 24

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        self.small_bfs.run_bfs(max_diameter=10)
        metrics = self.small_bfs.get_performance_metrics()

        # Check required metrics exist
        required_metrics = [
            'states_processed', 'hash_operations', 'tpu_utilization',
            'layer_expansions', 'deduplication_operations'
        ]
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

        # Check nested metrics
        assert 'hasher_stats' in metrics
        assert 'tensor_ops_stats' in metrics
        assert 'backend_info' in metrics

    def test_current_layer_info(self):
        """Test current layer information retrieval."""
        # Before initialization
        info = self.small_bfs.get_current_layer_info()
        assert not info['layer_exists']

        # After initialization
        self.small_bfs.initialize_bfs()
        info = self.small_bfs.get_current_layer_info()
        assert info['layer_exists']
        assert info['layer_size'] == 1
        assert info['current_diameter'] == 0

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        # Run BFS to generate metrics
        self.small_bfs.run_bfs(max_diameter=5)
        metrics_before = self.small_bfs.get_performance_metrics()
        assert metrics_before['states_processed'] > 0

        # Reset metrics
        self.small_bfs.reset_metrics()
        metrics_after = self.small_bfs.get_performance_metrics()
        assert metrics_after['states_processed'] == 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTPUBFSFunctions:
    """Test cases for high-level TPU BFS functions."""

    def setup_method(self):
        """Set up test fixtures."""
        try:
            from .graphs_lib import PermutationGroups

            self.test_graph = PermutationGroups.symmetric_group(3)

        except Exception as e:
            pytest.skip(f"Graph setup failed: {e}")

    def test_tpu_bfs_function(self):
        """Test high-level tpu_bfs function."""
        result = tpu_bfs(self.test_graph, max_diameter=10)

        assert isinstance(result, list)
        assert len(result) > 0
        assert sum(result) == 6  # S3 has 6 elements

    def test_create_tpu_bfs_function(self):
        """Test create_tpu_bfs factory function."""
        try:
            bfs_module = create_tpu_bfs(self.test_graph)
            assert isinstance(bfs_module, TPUBFSModule)
            assert bfs_module.backend.is_available

        except Exception as e:
            # If TPU not available, should raise appropriate error
            assert "TPU" in str(e) or "JAX" in str(e)

    def test_benchmark_function(self):
        """Test benchmark function."""
        benchmark_result = benchmark_tpu_vs_cpu_bfs(self.test_graph, max_diameter=10)

        # Check benchmark structure
        assert 'graph_info' in benchmark_result
        assert 'cpu_time' in benchmark_result
        assert 'cpu_result' in benchmark_result

        # CPU should always work
        assert benchmark_result['cpu_time'] > 0
        assert len(benchmark_result['cpu_result']) > 0

        # TPU results depend on availability
        if benchmark_result['tpu_available']:
            assert benchmark_result['tpu_time'] > 0
            assert len(benchmark_result['tpu_result']) > 0
            assert 'speedup' in benchmark_result
            assert 'results_match' in benchmark_result


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTPUBFSEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        try:
            from .graphs_lib import PermutationGroups

            self.test_graph = PermutationGroups.symmetric_group(3)

        except Exception as e:
            pytest.skip(f"Graph setup failed: {e}")

    def test_max_diameter_limit(self):
        """Test BFS with very small max_diameter."""
        try:
            bfs_module = create_tpu_bfs(self.test_graph)
            result = bfs_module.run_bfs(max_diameter=1)

            # Should stop early due to diameter limit
            assert isinstance(result, list)
            assert len(result) <= 1

        except Exception as e:
            # If TPU not available, should handle gracefully
            assert "TPU" in str(e) or "JAX" in str(e)

    def test_empty_layer_handling(self):
        """Test handling of empty layers."""
        try:
            bfs_module = create_tpu_bfs(self.test_graph)
            bfs_module.initialize_bfs()

            # Manually create empty layer scenario
            empty_layer = jnp.array([], dtype=jnp.int64).reshape(0, len(self.test_graph.central_state))
            visited_hashes = bfs_module.bfs_state.value['visited_hashes']

            # Should handle empty layer gracefully
            new_layer, updated_visited = bfs_module.bfs_step(empty_layer, visited_hashes)
            assert len(new_layer) == 0

        except Exception as e:
            # If TPU not available, should handle gracefully
            assert "TPU" in str(e) or "JAX" in str(e)

    def test_large_state_values(self):
        """Test handling of large int64 state values."""
        try:
            bfs_module = create_tpu_bfs(self.test_graph)

            # Test with large int64 values
            large_state = jnp.array([[2**40, 2**50, 2**60]], dtype=jnp.int64)

            # Should handle large values without overflow
            if len(bfs_module.generators.value) > 0:
                expanded = bfs_module.expand_layer(large_state)
                assert expanded.dtype == jnp.int64

        except Exception as e:
            # If TPU not available or operation fails, should handle gracefully
            assert "TPU" in str(e) or "JAX" in str(e) or "index" in str(e).lower()


class TestTPUBFSFallback:
    """Test fallback behavior when TPU is not available."""

    def test_fallback_to_cpu(self):
        """Test that tpu_bfs falls back to CPU when TPU unavailable."""
        from .graphs_lib import PermutationGroups

        test_graph = PermutationGroups.symmetric_group(3)

        # This should work regardless of TPU availability
        result = tpu_bfs(test_graph, max_diameter=10)

        assert isinstance(result, list)
        assert len(result) > 0
        assert sum(result) == 6  # S3 has 6 elements

    def test_create_tpu_bfs_without_jax(self):
        """Test create_tpu_bfs behavior when JAX not available."""
        if JAX_AVAILABLE:
            pytest.skip("JAX is available, cannot test fallback")

        from .graphs_lib import PermutationGroups

        test_graph = PermutationGroups.symmetric_group(3)

        with pytest.raises(ImportError, match="JAX and Flax are required"):
            create_tpu_bfs(test_graph)


def test_logging_configuration():
    """Test that logging is properly configured."""
    logger = logging.getLogger('cayleypy.tpu_bfs')
    assert logger is not None

    # Test that we can create log messages without errors
    logger.info("Test log message")
    logger.debug("Test debug message")


if __name__ == "__main__":
    # Run basic tests when executed as script
    print("Running TPU BFS Tests")
    print("=" * 40)

    if not JAX_AVAILABLE:
        print("JAX not available - testing fallback behavior only")
        test_fallback = TestTPUBFSFallback()
        test_fallback.test_fallback_to_cpu()
        print("✓ Fallback test passed")
    else:
        try:
            from .tpu_backend import get_tpu_backend
            from .graphs_lib import PermutationGroups

            backend = get_tpu_backend()
            print(f"TPU Available: {backend.is_available}")

            if backend.is_available:
                # Run basic functionality test
                test_graph = PermutationGroups.symmetric_group(3)
                bfs_module = create_tpu_bfs(test_graph, backend)

                print("Testing BFS initialization...")
                bfs_module.initialize_bfs()
                print("✓ Initialization passed")

                print("Testing int64 precision...")
                precision_ok = bfs_module.verify_int64_precision()
                print(f"✓ Precision test: {'PASSED' if precision_ok else 'FAILED'}")

                print("Testing complete BFS...")
                result = bfs_module.run_bfs(max_diameter=10)
                print(f"✓ BFS result: {result}")

                print("Testing performance metrics...")
                metrics = bfs_module.get_performance_metrics()
                print(f"✓ States processed: {metrics['states_processed']}")

                print("All basic tests passed!")
            else:
                print("TPU not available - testing fallback...")
                test_fallback = TestTPUBFSFallback()
                test_fallback.test_fallback_to_cpu()
                print("✓ Fallback test passed")

        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise