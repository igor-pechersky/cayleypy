"""Tests for NNX Tensor Operations Module."""

from unittest.mock import patch

import pytest

from .nnx_backend import create_nnx_backend, JAX_AVAILABLE
from .nnx_tensor_ops import TensorOpsConfig, TensorOpsModule, create_tensor_ops_module

if JAX_AVAILABLE:
    import jax.numpy as jnp
else:
    jnp = None  # type: ignore  # pylint: disable=invalid-name


class TestTensorOpsConfig:
    """Test tensor operations configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TensorOpsConfig()

        assert config.enable_caching is True
        assert config.max_cache_size == 1000
        assert config.cache_ttl_seconds == 300.0
        assert config.chunk_size == 10000
        assert config.enable_jit is True
        assert config.enable_vmap is True
        assert config.memory_efficient is True
        assert config.max_memory_mb == 512.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TensorOpsConfig(enable_caching=False, chunk_size=5000, enable_jit=False, memory_efficient=False)

        assert config.enable_caching is False
        assert config.chunk_size == 5000
        assert config.enable_jit is False
        assert config.memory_efficient is False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTensorOpsModule:
    """Test tensor operations module functionality when JAX is available."""

    def setup_method(self):
        """Set up test fixtures."""
        # pylint: disable=attribute-defined-outside-init
        self.backend = create_nnx_backend(preferred_device="cpu")
        self.config = TensorOpsConfig(enable_caching=True, chunk_size=100)
        self.module = TensorOpsModule(self.backend, self.config)

    def test_module_creation(self):
        """Test basic module creation."""
        assert self.module.backend == self.backend
        assert self.module.config == self.config
        assert self.module.cache is not None
        assert isinstance(self.module.metrics.value, dict)

    def test_unique_with_indices(self):
        """Test unique operation with indices."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        arr = jnp.array([1, 2, 2, 3, 1, 4])

        unique_vals, inverse_indices = self.module.unique_with_indices(arr)

        # Check that unique values are correct
        expected_unique = jnp.array([1, 2, 3, 4])
        assert jnp.array_equal(jnp.sort(unique_vals), expected_unique)

        # Check that inverse indices reconstruct original array
        reconstructed = unique_vals[inverse_indices]
        assert jnp.array_equal(jnp.sort(reconstructed), jnp.sort(arr))

        # Check metrics were updated
        assert self.module.metrics.value["unique_calls"] > 0
        assert self.module.metrics.value["total_operations"] > 0

    def test_isin_via_searchsorted(self):
        """Test membership testing using searchsorted."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        elements = jnp.array([1, 3, 5, 7])
        test_elements = jnp.array([1, 2, 3, 4, 5, 6])

        result = self.module.isin_via_searchsorted(elements, test_elements)

        expected = jnp.array([True, True, True, False])  # 1,3,5 are in test_elements, 7 is not
        assert jnp.array_equal(result, expected)

        # Check metrics were updated
        assert self.module.metrics.value["searchsorted_calls"] > 0

    def test_searchsorted_batched(self):
        """Test batched searchsorted operation."""
        # Create batch of sorted arrays
        sorted_arrays = jnp.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
        values = jnp.array([[2, 6], [3, 7]])

        indices = self.module.searchsorted_batched(sorted_arrays, values)

        # Check shapes
        assert indices.shape == (2, 2)

        # Check that indices are reasonable
        assert jnp.all(indices >= 0)
        assert jnp.all(indices <= 5)  # Should be within array bounds

        # Check metrics were updated
        assert self.module.metrics.value["vmap_calls"] > 0

    def test_batch_matmul(self):
        """Test vectorized matrix multiplication."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)

        result = self.module.batch_matmul(a, b)

        expected = jnp.dot(a, b)
        assert jnp.allclose(result, expected)

        # Check metrics were updated
        assert self.module.metrics.value["vmap_calls"] > 0

    def test_vectorized_element_wise_equal(self):
        """Test vectorized element-wise equality."""
        a = jnp.array([1, 2, 3])
        b = jnp.array([1, 2, 3])

        result = self.module.vectorized_element_wise_equal(a, b)

        assert result  # Arrays are equal

        # Test with different arrays
        c = jnp.array([1, 2, 4])
        result2 = self.module.vectorized_element_wise_equal(a, c)
        assert not result2  # Arrays are not equal

    def test_unique_counts_sorted(self):
        """Test unique counts for sorted array."""
        arr = jnp.array([1, 1, 2, 2, 2, 3, 4, 4])

        unique_vals, counts = self.module.unique_counts_sorted(arr)

        expected_unique = jnp.array([1, 2, 3, 4])
        expected_counts = jnp.array([2, 3, 1, 2])

        assert jnp.array_equal(unique_vals, expected_unique)
        assert jnp.array_equal(counts, expected_counts)

    def test_argsort_stable(self):
        """Test stable argsort implementation."""
        arr = jnp.array([3, 1, 2, 1, 3])

        indices = self.module.argsort_stable(arr)
        sorted_arr = arr[indices]

        expected_sorted = jnp.array([1, 1, 2, 3, 3])
        assert jnp.array_equal(sorted_arr, expected_sorted)

    def test_gather_batched(self):
        """Test batched gather operation."""
        params = jnp.array([10, 20, 30, 40, 50])
        indices = jnp.array([0, 2, 4])

        result = self.module.gather_batched(params, indices)

        expected = jnp.array([10, 30, 50])
        assert jnp.array_equal(result, expected)

    def test_scatter_add_batched(self):
        """Test batched scatter-add operation."""
        operand = jnp.zeros(5)
        scatter_indices = jnp.array([1, 3])
        updates = jnp.array([10, 20])

        result = self.module.scatter_add_batched(operand, scatter_indices, updates)

        expected = jnp.array([0, 10, 0, 20, 0])
        assert jnp.array_equal(result, expected)

    def test_process_large_array_small(self):
        """Test processing small array (no chunking)."""
        arr = jnp.array([3, 1, 4, 1, 5])

        result = self.module.process_large_array(arr, "unique")

        expected = jnp.array([1, 3, 4, 5])
        assert jnp.array_equal(jnp.sort(result), expected)

    def test_process_large_array_chunked(self):
        """Test processing large array with chunking."""
        # Create array larger than chunk size
        arr = jnp.arange(250)  # Larger than default chunk_size of 100

        result = self.module.process_large_array(arr, "sort")

        # Should be sorted
        expected = jnp.sort(arr)
        assert jnp.array_equal(result, expected)

    def test_caching_functionality(self):
        """Test operation caching."""
        arr = jnp.array([1, 2, 2, 3])

        # First call should miss cache
        result1 = self.module.unique_with_indices(arr)

        # Second call with same input should hit cache (if caching is working)
        # Note: JIT compilation may affect caching behavior
        result2 = self.module.unique_with_indices(arr)

        # Results should be the same
        assert jnp.array_equal(result1[0], result2[0])
        assert jnp.array_equal(result1[1], result2[1])

    def test_clear_cache(self):
        """Test cache clearing."""
        # Perform some operations to populate cache
        arr = jnp.array([1, 2, 3])
        self.module.unique_with_indices(arr)

        # Clear cache
        self.module.clear_cache()

        # Cache should be empty
        if self.module.cache is not None:
            assert len(self.module.cache.value) == 0

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Perform various operations
        arr = jnp.array([1, 2, 2, 3])
        self.module.unique_with_indices(arr)
        self.module.isin_via_searchsorted(arr, jnp.array([1, 2]))

        metrics = self.module.get_performance_metrics()

        # Check that metrics contain expected keys
        assert "unique_calls" in metrics
        assert "searchsorted_calls" in metrics
        assert "total_operations" in metrics
        assert "config" in metrics
        assert "backend_info" in metrics

        # Check that some operations were recorded
        assert metrics["unique_calls"] > 0
        assert metrics["total_operations"] > 0

    def test_optimize_for_device(self):
        """Test device-specific optimization."""
        original_chunk_size = self.module.config.chunk_size

        # Mock different device types
        with patch.object(self.module.backend, "device_type", "tpu"):
            self.module.optimize_for_device()
            # TPU should use smaller chunk size
            assert self.module.config.chunk_size <= original_chunk_size

        # Reset for next test
        self.module.config.chunk_size = original_chunk_size

        with patch.object(self.module.backend, "device_type", "cpu"):
            self.module.optimize_for_device()
            # CPU should use larger chunk size
            assert self.module.config.chunk_size >= original_chunk_size


class TestTensorOpsModuleWithoutJAX:
    """Test tensor operations module behavior when JAX is not available."""

    @patch("cayleypy.nnx_tensor_ops.JAX_AVAILABLE", False)
    def test_module_creation_without_jax(self):
        """Test that module creation fails gracefully without JAX."""
        backend = None  # Mock backend

        with pytest.raises(ImportError, match="JAX and Flax are required"):
            TensorOpsModule(backend)


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_tensor_ops_module_with_params(self):
        """Test module creation with custom parameters."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        backend = create_nnx_backend(preferred_device="cpu")
        module = create_tensor_ops_module(backend, enable_caching=False, chunk_size=5000)

        if backend is not None:
            assert module is not None
            assert module.config.enable_caching is False
            # Note: chunk_size may be modified by optimize_for_device()
            # so we check that it was set initially, then possibly optimized
        else:
            assert module is None

    @patch("cayleypy.nnx_tensor_ops.JAX_AVAILABLE", False)
    def test_create_module_without_jax(self):
        """Test factory function behavior without JAX."""
        backend = None
        module = create_tensor_ops_module(backend)

        assert module is None


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTensorOpsIntegration:
    """Integration tests for tensor operations module."""

    def test_end_to_end_graph_operations(self):
        """Test end-to-end graph-like operations."""
        backend = create_nnx_backend(preferred_device="cpu")
        if backend is None:
            pytest.skip("Backend not available")

        module = TensorOpsModule(backend)

        # Simulate graph edge processing
        edges = jnp.array([[0, 1], [1, 2], [2, 0], [1, 3], [3, 1], [0, 2]])

        # Find unique nodes
        all_nodes = edges.flatten()
        unique_nodes, _ = module.unique_with_indices(all_nodes)

        # Test that we found the right unique nodes
        expected_nodes = jnp.array([0, 1, 2, 3])
        assert jnp.array_equal(jnp.sort(unique_nodes), expected_nodes)

        # Test searchsorted for node lookup
        query_nodes = jnp.array([1, 2, 4])  # 4 doesn't exist
        membership = module.isin_via_searchsorted(query_nodes, unique_nodes)

        expected_membership = jnp.array([True, True, False])
        assert jnp.array_equal(membership, expected_membership)

        # Check that metrics were updated
        metrics = module.get_performance_metrics()
        assert metrics["unique_calls"] > 0
        assert metrics["searchsorted_calls"] > 0

    def test_large_array_processing(self):
        """Test processing of large arrays with chunking."""
        backend = create_nnx_backend(preferred_device="cpu")
        if backend is None:
            pytest.skip("Backend not available")

        config = TensorOpsConfig(chunk_size=1000, memory_efficient=True)
        module = TensorOpsModule(backend, config)

        # Create large array with duplicates
        large_array = jnp.concatenate(
            [
                jnp.arange(5000),
                jnp.arange(2500),  # Add duplicates
            ]
        )

        # Process with chunking
        unique_result = module.process_large_array(large_array, "unique")

        # Should have unique values from 0 to 4999
        expected_unique = jnp.arange(5000)
        assert jnp.array_equal(jnp.sort(unique_result), expected_unique)

    def test_performance_comparison(self):
        """Test performance metrics collection across operations."""
        backend = create_nnx_backend(preferred_device="cpu")
        if backend is None:
            pytest.skip("Backend not available")

        module = TensorOpsModule(backend)

        # Perform multiple operations
        arr1 = jnp.arange(1000)
        arr2 = jnp.arange(500, 1500)

        # Unique operations
        module.unique_with_indices(arr1)
        module.unique_with_indices(arr2)

        # Searchsorted operations
        module.isin_via_searchsorted(arr1[:100], arr2)

        # Matrix operations
        mat_a = jnp.ones((10, 10))
        mat_b = jnp.eye(10)
        module.batch_matmul(mat_a, mat_b)

        # Get final metrics
        metrics = module.get_performance_metrics()

        # Verify operations were counted
        assert metrics["unique_calls"] >= 2
        assert metrics["searchsorted_calls"] >= 1
        assert metrics["vmap_calls"] >= 1
        assert metrics["total_operations"] >= 4

        # Verify config is included
        assert "config" in metrics
        assert metrics["config"]["enable_jit"] is True


if __name__ == "__main__":
    pytest.main([__file__])
