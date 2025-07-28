"""Tests for NNX Hash Functions Module."""

from unittest.mock import patch, MagicMock

import pytest

from .nnx_backend import create_nnx_backend, JAX_AVAILABLE
from .nnx_hasher import NNXHasherConfig, NNXStateHasher, OptimizedNNXStateHasher, create_nnx_hasher

if JAX_AVAILABLE:
    import jax.numpy as jnp
else:
    jnp = None  # type: ignore  # pylint: disable=invalid-name


class TestNNXHasherConfig:
    """Test NNX hasher configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NNXHasherConfig()

        assert config.hash_bits == 64
        assert config.hash_seed == 42
        assert config.enable_caching is True
        assert config.max_cache_size == 10000
        assert config.cache_ttl_seconds == 600.0
        assert config.chunk_size == 50000
        assert config.enable_jit is True
        assert config.enable_vmap is True
        assert config.memory_efficient is True
        assert config.max_memory_mb == 1024.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = NNXHasherConfig(
            hash_bits=32,
            enable_caching=False,
            chunk_size=25000,
            memory_efficient=False,
        )

        assert config.hash_bits == 32
        assert config.enable_caching is False
        assert config.chunk_size == 25000
        assert config.memory_efficient is False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestNNXStateHasher:
    """Test NNX state hasher functionality when JAX is available."""

    def setup_method(self):
        """Set up test fixtures."""
        # pylint: disable=attribute-defined-outside-init
        self.backend = create_nnx_backend(preferred_device="cpu")
        self.state_size = 100
        self.config = NNXHasherConfig(enable_caching=True, chunk_size=1000)
        self.hasher = NNXStateHasher(self.state_size, self.backend, self.config)

    def test_hasher_creation(self):
        """Test basic hasher creation."""
        assert self.hasher.state_size == self.state_size
        assert self.hasher.backend == self.backend
        assert self.hasher.config == self.config
        assert hasattr(self.hasher, "hash_matrix")
        assert hasattr(self.hasher, "hash_cache")
        assert hasattr(self.hasher, "stats")

    def test_hash_matrix_shape(self):
        """Test hash matrix has correct shape."""
        expected_shape = (self.state_size, self.config.hash_bits)
        assert self.hasher.hash_matrix.value.shape == expected_shape
        assert self.hasher.hash_matrix.value.dtype == jnp.uint32

    def test_hash_single_state(self):
        """Test hashing a single state vector."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        state = jnp.arange(self.state_size, dtype=jnp.float32)
        hash_result = self.hasher.hash_state(state)

        # Check result properties
        assert isinstance(hash_result, jnp.ndarray)
        assert hash_result.shape == ()  # Scalar result
        assert hash_result.dtype in [jnp.int32, jnp.uint32]

        # Check statistics were updated
        assert self.hasher.stats.value["total_hashes"] > 0

    def test_hash_consistency(self):
        """Test that same state produces same hash."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        state = jnp.arange(self.state_size, dtype=jnp.float32)

        hash1 = self.hasher.hash_state(state)
        hash2 = self.hasher.hash_state(state)

        assert jnp.array_equal(hash1, hash2)

    def test_hash_different_states(self):
        """Test that different states produce different hashes."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        state1 = jnp.arange(self.state_size, dtype=jnp.float32)
        state2 = jnp.arange(self.state_size, dtype=jnp.float32) + 1

        hash1 = self.hasher.hash_state(state1)
        hash2 = self.hasher.hash_state(state2)

        # Different states should produce different hashes (with high probability)
        assert not jnp.array_equal(hash1, hash2)

    def test_hash_batch(self):
        """Test batch hashing functionality."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        batch_size = 10
        states = jnp.arange(batch_size * self.state_size, dtype=jnp.float32).reshape(batch_size, self.state_size)

        hash_results = self.hasher.hash_batch(states)

        # Check result properties
        assert hash_results.shape == (batch_size,)
        assert hash_results.dtype in [jnp.int32, jnp.uint32]

        # Check that different states in batch produce different hashes
        unique_hashes = jnp.unique(hash_results)
        assert len(unique_hashes) == batch_size  # All should be different

        # Check statistics were updated
        assert self.hasher.stats.value["total_hashes"] >= batch_size
        assert len(self.hasher.stats.value["batch_sizes"]) > 0

    def test_hash_large_batch_small(self):
        """Test large batch processing with small batch (no chunking)."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        batch_size = 50  # Smaller than chunk_size
        states = jnp.arange(batch_size * self.state_size, dtype=jnp.float32).reshape(batch_size, self.state_size)

        hash_results = self.hasher.hash_large_batch(states)

        assert hash_results.shape == (batch_size,)
        assert hash_results.dtype in [jnp.int32, jnp.uint32]

    def test_hash_large_batch_chunked(self):
        """Test large batch processing with chunking."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        batch_size = 2500  # Larger than chunk_size of 1000
        states = jnp.arange(batch_size * self.state_size, dtype=jnp.float32).reshape(batch_size, self.state_size)

        hash_results = self.hasher.hash_large_batch(states, chunk_size=500)

        assert hash_results.shape == (batch_size,)
        assert hash_results.dtype in [jnp.int32, jnp.uint32]

        # Verify results are consistent with non-chunked version for smaller subset
        subset_size = 100
        subset_states = states[:subset_size]
        subset_hashes_chunked = hash_results[:subset_size]
        subset_hashes_direct = self.hasher.hash_batch(subset_states)

        # Results should be the same (allowing for potential floating point differences)
        assert jnp.allclose(subset_hashes_chunked, subset_hashes_direct, rtol=1e-5)

    def test_caching_functionality(self):
        """Test hash result caching."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        state = jnp.arange(self.state_size, dtype=jnp.float32)

        # First call should miss cache
        hash1 = self.hasher.hash_state(state)

        # Second call should hit cache
        hash2 = self.hasher.hash_state(state)

        # Results should be the same
        assert jnp.array_equal(hash1, hash2)

        # Cache hits should have increased
        assert self.hasher.stats.value["cache_hits"] > 0

    def test_cache_stats(self):
        """Test cache statistics collection."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Perform some operations
        state1 = jnp.arange(self.state_size, dtype=jnp.float32)
        state2 = jnp.arange(self.state_size, dtype=jnp.float32) + 1

        self.hasher.hash_state(state1)
        self.hasher.hash_state(state2)
        self.hasher.hash_state(state1)  # Should hit cache

        stats = self.hasher.get_cache_stats()

        # Check that stats contain expected keys
        assert "total_hashes" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats
        assert "cache_size" in stats

        # Check that some operations were recorded
        assert stats["total_hashes"] > 0
        assert stats["cache_hits"] > 0

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Add some items to cache
        state = jnp.arange(self.state_size, dtype=jnp.float32)
        self.hasher.hash_state(state)

        # Verify cache has items
        assert len(self.hasher.hash_cache.value) > 0

        # Clear cache
        self.hasher.clear_cache()

        # Cache should be empty
        assert len(self.hasher.hash_cache.value) == 0
        assert self.hasher.stats.value["cache_hits"] == 0.0
        assert self.hasher.stats.value["cache_misses"] == 0.0

    def test_optimize_for_device(self):
        """Test device-specific optimization."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        original_chunk_size = self.hasher.config.chunk_size

        # Mock different device types
        with patch.object(self.hasher.backend, "device_type", "tpu"):
            self.hasher.optimize_for_device()
            # TPU should use smaller chunk size
            assert self.hasher.config.chunk_size <= original_chunk_size
            assert self.hasher.config.memory_efficient is True

        # Reset for next test
        self.hasher.config.chunk_size = original_chunk_size

        with patch.object(self.hasher.backend, "device_type", "cpu"):
            self.hasher.optimize_for_device()
            # CPU should use larger chunk size
            assert self.hasher.config.chunk_size >= original_chunk_size

    def test_cache_size_management(self):
        """Test that cache size is properly managed."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Create hasher with small cache size
        small_config = NNXHasherConfig(max_cache_size=3)
        small_hasher = NNXStateHasher(self.state_size, self.backend, small_config)

        # Add more items than cache size
        for i in range(5):
            state = jnp.arange(self.state_size, dtype=jnp.float32) + i
            small_hasher.hash_state(state)

        # Cache should not exceed max size
        assert len(small_hasher.hash_cache.value) <= small_config.max_cache_size


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestOptimizedNNXStateHasher:
    """Test optimized NNX state hasher functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # pylint: disable=attribute-defined-outside-init
        self.backend = create_nnx_backend(preferred_device="cpu")
        self.state_size = 100
        self.config = NNXHasherConfig(enable_caching=True, chunk_size=1000)
        self.hasher = OptimizedNNXStateHasher(self.state_size, self.backend, self.config)

    def test_optimized_hasher_creation(self):
        """Test optimized hasher creation."""
        assert isinstance(self.hasher, OptimizedNNXStateHasher)
        assert hasattr(self.hasher, "memory_stats")
        assert "peak_memory_mb" in self.hasher.memory_stats.value

    def test_memory_stats(self):
        """Test memory statistics collection."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        stats = self.hasher.get_memory_stats()

        # Check that memory stats contain expected keys
        assert "peak_memory_mb" in stats
        assert "current_memory_mb" in stats
        assert "memory_efficiency" in stats
        assert "gc_collections" in stats

    def test_hash_with_rematerialization(self):
        """Test hash computation with rematerialization."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Test large batch processing (should use rematerialization)
        batch_size = 2000
        states = jnp.arange(batch_size * self.state_size, dtype=jnp.float32).reshape(batch_size, self.state_size)

        hash_results = self.hasher.hash_large_batch(states)

        assert hash_results.shape == (batch_size,)
        assert hash_results.dtype in [jnp.int32, jnp.uint32]


class TestNNXHasherWithoutJAX:
    """Test NNX hasher behavior when JAX is not available."""

    @patch("cayleypy.nnx_hasher.JAX_AVAILABLE", False)
    def test_hasher_creation_without_jax(self):
        """Test that hasher creation fails gracefully without JAX."""
        backend = None  # Mock backend

        with pytest.raises(ImportError, match="JAX and Flax are required"):
            NNXStateHasher(100, backend)

    @patch("cayleypy.nnx_hasher.JAX_AVAILABLE", False)
    def test_optimized_hasher_creation_without_jax(self):
        """Test that optimized hasher creation fails gracefully without JAX."""
        backend = None  # Mock backend

        with pytest.raises(ImportError, match="JAX and Flax are required"):
            OptimizedNNXStateHasher(100, backend)


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_nnx_hasher_with_params(self):
        """Test hasher creation with custom parameters."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        backend = create_nnx_backend(preferred_device="cpu")
        hasher = create_nnx_hasher(
            state_size=50,
            backend=backend,
            optimized=False,
            enable_caching=False,
            chunk_size=25000,
        )

        if backend is not None:
            assert hasher is not None
            assert hasher.state_size == 50
            assert hasher.config.enable_caching is False
            # Note: chunk_size may be modified by optimize_for_device()
        else:
            assert hasher is None

    def test_create_optimized_hasher(self):
        """Test creation of optimized hasher."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        backend = create_nnx_backend(preferred_device="cpu")
        hasher = create_nnx_hasher(state_size=100, backend=backend, optimized=True, enable_caching=True)

        if backend is not None:
            assert hasher is not None
            assert isinstance(hasher, OptimizedNNXStateHasher)
        else:
            assert hasher is None

    @patch("cayleypy.nnx_hasher.JAX_AVAILABLE", False)
    def test_create_hasher_without_jax(self):
        """Test factory function behavior without JAX."""
        backend = None
        hasher = create_nnx_hasher(state_size=100, backend=backend)

        assert hasher is None

    def test_create_hasher_with_exception(self):
        """Test factory function with exception handling."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Mock backend that causes exception
        mock_backend = MagicMock()
        mock_backend.is_available.side_effect = Exception("Test exception")

        with patch("cayleypy.nnx_hasher.NNXStateHasher") as mock_hasher:
            mock_hasher.side_effect = Exception("Test exception")

            hasher = create_nnx_hasher(state_size=100, backend=mock_backend)
            assert hasher is None


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestNNXHasherIntegration:
    """Integration tests for NNX hasher functionality."""

    def test_end_to_end_hashing_workflow(self):
        """Test complete hashing workflow."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        backend = create_nnx_backend(preferred_device="cpu")
        if backend is None:
            pytest.skip("Backend not available")

        hasher = NNXStateHasher(state_size=50, backend=backend)

        # Test single state hashing
        state = jnp.arange(50, dtype=jnp.float32)
        single_hash = hasher.hash_state(state)
        assert single_hash.shape == ()

        # Test batch hashing
        batch_states = jnp.arange(500, dtype=jnp.float32).reshape(10, 50)
        batch_hashes = hasher.hash_batch(batch_states)
        assert batch_hashes.shape == (10,)

        # Test large batch hashing
        large_batch = jnp.arange(5000, dtype=jnp.float32).reshape(100, 50)
        large_hashes = hasher.hash_large_batch(large_batch, chunk_size=25)
        assert large_hashes.shape == (100,)

        # Verify consistency
        first_state_hash_single = hasher.hash_state(batch_states[0])
        first_state_hash_batch = batch_hashes[0]
        assert jnp.allclose(first_state_hash_single, first_state_hash_batch, rtol=1e-5)

    def test_performance_comparison(self):
        """Test performance metrics across different batch sizes."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        backend = create_nnx_backend(preferred_device="cpu")
        if backend is None:
            pytest.skip("Backend not available")

        hasher = NNXStateHasher(state_size=100, backend=backend)

        # Test different batch sizes
        batch_sizes = [10, 100, 1000]
        for batch_size in batch_sizes:
            states = jnp.arange(batch_size * 100, dtype=jnp.float32).reshape(batch_size, 100)
            hashes = hasher.hash_batch(states)
            assert hashes.shape == (batch_size,)

        # Get final statistics
        stats = hasher.get_cache_stats()
        assert stats["total_hashes"] >= sum(batch_sizes)
        assert "avg_batch_size" in stats
        assert "max_batch_size" in stats

    def test_memory_efficiency_comparison(self):
        """Test memory efficiency between regular and optimized hashers."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        backend = create_nnx_backend(preferred_device="cpu")
        if backend is None:
            pytest.skip("Backend not available")

        # Create both types of hashers
        regular_hasher = NNXStateHasher(state_size=100, backend=backend)
        optimized_hasher = OptimizedNNXStateHasher(state_size=100, backend=backend)

        # Test with large batch
        large_batch = jnp.arange(10000, dtype=jnp.float32).reshape(100, 100)

        # Hash with both
        regular_hashes = regular_hasher.hash_large_batch(large_batch)
        optimized_hashes = optimized_hasher.hash_large_batch(large_batch)

        # Results should be similar (allowing for implementation differences)
        assert regular_hashes.shape == optimized_hashes.shape
        assert regular_hashes.dtype == optimized_hashes.dtype

        # Both should have memory stats
        regular_stats = regular_hasher.get_cache_stats()
        optimized_stats = optimized_hasher.get_memory_stats()

        assert "memory_peak_mb" in regular_stats
        assert "peak_memory_mb" in optimized_stats


if __name__ == "__main__":
    pytest.main([__file__])
