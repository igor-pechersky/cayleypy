"""Unit tests for JAX state hashing system."""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from cayleypy.jax_hasher import (
    _splitmix64_jax, JAXStateHasher, JAXBatchHasher, create_hash_function,
    hash_state_collection, fast_hash_comparison, find_hash_duplicates,
    benchmark_hash_performance, JAX_AVAILABLE
)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestSplitMix64:
    """Test SplitMix64 hash function."""

    def test_splitmix64_basic(self):
        """Test basic SplitMix64 functionality."""
        x = jnp.array([1, 2, 3, 4, 5])
        result = _splitmix64_jax(x)
        
        # Should return different values for different inputs
        assert len(jnp.unique(result)) == len(x)
        
        # Should be deterministic
        result2 = _splitmix64_jax(x)
        assert jnp.array_equal(result, result2)

    def test_splitmix64_zero(self):
        """Test SplitMix64 with zero input."""
        x = jnp.array([0])
        result = _splitmix64_jax(x)
        
        # Should produce non-zero hash for zero input
        assert result[0] != 0

    def test_splitmix64_large_values(self):
        """Test SplitMix64 with large values."""
        x = jnp.array([2**60, 2**61, 2**62 - 1])
        result = _splitmix64_jax(x)
        
        # Should handle large values without overflow issues
        assert len(result) == 3
        assert jnp.all(jnp.isfinite(result))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXStateHasher:
    """Test JAXStateHasher class."""

    def test_identity_hasher(self):
        """Test hasher with single-element states (identity)."""
        hasher = JAXStateHasher(state_size=1, random_seed=42)
        
        assert hasher.is_identity
        
        states = jnp.array([[1], [2], [3]])
        hashes = hasher.hash_states(states)
        
        expected = jnp.array([1, 2, 3])
        assert jnp.array_equal(hashes, expected)

    def test_dot_product_hasher(self):
        """Test hasher with dot product method."""
        hasher = JAXStateHasher(state_size=3, random_seed=42, use_string_encoder=False)
        
        assert not hasher.is_identity
        assert not hasher.use_string_encoder
        
        states = jnp.array([[1, 2, 3], [4, 5, 6]])
        hashes = hasher.hash_states(states)
        
        # Should produce different hashes for different states
        assert hashes[0] != hashes[1]
        
        # Should be deterministic
        hashes2 = hasher.hash_states(states)
        assert jnp.array_equal(hashes, hashes2)

    def test_splitmix64_hasher(self):
        """Test hasher with SplitMix64 method."""
        hasher = JAXStateHasher(state_size=4, random_seed=42, use_string_encoder=True)
        
        assert not hasher.is_identity
        assert hasher.use_string_encoder
        
        states = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        hashes = hasher.hash_states(states)
        
        # Should produce different hashes for different states
        assert hashes[0] != hashes[1]
        
        # Should be deterministic
        hashes2 = hasher.hash_states(states)
        assert jnp.array_equal(hashes, hashes2)

    def test_hash_single_state(self):
        """Test hashing single state."""
        hasher = JAXStateHasher(state_size=3, random_seed=42)
        
        state = jnp.array([1, 2, 3])
        hash_value = hasher.hash_single_state(state)
        
        assert isinstance(hash_value, int)
        
        # Should be same as batch hashing
        batch_hash = hasher.hash_states(state.reshape(1, -1))
        assert hash_value == int(batch_hash[0])

    def test_different_seeds_different_hashes(self):
        """Test that different seeds produce different hashes."""
        hasher1 = JAXStateHasher(state_size=3, random_seed=42)
        hasher2 = JAXStateHasher(state_size=3, random_seed=123)
        
        states = jnp.array([[1, 2, 3], [4, 5, 6]])
        
        hashes1 = hasher1.hash_states(states)
        hashes2 = hasher2.hash_states(states)
        
        # Different seeds should produce different hashes
        assert not jnp.array_equal(hashes1, hashes2)

    def test_chunked_processing(self):
        """Test chunked processing for large arrays."""
        hasher = JAXStateHasher(state_size=5, random_seed=42, chunk_size=10)
        
        # Create array larger than chunk size
        large_states = jnp.arange(100).reshape(20, 5)
        hashes = hasher.hash_states(large_states)
        
        assert len(hashes) == 20
        
        # Should be same as processing in smaller chunks
        chunk1 = hasher.hash_states(large_states[:10])
        chunk2 = hasher.hash_states(large_states[10:])
        combined = jnp.concatenate([chunk1, chunk2])
        
        assert jnp.array_equal(hashes, combined)

    def test_state_shape_handling(self):
        """Test handling of different state shapes."""
        hasher = JAXStateHasher(state_size=4, random_seed=42)
        
        # 1D state
        state_1d = jnp.array([1, 2, 3, 4])
        hash_1d = hasher.hash_states(state_1d)
        
        # 2D state (single state)
        state_2d = jnp.array([[1, 2, 3, 4]])
        hash_2d = hasher.hash_states(state_2d)
        
        # Should produce same hash
        assert jnp.array_equal(hash_1d, hash_2d)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXBatchHasher:
    """Test JAXBatchHasher class."""

    def test_hash_multiple_batches(self):
        """Test hashing multiple batches."""
        base_hasher = JAXStateHasher(state_size=3, random_seed=42)
        batch_hasher = JAXBatchHasher(base_hasher, max_batch_size=5)
        
        batches = [
            jnp.array([[1, 2, 3], [4, 5, 6]]),
            jnp.array([[7, 8, 9]]),
            jnp.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
        ]
        
        results = batch_hasher.hash_multiple_batches(batches)
        
        assert len(results) == 3
        assert len(results[0]) == 2
        assert len(results[1]) == 1
        assert len(results[2]) == 3

    def test_hash_and_concatenate(self):
        """Test hashing and concatenating results."""
        base_hasher = JAXStateHasher(state_size=2, random_seed=42)
        batch_hasher = JAXBatchHasher(base_hasher)
        
        batches = [
            jnp.array([[1, 2], [3, 4]]),
            jnp.array([[5, 6]])
        ]
        
        result = batch_hasher.hash_and_concatenate(batches)
        
        assert len(result) == 3
        
        # Should be same as individual hashing
        expected = jnp.concatenate([
            base_hasher.hash_states(batches[0]),
            base_hasher.hash_states(batches[1])
        ])
        assert jnp.array_equal(result, expected)

    def test_large_batch_splitting(self):
        """Test automatic splitting of large batches."""
        base_hasher = JAXStateHasher(state_size=2, random_seed=42)
        batch_hasher = JAXBatchHasher(base_hasher, max_batch_size=10)
        
        # Create batch larger than max_batch_size
        large_batch = jnp.arange(30).reshape(15, 2)
        
        results = batch_hasher.hash_multiple_batches([large_batch])
        
        assert len(results) == 1
        assert len(results[0]) == 15
        
        # Should be same as direct hashing
        expected = base_hasher.hash_states(large_batch)
        assert jnp.array_equal(results[0], expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_hash_function(self):
        """Test creating hash function."""
        # Regular encoding
        hasher1 = create_hash_function(state_size=3, encoding_type="regular", random_seed=42)
        assert not hasher1.use_string_encoder
        
        # String encoding
        hasher2 = create_hash_function(state_size=3, encoding_type="string", random_seed=42)
        assert hasher2.use_string_encoder

    def test_hash_state_collection_array(self):
        """Test hashing state collection (single array)."""
        hasher = JAXStateHasher(state_size=2, random_seed=42)
        states = jnp.array([[1, 2], [3, 4], [5, 6]])
        
        result = hash_state_collection(states, hasher)
        expected = hasher.hash_states(states)
        
        assert jnp.array_equal(result, expected)

    def test_hash_state_collection_list(self):
        """Test hashing state collection (list of arrays)."""
        hasher = JAXStateHasher(state_size=2, random_seed=42)
        states = [
            [[1, 2], [3, 4]],
            [[5, 6]]
        ]
        
        result = hash_state_collection(states, hasher)
        
        assert len(result) == 3

    def test_fast_hash_comparison(self):
        """Test fast hash comparison."""
        hashes1 = jnp.array([1, 2, 3, 4])
        hashes2 = jnp.array([1, 5, 3, 7])
        
        result = fast_hash_comparison(hashes1, hashes2)
        expected = jnp.array([True, False, True, False])
        
        assert jnp.array_equal(result, expected)

    def test_find_hash_duplicates(self):
        """Test finding hash duplicates."""
        hashes = jnp.array([1, 2, 1, 3, 2, 4])
        
        unique_hashes, inverse, counts = find_hash_duplicates(hashes)
        
        expected_unique = jnp.array([1, 2, 3, 4])
        expected_counts = jnp.array([2, 2, 1, 1])
        
        assert jnp.array_equal(unique_hashes, expected_unique)
        assert jnp.array_equal(counts, expected_counts)
        
        # Reconstruct original from inverse
        reconstructed = unique_hashes[inverse]
        assert jnp.array_equal(reconstructed, hashes)

    def test_benchmark_hash_performance(self):
        """Test hash performance benchmarking."""
        hasher = JAXStateHasher(state_size=5, random_seed=42)
        test_states = jnp.arange(500).reshape(100, 5)
        
        stats = benchmark_hash_performance(hasher, test_states, num_iterations=3)
        
        assert "mean_time" in stats
        assert "min_time" in stats
        assert "max_time" in stats
        assert "states_per_second" in stats
        
        assert stats["mean_time"] > 0
        assert stats["states_per_second"] > 0


class TestJAXHasherNotAvailable:
    """Test behavior when JAX is not available."""

    def test_hasher_without_jax(self):
        """Test that JAXStateHasher raises ImportError when JAX not available."""
        with pytest.patch('cayleypy.jax_hasher.JAX_AVAILABLE', False):
            with pytest.raises(ImportError, match="JAX is not available"):
                JAXStateHasher(state_size=3)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXHasherPerformance:
    """Performance tests for JAX hasher."""

    def test_large_state_hashing(self):
        """Test hashing large number of states."""
        hasher = JAXStateHasher(state_size=10, random_seed=42)
        
        # Create large batch of states
        large_states = jnp.arange(10000).reshape(1000, 10)
        hashes = hasher.hash_states(large_states)
        
        assert len(hashes) == 1000
        
        # Should have good distribution (most hashes unique)
        unique_hashes = jnp.unique(hashes)
        collision_rate = 1.0 - len(unique_hashes) / len(hashes)
        assert collision_rate < 0.01  # Less than 1% collisions

    def test_chunked_vs_direct_performance(self):
        """Test that chunked processing gives same results as direct."""
        # Small chunk size to force chunking
        hasher = JAXStateHasher(state_size=8, random_seed=42, chunk_size=50)
        
        states = jnp.arange(800).reshape(100, 8)
        
        # Hash with chunking
        chunked_hashes = hasher.hash_states(states)
        
        # Hash directly (increase chunk size)
        hasher.chunk_size = 1000
        direct_hashes = hasher.hash_states(states)
        
        assert jnp.array_equal(chunked_hashes, direct_hashes)

    def test_string_encoder_vs_dot_product(self):
        """Test performance difference between encoding methods."""
        hasher_dot = JAXStateHasher(state_size=6, random_seed=42, use_string_encoder=False)
        hasher_split = JAXStateHasher(state_size=6, random_seed=42, use_string_encoder=True)
        
        states = jnp.arange(600).reshape(100, 6)
        
        hashes_dot = hasher_dot.hash_states(states)
        hashes_split = hasher_split.hash_states(states)
        
        # Both should produce valid hashes
        assert len(hashes_dot) == 100
        assert len(hashes_split) == 100
        
        # Should be different due to different methods
        assert not jnp.array_equal(hashes_dot, hashes_split)

    def test_memory_efficiency_large_arrays(self):
        """Test memory efficiency with very large arrays."""
        hasher = JAXStateHasher(state_size=20, random_seed=42, chunk_size=1000)
        
        # Create very large state array
        very_large_states = jnp.arange(20000).reshape(1000, 20)
        
        hashes = hasher.hash_states(very_large_states)
        
        assert len(hashes) == 1000
        
        # Check hash quality
        unique_hashes = jnp.unique(hashes)
        assert len(unique_hashes) > 990  # Very low collision rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])