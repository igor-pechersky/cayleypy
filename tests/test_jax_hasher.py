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
    _splitmix64_jax,
    JAXStateHasher,
    JAXBatchHasher,
    create_hash_function,
    hash_state_collection,
    fast_hash_comparison,
    find_hash_duplicates,
    benchmark_hash_performance,
    JAX_AVAILABLE,
)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
