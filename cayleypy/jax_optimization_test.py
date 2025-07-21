"""Test script to verify JAX/TPU optimizations are working correctly."""

import pytest
import numpy as np
from unittest.mock import patch

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from cayleypy.jax_tensor_ops import (
    vectorized_element_wise_equal, batch_isin_via_searchsorted,
    distributed_batch_matmul, memory_efficient_unique,
    optimized_chunked_operation, JAX_AVAILABLE
)

from cayleypy.jax_hasher import (
    JAXStateHasher, OptimizedJAXStateHasher, vectorized_hash_states,
    distributed_hash_states, memory_efficient_hash_large_batch
)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXOptimizations:
    """Test cases for JAX/TPU optimizations."""
    
    def test_vectorized_element_wise_equal(self):
        """Test vectorized element-wise equality."""
        a = jnp.array([[1, 2, 3], [4, 5, 6]])
        b = jnp.array([[1, 2, 3], [4, 5, 7]])
        
        result = vectorized_element_wise_equal(a, b)
        expected = jnp.array([[True, True, True], [True, True, False]])
        
        assert jnp.array_equal(result, expected)
    
    def test_batch_isin_via_searchsorted(self):
        """Test batch isin operation."""
        elements_batch = jnp.array([[1, 2, 3], [4, 5, 6]])
        test_elements = jnp.array([2, 4, 6])
        
        result = batch_isin_via_searchsorted(elements_batch, test_elements)
        expected = jnp.array([[False, True, False], [True, False, True]])
        
        assert jnp.array_equal(result, expected)
    
    def test_distributed_batch_matmul(self):
        """Test distributed batch matrix multiplication."""
        a = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = jnp.array([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
        
        result = distributed_batch_matmul(a, b)
        expected = jnp.matmul(a, b)
        
        assert jnp.allclose(result, expected)
    
    def test_memory_efficient_unique(self):
        """Test memory-efficient unique operation."""
        x = jnp.array([3, 1, 2, 1, 3, 2, 4])
        
        result = memory_efficient_unique(x, max_memory_gb=0.001)  # Force chunking
        expected = jnp.unique(x)
        
        assert jnp.array_equal(jnp.sort(result), expected)
    
    def test_optimized_chunked_operation(self):
        """Test optimized chunked operation."""
        array = jnp.arange(1000).reshape(100, 10)
        
        def sum_operation(chunk):
            return jnp.sum(chunk, axis=1)
        
        result = optimized_chunked_operation(array, sum_operation, chunk_size=30, use_scan=True)
        expected = jnp.sum(array, axis=1)
        
        assert jnp.array_equal(result, expected)
    
    def test_jax_state_hasher_vectorized(self):
        """Test vectorized state hashing."""
        hasher = JAXStateHasher(state_size=4, random_seed=42)
        states = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        
        # Test standard hashing
        result_standard = hasher.hash_states(states)
        
        # Test vectorized hashing
        result_vectorized = vectorized_hash_states(states, hasher)
        
        # Results should be the same
        assert jnp.array_equal(result_standard, result_vectorized)
    
    def test_optimized_jax_state_hasher(self):
        """Test optimized JAX state hasher."""
        hasher = OptimizedJAXStateHasher(state_size=4, random_seed=42, enable_sharding=False)
        states = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        
        result_standard = hasher.hash_states(states)
        result_optimized = hasher.hash_states_optimized(states)
        
        # Results should be the same
        assert jnp.array_equal(result_standard, result_optimized)
    
    def test_distributed_hash_states(self):
        """Test distributed state hashing."""
        hasher = JAXStateHasher(state_size=4, random_seed=42)
        states = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        
        result_standard = hasher.hash_states(states)
        result_distributed = distributed_hash_states(states, hasher)
        
        # Results should be the same
        assert jnp.array_equal(result_standard, result_distributed)
    
    def test_memory_efficient_hash_large_batch(self):
        """Test memory-efficient hashing for large batches."""
        hasher = JAXStateHasher(state_size=4, random_seed=42)
        states = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        
        result_standard = hasher.hash_states(states)
        result_efficient = memory_efficient_hash_large_batch(states, hasher, max_memory_gb=0.001)
        
        # Results should be the same
        assert jnp.array_equal(result_standard, result_efficient)
    
    def test_splitmix64_optimization(self):
        """Test SplitMix64 optimization with string encoder."""
        hasher = JAXStateHasher(state_size=4, random_seed=42, use_string_encoder=True)
        states = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int64)
        
        result = hasher.hash_states(states)
        
        # Should produce valid hash values
        assert result.shape == (2,)
        assert result.dtype == jnp.int64
    
    def test_identity_hash_optimization(self):
        """Test identity hash optimization for single-element states."""
        hasher = JAXStateHasher(state_size=1, random_seed=42)
        states = jnp.array([[1], [2], [3]])
        
        result = hasher.hash_states(states)
        expected = jnp.array([1, 2, 3])
        
        assert jnp.array_equal(result, expected)


if __name__ == "__main__":
    # Run a quick test to verify optimizations
    if JAX_AVAILABLE:
        print("Testing JAX/TPU optimizations...")
        
        # Test vectorized operations
        a = jnp.array([[1, 2], [3, 4]])
        b = jnp.array([[1, 2], [3, 5]])
        result = vectorized_element_wise_equal(a, b)
        print(f"Vectorized equality test: {result}")
        
        # Test optimized hasher
        hasher = OptimizedJAXStateHasher(state_size=3, random_seed=42)
        states = jnp.array([[1, 2, 3], [4, 5, 6]])
        hashes = hasher.hash_states_optimized(states)
        print(f"Optimized hashing test: {hashes}")
        
        print("JAX/TPU optimizations are working correctly!")
    else:
        print("JAX not available - skipping optimization tests")