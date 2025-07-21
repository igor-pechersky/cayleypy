"""Unit tests for JAX tensor operations."""

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
    unique_with_indices, gather_along_axis, searchsorted, isin_via_searchsorted,
    tensor_split, sort_with_indices, concatenate_arrays, stack_arrays,
    zeros_like, ones_like, full_like, arange, batch_matmul, chunked_operation,
    to_jax_array, ensure_jax_array, advanced_indexing, boolean_indexing,
    element_wise_equal, element_wise_not_equal, logical_and, logical_or, logical_not,
    JAX_AVAILABLE
)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXTensorOps:
    """Test cases for JAX tensor operations."""

    def test_unique_with_indices_basic(self):
        """Test basic unique functionality."""
        x = jnp.array([3, 1, 2, 1, 3, 2])
        unique_vals = unique_with_indices(x)
        
        expected = jnp.array([1, 2, 3])
        assert jnp.array_equal(unique_vals, expected)

    def test_unique_with_indices_return_inverse(self):
        """Test unique with inverse indices."""
        x = jnp.array([3, 1, 2, 1, 3, 2])
        unique_vals, inverse = unique_with_indices(x, return_inverse=True)
        
        expected_unique = jnp.array([1, 2, 3])
        expected_inverse = jnp.array([2, 0, 1, 0, 2, 1])
        
        assert jnp.array_equal(unique_vals, expected_unique)
        assert jnp.array_equal(inverse, expected_inverse)

    def test_unique_with_indices_return_counts(self):
        """Test unique with counts."""
        x = jnp.array([3, 1, 2, 1, 3, 2])
        unique_vals, counts = unique_with_indices(x, return_counts=True)
        
        expected_unique = jnp.array([1, 2, 3])
        expected_counts = jnp.array([2, 2, 2])
        
        assert jnp.array_equal(unique_vals, expected_unique)
        assert jnp.array_equal(counts, expected_counts)

    def test_unique_with_indices_all_returns(self):
        """Test unique with both inverse and counts."""
        x = jnp.array([3, 1, 2, 1, 3, 2])
        unique_vals, inverse, counts = unique_with_indices(x, return_inverse=True, return_counts=True)
        
        expected_unique = jnp.array([1, 2, 3])
        expected_inverse = jnp.array([2, 0, 1, 0, 2, 1])
        expected_counts = jnp.array([2, 2, 2])
        
        assert jnp.array_equal(unique_vals, expected_unique)
        assert jnp.array_equal(inverse, expected_inverse)
        assert jnp.array_equal(counts, expected_counts)

    def test_unique_empty_array(self):
        """Test unique with empty array."""
        x = jnp.array([])
        unique_vals = unique_with_indices(x)
        assert len(unique_vals) == 0

    def test_gather_along_axis_2d(self):
        """Test gather operation on 2D arrays."""
        input_array = jnp.array([[1, 2, 3], [4, 5, 6]])
        indices = jnp.array([[0, 2], [1, 0]])
        
        result = gather_along_axis(input_array, indices, axis=1)
        expected = jnp.array([[1, 3], [5, 4]])
        
        assert jnp.array_equal(result, expected)

    def test_searchsorted_basic(self):
        """Test searchsorted functionality."""
        sorted_array = jnp.array([1, 3, 5, 7, 9])
        values = jnp.array([2, 4, 6, 8])
        
        result = searchsorted(sorted_array, values)
        expected = jnp.array([1, 2, 3, 4])
        
        assert jnp.array_equal(result, expected)

    def test_isin_via_searchsorted(self):
        """Test isin functionality using searchsorted."""
        elements = jnp.array([1, 2, 3, 4, 5])
        test_elements = jnp.array([2, 4, 6])
        
        result = isin_via_searchsorted(elements, test_elements)
        expected = jnp.array([False, True, False, True, False])
        
        assert jnp.array_equal(result, expected)

    def test_isin_empty_test_elements(self):
        """Test isin with empty test elements."""
        elements = jnp.array([1, 2, 3])
        test_elements = jnp.array([])
        
        result = isin_via_searchsorted(elements, test_elements)
        expected = jnp.array([False, False, False])
        
        assert jnp.array_equal(result, expected)

    def test_tensor_split(self):
        """Test tensor splitting."""
        array = jnp.arange(12).reshape(4, 3)
        splits = tensor_split(array, 2, axis=0)
        
        assert len(splits) == 2
        assert splits[0].shape == (2, 3)
        assert splits[1].shape == (2, 3)

    def test_sort_with_indices(self):
        """Test sorting with indices."""
        array = jnp.array([3, 1, 4, 1, 5])
        sorted_vals, indices = sort_with_indices(array)
        
        expected_vals = jnp.array([1, 1, 3, 4, 5])
        expected_indices = jnp.array([1, 3, 0, 2, 4])
        
        assert jnp.array_equal(sorted_vals, expected_vals)
        assert jnp.array_equal(indices, expected_indices)

    def test_concatenate_arrays(self):
        """Test array concatenation."""
        arrays = [jnp.array([1, 2]), jnp.array([3, 4]), jnp.array([5, 6])]
        result = concatenate_arrays(arrays, axis=0)
        expected = jnp.array([1, 2, 3, 4, 5, 6])
        
        assert jnp.array_equal(result, expected)

    def test_stack_arrays(self):
        """Test array stacking."""
        arrays = [jnp.array([1, 2]), jnp.array([3, 4])]
        result = stack_arrays(arrays, axis=0)
        expected = jnp.array([[1, 2], [3, 4]])
        
        assert jnp.array_equal(result, expected)

    def test_zeros_like(self):
        """Test zeros_like functionality."""
        array = jnp.array([[1, 2], [3, 4]])
        result = zeros_like(array)
        expected = jnp.zeros((2, 2))
        
        assert jnp.array_equal(result, expected)

    def test_ones_like(self):
        """Test ones_like functionality."""
        array = jnp.array([[1, 2], [3, 4]])
        result = ones_like(array)
        expected = jnp.ones((2, 2))
        
        assert jnp.array_equal(result, expected)

    def test_full_like(self):
        """Test full_like functionality."""
        array = jnp.array([[1, 2], [3, 4]])
        result = full_like(array, 7)
        expected = jnp.full((2, 2), 7)
        
        assert jnp.array_equal(result, expected)

    def test_arange(self):
        """Test arange functionality."""
        result = arange(5)
        expected = jnp.array([0, 1, 2, 3, 4])
        assert jnp.array_equal(result, expected)
        
        result = arange(2, 8, 2)
        expected = jnp.array([2, 4, 6])
        assert jnp.array_equal(result, expected)

    def test_batch_matmul(self):
        """Test batch matrix multiplication."""
        a = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = jnp.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
        
        result = batch_matmul(a, b)
        
        # First batch: identity multiplication
        assert jnp.array_equal(result[0], a[0])
        # Second batch: multiplication by 2*identity
        assert jnp.array_equal(result[1], 2 * a[1])

    def test_chunked_operation(self):
        """Test chunked operation processing."""
        large_array = jnp.arange(1000).reshape(100, 10)
        
        def sum_operation(chunk):
            return jnp.sum(chunk, axis=1)
        
        result = chunked_operation(large_array, sum_operation, chunk_size=30)
        expected = jnp.sum(large_array, axis=1)
        
        assert jnp.array_equal(result, expected)

    def test_to_jax_array(self):
        """Test conversion to JAX array."""
        # From list
        result = to_jax_array([1, 2, 3])
        expected = jnp.array([1, 2, 3])
        assert jnp.array_equal(result, expected)
        
        # From numpy array
        np_array = np.array([4, 5, 6])
        result = to_jax_array(np_array)
        expected = jnp.array([4, 5, 6])
        assert jnp.array_equal(result, expected)

    def test_ensure_jax_array(self):
        """Test ensuring input is JAX array."""
        # Already JAX array
        jax_array = jnp.array([1, 2, 3])
        result = ensure_jax_array(jax_array)
        assert result is jax_array
        
        # Convert from list
        result = ensure_jax_array([4, 5, 6])
        expected = jnp.array([4, 5, 6])
        assert jnp.array_equal(result, expected)

    def test_boolean_indexing(self):
        """Test boolean indexing."""
        array = jnp.array([1, 2, 3, 4, 5])
        mask = jnp.array([True, False, True, False, True])
        
        result = boolean_indexing(array, mask)
        expected = jnp.array([1, 3, 5])
        
        assert jnp.array_equal(result, expected)

    def test_element_wise_operations(self):
        """Test element-wise comparison operations."""
        a = jnp.array([1, 2, 3])
        b = jnp.array([1, 3, 2])
        
        # Equality
        eq_result = element_wise_equal(a, b)
        expected_eq = jnp.array([True, False, False])
        assert jnp.array_equal(eq_result, expected_eq)
        
        # Inequality
        neq_result = element_wise_not_equal(a, b)
        expected_neq = jnp.array([False, True, True])
        assert jnp.array_equal(neq_result, expected_neq)

    def test_logical_operations(self):
        """Test logical operations."""
        a = jnp.array([True, False, True])
        b = jnp.array([True, True, False])
        
        # AND
        and_result = logical_and(a, b)
        expected_and = jnp.array([True, False, False])
        assert jnp.array_equal(and_result, expected_and)
        
        # OR
        or_result = logical_or(a, b)
        expected_or = jnp.array([True, True, True])
        assert jnp.array_equal(or_result, expected_or)
        
        # NOT
        not_result = logical_not(a)
        expected_not = jnp.array([False, True, False])
        assert jnp.array_equal(not_result, expected_not)


class TestJAXNotAvailable:
    """Test behavior when JAX is not available."""

    @patch('cayleypy.jax_tensor_ops.JAX_AVAILABLE', False)
    def test_operations_without_jax(self):
        """Test that operations raise ImportError when JAX not available."""
        with pytest.raises(ImportError, match="JAX is not available"):
            unique_with_indices(jnp.array([1, 2, 3]))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXTensorOpsPerformance:
    """Performance tests for JAX tensor operations."""

    def test_large_unique_operation(self):
        """Test unique operation on large arrays."""
        # Create large array with some duplicates
        large_array = jnp.concatenate([
            jnp.arange(10000),
            jnp.arange(5000),  # Add some duplicates
        ])
        
        unique_vals = unique_with_indices(large_array)
        
        # Should have 10000 unique values
        assert len(unique_vals) == 10000
        assert jnp.array_equal(unique_vals, jnp.arange(10000))

    def test_large_gather_operation(self):
        """Test gather operation on large arrays."""
        large_input = jnp.arange(10000).reshape(1000, 10)
        indices = jnp.tile(jnp.array([0, 5, 9]), (1000, 1))
        
        result = gather_along_axis(large_input, indices, axis=1)
        
        assert result.shape == (1000, 3)
        # Check first row
        assert jnp.array_equal(result[0], jnp.array([0, 5, 9]))

    def test_chunked_operation_memory_efficiency(self):
        """Test that chunked operations work with large arrays."""
        # Create array larger than typical chunk size
        large_array = jnp.arange(100000).reshape(10000, 10)
        
        def mean_operation(chunk):
            return jnp.mean(chunk, axis=1)
        
        result = chunked_operation(large_array, mean_operation, chunk_size=1000)
        expected = jnp.mean(large_array, axis=1)
        
        assert jnp.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])