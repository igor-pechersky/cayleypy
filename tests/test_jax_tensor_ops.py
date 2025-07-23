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
    unique_with_indices,
    gather_along_axis,
    searchsorted,
    isin_via_searchsorted,
    tensor_split,
    sort_with_indices,
    concatenate_arrays,
    stack_arrays,
    zeros_like,
    ones_like,
    full_like,
    arange,
    batch_matmul,
    chunked_operation,
    to_jax_array,
    ensure_jax_array,
    advanced_indexing,
    boolean_indexing,
    element_wise_equal,
    element_wise_not_equal,
    logical_and,
    logical_or,
    logical_not,
    JAX_AVAILABLE,
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

    def test_gather_along_axis_2d(self):
        """Test gather operation on 2D arrays."""
        input_array = jnp.array([[1, 2, 3], [4, 5, 6]])
        indices = jnp.array([[0, 2], [1, 0]])

        result = gather_along_axis(input_array, indices, axis=1)
        expected = jnp.array([[1, 3], [5, 4]])

        assert jnp.array_equal(result, expected)

    def test_isin_via_searchsorted(self):
        """Test isin functionality using searchsorted."""
        elements = jnp.array([1, 2, 3, 4, 5])
        test_elements = jnp.array([2, 4, 6])

        result = isin_via_searchsorted(elements, test_elements)
        expected = jnp.array([False, True, False, True, False])

        assert jnp.array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
