"""JAX equivalents for PyTorch tensor operations used in CayleyPy.

This module provides JAX implementations of key tensor operations that replace
PyTorch functionality, optimized for TPU/GPU computation with JIT compilation.
"""

from typing import Tuple, Union, Optional
import warnings

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


def _check_jax_available():
    """Check if JAX is available and raise error if not."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not available. Install with: pip install jax[tpu] or pip install jax[cuda]"
        )


def unique_with_indices(x: jnp.ndarray, return_inverse: bool = False, return_counts: bool = False) -> Union[
    jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
]:
    """JAX equivalent of torch.unique with return_inverse and return_counts support.
    
    Args:
        x: Input array to find unique elements
        return_inverse: If True, return inverse indices
        return_counts: If True, return counts of unique elements
        
    Returns:
        unique_values: Unique elements in sorted order
        inverse_indices: (optional) Indices to reconstruct original array
        counts: (optional) Count of each unique element
    """
    _check_jax_available()
    
    # Use JAX's built-in unique for simple case
    if not return_inverse and not return_counts:
        return jnp.unique(x)
    
    # For complex cases, avoid JIT to handle dynamic shapes
    x_flat = x.flatten()
    sorted_indices = jnp.argsort(x_flat)
    sorted_values = x_flat[sorted_indices]
    
    if len(sorted_values) == 0:
        empty_vals = jnp.array([], dtype=x.dtype)
        empty_indices = jnp.array([], dtype=jnp.int32)
        if return_inverse and return_counts:
            return empty_vals, empty_indices, empty_indices
        elif return_inverse:
            return empty_vals, empty_indices
        elif return_counts:
            return empty_vals, empty_indices
        else:
            return empty_vals
    
    # Find unique elements
    diff_mask = jnp.concatenate([
        jnp.array([True]),
        sorted_values[1:] != sorted_values[:-1]
    ])
    
    # Use jnp.where with size parameter to avoid dynamic shapes
    unique_positions = jnp.where(diff_mask)[0]
    unique_values = sorted_values[unique_positions]
    
    results = [unique_values]
    
    if return_inverse:
        # Create inverse mapping
        cumsum_mask = jnp.cumsum(diff_mask) - 1
        inverse_indices = jnp.zeros_like(x_flat, dtype=jnp.int32)
        inverse_indices = inverse_indices.at[sorted_indices].set(cumsum_mask)
        inverse_indices = inverse_indices.reshape(x.shape)
        results.append(inverse_indices)
    
    if return_counts:
        # Calculate counts
        next_positions = jnp.concatenate([unique_positions[1:], jnp.array([len(sorted_values)])])
        counts = next_positions - unique_positions
        results.append(counts)
    
    return tuple(results) if len(results) > 1 else results[0]


def gather_along_axis(input_array: jnp.ndarray, indices: jnp.ndarray, axis: int = 1) -> jnp.ndarray:
    """JAX equivalent of torch.gather for gathering elements along specified axis.
    
    Args:
        input_array: Source array to gather from
        indices: Indices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered elements
    """
    _check_jax_available()
    
    # Remove JIT to avoid tracer issues with dynamic axis
    # Handle negative axis
    if axis < 0:
        axis = input_array.ndim + axis
    
    # Use jnp.take_along_axis which is designed for this purpose
    return jnp.take_along_axis(input_array, indices, axis=axis)


@jit
def searchsorted(sorted_sequence: jnp.ndarray, values: jnp.ndarray, side: str = 'left') -> jnp.ndarray:
    """JAX equivalent of torch.searchsorted for finding insertion points.
    
    Args:
        sorted_sequence: Sorted 1-D array
        values: Values to find insertion points for
        side: 'left' or 'right' for insertion side
        
    Returns:
        Insertion indices
    """
    _check_jax_available()
    
    return jnp.searchsorted(sorted_sequence, values, side=side)


def isin_via_searchsorted(elements: jnp.ndarray, test_elements_sorted: jnp.ndarray) -> jnp.ndarray:
    """JAX equivalent of the optimized isin function using searchsorted.
    
    This is a direct port of the PyTorch version for compatibility.
    
    Args:
        elements: Elements to test for membership
        test_elements_sorted: Sorted array to test membership against
        
    Returns:
        Boolean array indicating membership
    """
    _check_jax_available()
    
    # Handle empty test_elements case directly
    if len(test_elements_sorted) == 0:
        return jnp.zeros_like(elements, dtype=bool)
    
    # Find insertion points
    indices = jnp.searchsorted(test_elements_sorted, elements)
    # Clamp indices to valid range
    indices = jnp.clip(indices, 0, len(test_elements_sorted) - 1)
    # Check if elements match at insertion points
    return test_elements_sorted[indices] == elements


def tensor_split(array: jnp.ndarray, sections: int, axis: int = 0) -> list:
    """JAX equivalent of torch.tensor_split.
    
    Args:
        array: Array to split
        sections: Number of sections to split into
        axis: Axis along which to split
        
    Returns:
        List of split arrays
    """
    _check_jax_available()
    
    # Remove JIT to avoid concrete value requirements
    return jnp.array_split(array, sections, axis=axis)


@jit
def sort_with_indices(array: jnp.ndarray, axis: int = -1, stable: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX equivalent of torch.sort returning both values and indices.
    
    Args:
        array: Array to sort
        axis: Axis along which to sort
        stable: Whether to use stable sorting
        
    Returns:
        Tuple of (sorted_values, sort_indices)
    """
    _check_jax_available()
    
    indices = jnp.argsort(array, axis=axis, stable=stable)
    sorted_values = jnp.take_along_axis(array, indices, axis=axis)
    
    return sorted_values, indices


def concatenate_arrays(arrays: list, axis: int = 0) -> jnp.ndarray:
    """JAX equivalent of torch.cat/torch.hstack/torch.vstack.
    
    Args:
        arrays: List of arrays to concatenate
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated array
    """
    _check_jax_available()
    
    # Remove JIT to avoid tracer issues
    return jnp.concatenate(arrays, axis=axis)


def stack_arrays(arrays: list, axis: int = 0) -> jnp.ndarray:
    """JAX equivalent of torch.stack.
    
    Args:
        arrays: List of arrays to stack
        axis: Axis along which to stack
        
    Returns:
        Stacked array
    """
    _check_jax_available()
    
    # Remove JIT to avoid tracer issues
    return jnp.stack(arrays, axis=axis)


@jit
def zeros_like(array: jnp.ndarray, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """JAX equivalent of torch.zeros_like.
    
    Args:
        array: Reference array for shape and dtype
        dtype: Optional dtype override
        
    Returns:
        Zero array with same shape
    """
    _check_jax_available()
    
    return jnp.zeros_like(array, dtype=dtype)


@jit
def ones_like(array: jnp.ndarray, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """JAX equivalent of torch.ones_like.
    
    Args:
        array: Reference array for shape and dtype
        dtype: Optional dtype override
        
    Returns:
        Ones array with same shape
    """
    _check_jax_available()
    
    return jnp.ones_like(array, dtype=dtype)


@jit
def full_like(array: jnp.ndarray, fill_value: Union[int, float], dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """JAX equivalent of torch.full_like.
    
    Args:
        array: Reference array for shape
        fill_value: Value to fill with
        dtype: Optional dtype override
        
    Returns:
        Array filled with specified value
    """
    _check_jax_available()
    
    return jnp.full_like(array, fill_value, dtype=dtype)


def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype: jnp.dtype = jnp.int32) -> jnp.ndarray:
    """JAX equivalent of torch.arange.
    
    Args:
        start: Start value (or stop if stop is None)
        stop: Stop value
        step: Step size
        dtype: Data type
        
    Returns:
        Range array
    """
    _check_jax_available()
    
    # Remove JIT to avoid concrete value requirements
    if stop is None:
        stop = start
        start = 0
    
    return jnp.arange(start, stop, step, dtype=dtype)


@jit
def batch_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Vectorized batch matrix multiplication.
    
    Args:
        a: First batch of matrices
        b: Second batch of matrices
        
    Returns:
        Batch matrix multiplication result
    """
    _check_jax_available()
    
    return jnp.matmul(a, b)


def chunked_operation(array: jnp.ndarray, operation_fn, chunk_size: int = 2**18) -> jnp.ndarray:
    """Apply operation to array in chunks for memory efficiency.
    
    Args:
        array: Input array
        operation_fn: Function to apply to each chunk
        chunk_size: Size of each chunk
        
    Returns:
        Result of applying operation to all chunks
    """
    _check_jax_available()
    
    if array.shape[0] <= chunk_size:
        return operation_fn(array)
    
    # Split into chunks and process
    num_chunks = (array.shape[0] + chunk_size - 1) // chunk_size
    chunks = jnp.array_split(array, num_chunks, axis=0)
    
    # Process each chunk and concatenate results
    results = [operation_fn(chunk) for chunk in chunks]
    
    return jnp.concatenate(results, axis=0)


def to_jax_array(data, dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
    """Convert various data types to JAX arrays.
    
    Args:
        data: Input data (list, tuple, numpy array, etc.)
        dtype: Optional dtype specification
        
    Returns:
        JAX array
    """
    _check_jax_available()
    
    return jnp.array(data, dtype=dtype)


def ensure_jax_array(data) -> jnp.ndarray:
    """Ensure input is a JAX array, converting if necessary.
    
    Args:
        data: Input data
        
    Returns:
        JAX array
    """
    _check_jax_available()
    
    if isinstance(data, jnp.ndarray):
        return data
    return jnp.array(data)


@jit
def advanced_indexing(array: jnp.ndarray, indices: tuple) -> jnp.ndarray:
    """Advanced indexing operation for JAX arrays.
    
    Args:
        array: Source array
        indices: Tuple of index arrays
        
    Returns:
        Indexed array
    """
    _check_jax_available()
    
    return array[indices]


def boolean_indexing(array: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Boolean indexing for JAX arrays.
    
    Args:
        array: Source array
        mask: Boolean mask
        
    Returns:
        Filtered array
    """
    _check_jax_available()
    
    # Use jnp.where to avoid boolean indexing issues
    indices = jnp.where(mask)[0]
    return array[indices]


@jit
def element_wise_equal(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Element-wise equality comparison.
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Boolean array of comparisons
    """
    _check_jax_available()
    
    return a == b


@jit
def element_wise_not_equal(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Element-wise inequality comparison.
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Boolean array of comparisons
    """
    _check_jax_available()
    
    return a != b


@jit
def logical_and(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Element-wise logical AND.
    
    Args:
        a: First boolean array
        b: Second boolean array
        
    Returns:
        Boolean array result
    """
    _check_jax_available()
    
    return jnp.logical_and(a, b)


@jit
def logical_or(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Element-wise logical OR.
    
    Args:
        a: First boolean array
        b: Second boolean array
        
    Returns:
        Boolean array result
    """
    _check_jax_available()
    
    return jnp.logical_or(a, b)


@jit
def logical_not(a: jnp.ndarray) -> jnp.ndarray:
    """Element-wise logical NOT.
    
    Args:
        a: Boolean array
        
    Returns:
        Boolean array result
    """
    _check_jax_available()
    
    return jnp.logical_not(a)