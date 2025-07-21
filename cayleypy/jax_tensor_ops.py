"""JAX equivalents for PyTorch tensor operations used in CayleyPy.

This module provides JAX implementations of key tensor operations that replace
PyTorch functionality, optimized for TPU/GPU computation with JIT compilation.
"""

from typing import Tuple, Union, Optional, TYPE_CHECKING, Any
import warnings

if TYPE_CHECKING:
    import jax.numpy as jnp
    JaxArray = jnp.ndarray
else:
    JaxArray = Any

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax
    from jax.experimental import pjit
    from jax.experimental.pjit import PartitionSpec as P
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    lax = None
    pjit = None
    P = None
    # Create dummy decorators when JAX is not available
    def jit(func):
        return func
    def vmap(func):
        return func


def _check_jax_available():
    """Check if JAX is available and raise error if not."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not available. Install with: pip install jax[tpu] or pip install jax[cuda]"
        )


@jit
def unique_with_indices(x: JaxArray, return_inverse: bool = False, return_counts: bool = False) -> Union[
    JaxArray, Tuple[JaxArray, JaxArray], Tuple[JaxArray, JaxArray, JaxArray]
]:
    """JAX equivalent of torch.unique with return_inverse and return_counts support.
    
    Optimized for JIT compilation with static shapes and TPU efficiency.
    
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
    
    x_flat = x.flatten()
    n = x_flat.shape[0]
    
    # Handle empty array case
    def empty_case():
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
    
    def non_empty_case():
        sorted_indices = jnp.argsort(x_flat)
        sorted_values = x_flat[sorted_indices]
        
        # Find unique elements using lax.scan for JIT compatibility
        def scan_fn(carry, i):
            prev_val, unique_mask = carry
            current_val = sorted_values[i]
            is_unique = (i == 0) | (current_val != prev_val)
            new_mask = unique_mask.at[i].set(is_unique)
            return (current_val, new_mask), None
        
        init_mask = jnp.zeros(n, dtype=bool)
        (_, diff_mask), _ = lax.scan(scan_fn, (sorted_values[0], init_mask), jnp.arange(n))
        
        # Get unique positions and values
        unique_positions = jnp.where(diff_mask, size=n, fill_value=n)[0]
        num_unique = jnp.sum(diff_mask)
        unique_positions = unique_positions[:num_unique]
        unique_values = sorted_values[unique_positions]
        
        results = [unique_values]
        
        if return_inverse:
            # Create inverse mapping using cumsum
            cumsum_mask = jnp.cumsum(diff_mask) - 1
            inverse_indices = jnp.zeros_like(x_flat, dtype=jnp.int32)
            inverse_indices = inverse_indices.at[sorted_indices].set(cumsum_mask)
            inverse_indices = inverse_indices.reshape(x.shape)
            results.append(inverse_indices)
        
        if return_counts:
            # Calculate counts efficiently
            next_positions = jnp.concatenate([unique_positions[1:], jnp.array([n])])
            counts = next_positions - unique_positions
            results.append(counts)
        
        return tuple(results) if len(results) > 1 else results[0]
    
    return lax.cond(n == 0, empty_case, non_empty_case)


@jit
def gather_along_axis(input_array: JaxArray, indices: JaxArray, axis: int = 1) -> JaxArray:
    """JAX equivalent of torch.gather for gathering elements along specified axis.
    
    Optimized for JIT compilation with static axis handling.
    
    Args:
        input_array: Source array to gather from
        indices: Indices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered elements
    """
    _check_jax_available()
    
    # Handle negative axis at compile time
    axis = axis if axis >= 0 else input_array.ndim + axis
    
    # Use jnp.take_along_axis which is TPU-optimized
    return jnp.take_along_axis(input_array, indices, axis=axis)


@jit
def searchsorted(sorted_sequence: JaxArray, values: JaxArray, side: str = 'left') -> JaxArray:
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


@jit
def isin_via_searchsorted(elements: JaxArray, test_elements_sorted: JaxArray) -> JaxArray:
    """JAX equivalent of the optimized isin function using searchsorted.
    
    Optimized for JIT compilation and TPU efficiency.
    
    Args:
        elements: Elements to test for membership
        test_elements_sorted: Sorted array to test membership against
        
    Returns:
        Boolean array indicating membership
    """
    _check_jax_available()
    
    def empty_case():
        return jnp.zeros_like(elements, dtype=bool)
    
    def non_empty_case():
        # Find insertion points
        indices = jnp.searchsorted(test_elements_sorted, elements)
        # Clamp indices to valid range
        indices = jnp.clip(indices, 0, len(test_elements_sorted) - 1)
        # Check if elements match at insertion points
        return test_elements_sorted[indices] == elements
    
    return lax.cond(
        len(test_elements_sorted) == 0,
        empty_case,
        non_empty_case
    )


def tensor_split(array: JaxArray, sections: int, axis: int = 0) -> list:
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
def sort_with_indices(array: JaxArray, axis: int = -1, stable: bool = True) -> Tuple[JaxArray, JaxArray]:
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


@jit
def concatenate_arrays(arrays: list, axis: int = 0) -> JaxArray:
    """JAX equivalent of torch.cat/torch.hstack/torch.vstack.
    
    Optimized for JIT compilation and TPU efficiency.
    
    Args:
        arrays: List of arrays to concatenate
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated array
    """
    _check_jax_available()
    
    return jnp.concatenate(arrays, axis=axis)


@jit
def stack_arrays(arrays: list, axis: int = 0) -> JaxArray:
    """JAX equivalent of torch.stack.
    
    Optimized for JIT compilation and TPU efficiency.
    
    Args:
        arrays: List of arrays to stack
        axis: Axis along which to stack
        
    Returns:
        Stacked array
    """
    _check_jax_available()
    
    return jnp.stack(arrays, axis=axis)


@jit
def zeros_like(array: JaxArray, dtype: Optional[Any] = None) -> JaxArray:
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
def ones_like(array: JaxArray, dtype: Optional[Any] = None) -> JaxArray:
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
def full_like(array: JaxArray, fill_value: Union[int, float], dtype: Optional[Any] = None) -> JaxArray:
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


@jit
def arange(start: int, stop: Optional[int] = None, step: int = 1, dtype: Any = None) -> JaxArray:
    """JAX equivalent of torch.arange.
    
    Optimized for JIT compilation with static parameter handling.
    
    Args:
        start: Start value (or stop if stop is None)
        stop: Stop value
        step: Step size
        dtype: Data type
        
    Returns:
        Range array
    """
    _check_jax_available()
    
    # Handle None case with lax.cond for JIT compatibility
    def with_stop():
        return jnp.arange(start, stop, step, dtype=dtype)
    
    def without_stop():
        return jnp.arange(0, start, step, dtype=dtype)
    
    return lax.cond(stop is None, without_stop, with_stop)


@jit
def batch_matmul(a: JaxArray, b: JaxArray) -> JaxArray:
    """Vectorized batch matrix multiplication.
    
    Args:
        a: First batch of matrices
        b: Second batch of matrices
        
    Returns:
        Batch matrix multiplication result
    """
    _check_jax_available()
    
    return jnp.matmul(a, b)


def chunked_operation(array: JaxArray, operation_fn, chunk_size: int = 2**18) -> JaxArray:
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


def to_jax_array(data, dtype: Optional[Any] = None) -> JaxArray:
    """Convert various data types to JAX arrays.
    
    Args:
        data: Input data (list, tuple, numpy array, etc.)
        dtype: Optional dtype specification
        
    Returns:
        JAX array
    """
    _check_jax_available()
    
    return jnp.array(data, dtype=dtype)


def ensure_jax_array(data) -> JaxArray:
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
def advanced_indexing(array: JaxArray, indices: tuple) -> JaxArray:
    """Advanced indexing operation for JAX arrays.
    
    Args:
        array: Source array
        indices: Tuple of index arrays
        
    Returns:
        Indexed array
    """
    _check_jax_available()
    
    return array[indices]


def boolean_indexing(array: JaxArray, mask: JaxArray) -> JaxArray:
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
def element_wise_equal(a: JaxArray, b: JaxArray) -> JaxArray:
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
def element_wise_not_equal(a: JaxArray, b: JaxArray) -> JaxArray:
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
def logical_and(a: JaxArray, b: JaxArray) -> JaxArray:
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
def logical_or(a: JaxArray, b: JaxArray) -> JaxArray:
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
def logical_not(a: JaxArray) -> JaxArray:
    """Element-wise logical NOT.
    
    Args:
        a: Boolean array
        
    Returns:
        Boolean array result
    """
    _check_jax_available()
    
    return jnp.logical_not(a)


# TPU Sharding and Parallelization Functions
def create_sharded_array(array: JaxArray, axis: int = 0) -> JaxArray:
    """Create a sharded array for multi-device computation.
    
    Args:
        array: Input array to shard
        axis: Axis along which to shard
        
    Returns:
        Sharded array
    """
    _check_jax_available()
    
    if not JAX_AVAILABLE or pjit is None:
        return array
    
    # Create partition spec for sharding along specified axis
    partition_spec = P('devices') if axis == 0 else P(None, 'devices')
    
    @pjit(in_axis_resources=partition_spec, out_axis_resources=partition_spec)
    def shard_fn(x):
        return x
    
    return shard_fn(array)


@jit
def distributed_batch_operation(arrays: JaxArray, operation_fn) -> JaxArray:
    """Apply operation to arrays with automatic distribution across devices.
    
    Args:
        arrays: Input arrays
        operation_fn: Function to apply
        
    Returns:
        Result of distributed operation
    """
    _check_jax_available()
    
    # Use vmap for vectorization across batch dimension
    vectorized_op = vmap(operation_fn, in_axes=0, out_axes=0)
    return vectorized_op(arrays)


# Vectorized Operations using vmap
@jit
def vectorized_searchsorted(sorted_sequences: JaxArray, values: JaxArray) -> JaxArray:
    """Vectorized searchsorted operation across multiple sequences.
    
    Args:
        sorted_sequences: Batch of sorted sequences, shape (batch, seq_len)
        values: Values to search for, shape (batch, num_values)
        
    Returns:
        Insertion indices for each sequence
    """
    _check_jax_available()
    
    vectorized_fn = vmap(jnp.searchsorted, in_axes=(0, 0), out_axes=0)
    return vectorized_fn(sorted_sequences, values)


@jit
def vectorized_unique(arrays: JaxArray) -> JaxArray:
    """Vectorized unique operation across multiple arrays.
    
    Args:
        arrays: Batch of arrays, shape (batch, array_len)
        
    Returns:
        Unique elements for each array (padded to same length)
    """
    _check_jax_available()
    
    vectorized_fn = vmap(jnp.unique, in_axes=0, out_axes=0)
    return vectorized_fn(arrays)


@jit
def vectorized_gather(input_arrays: JaxArray, indices: JaxArray, axis: int = -1) -> JaxArray:
    """Vectorized gather operation across multiple arrays.
    
    Args:
        input_arrays: Batch of input arrays
        indices: Batch of indices to gather
        axis: Axis along which to gather
        
    Returns:
        Gathered elements for each array
    """
    _check_jax_available()
    
    def single_gather(arr, idx):
        return jnp.take_along_axis(arr, idx, axis=axis)
    
    vectorized_fn = vmap(single_gather, in_axes=(0, 0), out_axes=0)
    return vectorized_fn(input_arrays, indices)


# Memory-Efficient Operations for Large Arrays
@jit
def memory_efficient_matmul(a: JaxArray, b: JaxArray, chunk_size: int = 2**16) -> JaxArray:
    """Memory-efficient matrix multiplication with automatic chunking.
    
    Args:
        a: First matrix
        b: Second matrix
        chunk_size: Size of chunks for processing
        
    Returns:
        Matrix multiplication result
    """
    _check_jax_available()
    
    def chunk_matmul(a_chunk):
        return jnp.matmul(a_chunk, b)
    
    # Use lax.map for efficient chunked processing
    if a.shape[0] > chunk_size:
        chunks = jnp.array_split(a, (a.shape[0] + chunk_size - 1) // chunk_size)
        results = lax.map(chunk_matmul, jnp.stack(chunks))
        return jnp.concatenate(results, axis=0)
    else:
        return jnp.matmul(a, b)


@jit
def optimized_boolean_indexing(array: JaxArray, mask: JaxArray, max_size: Optional[int] = None) -> JaxArray:
    """Optimized boolean indexing with static shape handling.
    
    Args:
        array: Source array
        mask: Boolean mask
        max_size: Maximum expected output size for static shapes
        
    Returns:
        Filtered array with static shape
    """
    _check_jax_available()
    
    if max_size is not None:
        # Use static shape version for better TPU performance
        indices = jnp.where(mask, size=max_size, fill_value=0)[0]
        return array[indices]
    else:
        # Fallback to dynamic version
        indices = jnp.where(mask)[0]
        return array[indices]


# Advanced JAX/TPU optimizations

@vmap
def vectorized_element_wise_equal(a: JaxArray, b: JaxArray) -> JaxArray:
    """Vectorized element-wise equality using vmap."""
    return a == b


@vmap
def vectorized_hash_single_state(state: JaxArray, hash_params: dict) -> JaxArray:
    """Vectorized single state hashing using vmap."""
    # This would be used with a hash function
    return jnp.sum(state)  # Placeholder implementation


def batch_isin_via_searchsorted(elements_batch: JaxArray, test_elements_sorted: JaxArray) -> JaxArray:
    """Batch version of isin using vmap for multiple element arrays."""
    _check_jax_available()
    
    vectorized_isin = vmap(isin_via_searchsorted, in_axes=(0, None))
    return vectorized_isin(elements_batch, test_elements_sorted)


def batch_unique_with_indices(x_batch: JaxArray) -> Tuple[JaxArray, JaxArray, JaxArray]:
    """Batch version of unique_with_indices using vmap."""
    _check_jax_available()
    
    vectorized_unique = vmap(lambda x: unique_with_indices(x, return_inverse=True, return_counts=True))
    return vectorized_unique(x_batch)


# TPU sharding support
if JAX_AVAILABLE and pjit is not None:
    @pjit(
        in_axis_resources=(P('batch'), P('batch')),
        out_axis_resources=P('batch')
    )
    def distributed_batch_matmul(a: JaxArray, b: JaxArray) -> JaxArray:
        """Distributed batch matrix multiplication across TPU cores."""
        return jnp.matmul(a, b)
    
    @pjit(
        in_axis_resources=(P('batch'),),
        out_axis_resources=P('batch')
    )
    def distributed_sort_with_indices(array: JaxArray) -> Tuple[JaxArray, JaxArray]:
        """Distributed sorting across TPU cores."""
        return sort_with_indices(array)
    
    @pjit(
        in_axis_resources=(P('batch'), None),
        out_axis_resources=P('batch')
    )
    def distributed_isin_via_searchsorted(elements: JaxArray, test_elements_sorted: JaxArray) -> JaxArray:
        """Distributed isin operation across TPU cores."""
        return isin_via_searchsorted(elements, test_elements_sorted)
else:
    # Fallback implementations when pjit is not available
    def distributed_batch_matmul(a: JaxArray, b: JaxArray) -> JaxArray:
        return batch_matmul(a, b)
    
    def distributed_sort_with_indices(array: JaxArray) -> Tuple[JaxArray, JaxArray]:
        return sort_with_indices(array)
    
    def distributed_isin_via_searchsorted(elements: JaxArray, test_elements_sorted: JaxArray) -> JaxArray:
        return isin_via_searchsorted(elements, test_elements_sorted)


# Memory-efficient operations for large arrays
def memory_efficient_unique(x: JaxArray, max_memory_gb: float = 4.0) -> JaxArray:
    """Memory-efficient unique operation for very large arrays."""
    _check_jax_available()
    
    # Estimate memory usage (rough approximation)
    element_size = x.dtype.itemsize
    array_size_gb = (x.size * element_size) / (1024**3)
    
    if array_size_gb <= max_memory_gb:
        return jnp.unique(x)
    
    # Process in chunks for large arrays
    chunk_size = int(max_memory_gb * (1024**3) / element_size)
    chunks = jnp.array_split(x.flatten(), max(1, x.size // chunk_size))
    
    # Get unique elements from each chunk
    unique_chunks = [jnp.unique(chunk) for chunk in chunks]
    
    # Combine and get final unique elements
    combined = jnp.concatenate(unique_chunks)
    return jnp.unique(combined)


def optimized_chunked_operation(array: JaxArray, operation_fn, chunk_size: int = 2**18, 
                               use_scan: bool = True) -> JaxArray:
    """Optimized chunked operation using lax.scan for better TPU performance."""
    _check_jax_available()
    
    if array.shape[0] <= chunk_size:
        return operation_fn(array)
    
    if use_scan and JAX_AVAILABLE:
        # Use lax.scan for better compilation
        num_chunks = (array.shape[0] + chunk_size - 1) // chunk_size
        chunks = jnp.array_split(array, num_chunks, axis=0)
        
        def scan_fn(carry, chunk):
            result = operation_fn(chunk)
            return carry, result
        
        _, results = lax.scan(scan_fn, None, jnp.stack(chunks))
        return jnp.concatenate(results, axis=0)
    else:
        # Fallback to original implementation
        return chunked_operation(array, operation_fn, chunk_size)


# Gradient checkpointing for memory efficiency
if JAX_AVAILABLE:
    try:
        from jax.experimental import remat
        
        @remat
        def memory_efficient_batch_matmul(a: JaxArray, b: JaxArray) -> JaxArray:
            """Memory-efficient batch matmul with gradient checkpointing."""
            return jnp.matmul(a, b)
        
        @remat
        def memory_efficient_sort_with_indices(array: JaxArray) -> Tuple[JaxArray, JaxArray]:
            """Memory-efficient sorting with gradient checkpointing."""
            return sort_with_indices(array)
    except ImportError:
        # Fallback if remat is not available
        def memory_efficient_batch_matmul(a: JaxArray, b: JaxArray) -> JaxArray:
            return batch_matmul(a, b)
        
        def memory_efficient_sort_with_indices(array: JaxArray) -> Tuple[JaxArray, JaxArray]:
            return sort_with_indices(array)