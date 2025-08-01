"""TPU Tensor Operations for CayleyPy.

This module provides TPU-accelerated tensor operations with native int64 support,
optimized for TPU v6e (Trillium) architecture with 256x256 systolic arrays.
"""

import logging
from typing import Tuple, Dict, Any, Optional

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    nnx = None  # type: ignore

from .tpu_backend import TPUBackend
from .tpu_kernel_cache import TPUKernelCache, create_kernel_signature


# JIT-compiled helper functions for TPU operations
@jax.jit
def _unique_with_indices_jit(arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled unique operation using JAX's built-in unique."""
    # Use JAX's built-in unique function with size parameter for JIT compatibility
    # Assume worst case: all elements are unique
    max_unique = arr.shape[0]
    unique_vals, unique_indices = jnp.unique(
        arr, return_index=True, size=max_unique, fill_value=jnp.iinfo(arr.dtype).min
    )

    # Return the full arrays - caller can handle filtering if needed
    return unique_vals, unique_indices


@jax.jit
def _isin_jit(elements: jnp.ndarray, test_elements: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled membership testing."""
    # Sort test elements for binary search
    sorted_test = jnp.sort(test_elements)
    indices = jnp.searchsorted(sorted_test, elements)
    indices = jnp.clip(indices, 0, len(sorted_test) - 1)

    # Check membership
    return sorted_test[indices] == elements


@jax.jit
def _batch_apply_permutation_jit(states: jnp.ndarray, perm: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled batch permutation application."""
    # Use vmap for automatic vectorization across TPU v6e cores
    return jax.vmap(lambda state: state[perm])(states)


@jax.jit
def _batch_matrix_multiply_jit(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled batch matrix multiplication."""
    # TPU v6e has enhanced systolic array capabilities
    return jax.vmap(jnp.dot, in_axes=(0, None))(a, b)


@jax.jit
def _deduplicate_int64_states_jit(states: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled state deduplication."""
    # Convert states to int64 hashes for precise deduplication
    state_hashes = jax.vmap(lambda s: jnp.sum(s.astype(jnp.int64) * jnp.arange(len(s), dtype=jnp.int64)))(states)

    _, unique_indices = _unique_with_indices_jit(state_hashes)

    # Filter out fill values by checking for valid indices
    valid_mask = unique_indices >= 0
    valid_indices = jnp.where(valid_mask, unique_indices, 0)  # Replace invalid with 0

    # Use lax.dynamic_index_in_dim for dynamic indexing
    from jax import lax  # pylint: disable=import-outside-toplevel

    result_states = jax.vmap(lambda idx: lax.dynamic_index_in_dim(states, idx, axis=0))(valid_indices)

    # Only return states corresponding to valid indices
    return jnp.where(valid_mask[:, None], result_states, 0)


class TPUTensorOpsModule(nnx.Module):
    """NNX module for TPU-accelerated tensor operations with native int64 support."""

    def __init__(self, backend: TPUBackend, rngs: Optional[nnx.Rngs] = None):
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for TPU tensor operations")

        self.backend = backend

        # Cache for frequently computed operations
        self.operation_cache: nnx.Variable[Dict[str, Any]] = nnx.Variable({})

        # Performance metrics
        self.metrics = nnx.Variable(
            {
                "operations_count": 0,
                "cache_hits": 0,
                "total_elements_processed": 0,
                "int64_operations": 0,
                "systolic_array_utilization": 0.0,
                "memory_peak_mb": 0.0,
                "unique_operations": 0,
                "isin_operations": 0,
                "permutation_operations": 0,
                "deduplication_operations": 0,
            }
        )

        self.logger = logging.getLogger(__name__)

        # Initialize RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(42)
        self.rngs = rngs

        # Initialize kernel cache for tensor operations
        self.kernel_cache = TPUKernelCache(backend, rngs=rngs)

        self.logger.info("TPU Tensor Operations Module initialized")

    def unique_with_indices(self, arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """TPU-optimized unique operation with native int64 support."""
        # Track int64 operations
        if arr.dtype == jnp.int64:
            self.metrics.value["int64_operations"] += 1

        # Create kernel signature for caching
        signature = create_kernel_signature(
            graph=None,  # Tensor operations are graph-independent
            operation_type="unique_with_indices",
            batch_size=len(arr),
            dtype=str(arr.dtype),
        )

        # Get or compile cached kernel
        def compile_unique_kernel():
            return jax.jit(_unique_with_indices_jit)

        cached_kernel = self.kernel_cache.get_or_compile_kernel(signature, compile_unique_kernel)

        # Use cached kernel
        unique_vals, unique_indices = cached_kernel(arr)

        # Update metrics
        self.metrics.value["operations_count"] += 1
        self.metrics.value["unique_operations"] += 1
        self.metrics.value["total_elements_processed"] += len(arr)

        return unique_vals, unique_indices

    def isin(self, elements: jnp.ndarray, test_elements: jnp.ndarray) -> jnp.ndarray:
        """TPU-optimized membership testing with int64 support."""
        # Track operations
        self.metrics.value["isin_operations"] += 1
        if elements.dtype == jnp.int64:
            self.metrics.value["int64_operations"] += 1

        # Use JIT-compiled function
        result = _isin_jit(elements, test_elements)

        # Update metrics
        self.metrics.value["operations_count"] += 1
        self.metrics.value["total_elements_processed"] += len(elements)

        return result

    def batch_apply_permutation(self, states: jnp.ndarray, perm: jnp.ndarray) -> jnp.ndarray:
        """Apply permutation to batch of states using TPU's systolic array."""
        # Track operations
        self.metrics.value["permutation_operations"] += 1
        if states.dtype == jnp.int64:
            self.metrics.value["int64_operations"] += 1

        # Use JIT-compiled function
        result = _batch_apply_permutation_jit(states, perm)

        # Update metrics
        self.metrics.value["operations_count"] += 1
        self.metrics.value["systolic_array_utilization"] += 1.0
        self.metrics.value["total_elements_processed"] += states.size

        return result

    def batch_matrix_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Leverage TPU v6e's 256x256 systolic array for matrix operations."""
        # Track operations
        if jnp.int64 in (a.dtype, b.dtype):
            self.metrics.value["int64_operations"] += 1

        # Use JIT-compiled function
        result = _batch_matrix_multiply_jit(a, b)

        # Track systolic array utilization
        self.metrics.value["systolic_array_utilization"] += 1.0
        self.metrics.value["operations_count"] += 1
        self.metrics.value["total_elements_processed"] += a.size + b.size

        return result

    def deduplicate_int64_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Remove duplicate states using native int64 operations on TPU."""
        # Track operations
        self.metrics.value["deduplication_operations"] += 1
        self.metrics.value["int64_operations"] += 1

        # Use a better hash function to avoid collisions
        # This is a simple polynomial hash with a large prime
        def better_hash(s):
            prime = jnp.int64(1000000007)
            hash_val = jnp.int64(0)
            for i in range(len(s)):
                hash_val = hash_val * prime + s[i].astype(jnp.int64)
            return hash_val

        state_hashes = jax.vmap(better_hash)(states)

        # Get unique hashes and their indices, then filter out fill values
        unique_hashes, unique_indices = self.unique_with_indices(state_hashes)

        # Filter out fill values (negative values from jnp.unique padding)
        fill_value = jnp.iinfo(state_hashes.dtype).min
        valid_mask = unique_hashes != fill_value
        valid_indices = unique_indices[valid_mask]

        # Return states corresponding to valid unique indices
        if len(valid_indices) > 0:
            result = states[valid_indices]
        else:
            # Return empty array with correct shape
            result = jnp.array([], dtype=jnp.int64).reshape(0, states.shape[1])

        # Update metrics
        self.metrics.value["operations_count"] += 1
        self.metrics.value["total_elements_processed"] += states.size

        return result

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return dict(self.metrics.value)

    def reset_metrics(self):
        """Reset performance metrics."""
        for key in self.metrics.value:
            if isinstance(self.metrics.value[key], (int, float)):
                self.metrics.value[key] = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get operation cache statistics."""
        total_ops = self.metrics.value["operations_count"]
        cache_hits = self.metrics.value["cache_hits"]

        return {
            "cache_size": len(self.operation_cache.value),
            "cache_hit_rate": cache_hits / max(1, total_ops),
            "total_operations": total_ops,
            "cache_hits": cache_hits,
        }

    def clear_cache(self):
        """Clear operation cache."""
        self.operation_cache.value.clear()
        self.logger.info("Operation cache cleared")


def create_tpu_tensor_ops(backend: Optional[TPUBackend] = None) -> TPUTensorOpsModule:
    """Factory function to create TPU tensor operations module."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX and Flax are required for TPU tensor operations. " + "Install with: pip install 'cayleypy[jax-tpu]'"
        )

    if backend is None:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel

        backend = get_tpu_backend()

    return TPUTensorOpsModule(backend)


def test_int64_operations():
    """Test native int64 operations on TPU."""
    if not JAX_AVAILABLE:
        print("JAX not available - cannot test TPU int64 operations")
        return False

    try:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel

        # Get TPU backend
        backend = get_tpu_backend()
        if not backend.is_available:
            print("TPU not available - cannot test int64 operations")
            return False

        # Create tensor ops module
        tensor_ops = TPUTensorOpsModule(backend)

        # Test 1: Large int64 values that exceed int32 range
        print("Testing native int64 support on TPU v6e...")
        large_vals = jnp.array([2**40, 2**50, 2**60], dtype=jnp.int64)
        print(f"Large int64 values: {large_vals}")
        print(f"Dtype: {large_vals.dtype}")

        # Test 2: int64 arithmetic
        result = large_vals + jnp.int64(1)
        print(f"After adding 1: {result}")
        print(f"Result dtype: {result.dtype}")

        # Test 3: Unique operation with int64
        test_array = jnp.array([2**40, 2**40, 2**50, 2**50, 2**60], dtype=jnp.int64)
        unique_vals, unique_indices = tensor_ops.unique_with_indices(test_array)
        print(f"Unique values: {unique_vals}")
        print(f"Unique indices: {unique_indices}")

        # Test 4: Membership testing with int64
        elements = jnp.array([2**40, 2**45], dtype=jnp.int64)
        test_elements = jnp.array([2**40, 2**50, 2**60], dtype=jnp.int64)
        membership = tensor_ops.isin(elements, test_elements)
        print(f"Membership test: {membership}")

        # Test 5: Batch permutation with int64
        states = jnp.array([[2**40, 2**50], [2**60, 2**45]], dtype=jnp.int64)
        perm = jnp.array([1, 0])
        permuted = tensor_ops.batch_apply_permutation(states, perm)
        print(f"Permuted states: {permuted}")

        # Get performance metrics
        metrics = tensor_ops.get_performance_metrics()
        print(f"int64 operations performed: {metrics['int64_operations']}")

        print("✓ All int64 operations completed successfully on TPU v6e!")
        return True

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"✗ int64 operations failed: {e}")
        return False


if __name__ == "__main__":
    # Test int64 operations when run as script
    test_int64_operations()
