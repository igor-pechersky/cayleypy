"""NNX Tensor Operations Module for JAX/GPU/TPU acceleration in CayleyPy.

This module provides core tensor operations implemented as NNX modules with automatic
state management, caching, and performance tracking. It includes optimized implementations
for unique element detection, searchsorted operations, and vectorized computations.
"""

import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

try:
    import jax.numpy as jnp
    from flax import nnx
    from .nnx_backend import NNXBackend, JAX_AVAILABLE
except ImportError:
    JAX_AVAILABLE = False
    jnp = None  # type: ignore
    nnx = None  # type: ignore


@dataclass
class TensorOpsConfig:
    """Configuration for tensor operations module."""

    # Caching settings
    enable_caching: bool = True
    max_cache_size: int = 1000
    cache_ttl_seconds: float = 300.0  # 5 minutes

    # Performance settings
    chunk_size: int = 10000
    enable_jit: bool = True
    enable_vmap: bool = True

    # Memory management
    memory_efficient: bool = True
    max_memory_mb: float = 512.0


class TensorOpsModule(nnx.Module if JAX_AVAILABLE else object):  # type: ignore
    """NNX module for tensor operations with automatic state management and caching.

    This module provides optimized implementations of core tensor operations needed
    for graph algorithms, including unique element detection, searchsorted operations,
    and vectorized computations. All operations are JIT-compiled and support automatic
    batching through vmap.
    """

    def __init__(self, backend: NNXBackend, config: Optional[TensorOpsConfig] = None, rngs: Optional[Any] = None):
        """Initialize tensor operations module.

        Args:
            backend: NNX backend for device management and configuration
            config: Configuration for tensor operations. If None, uses default.
            rngs: Random number generators for NNX. If None, creates default.
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX and Flax are required for NNX tensor operations. Install with: pip install 'cayleypy[jax]'"
            )

        self.backend = backend
        self.config = config or TensorOpsConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(42)
        self.rngs = rngs

        # Operation cache for frequently used computations
        self.cache: Optional[nnx.Variable] = nnx.Variable({}) if self.config.enable_caching else None

        # Performance metrics tracking
        self.metrics = nnx.Variable(
            {
                "unique_calls": 0.0,
                "searchsorted_calls": 0.0,
                "vmap_calls": 0.0,
                "cache_hits": 0.0,
                "cache_misses": 0.0,
                "total_operations": 0.0,
                "memory_peak_mb": 0.0,
                "compilation_time_ms": 0.0,
            }
        )

        # Compilation cache for JIT functions
        self._compiled_functions: Dict[str, Any] = {}

        self.logger.info(
            "TensorOpsModule initialized with caching=%s, JIT=%s", self.config.enable_caching, self.config.enable_jit
        )

    def _get_cache_key(self, operation: str, *args, **kwargs) -> str:
        """Generate cache key for operation with arguments."""
        if not self.config.enable_caching:
            return ""

        try:
            # Create a simple hash-based key from operation name and array shapes/dtypes
            key_parts = [operation]
            for arg in args:
                if hasattr(arg, "shape") and hasattr(arg, "dtype"):
                    key_parts.append(f"{arg.shape}_{arg.dtype}")
                else:
                    key_parts.append(str(hash(str(arg))))

            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={hash(str(v))}")

            return "_".join(key_parts)
        except Exception:  # pylint: disable=broad-exception-caught
            # If hashing fails, disable caching for this operation
            return ""

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve result from cache if available."""
        if not self.config.enable_caching or not cache_key or self.cache is None:
            return None

        if cache_key in self.cache.value:
            self.metrics.value["cache_hits"] += 1.0
            return self.cache.value[cache_key]

        self.metrics.value["cache_misses"] += 1.0
        return None

    def _store_in_cache(self, cache_key: str, result: Any) -> None:
        """Store result in cache with size management."""
        if not self.config.enable_caching or not cache_key or self.cache is None:
            return

        # Simple cache size management - remove oldest entries if needed
        if len(self.cache.value) >= self.config.max_cache_size:
            # Remove first (oldest) entry
            oldest_key = next(iter(self.cache.value))
            del self.cache.value[oldest_key]

        self.cache.value[cache_key] = result

    def unique_with_indices(
        self,
        arr: jnp.ndarray,
        return_inverse: bool = True,
        return_counts: bool = False,
        size: Optional[int] = None,  # pylint: disable=unused-argument
    ) -> Tuple[jnp.ndarray, ...]:
        """Optimized unique operation with indices and optional caching.

        Args:
            arr: Input array to find unique elements in
            return_inverse: Whether to return inverse indices
            return_counts: Whether to return counts of unique elements
            size: Maximum number of unique elements (for compatibility)

        Returns:
            Tuple containing unique elements and optionally inverse indices and counts
        """
        self.metrics.value["unique_calls"] += 1.0
        self.metrics.value["total_operations"] += 1.0

        # Use JAX's built-in unique function without JIT for dynamic shapes
        # JIT compilation is problematic for unique operations due to dynamic output sizes
        return jnp.unique(arr, return_inverse=return_inverse, return_counts=return_counts)

    @nnx.jit
    def isin_via_searchsorted(self, elements: jnp.ndarray, test_elements: jnp.ndarray) -> jnp.ndarray:
        """Fast membership testing using searchsorted with NNX optimization.

        This is more efficient than jnp.isin for large sorted arrays.

        Args:
            elements: Elements to test for membership
            test_elements: Sorted array to test membership against

        Returns:
            Boolean array indicating membership
        """
        self.metrics.value["searchsorted_calls"] += 1.0
        self.metrics.value["total_operations"] += 1.0

        # Ensure test_elements is sorted for searchsorted to work correctly
        sorted_test = jnp.sort(test_elements)

        # Find insertion points
        indices = jnp.searchsorted(sorted_test, elements)

        # Clamp indices to valid range
        indices = jnp.clip(indices, 0, len(sorted_test) - 1)

        # Check if elements match at found positions
        return sorted_test[indices] == elements

    @nnx.jit
    def searchsorted_batched(self, sorted_arrays: jnp.ndarray, values: jnp.ndarray, side: str = "left") -> jnp.ndarray:
        """Batched searchsorted operation using vmap.

        Args:
            sorted_arrays: Batch of sorted arrays with shape (batch_size, array_size)
            values: Values to search for with shape (batch_size, num_values)
            side: 'left' or 'right' for insertion side

        Returns:
            Indices array with shape (batch_size, num_values)
        """
        self.metrics.value["vmap_calls"] += 1.0
        self.metrics.value["total_operations"] += 1.0

        # Use vmap to vectorize searchsorted across the batch dimension
        def single_searchsorted(sorted_arr, vals):
            return jnp.searchsorted(sorted_arr, vals, side=side)

        vectorized_searchsorted = nnx.vmap(single_searchsorted, in_axes=(0, 0))
        return vectorized_searchsorted(sorted_arrays, values)

    @nnx.jit
    def batch_matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Vectorized matrix multiplication using NNX vmap.

        Args:
            a: First matrix operand
            b: Second matrix operand

        Returns:
            Matrix product result
        """
        self.metrics.value["vmap_calls"] += 1.0
        self.metrics.value["total_operations"] += 1.0

        if self.config.enable_vmap and a.ndim > 2 and b.ndim > 2:
            # Use vmap for batched operations
            vectorized_dot = nnx.vmap(jnp.dot, in_axes=(0, 0))
            return vectorized_dot(a, b)
        else:
            # Regular matrix multiplication
            return jnp.dot(a, b)

    @nnx.jit
    def vectorized_element_wise_equal(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Vectorized element-wise equality comparison.

        Args:
            a: First array operand
            b: Second array operand

        Returns:
            Boolean array indicating element-wise equality
        """
        self.metrics.value["vmap_calls"] += 1.0
        self.metrics.value["total_operations"] += 1.0

        if self.config.enable_vmap and a.ndim > 1 and b.ndim > 1:
            # Use vmap for batched comparisons
            vectorized_equal = nnx.vmap(jnp.array_equal, in_axes=(0, 0))
            return vectorized_equal(a, b)
        else:
            # Regular element-wise equality
            return jnp.array_equal(a, b)

    def unique_counts_sorted(self, arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get unique elements and their counts from a sorted array efficiently.

        This is optimized for the case where the input array is already sorted,
        which is common in graph algorithms.

        Args:
            arr: Sorted input array

        Returns:
            Tuple of (unique_elements, counts)
        """
        self.metrics.value["unique_calls"] += 1.0
        self.metrics.value["total_operations"] += 1.0

        if len(arr) == 0:
            return jnp.array([]), jnp.array([])

        # Use JAX's unique with return_counts (no JIT due to dynamic shapes)
        return jnp.unique(arr, return_counts=True)

    @nnx.jit
    def argsort_stable(self, arr: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        """Stable argsort implementation with performance tracking.

        Args:
            arr: Array to sort
            axis: Axis along which to sort

        Returns:
            Indices that would sort the array
        """
        self.metrics.value["total_operations"] += 1.0
        return jnp.argsort(arr, axis=axis, stable=True)

    @nnx.jit
    def gather_batched(
        self,
        params: jnp.ndarray,
        indices: jnp.ndarray,
        axis: int = 0,
        batch_dims: int = 0,  # pylint: disable=unused-argument
    ) -> jnp.ndarray:
        """Batched gather operation with advanced indexing.

        Args:
            params: Array to gather from
            indices: Indices to gather
            axis: Axis to gather along
            batch_dims: Number of batch dimensions

        Returns:
            Gathered elements
        """
        self.metrics.value["total_operations"] += 1.0

        # Use JAX's advanced indexing which is optimized for XLA
        return jnp.take(params, indices, axis=axis)

    @nnx.jit
    def scatter_add_batched(
        self, operand: jnp.ndarray, scatter_indices: jnp.ndarray, updates: jnp.ndarray
    ) -> jnp.ndarray:
        """Batched scatter-add operation for efficient sparse updates.

        Args:
            operand: Array to scatter into
            scatter_indices: Indices where to scatter
            updates: Values to scatter

        Returns:
            Updated array with scattered values added
        """
        self.metrics.value["total_operations"] += 1.0

        return operand.at[scatter_indices].add(updates)

    def process_large_array(
        self, arr: jnp.ndarray, operation: str, chunk_size: Optional[int] = None, **kwargs
    ) -> jnp.ndarray:
        """Process large arrays in chunks to manage memory usage.

        Args:
            arr: Large input array
            operation: Name of operation to apply ('unique', 'sort', etc.)
            chunk_size: Size of chunks to process. If None, uses config default.
            **kwargs: Additional arguments for the operation

        Returns:
            Processed array result
        """
        if not self.config.memory_efficient:
            # Process entire array at once
            return self._apply_operation(arr, operation, **kwargs)

        chunk_size = chunk_size or self.config.chunk_size

        if len(arr) <= chunk_size:
            return self._apply_operation(arr, operation, **kwargs)

        # Process in chunks and combine results
        results = []
        for i in range(0, len(arr), chunk_size):
            chunk = arr[i : i + chunk_size]
            chunk_result = self._apply_operation(chunk, operation, **kwargs)
            results.append(chunk_result)

        # Combine chunk results based on operation type
        if operation == "unique":
            # For unique, we need to find unique across all chunks
            combined = jnp.concatenate(results)
            return self.unique_with_indices(combined, **kwargs)[0]  # Return only unique values
        elif operation == "sort":
            # For sort, merge sorted chunks
            return self._merge_sorted_chunks(results)
        else:
            # Default: concatenate results
            return jnp.concatenate(results)

    def _apply_operation(self, arr: jnp.ndarray, operation: str, **kwargs) -> jnp.ndarray:
        """Apply specified operation to array."""
        if operation == "unique":
            # Provide size parameter for JIT compatibility
            size = kwargs.get("size", len(arr))
            return self.unique_with_indices(arr, size=size, **kwargs)[0]
        elif operation == "sort":
            return jnp.sort(arr)
        elif operation == "argsort":
            return self.argsort_stable(arr, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    @nnx.jit
    def _merge_sorted_chunks(self, sorted_chunks: list) -> jnp.ndarray:
        """Merge multiple sorted arrays into a single sorted array."""
        if len(sorted_chunks) == 1:
            return sorted_chunks[0]

        # Simple merge - for production, could use a more efficient k-way merge
        result = sorted_chunks[0]
        for chunk in sorted_chunks[1:]:
            combined = jnp.concatenate([result, chunk])
            result = jnp.sort(combined)

        return result

    def clear_cache(self) -> None:
        """Clear the operation cache."""
        if self.cache is not None:
            self.cache.value.clear()
            self.logger.info("Tensor operations cache cleared")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics: Dict[str, Any] = dict(self.metrics.value)

        # Add cache statistics
        if self.cache is not None:
            cache_size = len(self.cache.value)
            total_cache_ops = metrics["cache_hits"] + metrics["cache_misses"]
            hit_rate = float(metrics["cache_hits"]) / max(1, total_cache_ops)

            metrics.update(
                {
                    "cache_size": cache_size,
                    "cache_hit_rate": hit_rate,
                    "cache_efficiency": hit_rate * cache_size / max(1, self.config.max_cache_size),
                }
            )

        # Add configuration info
        config_dict: Dict[str, Any] = {
            "enable_caching": self.config.enable_caching,
            "enable_jit": self.config.enable_jit,
            "chunk_size": float(self.config.chunk_size),
            "memory_efficient": self.config.memory_efficient,
        }

        backend_info: Optional[Dict[str, Any]] = self.backend.get_device_info() if self.backend else None

        metrics.update(
            {
                "config": config_dict,
                "backend_info": backend_info,
            }
        )

        return metrics

    def optimize_for_device(self) -> None:
        """Optimize tensor operations for the current device."""
        if not self.backend or not self.backend.is_available():
            return

        device_type = self.backend.device_type

        if device_type == "tpu":
            # TPU optimizations
            self.config.chunk_size = min(self.config.chunk_size, 8192)
            self.config.memory_efficient = True
            self.logger.info("Optimized tensor operations for TPU")
        elif device_type == "gpu":
            # GPU optimizations
            self.config.chunk_size = min(self.config.chunk_size, 16384)
            self.logger.info("Optimized tensor operations for GPU")
        else:
            # CPU optimizations
            self.config.chunk_size = max(self.config.chunk_size, 32768)
            self.logger.info("Optimized tensor operations for CPU")


def create_tensor_ops_module(
    backend: NNXBackend, enable_caching: bool = True, chunk_size: int = 10000, **kwargs
) -> Optional[TensorOpsModule]:
    """Factory function to create tensor operations module with error handling.

    Args:
        backend: NNX backend for device management
        enable_caching: Whether to enable operation caching
        chunk_size: Default chunk size for large array processing
        **kwargs: Additional configuration options

    Returns:
        TensorOpsModule instance if successful, None if JAX is not available
    """
    if not JAX_AVAILABLE:
        return None

    try:
        config = TensorOpsConfig(enable_caching=enable_caching, chunk_size=chunk_size, **kwargs)

        module = TensorOpsModule(backend, config)
        module.optimize_for_device()

        return module

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).warning("Failed to create tensor operations module: %s", e)
        return None
