"""NNX Hash Functions Module for JAX/GPU/TPU acceleration in CayleyPy.

This module provides hash functions implemented as NNX modules with automatic
state management, caching, and performance tracking. It includes optimized implementations
for state hashing with vectorization and memory-efficient batch processing.
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from .nnx_backend import NNXBackend, JAX_AVAILABLE
    from .hasher import StateHasher
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    nnx = None  # type: ignore


@dataclass
class NNXHasherConfig:
    """Configuration for NNX hash functions."""

    # Hash matrix settings
    hash_bits: int = 64
    hash_seed: int = 42

    # Caching settings
    enable_caching: bool = True
    max_cache_size: int = 10000
    cache_ttl_seconds: float = 600.0  # 10 minutes

    # Performance settings
    chunk_size: int = 50000
    enable_jit: bool = True
    enable_vmap: bool = True

    # Memory management
    memory_efficient: bool = True
    max_memory_mb: float = 1024.0


if JAX_AVAILABLE:

    class NNXStateHasher(nnx.Module):
        """NNX-based state hashing with automatic state management and caching.

        This module provides optimized hash functions for state vectors using JAX
        acceleration. It includes automatic caching, performance tracking, and
        memory-efficient batch processing.
        """

        def __init__(
            self,
            state_size: int,
            backend: NNXBackend,
            config: Optional[NNXHasherConfig] = None,
            rngs: Optional[nnx.Rngs] = None,
        ):
            """Initialize NNX state hasher.

            Args:
                state_size: Size of state vectors to hash
                backend: NNX backend for device management
                config: Configuration for hasher. If None, uses default.
                rngs: Random number generators for NNX. If None, creates default.
            """
            if not JAX_AVAILABLE:
                raise ImportError(
                    "JAX and Flax are required for NNX hash functions. " "Install with: pip install 'cayleypy[jax]'"
                )

            self.state_size = state_size
            self.backend = backend
            self.config = config or NNXHasherConfig()
            self.logger = logging.getLogger(__name__)

            # Initialize RNGs if not provided
            if rngs is None:
                rngs = nnx.Rngs(self.config.hash_seed)
            self.rngs = rngs

            # Hash matrix as NNX Parameter - automatically managed and sharded
            hash_key = rngs.params()
            self.hash_matrix = nnx.Param(
                jax.random.randint(hash_key, (state_size, self.config.hash_bits), 0, 2**31 - 1, dtype=jnp.uint32)
            )

            # Apply sharding if backend supports it
            if hasattr(backend, "sharding") and backend.sharding is not None:
                self.hash_matrix = nnx.Param(self.hash_matrix.value, sharding=backend.sharding)

            # Hash result cache using NNX Variable
            self.hash_cache: nnx.Variable = nnx.Variable({}) if self.config.enable_caching else nnx.Variable({})

            # Statistics tracking as NNX Variables
            self.stats = nnx.Variable(
                {
                    "total_hashes": 0.0,
                    "cache_hits": 0.0,
                    "cache_misses": 0.0,
                    "batch_sizes": [],
                    "memory_peak_mb": 0.0,
                    "computation_time_ms": 0.0,
                }
            )

            self.logger.info(
                "NNXStateHasher initialized with state_size=%d, hash_bits=%d, caching=%s",
                state_size,
                self.config.hash_bits,
                self.config.enable_caching,
            )

        def _get_cache_key(self, state: jnp.ndarray) -> str:
            """Generate cache key for state vector."""
            if not self.config.enable_caching:
                return ""

            try:
                # Use a fast hash of the state bytes for cache key
                state_bytes = state.tobytes()
                return str(hash(state_bytes))
            except Exception:  # pylint: disable=broad-exception-caught
                # If hashing fails, disable caching for this state
                return ""

        def _get_from_cache(self, cache_key: str) -> Optional[jnp.ndarray]:
            """Retrieve hash result from cache if available."""
            if not self.config.enable_caching or not cache_key:
                return None

            if cache_key in self.hash_cache.value:
                self.stats.value["cache_hits"] = float(self.stats.value["cache_hits"]) + 1.0
                return self.hash_cache.value[cache_key]

            self.stats.value["cache_misses"] = float(self.stats.value["cache_misses"]) + 1.0
            return None

        def _store_in_cache(self, cache_key: str, result: jnp.ndarray) -> None:
            """Store hash result in cache with size management."""
            if not self.config.enable_caching or not cache_key:
                return

            # Simple cache size management - remove oldest entries if needed
            if len(self.hash_cache.value) >= self.config.max_cache_size:
                # Remove first (oldest) entry
                oldest_key = next(iter(self.hash_cache.value))
                del self.hash_cache.value[oldest_key]

            self.hash_cache.value[cache_key] = result

        @nnx.jit
        def _compute_hash(self, state: jnp.ndarray) -> jnp.ndarray:
            """Compute hash using matrix multiplication and bit operations."""
            # Ensure state is the right type for computation
            state_uint32 = state.astype(jnp.uint32)

            # Matrix multiplication for hash computation
            hash_result = jnp.dot(state_uint32, self.hash_matrix.value)

            # Apply additional mixing for better distribution
            hash_result = hash_result ^ (hash_result >> 16)
            hash_result = hash_result * jnp.uint32(0x85EBCA6B)
            hash_result = hash_result ^ (hash_result >> 13)
            hash_result = hash_result * jnp.uint32(0xC2B2AE35)
            hash_result = hash_result ^ (hash_result >> 16)

            # Reduce to final hash value
            return jnp.bitwise_xor.reduce(hash_result) % (2**31 - 1)

        def hash_state(self, state: jnp.ndarray) -> jnp.ndarray:
            """Hash a single state vector with caching.

            Args:
                state: State vector to hash

            Returns:
                Hash value as JAX array
            """
            # Check cache first (outside JIT context)
            cache_key = self._get_cache_key(state)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute hash using JIT-compiled function
            hash_result = self._compute_hash(state)

            # Update cache and stats (outside JIT context)
            self._store_in_cache(cache_key, hash_result)
            self.stats.value["total_hashes"] = float(self.stats.value["total_hashes"]) + 1.0

            return hash_result

        def hash_batch(self, states: jnp.ndarray) -> jnp.ndarray:
            """Hash a batch of states using JAX vectorization.

            Args:
                states: Batch of state vectors with shape (batch_size, state_size)

            Returns:
                Hash values with shape (batch_size,)
            """
            # Extract hash matrix value outside of vmap to avoid state access issues
            hash_matrix = self.hash_matrix.value

            # Use JAX vmap with pure function
            def compute_hash_pure(state):
                # Ensure state is the right type for computation
                state_uint32 = state.astype(jnp.uint32)

                # Matrix multiplication for hash computation
                hash_result = jnp.dot(state_uint32, hash_matrix)

                # Apply additional mixing for better distribution
                hash_result = hash_result ^ (hash_result >> 16)
                hash_result = hash_result * jnp.uint32(0x85EBCA6B)
                hash_result = hash_result ^ (hash_result >> 13)
                hash_result = hash_result * jnp.uint32(0xC2B2AE35)
                hash_result = hash_result ^ (hash_result >> 16)

                # Reduce to final hash value
                return jnp.bitwise_xor.reduce(hash_result) % (2**31 - 1)

            vectorized_hash = jax.vmap(compute_hash_pure, in_axes=0, out_axes=0)
            hash_results = vectorized_hash(states)

            # Update statistics (outside of JIT context)
            batch_size = states.shape[0]
            batch_sizes_list = list(self.stats.value["batch_sizes"])
            batch_sizes_list.append(float(batch_size))
            self.stats.value["batch_sizes"] = batch_sizes_list
            self.stats.value["total_hashes"] = float(self.stats.value["total_hashes"]) + float(batch_size)

            return hash_results

        def hash_large_batch(self, states: jnp.ndarray, chunk_size: Optional[int] = None) -> jnp.ndarray:
            """Process large batches with automatic chunking and memory management.

            Args:
                states: Large batch of state vectors
                chunk_size: Size of chunks to process. If None, uses config default.

            Returns:
                Hash values for all states
            """
            if not self.config.memory_efficient:
                # Process entire batch at once
                return self.hash_batch(states)

            chunk_size = chunk_size or self.config.chunk_size
            batch_size = states.shape[0]

            if batch_size <= chunk_size:
                return self.hash_batch(states)

            # Process in chunks manually (simpler approach)
            results = []
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk = states[i:end_idx]
                chunk_hashes = self.hash_batch(chunk)
                results.append(chunk_hashes)

            # Concatenate all results
            return jnp.concatenate(results, axis=0)

        def get_cache_stats(self) -> Dict[str, Any]:
            """Get comprehensive caching and performance statistics."""
            stats_dict = dict(self.stats.value)
            total_ops = float(stats_dict["total_hashes"])
            cache_hits = float(stats_dict["cache_hits"])
            cache_misses = float(stats_dict["cache_misses"])
            total_cache_ops = cache_hits + cache_misses

            stats = {
                "total_hashes": int(total_ops),
                "cache_hits": int(cache_hits),
                "cache_misses": int(cache_misses),
                "cache_hit_rate": float(cache_hits) / max(1.0, total_cache_ops),
                "cache_size": len(self.hash_cache.value),
                "max_cache_size": self.config.max_cache_size,
                "memory_peak_mb": float(stats_dict["memory_peak_mb"]),
            }

            # Add batch size statistics
            batch_sizes_list = list(stats_dict["batch_sizes"])
            if batch_sizes_list:
                batch_sizes = jnp.array(batch_sizes_list)
                stats.update(
                    {
                        "avg_batch_size": float(jnp.mean(batch_sizes)),
                        "max_batch_size": float(jnp.max(batch_sizes)),
                        "total_batches": len(batch_sizes_list),
                    }
                )

            return stats

        def clear_cache(self) -> None:
            """Clear the hash cache and reset statistics."""
            self.hash_cache.value.clear()
            stats_dict = dict(self.stats.value)
            stats_dict.update(
                {
                    "cache_hits": 0.0,
                    "cache_misses": 0.0,
                    "batch_sizes": [],
                }
            )
            self.stats.value = stats_dict
            self.logger.info("Hash cache cleared")

        def optimize_for_device(self) -> None:
            """Optimize hash function for the current device."""
            if not self.backend or not self.backend.is_available():
                return

            device_type = self.backend.device_type

            if device_type == "tpu":
                # TPU optimizations
                self.config.chunk_size = min(self.config.chunk_size, 32768)
                self.config.memory_efficient = True
                self.logger.info("Optimized hash function for TPU")
            elif device_type == "gpu":
                # GPU optimizations
                self.config.chunk_size = min(self.config.chunk_size, 65536)
                self.logger.info("Optimized hash function for GPU")
            else:
                # CPU optimizations
                self.config.chunk_size = max(self.config.chunk_size, 100000)
                self.logger.info("Optimized hash function for CPU")

    class OptimizedNNXStateHasher(NNXStateHasher):
        """Memory-optimized version with advanced NNX features and rematerialization."""

        def __init__(
            self,
            state_size: int,
            backend: NNXBackend,
            config: Optional[NNXHasherConfig] = None,
            rngs: Optional[nnx.Rngs] = None,
        ):
            """Initialize optimized NNX state hasher with advanced features."""
            super().__init__(state_size, backend, config, rngs)

            # Enhanced memory tracking
            self.memory_stats = nnx.Variable(
                {
                    "peak_memory_mb": 0.0,
                    "current_memory_mb": 0.0,
                    "memory_efficiency": 0.0,
                    "gc_collections": 0.0,
                }
            )

        def hash_large_batch(self, states: jnp.ndarray, chunk_size: Optional[int] = None) -> jnp.ndarray:
            """Memory-efficient large batch processing with rematerialization."""
            # Use the parent implementation but with memory tracking
            # Track memory usage before processing
            result = super().hash_large_batch(states, chunk_size)
            # Could add memory tracking here in the future
            return result

        @nnx.jit
        def _compute_hash_with_checkpointing(self, state: jnp.ndarray) -> jnp.ndarray:
            """Compute hash with gradient checkpointing for memory efficiency."""

            # Use checkpointing for memory-intensive operations
            def hash_computation(s):
                return self._compute_hash(s)

            return jax.checkpoint(hash_computation)(state)

        def get_memory_stats(self) -> Dict[str, Any]:
            """Get detailed memory usage statistics."""
            return dict(self.memory_stats.value)

else:
    # When JAX is not available, the classes are not defined
    # The factory function will return None
    pass


def create_nnx_hasher(
    state_size: int,
    backend: NNXBackend,
    optimized: bool = False,
    enable_caching: bool = True,
    chunk_size: int = 50000,
    **kwargs,
) -> Optional[Any]:
    """Factory function to create NNX hash function with error handling.

    Args:
        state_size: Size of state vectors to hash
        backend: NNX backend for device management
        optimized: Whether to use optimized version with rematerialization
        enable_caching: Whether to enable hash result caching
        chunk_size: Default chunk size for large batch processing
        **kwargs: Additional configuration options

    Returns:
        NNXStateHasher instance if successful, None if JAX is not available
    """
    if not JAX_AVAILABLE:
        return None

    try:
        config = NNXHasherConfig(enable_caching=enable_caching, chunk_size=chunk_size, **kwargs)

        if optimized:
            hasher = OptimizedNNXStateHasher(state_size, backend, config)
        else:
            hasher = NNXStateHasher(state_size, backend, config)

        hasher.optimize_for_device()
        return hasher

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).warning("Failed to create NNX hasher: %s", e)
        return None
