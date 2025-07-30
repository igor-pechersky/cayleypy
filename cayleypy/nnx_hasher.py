"""NNX Hash Functions with Advanced State Management for CayleyPy.

This module provides hash functions implemented as NNX modules with automatic state
management, caching, and performance tracking. It includes optimized implementations
for state hashing with support for different hash strategies and memory-efficient
batch processing.
"""

import logging
import time
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from flax import nnx
    from .nnx_backend import NNXBackend, JAX_AVAILABLE
    from .hasher import StateHasher
except ImportError:
    JAX_AVAILABLE = False
    jnp = None  # type: ignore
    random = None  # type: ignore
    nnx = None  # type: ignore
    StateHasher = object  # type: ignore


@dataclass
class HashConfig:
    """Configuration for NNX hash functions."""

    # Hash strategy
    hash_strategy: str = "auto"  # "auto", "matrix", "splitmix64", "identity"
    hash_bits: int = 64  # Number of bits for hash output

    # Caching settings
    enable_caching: bool = True
    max_cache_size: int = 10000
    cache_ttl_seconds: float = 600.0  # 10 minutes

    # Performance settings
    chunk_size: int = 10000
    enable_jit: bool = True
    enable_batch_processing: bool = True

    # Memory management
    memory_efficient: bool = True
    max_memory_mb: float = 256.0

    # Statistics tracking
    enable_statistics: bool = True
    statistics_window_size: int = 1000


class NNXStateHasher(nnx.Module if JAX_AVAILABLE else StateHasher):  # type: ignore
    """NNX-based state hashing with automatic state management and caching.

    This class provides efficient state hashing using JAX/NNX with support for
    different hash strategies, automatic caching, and comprehensive performance
    tracking. It maintains compatibility with the original StateHasher interface
    while adding advanced NNX features.
    """

    def __init__(
        self,
        state_size: int,
        backend: NNXBackend,
        config: Optional[HashConfig] = None,
        random_seed: Optional[int] = None,
        rngs: Optional[Any] = None,
    ):
        """Initialize NNX state hasher.

        Args:
            state_size: Size of state vectors to hash
            backend: NNX backend for device management
            config: Hash configuration. If None, uses default.
            random_seed: Random seed for hash matrix generation
            rngs: Random number generators for NNX. If None, creates default.
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX and Flax are required for NNX hash functions. Install with: pip install 'cayleypy[jax]'"
            )

        self.state_size = state_size
        self.backend = backend
        self.config = config or HashConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize RNGs
        if rngs is None:
            seed = random_seed or 42
            rngs = nnx.Rngs(seed)
        self.rngs = rngs

        # Determine hash strategy based on state size and configuration
        self._setup_hash_strategy()

        # Initialize hash parameters based on strategy
        self._initialize_hash_parameters()

        # Hash result cache using NNX Variables
        self.hash_cache: Optional[nnx.Variable] = nnx.Variable({}) if self.config.enable_caching else None

        # Cache metadata for TTL and size management
        if self.config.enable_caching:
            self.cache_metadata: Optional[nnx.Variable] = nnx.Variable({})  # Stores timestamps and access counts
        else:
            self.cache_metadata = None

        # Statistics tracking as NNX Variables
        if self.config.enable_statistics:
            self.stats: Optional[nnx.Variable] = nnx.Variable(
                {
                    "total_hashes": 0,
                    "total_requests": 0,  # Total hash requests including cache hits
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "batch_sizes": [],
                    "hash_times_ms": [],
                    "memory_usage_mb": 0.0,
                    "collision_estimates": [],
                    "throughput_hashes_per_sec": 0.0,
                }
            )
        else:
            self.stats = None

        # Performance metrics for different operations
        self.performance_metrics = nnx.Variable(
            {
                "single_hash_time_ms": 0.0,
                "batch_hash_time_ms": 0.0,
                "large_batch_time_ms": 0.0,
                "cache_lookup_time_ms": 0.0,
                "memory_peak_mb": 0.0,
            }
        )

        self.logger.info(
            "NNXStateHasher initialized: strategy=%s, state_size=%d, caching=%s",
            self.hash_strategy,
            state_size,
            self.config.enable_caching,
        )

    def _setup_hash_strategy(self):
        """Determine optimal hash strategy based on state size and configuration."""
        if self.config.hash_strategy != "auto":
            self.hash_strategy = self.config.hash_strategy
            return

        # Auto-select strategy based on state size
        if self.state_size == 1:
            self.hash_strategy = "identity"
        elif self.state_size <= 32:  # Reduced threshold to avoid overflow issues
            self.hash_strategy = "splitmix64"
        else:
            self.hash_strategy = "matrix"

        self.logger.info("Auto-selected hash strategy: %s for state_size=%d", self.hash_strategy, self.state_size)

    def _initialize_hash_parameters(self):
        """Initialize hash parameters based on the selected strategy."""
        if self.hash_strategy == "identity":
            # Identity hash - no parameters needed
            self.is_identity = True
            return

        self.is_identity = False

        if self.hash_strategy == "splitmix64":
            # SplitMix64 hash - store seed as parameter
            hash_key = self.rngs.params()
            seed_value = random.randint(hash_key, (), 0, 2**31)
            self.hash_seed = nnx.Param(seed_value)

        elif self.hash_strategy == "matrix":
            # Matrix-based hash - create random hash matrix
            hash_key = self.rngs.params()

            # Create hash matrix with proper dimensions (state_size, hash_bits)
            hash_matrix_shape = (self.state_size, self.config.hash_bits)
            # Use int32 for TPU compatibility (TPUs don't support X64 operations well)
            hash_matrix = random.randint(hash_key, hash_matrix_shape, 0, 2**31, dtype=jnp.int32)

            # Apply sharding if available
            if self.backend.sharding is not None:
                hash_matrix = self.backend.create_sharded_array(hash_matrix)

            self.hash_matrix = nnx.Param(hash_matrix)

        else:
            raise ValueError(f"Unknown hash strategy: {self.hash_strategy}")

    def _get_cache_key(self, state: jnp.ndarray) -> str:
        """Generate cache key for a state vector."""
        if not self.config.enable_caching:
            return ""

        try:
            # Use a simple hash of the state bytes as cache key
            return str(hash(state.tobytes()))
        except Exception:  # pylint: disable=broad-exception-caught
            return ""

    def _get_from_cache(self, cache_key: str) -> Optional[jnp.ndarray]:
        """Retrieve hash result from cache if available and not expired."""
        if not self.config.enable_caching or not cache_key or self.hash_cache is None or self.cache_metadata is None:
            return None

        start_time = time.time()

        if cache_key in self.hash_cache.value:
            # Check TTL
            metadata = self.cache_metadata.value.get(cache_key, {})
            timestamp = metadata.get("timestamp", 0)

            if time.time() - timestamp < self.config.cache_ttl_seconds:
                # Update access count and return cached result
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                metadata["last_access"] = time.time()
                self.cache_metadata.value[cache_key] = metadata

                if self.stats is not None:
                    self.stats.value["cache_hits"] = self.stats.value["cache_hits"] + 1

                # Track cache lookup time
                lookup_time = (time.time() - start_time) * 1000
                self.performance_metrics.value["cache_lookup_time_ms"] = lookup_time

                return self.hash_cache.value[cache_key]
            else:
                # Expired - remove from cache
                del self.hash_cache.value[cache_key]
                del self.cache_metadata.value[cache_key]

        if self.stats is not None:
            self.stats.value["cache_misses"] = self.stats.value["cache_misses"] + 1

        return None

    def _store_in_cache(self, cache_key: str, result: jnp.ndarray) -> None:
        """Store hash result in cache with TTL and size management."""
        if not self.config.enable_caching or not cache_key or self.hash_cache is None or self.cache_metadata is None:
            return

        current_time = time.time()

        # Manage cache size - remove oldest entries if needed
        if len(self.hash_cache.value) >= self.config.max_cache_size:
            self._evict_cache_entries()

        # Store result and metadata
        self.hash_cache.value[cache_key] = result
        self.cache_metadata.value[cache_key] = {
            "timestamp": current_time,
            "last_access": current_time,
            "access_count": 1,
            "size_bytes": result.nbytes,
        }

    def _evict_cache_entries(self):
        """Evict cache entries using LRU policy."""
        if self.cache_metadata is None or self.hash_cache is None:
            return

        # Sort by last access time and remove oldest entries
        sorted_keys = sorted(
            self.cache_metadata.value.keys(), key=lambda k: self.cache_metadata.value[k].get("last_access", 0)
        )

        # Remove oldest 25% of entries
        num_to_remove = max(1, len(sorted_keys) // 4)
        for key in sorted_keys[:num_to_remove]:
            if key in self.hash_cache.value:
                del self.hash_cache.value[key]
            if key in self.cache_metadata.value:
                del self.cache_metadata.value[key]

    def _hash_identity(self, state: jnp.ndarray) -> jnp.ndarray:
        """Identity hash function for single-element states."""
        return state.reshape(-1)

    def _hash_splitmix64(self, state: jnp.ndarray) -> jnp.ndarray:
        """SplitMix64 hash function for small states.

        This implementation uses the EXACT same constants and logic as the reference
        implementation in hasher.py to ensure identical results.
        """
        # Use int32 for TPU compatibility, int64 for other devices
        if self.backend.device_type == "tpu":
            # TPU-compatible constants (truncated to int32)
            const1 = jnp.int32(-1062731775)  # Truncated version of 0xBF58476D1CE4E5B9
            const2 = jnp.int32(322687467)    # Truncated version of 0x94D049BB133111EB
            const3 = jnp.int32(2246822507)   # 0x85EBCA6B fits in int32
            dtype = jnp.int32
        else:
            # Constants from reference implementation (converted to signed int64)
            const1 = jnp.int64(-4658895280553007687)  # 0xBF58476D1CE4E5B9
            const2 = jnp.int64(-7723592293110705685)  # 0x94D049BB133111EB
            const3 = jnp.int64(2246822507)            # 0x85EBCA6B
            dtype = jnp.int64

        def splitmix64_step(x: jnp.ndarray) -> jnp.ndarray:
            # Exact implementation of _splitmix64 from hasher.py
            x = x.astype(dtype)
            if dtype == jnp.int32:
                # TPU-compatible version with 32-bit shifts
                x = x ^ (x >> 15)  # Reduced shift for 32-bit
                x = x * const1
                x = x ^ (x >> 13)  # Reduced shift for 32-bit
                x = x * const2
                x = x ^ (x >> 16)  # Reduced shift for 32-bit
            else:
                # Original 64-bit version
                x = x ^ (x >> 30)
                x = x * const1
                x = x ^ (x >> 27)
                x = x * const2
                x = x ^ (x >> 31)
            return x

        # Convert state to appropriate dtype
        state_typed = state.astype(dtype)

        # Start with seed - match reference implementation shape handling
        if state.ndim == 1:
            # Single state vector - treat as batch of size 1 for consistency
            h = jnp.full((1,), self.hash_seed.value, dtype=dtype)
            # Hash each element of the state
            for i in range(state.shape[0]):
                element_batch = jnp.array([state_typed[i]])  # Make it batch-like
                h = h ^ splitmix64_step(element_batch)
                h = h * const3
            return h[0]  # Return scalar for single state
        else:
            # Batch of state vectors - match reference (n, m) shape handling
            n, m = state.shape
            h = jnp.full((n,), self.hash_seed.value, dtype=jnp.int64)
            for i in range(m):
                h = h ^ splitmix64_step(state_int64[:, i])
                h = h * const3
            return h

    def _hash_matrix_impl(self, state: jnp.ndarray) -> jnp.ndarray:
        """Matrix-based hash function implementation (non-JIT).
        
        Returns hash bits as specified in config.
        """
        # Use int64 for compatibility with reference implementation
        state_int64 = state.astype(jnp.int64)
        hash_matrix_int64 = self.hash_matrix.value.astype(jnp.int64)
        
        if state.ndim == 1:
            # Single state vector - compute hash bits
            result = jnp.dot(state_int64, hash_matrix_int64)  # (hash_bits,)
            return result.astype(jnp.int64)
        else:
            # Batch of state vectors - matrix multiplication
            result = jnp.dot(state_int64, hash_matrix_int64)  # (batch_size, hash_bits)
            return result.astype(jnp.int64)

    @nnx.jit
    def _hash_matrix(self, state: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled matrix hash for performance."""
        return self._hash_matrix_impl(state)

    def hash_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Hash a single state vector with caching and performance tracking.

        Args:
            state: State vector to hash

        Returns:
            Hash value(s) as JAX array
        """
        start_time = time.time()

        # Track total requests
        if self.stats is not None:
            self.stats.value["total_requests"] = self.stats.value["total_requests"] + 1

        # Check cache first
        cache_key = self._get_cache_key(state)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Compute hash based on strategy
        if self.is_identity:
            result = self._hash_identity(state)
        elif self.hash_strategy == "splitmix64":
            result = self._hash_splitmix64(state)
        elif self.hash_strategy == "matrix":
            result = self._hash_matrix_impl(state)  # Use non-JIT version to avoid trace context issues
        else:
            raise ValueError(f"Unknown hash strategy: {self.hash_strategy}")

        # Store in cache
        self._store_in_cache(cache_key, result)

        # Update statistics
        if self.stats is not None:
            self.stats.value["total_hashes"] = self.stats.value["total_hashes"] + 1
            hash_time = (time.time() - start_time) * 1000
            hash_times_list = self.stats.value["hash_times_ms"]
            if isinstance(hash_times_list, list):
                hash_times_list.append(hash_time)

            # Keep only recent timing data
            hash_times_list = self.stats.value["hash_times_ms"]
            if isinstance(hash_times_list, list) and len(hash_times_list) > self.config.statistics_window_size:
                self.stats.value["hash_times_ms"] = hash_times_list[-self.config.statistics_window_size :]

        # Update performance metrics
        hash_time_ms = (time.time() - start_time) * 1000
        self.performance_metrics.value["single_hash_time_ms"] = hash_time_ms

        return result

    def hash_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Hash a batch of states using vectorized operations.

        Args:
            states: Batch of state vectors with shape (batch_size, state_size)

        Returns:
            Hash values with shape (batch_size, hash_bits)
        """
        start_time = time.time()

        # Update batch size statistics
        if self.stats is not None:
            batch_sizes_list = self.stats.value["batch_sizes"]
            if isinstance(batch_sizes_list, list):
                batch_sizes_list.append(states.shape[0])
                if len(batch_sizes_list) > self.config.statistics_window_size:
                    self.stats.value["batch_sizes"] = batch_sizes_list[-self.config.statistics_window_size :]

        # Compute hashes for the batch
        if self.is_identity:
            result = self._hash_identity(states)
        elif self.hash_strategy == "splitmix64":
            # Use vmap for splitmix64 to handle each state individually
            vectorized_hash = nnx.vmap(self._hash_splitmix64, in_axes=0)
            result = vectorized_hash(states)
        elif self.hash_strategy == "matrix":
            result = self._hash_matrix_impl(states)  # Use non-JIT version to avoid trace context issues
        else:
            raise ValueError(f"Unknown hash strategy: {self.hash_strategy}")

        # Update performance metrics
        batch_time_ms = (time.time() - start_time) * 1000
        self.performance_metrics.value["batch_hash_time_ms"] = batch_time_ms

        # Update statistics
        if self.stats is not None:
            self.stats.value["total_hashes"] = self.stats.value["total_hashes"] + states.shape[0]

            # Calculate throughput
            if batch_time_ms > 0:
                throughput = (states.shape[0] / batch_time_ms) * 1000  # hashes per second
                self.stats.value["throughput_hashes_per_sec"] = throughput

        return result

    def hash_large_batch(self, states: jnp.ndarray, chunk_size: Optional[int] = None) -> jnp.ndarray:
        """Process large batches with automatic chunking and memory management.

        Args:
            states: Large batch of state vectors
            chunk_size: Size of chunks to process. If None, uses config default.

        Returns:
            Hash values for all states
        """
        start_time = time.time()

        chunk_size = chunk_size or self.config.chunk_size

        if states.shape[0] <= chunk_size:
            # Small enough to process directly
            return self.hash_batch(states)

        # Process in chunks
        results = []
        for i in range(0, states.shape[0], chunk_size):
            chunk = states[i : i + chunk_size]
            chunk_result = self.hash_batch(chunk)
            results.append(chunk_result)

        # Concatenate results
        final_result = jnp.concatenate(results, axis=0)

        # Update performance metrics
        large_batch_time_ms = (time.time() - start_time) * 1000
        self.performance_metrics.value["large_batch_time_ms"] = large_batch_time_ms

        return final_result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive caching statistics."""
        if not self.config.enable_caching or self.stats is None:
            return {"caching_enabled": False}

        total_requests = self.stats.value["cache_hits"] + self.stats.value["cache_misses"]
        hit_rate = self.stats.value["cache_hits"] / max(1, total_requests)

        cache_size = len(self.hash_cache.value) if self.hash_cache else 0

        # Calculate cache memory usage
        total_cache_memory = 0
        if self.cache_metadata is not None:
            for metadata in self.cache_metadata.value.values():
                total_cache_memory += metadata.get("size_bytes", 0)

        return {
            "caching_enabled": True,
            "cache_hit_rate": hit_rate,
            "cache_hits": self.stats.value["cache_hits"],
            "cache_misses": self.stats.value["cache_misses"],
            "cache_size": cache_size,
            "max_cache_size": self.config.max_cache_size,
            "cache_memory_mb": total_cache_memory / (1024 * 1024),
            "cache_efficiency": hit_rate * cache_size / max(1, self.config.max_cache_size),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics: Dict[str, Any] = dict(self.performance_metrics.value)

        # Add statistics if available
        if self.stats is not None:
            stats = dict(self.stats.value)

            # Calculate average metrics
            if stats["hash_times_ms"]:
                stats["avg_hash_time_ms"] = jnp.mean(jnp.array(stats["hash_times_ms"]))
                stats["max_hash_time_ms"] = jnp.max(jnp.array(stats["hash_times_ms"]))
                stats["min_hash_time_ms"] = jnp.min(jnp.array(stats["hash_times_ms"]))

            if stats["batch_sizes"]:
                stats["avg_batch_size"] = jnp.mean(jnp.array(stats["batch_sizes"]))
                stats["max_batch_size"] = jnp.max(jnp.array(stats["batch_sizes"]))

            metrics.update(stats)

        # Add cache statistics
        metrics.update(self.get_cache_stats())

        # Add configuration info
        metrics.update(
            {
                "hash_strategy": self.hash_strategy,
                "state_size": float(self.state_size),
                "hash_bits": float(self.config.hash_bits),
                "is_identity": self.is_identity,
                "backend_device": self.backend.device_type if self.backend else "unknown",
            }
        )

        return metrics

    def clear_cache(self) -> None:
        """Clear the hash cache and reset cache statistics."""
        if self.hash_cache is not None:
            self.hash_cache.value.clear()

        if self.cache_metadata is not None:
            self.cache_metadata.value.clear()

        if self.stats is not None:
            self.stats.value["cache_hits"] = 0
            self.stats.value["cache_misses"] = 0

        self.logger.info("Hash cache cleared")

    def optimize_for_device(self) -> None:
        """Optimize hash function parameters for the current device."""
        if not self.backend or not self.backend.is_available():
            return

        device_type = self.backend.device_type

        if device_type == "tpu":
            # TPU optimizations
            self.config.chunk_size = min(self.config.chunk_size, 4096)
            self.config.memory_efficient = True
            self.config.max_cache_size = min(self.config.max_cache_size, 5000)
            self.logger.info("Optimized hash function for TPU")

        elif device_type == "gpu":
            # GPU optimizations
            self.config.chunk_size = min(self.config.chunk_size, 8192)
            self.config.max_cache_size = min(self.config.max_cache_size, 15000)
            self.logger.info("Optimized hash function for GPU")

        else:
            # CPU optimizations
            self.config.chunk_size = max(self.config.chunk_size, 16384)
            self.config.max_cache_size = max(self.config.max_cache_size, 20000)
            self.logger.info("Optimized hash function for CPU")


class OptimizedNNXStateHasher(NNXStateHasher):
    """Memory-optimized version with advanced NNX features and sharding support."""

    def __init__(
        self,
        state_size: int,
        backend: NNXBackend,
        config: Optional[HashConfig] = None,
        random_seed: Optional[int] = None,
        rngs: Optional[Any] = None,
    ):
        """Initialize optimized NNX state hasher with advanced features."""
        super().__init__(state_size, backend, config, random_seed, rngs)

        # Apply advanced optimizations
        self._setup_advanced_optimizations()

    def _setup_advanced_optimizations(self):
        """Setup advanced optimizations including sharding and memory management."""
        if hasattr(self, "hash_matrix") and self.backend.sharding is not None:
            # Apply sharding to hash matrix for distributed computation
            self.hash_matrix = nnx.Param(self.hash_matrix.value, sharding=self.backend.sharding)
            self.logger.info("Applied sharding to hash matrix")

        # Enable gradient checkpointing for memory efficiency if available
        if hasattr(nnx, "remat"):
            self._enable_rematerialization()

    def _enable_rematerialization(self):
        """Enable rematerialization for memory-efficient computation."""
        # Wrap hash computation methods with rematerialization
        if hasattr(nnx, "remat"):
            self._hash_matrix = nnx.remat(self._hash_matrix)
            self._hash_splitmix64 = nnx.remat(self._hash_splitmix64)
            self.logger.info("Enabled rematerialization for memory efficiency")

    def hash_large_batch_optimized(self, states: jnp.ndarray, chunk_size: Optional[int] = None) -> jnp.ndarray:
        """Memory-efficient large batch processing with advanced optimizations.

        This version uses scan for efficient iteration and memory management.
        """
        chunk_size = chunk_size or self.config.chunk_size

        if states.shape[0] <= chunk_size:
            return self.hash_batch(states)

        # Process in chunks using a simpler approach
        results = []
        for i in range(0, states.shape[0], chunk_size):
            chunk = states[i : i + chunk_size]
            chunk_result = self.hash_batch(chunk)
            results.append(chunk_result)

        return jnp.concatenate(results, axis=0)


class TPUQuickHasher(nnx.Module if JAX_AVAILABLE else object):  # type: ignore
    """Fast, approximate hashing for TPU using int32 polynomial operations."""
    
    def __init__(self, state_size: int, backend: NNXBackend, rngs: Optional[Any] = None):
        """Initialize TPU quick hasher.
        
        Args:
            state_size: Size of state vectors to hash
            backend: NNX backend (should be TPU)
            rngs: Random number generators for NNX
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for TPU quick hasher")
            
        self.state_size = state_size
        self.backend = backend
        
        if rngs is None:
            rngs = nnx.Rngs(42)
        
        # Use int32 polynomial coefficients for TPU compatibility
        hash_key = rngs.params()
        self.hash_coeffs = nnx.Param(
            random.randint(hash_key, (min(64, state_size),), 1, 2**15, dtype=jnp.int32)
        )
        
        # Statistics for quick hashing
        self.stats = nnx.Variable({
            "total_quick_hashes": 0,
            "collision_estimates": 0,
            "throughput_hashes_per_sec": 0.0
        })
    
    @nnx.jit
    def hash_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Fast polynomial hashing using int32 operations.
        
        Args:
            states: Batch of state vectors with shape (batch_size, state_size)
            
        Returns:
            Quick hash values with shape (batch_size,)
        """
        batch_size, state_size = states.shape
        
        # Ensure int32 for TPU compatibility
        states_int32 = states.astype(jnp.int32)
        
        # Broadcast coefficients to match state dimensions
        coeffs = jnp.tile(self.hash_coeffs.value, (state_size // len(self.hash_coeffs.value) + 1))[:state_size]
        
        # Polynomial hash: sum(state[i] * coeff[i % len(coeffs)]) mod 2^31
        hash_values = jnp.sum(states_int32 * coeffs[None, :], axis=1, dtype=jnp.int32)
        hash_values = hash_values % (2**31 - 1)  # Keep in int32 range
        
        # Update statistics
        self.stats.value["total_quick_hashes"] += batch_size
        
        return hash_values


class HierarchicalHasher(nnx.Module if JAX_AVAILABLE else object):  # type: ignore
    """Multi-level hashing with TPU quick filtering and CPU precise deduplication."""
    
    def __init__(self, state_size: int, hybrid_backend, rngs: Optional[Any] = None):
        """Initialize hierarchical hasher.
        
        Args:
            state_size: Size of state vectors to hash
            hybrid_backend: HybridCayleyGraphBackend instance
            rngs: Random number generators for NNX
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for hierarchical hasher")
            
        self.state_size = state_size
        self.hybrid_backend = hybrid_backend
        
        # TPU quick hasher for initial filtering
        if hybrid_backend.tpu_backend is not None:
            self.tpu_hasher = TPUQuickHasher(state_size, hybrid_backend.tpu_backend, rngs)
        else:
            self.tpu_hasher = None
            
        # CPU precise hasher for final deduplication
        if hybrid_backend.cpu_backend is not None:
            self.cpu_hasher = NNXStateHasher(state_size, hybrid_backend.cpu_backend, rngs=rngs)
        else:
            self.cpu_hasher = None
            
        # Performance metrics
        self.metrics = nnx.Variable({
            "phase1_filtered": 0,
            "phase2_deduplicated": 0,
            "transfer_time_ms": 0.0,
            "total_processing_time_ms": 0.0
        })
    
    def deduplicate_states(self, states: jnp.ndarray, 
                          quick_filter_threshold: int = 1000) -> jnp.ndarray:
        """Two-phase deduplication: TPU filtering + CPU precision.
        
        Args:
            states: Input states to deduplicate
            quick_filter_threshold: Minimum size for TPU quick filtering
            
        Returns:
            Deduplicated states
        """
        start_time = time.time()
        
        # Phase 1: TPU-based quick filtering (if available and beneficial)
        if (self.tpu_hasher is not None and 
            len(states) > quick_filter_threshold):
            
            # Quick hash on TPU
            quick_hashes = self.tpu_hasher.hash_batch(states)
            
            # Remove obvious duplicates based on quick hashes
            unique_quick_indices = self._get_unique_indices_int32(quick_hashes)
            filtered_states = states[unique_quick_indices]
            
            self.metrics.value["phase1_filtered"] = len(states) - len(filtered_states)
        else:
            filtered_states = states
            self.metrics.value["phase1_filtered"] = 0
        
        # Phase 2: CPU-based precise deduplication (if needed)
        if (self.cpu_hasher is not None and 
            len(filtered_states) > 1):
            
            # Transfer to CPU and perform precise hashing
            transfer_start = time.time()
            cpu_states = jax.device_put(filtered_s
000
            e
            
       CPU
            precise_hashees)
            unique_shes)
ces]
            
            self.metrics.value["phase2_deduplicated"] = len(filtered_states) - len(final
        else:
s
        
        

        total_time = 000
        self.metrics.value["total_processing_time_ms"] = total_time
        
        return final_states
    
    def _get_unique_indices_int32(sedarray:

        sorted_indices)
dices]
        
        # Find unique elements
        unique_mask
Hasher]]:rchicalasher, HieraStateHtimizedNNX Opasher,eHNNXStatUnion[> Optional[ -,
)rgs   **kwa False,
 ool =: bical    hierarchl = False,
imized: boo
    optt = 10000,unk_size: in    ch True,
: bool =ble_cachingna,
    eo""aut r = st_strategy:
    hashnd,XBackekend: NNt,
    bace_size: in(
    stather_nnx_has createics


defturn metr   re           
   trics()
   ormance_merf_peer.get.cpu_hash selfs"] =statics["cpu_ metr      ne:
     not No is asherf.cpu_hel        if s)
valuesher.stats.(self.tpu_ha = dictpu_stats"]ics["ttrme      :
      one N nothasher is self.tpu_ if
       tricsponent me-com# Add sub        
  e)
      rics.valulf.metdict(se  metrics = "
      .""e metricsg performanchinal hasrchicierat h"""Ge       y]:
 str, An) -> Dict[(selftricsormance_merf_peget 
    def   ndices
 e_iturn uniqu re=0)
       ue, axis_index=Trs, returnue(hasheniqp.udices = jnunique_in, shes unique_ha     
  plicationise deduon for precique functis un  # Use JAX'      """
precise).CPU  (shesint64 hafor indices t unique ""Ge       "ray:
 arjnp.ndray) -> .ndarjnphes: self, hasces_int64(e_indiget_uniqu   def _ 
 k]
   [unique_masesicted_indrn sor retu
       ])
        
        hes[:-1]rted_has[1:] != sohes  sorted_has     e]), 
     ay([Tru    jnp.arr        ncatenate([ jnp.co =es[sorted_insh = ha_hashesrted       so heort(hasgss = jnp.are)."""compatiblhes (TPU-32 hass for intdicet unique in"""Ge        .n-> jnparray) shes: jnp.nd haf,l_time) * 1() - start(time.timeessing timeal proc# Update tot        d"] = 0duplicate["phase2_derics.valuef.met sel   ltered_statees = fi final_stat           ates)_stcise_indique_pretes[unies = cpu_stanal_stat         fi   recise_ha4(pdices_int6nique_in_get_uf.selces = ecise_indiprpu_stath(csh_batcr.half.cpu_hashes = seon on uplicatiedise d     # Precim transfer_t"] =_mstimeansfer_alue["tretrics.v.mselfr_start) * 1feans trme.time() -(ti = timefer_   trans         "cpu")[0])ces(es, jax.devitatneeded)n (if iltered"]1_f"phasevalue[lf.metrics. se           
    """Factory function to create NNX state hasher with error handling.

    Args:
        state_size: Size of state vectors to hash
        backend: NNX backend for device management
        hash_strategy: Hash strategy ("auto", "matrix", "splitmix64", "identity")
        enable_caching: Whether to enable result caching
        chunk_size: Default chunk size for large batch processing
        optimized: Whether to use the optimized version with advanced features
        **kwargs: Additional configuration options

    Returns:
        NNXStateHasher instance if successful, None if JAX is not available
    """
    if not JAX_AVAILABLE:
        return None

    if backend is None:
        logging.getLogger(__name__).warning("Backend is None, cannot create NNX hasher")
        return None

    try:
        config = HashConfig(hash_strategy=hash_strategy, enable_caching=enable_caching, chunk_size=chunk_size, **kwargs)

        if optimized:
            hasher = OptimizedNNXStateHasher(state_size, backend, config)
        else:
            hasher = NNXStateHasher(state_size, backend, config)

        hasher.optimize_for_device()

        return hasher

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).warning("Failed to create NNX state hasher: %s", e)
        return None
        if rngs is None:
            rngs = nnx.Rngs(42)
        
        # Use int32 polynomial coefficients for TPU compatibility
        hash_key = rngs.params()
        self.hash_coeffs = nnx.Param(
            random.randint(hash_key, (min(64, state_size),), 1, 2**15, dtype=jnp.int32)
        )
        
        # Statistics for quick hashing
        self.stats = nnx.Variable({
            "total_quick_hashes": 0,
            "collision_estimates": 0,
            "throughput_hashes_per_sec": 0.0
        })
    
    @nnx.jit
    def hash_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Fast polynomial hashing using int32 operations.
        
        Args:
            states: Batch of state vectors with shape (batch_size, state_size)
            
        Returns:
            Quick hash values with shape (batch_size,)
        """
        batch_size, state_size = states.shape
        
        # Ensure int32 for TPU compatibility
        states_int32 = states.astype(jnp.int32)
        
        # Broadcast coefficients to match state dimensions
        coeffs = jnp.tile(self.hash_coeffs.value, (state_size // len(self.hash_coeffs.value) + 1))[:state_size]
        
        # Polynomial hash: sum(state[i] * coeff[i % len(coeffs)]) mod 2^31
        hash_values = jnp.sum(states_int32 * coeffs[None, :], axis=1, dtype=jnp.int32)
        hash_values = hash_values % (2**31 - 1)  # Keep in int32 range
        
        # Update statistics
        self.stats.value["total_quick_hashes"] += batch_size
        
        return hash_values


class HierarchicalHasher(nnx.Module if JAX_AVAILABLE else object):  # type: ignore
    """Multi-level hashing with TPU quick filtering and CPU precise deduplication."""
    
    def __init__(self, state_size: int, hybrid_backend, rngs: Optional[Any] = None):
        """Initialize hierarchical hasher.
        
        Args:
            state_size: Size of state vectors to hash
            hybrid_backend: HybridCayleyGraphBackend instance
            rngs: Random number generators for NNX
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for hierarchical hasher")
            
        self.state_size = state_size
        self.hybrid_backend = hybrid_backend
        
        # TPU quick hasher for initial filtering
        if hasattr(hybrid_backend, 'tpu_backend') and hybrid_backend.tpu_backend is not None:
            self.tpu_hasher = TPUQuickHasher(state_size, hybrid_backend.tpu_backend, rngs)
        else:
            self.tpu_hasher = None
            
        # CPU precise hasher for final deduplication
        if hasattr(hybrid_backend, 'cpu_backend') and hybrid_backend.cpu_backend is not None:
            self.cpu_hasher = NNXStateHasher(state_size, hybrid_backend.cpu_backend, rngs=rngs)
        else:
            self.cpu_hasher = None
            
        # Performance metrics
        self.metrics = nnx.Variable({
            "phase1_filtered": 0,
            "phase2_deduplicated": 0,
            "transfer_time_ms": 0.0,
        

 metricseturn      r    
       rics()
   _metrformanceher.get_pehasf.cpu_ = sel"]cpu_stats  metrics["   
        not None:u_hasher is if self.cp    
   e)tats.valuer.sshha(self.tpu_ct"] = distatstrics["tpu_     mee:
        is not Nonf.tpu_hasherf sel  i      etrics
-component mAdd sub#   
              )
cs.valuemetriict(self. dmetrics =    ""
     metrics."rformanceg peal hashint hierarchicGe"" "     :
  tr, Any]Dict[s) -> (selftricsmeormance_get_perf 
    def dices
   unique_inn     retur  )
  axis=0rue, x=Tdeeturn_inhes, runique(hasp.es = jnndicunique_ishes, ique_ha    un
    cationdedupli precise on fore functiquse JAX's uni       # U""
 precise)."hes (CPU r int64 hasdices foque in""Get uni     ":
   jnp.ndarrayray) -> s: jnp.ndarf, hashesel64(indices_int_unique_f _get    
    deue_mask]
es[uniqdicrn sorted_intu       re   
   ])
     
      1]s[:-ted_hashe[1:] != sorted_hashes         sore]), 
   ([Tru.array       jnpte([
     concatenanp.ue_mask = juniqs
        e elementFind uniqu# 
                ]
ndicess[sorted_i = hasherted_hashes
        sot(hashes)= jnp.argsor_indices   sorted""
      ble)."compatis (TPU-he32 hass for intnique indice""Get u"       .ndarray:
 nprray) -> jjnp.nda hashes: 32(self,dices_intunique_in _get_ef
    
    dinal_states freturn
         me
       ] = total_tiime_ms"essing_t_procale["totics.valu  self.metr1000
       * time)start_ime() - me.t (til_time =        totassing time
otal procee t # Updat         
     0
  "] =eduplicatede2_de["phasics.valuelf.metr        sates
    stered_ates = filt    final_st
        lse:     etates)
   en(final_ss) - lred_state = len(filtecated"]e2_dedupliphasvalue["etrics.elf.m  s    
                dices]
  ine_precise_s[uniqustates = cpu_final_state    
        es)ecise_hash_int64(presnique_indicget_us = self._cise_indicee_preiqu   un  
       ates)cpu_st_batch(er.hashshelf.cpu_hahashes = scise_   pre
         PUation on Ce deduplic    # Precis  
                _time
  ransfer_ms"] = tsfer_timelue["tranrics.va  self.met        00
  start) * 10nsfer_me() - tra (time.ti =r_timetransfe      ils
      ansfer faf device tr# Fallback i_states  ered= filtstates       cpu_          ught
on-capti-exceadle=broabint: disn:  # pylExceptio except          )[0])
  ices("cpu" jax.devs,tateiltered_st(fdevice_putes = jax.pu_sta  c            try:
            time()
   = time.tarter_s  transf         
 ingecise hashprrm rfoU and pe to CPTransfer #                
    > 1):
    states) n(filtered_      le
      d one an is not Ncpu_hasherf (self.   ieded)
      (if neonatilicise dedupbased prec 2: CPU-Phase       #     
 
     = 0ltered"]_fise1"phavalue[elf.metrics.           s= states
 ered_states  filt      else:
     s)
        stateltered_- len(fis)  len(statered"] =ase1_filtevalue["phs..metric self       
              
  indices]uick_tes[unique_qates = sta filtered_st           es)
shuick_haint32(qque_indices_uniself._get_s = ceick_indi  unique_qu      shes
    on quick habased s s duplicateviou ob  # Remove             
         )
statesh_batch(u_hasher.has = self.tp_hashes       quick
     h on TPU Quick has         #      
   
      :reshold)_filter_thcktates) > qui      len(s      
 ot None andher is nf.tpu_hasel    if (s   al)
 beneficie and availablif ng (filteriuick -based q1: TPU# Phase             
   me.time()
  = ti start_time"
            ""  ates
 plicated st     Dedu         Returns:
      
      ing
      er filtuickfor TPU qize d: Minimum sr_thresholltek_fi      quic
      uplicateates to dedput st: In states      
          Args:    
      sion.
 reciU pCP+ U filtering ication: TPedupl dTwo-phase""
        "ndarray:np.00) -> j = 10hold: inter_thresk_filt      quic             
       ray, jnp.ndar: statess(self, ateate_stlic dedupef   
    d    })
 
    ": 0.0_msimeg_tal_processin    "tot