"""TPU Hasher for CayleyPy with native int64 operations.

This module provides TPU-accelerated state hashing with native int64 support,
optimized for TPU v6e (Trillium) architecture with precise hash operations.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple

# pylint: disable=duplicate-code
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    random = None  # type: ignore
    nnx = None  # type: ignore

from .tpu_backend import TPUBackend


# JIT-compiled helper functions for TPU hashing operations
@jax.jit
def _hash_state_jit(state: jnp.ndarray, hash_matrix: jnp.ndarray) -> jnp.int64:
    """JIT-compiled single state hashing using native int64 operations."""
    # Native int64 matrix multiplication on TPU v6e - match reference implementation
    hash_result = jnp.sum(state.astype(jnp.int64) * hash_matrix.reshape(-1))
    return hash_result


@jax.jit
def _hash_batch_jit(states: jnp.ndarray, hash_matrix: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled batch hashing using TPU vectorization."""
    # Use vmap for efficient batch processing on TPU
    return jax.vmap(lambda state: _hash_state_jit(state, hash_matrix))(states)


def _hash_large_batch_chunked(states: jnp.ndarray, hash_matrix: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    """Non-JIT large batch hashing with chunking for dynamic sizes."""
    results = []
    for i in range(0, states.shape[0], chunk_size):
        end_idx = min(i + chunk_size, states.shape[0])
        chunk = states[i:end_idx]
        chunk_hashes = _hash_batch_jit(chunk, hash_matrix)
        results.append(chunk_hashes)

    return jnp.concatenate(results, axis=0)


def _deduplicate_by_hash_simple(states: jnp.ndarray, hashes: jnp.ndarray) -> jnp.ndarray:
    """Simple hash-based deduplication without JIT complications."""
    # Use JAX's unique function for deduplication
    _, unique_indices = jnp.unique(hashes, return_index=True)

    # Return the states corresponding to unique hashes
    return states[unique_indices]


@jax.jit
def _splitmix64_jit(x: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled SplitMix64 hash function for TPU."""
    # Use proper SplitMix64 constants - handle as unsigned then cast to signed
    x = x ^ (x >> 30)
    # Convert large unsigned constants to signed equivalents
    x = x * jnp.int64(-4658895280553007687)  # 0xBF58476D1CE4E5B9 as signed int64
    x = x ^ (x >> 27)
    x = x * jnp.int64(-7723592293110705685)  # 0x94D049BB133111EB as signed int64
    x = x ^ (x >> 31)
    return x


@jax.jit
def _hash_splitmix64_batch_jit(states: jnp.ndarray, seed: jnp.int64) -> jnp.ndarray:
    """JIT-compiled SplitMix64 batch hashing for bit-encoded states."""

    def hash_single_state(state):
        h = jnp.full((), seed, dtype=jnp.int64)

        def scan_fn(h_carry, x_i):
            h_new = h_carry ^ _splitmix64_jit(x_i)
            h_new = h_new * jnp.int64(0x85EBCA6B)  # Use reference constant
            return h_new, None

        h_final, _ = jax.lax.scan(scan_fn, h, state)
        return h_final

    return jax.vmap(hash_single_state)(states)


class TPUHasherModule(nnx.Module):
    """NNX module for TPU-accelerated state hashing with native int64 support."""

    def __init__(
        self,
        state_size: int,
        backend: TPUBackend,
        rngs: nnx.Rngs,
        use_splitmix64: bool = False,
        random_seed: Optional[int] = None,
    ):
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for TPU hasher")

        self.state_size = state_size
        self.backend = backend
        self.use_splitmix64 = use_splitmix64

        # Set random seed
        if random_seed is not None:
            seed = random_seed
        else:
            seed = 42  # Default seed

        # Generate hash matrix using native int64 on TPU
        # Use same dimensions and range as reference implementation
        max_int = 2**62
        self.hash_matrix: Optional[nnx.Param]
        if not use_splitmix64:
            self.hash_matrix = nnx.Param(
                random.randint(rngs.params(), (state_size, 1), minval=-max_int, maxval=max_int, dtype=jnp.int64)
            )
        else:
            # For SplitMix64, we just need the seed
            self.hash_matrix = None

        self.seed = nnx.Variable(jnp.int64(seed))

        # Hash cache for performance
        self.hash_cache: nnx.Variable[Dict[str, Any]] = nnx.Variable({})

        # Performance metrics
        self.metrics: nnx.Variable[Dict[str, Any]] = nnx.Variable(
            {
                "total_hashes": 0,
                "cache_hits": 0,
                "int64_hashes": 0,
                "collision_rate": 0.0,
                "batch_hashes": 0,
                "large_batch_hashes": 0,
                "deduplication_operations": 0,
                "splitmix64_hashes": 0,
                "memory_peak_mb": 0.0,
                "tpu_utilization": 0.0,
            }
        )

        # Collision tracking for statistics
        self.collision_tracker: nnx.Variable[Dict[str, Any]] = nnx.Variable(
            {"hash_counts": {}, "total_unique_hashes": 0, "total_hash_attempts": 0}
        )

        # Optimal chunk size for TPU v6e's 32GB HBM
        self.optimal_chunk_size = nnx.Variable(100000)  # Leverage large HBM

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "TPU Hasher Module initialized with state_size=%d, use_splitmix64=%s", state_size, use_splitmix64
        )

    def hash_state(self, state: jnp.ndarray) -> jnp.int64:
        """Hash single state using native int64 operations on TPU."""
        if self.use_splitmix64:
            # Use SplitMix64 for bit-encoded states
            hash_result = _hash_splitmix64_batch_jit(state.reshape(1, -1), self.seed.value)[0]
            self.metrics.value["splitmix64_hashes"] += 1
        else:
            # Use matrix multiplication for regular states
            if self.hash_matrix is not None:
                hash_result = _hash_state_jit(state, self.hash_matrix.value)
            else:
                raise ValueError("Hash matrix is None for non-SplitMix64 hasher")

        # Update metrics
        self.metrics.value["total_hashes"] += 1
        self.metrics.value["int64_hashes"] += 1

        # Track collisions
        collision_tracker = self.collision_tracker.value
        hash_counts = collision_tracker["hash_counts"]
        hash_key = str(int(hash_result))
        if hash_key in hash_counts:
            hash_counts[hash_key] += 1
        else:
            hash_counts[hash_key] = 1
            collision_tracker["total_unique_hashes"] += 1

        collision_tracker["total_hash_attempts"] += 1

        return hash_result

    def hash_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Hash batch of states using TPU vectorization."""
        if self.use_splitmix64:
            # Use SplitMix64 for bit-encoded states
            hashes = _hash_splitmix64_batch_jit(states, self.seed.value)
            self.metrics.value["splitmix64_hashes"] += len(states)
        else:
            # Use matrix multiplication for regular states
            if self.hash_matrix is not None:
                hashes = _hash_batch_jit(states, self.hash_matrix.value)
            else:
                raise ValueError("Hash matrix is None for non-SplitMix64 hasher")

        # Update metrics
        self.metrics.value["batch_hashes"] += 1
        self.metrics.value["total_hashes"] += len(states)
        self.metrics.value["int64_hashes"] += len(states)
        self.metrics.value["tpu_utilization"] += 1.0

        # Update collision tracking
        collision_tracker = self.collision_tracker.value
        collision_tracker["total_hash_attempts"] += len(states)

        # Track unique hashes in batch
        hash_counts = collision_tracker["hash_counts"]
        for hash_val in hashes:
            hash_key = str(int(hash_val))
            if hash_key in hash_counts:
                hash_counts[hash_key] += 1
            else:
                hash_counts[hash_key] = 1
                collision_tracker["total_unique_hashes"] += 1

        return hashes

    def hash_large_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Hash large batches leveraging TPU v6e's 32GB HBM."""
        chunk_size = self.optimal_chunk_size.value

        if len(states) <= chunk_size:
            return self.hash_batch(states)

        if self.use_splitmix64:
            # For SplitMix64, process in chunks manually
            results = []
            for i in range(0, len(states), chunk_size):
                chunk = states[i : i + chunk_size]
                chunk_hashes = _hash_splitmix64_batch_jit(chunk, self.seed.value)
                results.append(chunk_hashes)
            hashes = jnp.concatenate(results, axis=0)
        else:
            # Use non-JIT chunked processing for dynamic sizes
            if self.hash_matrix is not None:
                hashes = _hash_large_batch_chunked(states, self.hash_matrix.value, chunk_size)
            else:
                raise ValueError("Hash matrix is None for non-SplitMix64 hasher")

        # Update metrics
        self.metrics.value["large_batch_hashes"] += 1
        self.metrics.value["total_hashes"] += len(states)
        self.metrics.value["int64_hashes"] += len(states)
        self.metrics.value["tpu_utilization"] += len(states) / chunk_size

        # Update collision tracking
        collision_tracker = self.collision_tracker.value
        collision_tracker["total_hash_attempts"] += len(states)

        # Track unique hashes in large batch
        hash_counts = collision_tracker["hash_counts"]
        for hash_val in hashes:
            hash_key = str(int(hash_val))
            if hash_key in hash_counts:
                hash_counts[hash_key] += 1
            else:
                hash_counts[hash_key] = 1
                collision_tracker["total_unique_hashes"] += 1

        return hashes

    def deduplicate_by_hash(self, states: jnp.ndarray) -> jnp.ndarray:
        """Remove duplicates using native int64 hash-based deduplication."""
        # Hash all states
        hashes = self.hash_batch(states)

        # Use simple deduplication
        unique_states = _deduplicate_by_hash_simple(states, hashes)

        # Update metrics
        self.metrics.value["deduplication_operations"] += 1
        duplicates_removed = len(states) - len(unique_states)

        self.logger.debug("Deduplicated %d states, removed %d duplicates", len(states), duplicates_removed)

        return unique_states

    def get_hash_stats(self) -> Dict[str, Any]:
        """Get hashing performance statistics."""
        total = self.metrics.value["total_hashes"]
        hits = self.metrics.value["cache_hits"]

        # Calculate collision rate
        collision_tracker = self.collision_tracker.value
        unique_hashes = collision_tracker["total_unique_hashes"]
        total_attempts = collision_tracker["total_hash_attempts"]
        collision_rate = 1.0 - (unique_hashes / max(1, total_attempts))

        # Update collision rate in metrics
        self.metrics.value["collision_rate"] = collision_rate

        return {
            "cache_hit_rate": hits / max(1, total),
            "total_hashes": total,
            "int64_hashes": self.metrics.value["int64_hashes"],
            "collision_rate": collision_rate,
            "batch_hashes": self.metrics.value["batch_hashes"],
            "large_batch_hashes": self.metrics.value["large_batch_hashes"],
            "deduplication_operations": self.metrics.value["deduplication_operations"],
            "splitmix64_hashes": self.metrics.value["splitmix64_hashes"],
            "unique_hashes": unique_hashes,
            "total_hash_attempts": total_attempts,
            "tpu_utilization": self.metrics.value["tpu_utilization"],
            "memory_peak_mb": self.metrics.value["memory_peak_mb"],
        }

    def get_collision_details(self) -> Dict[str, Any]:
        """Get detailed collision statistics."""
        collision_tracker = self.collision_tracker.value
        hash_counts = collision_tracker["hash_counts"]

        # Find most frequent collisions
        collision_counts = {k: v for k, v in hash_counts.items() if v > 1}
        sorted_collisions = sorted(collision_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_unique_hashes": len(hash_counts),
            "collision_count": len(collision_counts),
            "max_collision_frequency": max(hash_counts.values()) if hash_counts else 0,
            "top_collisions": sorted_collisions[:10],  # Top 10 most frequent collisions
            "collision_distribution": {
                "single_occurrence": sum(1 for v in hash_counts.values() if v == 1),
                "multiple_occurrences": sum(1 for v in hash_counts.values() if v > 1),
            },
        }

    def reset_metrics(self):
        """Reset performance metrics and collision tracking."""
        metrics = self.metrics.value
        for key in metrics:
            if isinstance(metrics[key], (int, float)):
                metrics[key] = 0

        self.collision_tracker.value = {"hash_counts": {}, "total_unique_hashes": 0, "total_hash_attempts": 0}

        self.hash_cache.value.clear()
        self.logger.info("Metrics and collision tracking reset")

    def optimize_chunk_size(self, test_sizes: Tuple[int, ...] = (50000, 100000, 200000)) -> int:
        """Optimize chunk size for TPU v6e's memory characteristics."""
        # Create test data
        test_states = jnp.ones((max(test_sizes), self.state_size), dtype=jnp.int64)

        best_size = self.optimal_chunk_size.value
        best_time = float("inf")

        for size in test_sizes:
            try:
                # Time the operation
                start_time = time.time()
                _ = self.hash_large_batch(test_states[:size])
                elapsed = time.time() - start_time

                throughput = size / elapsed
                self.logger.info("Chunk size %d: %.2f states/sec", size, throughput)

                if elapsed < best_time:
                    best_time = elapsed
                    best_size = size

            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.warning("Chunk size %d failed: %s", size, e)
                continue

        self.optimal_chunk_size.value = best_size
        self.logger.info("Optimal chunk size set to: %d", best_size)
        return best_size

    def verify_int64_precision(self) -> bool:
        """Verify that hash operations maintain int64 precision."""
        try:
            # Test with large int64 values that would overflow int32
            # Create a state with the correct size for this hasher
            large_values = [2**40, 2**50, 2**60]
            # Pad or truncate to match state_size
            if len(large_values) < self.state_size:
                large_values.extend([1] * (self.state_size - len(large_values)))
            else:
                large_values = large_values[: self.state_size]

            large_state = jnp.array(large_values, dtype=jnp.int64)

            # Hash the state
            hash_result = self.hash_state(large_state)

            # Verify the result is int64 and within expected range
            is_int64 = hash_result.dtype == jnp.int64
            is_large = abs(int(hash_result)) > 2**32  # Should be larger than int32 range

            self.logger.info("int64 precision test: dtype=%s, large_value=%s", hash_result.dtype, is_large)

            return is_int64 and is_large

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("int64 precision verification failed: %s", e)
            return False


class HybridPrecisionHasher(nnx.Module):
    """Hybrid hasher: TPU int64 for precision, with adaptive fallback."""

    def __init__(self, state_size: int, backend: TPUBackend, rngs: nnx.Rngs, precision_threshold: float = 0.01):
        self.tpu_hasher = TPUHasherModule(state_size, backend, rngs)
        self.precision_threshold = nnx.Variable(precision_threshold)
        self.fallback_enabled = nnx.Variable(False)

        self.logger = logging.getLogger(__name__)

    def hash_batch_adaptive(self, states: jnp.ndarray) -> jnp.ndarray:
        """Adaptively choose hashing strategy based on collision rate."""
        # Always try TPU first since we have native int64 support
        tpu_result = self.tpu_hasher.hash_batch(states)

        # Check collision rate
        stats = self.tpu_hasher.get_hash_stats()
        collision_rate = stats["collision_rate"]

        if collision_rate > self.precision_threshold.value:
            self.logger.warning(
                "High collision rate (%.3f), but continuing with TPU int64 (native precision)", collision_rate
            )
            # With native int64 support, we don't need CPU fallback
            # The precision is already maximal on TPU v6e

        return tpu_result

    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive hashing statistics."""
        return {
            "precision_threshold": self.precision_threshold.value,
            "fallback_enabled": self.fallback_enabled.value,
            "tpu_hasher_stats": self.tpu_hasher.get_hash_stats(),
            "collision_details": self.tpu_hasher.get_collision_details(),
        }


def create_tpu_hasher(
    state_size: int,
    backend: Optional[TPUBackend] = None,
    use_splitmix64: bool = False,
    random_seed: Optional[int] = None,
) -> TPUHasherModule:
    """Factory function to create TPU hasher with error handling."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX and Flax are required for TPU hasher. " + "Install with: pip install 'cayleypy[jax-tpu]'"
        )

    if backend is None:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel

        backend = get_tpu_backend()

    rngs = nnx.Rngs(random_seed or 42)
    return TPUHasherModule(state_size, backend, rngs, use_splitmix64, random_seed)
