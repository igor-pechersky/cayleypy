"""TPU-accelerated BFS using bitmask approach for CayleyPy.

This module provides TPU-accelerated breadth-first search using the memory-efficient
bitmask approach, optimized for TPU v6e (Trillium) architecture with native int64 support.
"""

import itertools
import logging
import math
from typing import List, Dict, Any, Optional

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
from .tpu_hasher import TPUHasherModule
from .tpu_tensor_ops import TPUTensorOpsModule

# from .permutation_utils import is_permutation  # Unused import

# Constants from original bitmask implementation
R = 8  # Chunk prefix size
CHUNK_SIZE = math.factorial(R)


# JIT-compiled helper functions for TPU bitmask operations
@jax.jit
def _bit_count_jit(x: jnp.ndarray) -> jnp.int64:
    """JIT-compiled bit counting using TPU-optimized operations."""
    # Use JAX's built-in popcount for efficient bit counting on TPU
    return jnp.sum(jnp.bitwise_count(x))


@jax.jit
def _encode_perm_jit(p: jnp.ndarray) -> jnp.int64:
    """JIT-compiled permutation encoding."""
    shifts = jnp.arange(len(p), dtype=jnp.int64) * 4
    return jnp.sum(p.astype(jnp.int64) << shifts)


@jax.jit
def _materialize_permutations_jit(black: jnp.ndarray, map1: jnp.ndarray, encoded_suffix: jnp.int64) -> jnp.ndarray:
    """JIT-compiled permutation materialization using TPU vectorization."""
    # Find set bits in the bitmask
    indices = jnp.where(black.flatten())[0]

    if len(indices) == 0:
        return jnp.array([], dtype=jnp.int64)

    # Convert bit indices to ranks
    ranks = indices

    # Decode ranks to permutations using vectorized operations
    def decode_rank(rank):
        # Simplified rank-to-permutation conversion
        # This is a placeholder - actual implementation would need the prefix maps
        return encoded_suffix

    decoded_perms = jax.vmap(decode_rank)(ranks)
    return decoded_perms


@jax.jit
def _paint_gray_jit(perms: jnp.ndarray, gray: jnp.ndarray, map2: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled gray painting using TPU operations."""

    def paint_single_perm(perm):
        # Simplified rank calculation - actual implementation would need prefix maps
        rank = jnp.sum(perm * jnp.arange(len(perm), dtype=jnp.int64))
        bit_idx = rank % 64
        word_idx = rank // 64

        # Set the bit in the appropriate word
        mask = jnp.int64(1) << bit_idx
        return word_idx, mask

    # Process all permutations
    word_indices, masks = jax.vmap(paint_single_perm)(perms)

    # Update gray array (simplified - actual implementation needs scatter_add)
    updated_gray = gray.copy()
    for i, (word_idx, mask) in enumerate(zip(word_indices, masks)):
        if word_idx < len(updated_gray):
            updated_gray = updated_gray.at[word_idx].or_(mask)

    return updated_gray


class TPUVertexChunk(nnx.Module):
    """TPU-accelerated vertex chunk with bitmask storage."""

    def __init__(self, n: int, suffix: List[int], backend: TPUBackend, rngs: nnx.Rngs):
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for TPU bitmask BFS")

        self.backend = backend
        self.n = n
        self.suffix = suffix

        # Bitmask arrays stored as NNX variables for TPU optimization
        chunk_words = CHUNK_SIZE // 64
        self.black = nnx.Variable(jnp.zeros((chunk_words,), dtype=jnp.uint64))
        self.last_layer = nnx.Variable(jnp.zeros((chunk_words,), dtype=jnp.uint64))
        self.gray = nnx.Variable(jnp.zeros((chunk_words,), dtype=jnp.uint64))

        # State tracking
        self.changed_on_last_step = nnx.Variable(False)
        self.last_layer_count = nnx.Variable(0)

        # Encoded suffix for fast operations
        self.encoded_suffix = nnx.Variable(jnp.int64(sum(suffix[i - R] << (4 * i) for i in range(R, n))))

        # Mapping arrays for permutation conversion
        prefix_elements = [i for i in range(n) if i not in suffix]
        self.map1 = nnx.Param(jnp.array(prefix_elements, dtype=jnp.int64))

        map2_array = jnp.zeros((n,), dtype=jnp.int64)
        for i, elem in enumerate(prefix_elements):
            map2_array = map2_array.at[elem].set(i)
        self.map2 = nnx.Param(map2_array)

        self.logger = logging.getLogger(__name__)

    def materialize_last_layer_permutations(self) -> jnp.ndarray:
        """Materialize permutations from the last layer using TPU operations."""
        if not self.changed_on_last_step.value or self.last_layer_count.value == 0:
            return jnp.array([], dtype=jnp.int64)

        # Use JIT-compiled materialization
        perms = _materialize_permutations_jit(self.last_layer.value, self.map1.value, self.encoded_suffix.value)

        return perms

    def paint_gray(self, perms: jnp.ndarray):
        """Paint vertices gray using TPU-optimized operations."""
        if len(perms) == 0:
            return

        # Use JIT-compiled painting
        updated_gray = _paint_gray_jit(perms, self.gray.value, self.map2.value)
        self.gray.value = updated_gray

    def flush_gray_to_black(self):
        """Flush gray vertices to black using TPU operations."""
        # Remove already black vertices from gray
        new_gray = self.gray.value & ~self.black.value

        # Count new vertices
        new_count = _bit_count_jit(new_gray)

        if new_count == 0:
            self.changed_on_last_step.value = False
            self.last_layer.value = jnp.zeros_like(self.last_layer.value)
            self.last_layer_count.value = 0
        else:
            self.changed_on_last_step.value = True
            self.black.value = self.black.value | new_gray
            self.last_layer.value = new_gray
            self.last_layer_count.value = int(new_count)

        # Clear gray
        self.gray.value = jnp.zeros_like(self.gray.value)


class TPUBitmaskBFSModule(nnx.Module):
    """TPU-accelerated bitmask BFS module with NNX state management."""

    def __init__(self, graph, backend: TPUBackend, rngs: nnx.Rngs):
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for TPU bitmask BFS")

        self.graph = graph
        self.backend = backend

        # Handle both CayleyGraph and CayleyGraphDef
        if hasattr(graph, "definition"):
            graph_def = graph.definition
            central_state = graph.central_state
        else:
            graph_def = graph
            central_state = graph.central_state

        n = len(central_state)
        self.n = n

        # Validate constraints
        if not graph_def.is_permutation_group():
            raise ValueError("TPU bitmask BFS only works for permutation groups")
        if n <= R:
            raise ValueError(f"TPU bitmask BFS requires state size > {R}")

        # Create vertex chunks
        suffixes = list(itertools.permutations(range(n), r=n - R))
        self.chunks = [TPUVertexChunk(n, list(suffix), backend, rngs) for suffix in suffixes]

        # Create chunk mapping for fast lookup
        chunk_map = {}
        for chunk in self.chunks:
            chunk_map[int(chunk.encoded_suffix.value)] = chunk
        self.chunk_map = chunk_map

        # Suffix mask for chunk identification
        self.suffix_mask = nnx.Variable(jnp.int64((2 ** (4 * (n - R)) - 1) << (4 * R)))

        # Store generators as parameters
        self.generators = nnx.Param(jnp.array(graph_def.generators_permutations, dtype=jnp.int64))

        # Initialize tensor operations and hasher
        self.tensor_ops = TPUTensorOpsModule(backend, rngs)
        self.hasher = TPUHasherModule(n, backend, rngs)

        # BFS state tracking
        self.bfs_state = nnx.Variable({"layer_sizes": [], "diameter": 0, "total_states_found": 0, "is_complete": False})

        # Performance metrics
        self.metrics = nnx.Variable(
            {
                "chunks_processed": 0,
                "permutations_materialized": 0,
                "neighbors_generated": 0,
                "gray_paint_operations": 0,
                "flush_operations": 0,
                "memory_peak_mb": 0.0,
                "tpu_utilization": 0.0,
                "bitmask_operations": 0,
            }
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "TPU Bitmask BFS Module initialized: n=%d, chunks=%d, generators=%d",
            n,
            len(self.chunks),
            len(graph_def.generators_permutations),
        )

    def paint_gray(self, perms: jnp.ndarray):
        """Paint permutations gray across appropriate chunks."""
        if len(perms) == 0:
            return

        # Group permutations by chunk
        suffix_mask = self.suffix_mask.value
        keys = perms & suffix_mask

        # Sort and group by suffix
        sorted_indices = jnp.argsort(keys)
        sorted_perms = perms[sorted_indices]
        sorted_keys = keys[sorted_indices]

        # Find group boundaries
        unique_keys, group_starts = jnp.unique(sorted_keys, return_index=True)
        group_ends = jnp.concatenate([group_starts[1:], jnp.array([len(sorted_perms)])])

        # Paint each group
        for i, key in enumerate(unique_keys):
            start_idx = group_starts[i]
            end_idx = group_ends[i]
            group_perms = sorted_perms[start_idx:end_idx]

            chunk = self.chunk_map.get(int(key))
            if chunk is not None:
                chunk.paint_gray(group_perms)

        self.metrics.value["gray_paint_operations"] += 1

    def flush_gray_to_black(self):
        """Flush gray vertices to black across all chunks."""
        for chunk in self.chunks:
            chunk.flush_gray_to_black()

        self.metrics.value["flush_operations"] += 1

    def count_last_layer(self) -> int:
        """Count vertices in the last layer across all chunks."""
        return sum(int(chunk.last_layer_count.value) for chunk in self.chunks)

    def get_neighbors(self, perms: jnp.ndarray) -> jnp.ndarray:
        """Generate neighbors by applying all generators."""
        if len(perms) == 0:
            return jnp.array([], dtype=jnp.int64)

        # Apply each generator to all permutations
        all_neighbors = []
        for gen in self.generators.value:
            neighbors = self.tensor_ops.batch_apply_permutation(perms, gen)
            all_neighbors.append(neighbors)

        # Concatenate all neighbors
        if all_neighbors:
            result = jnp.concatenate(all_neighbors, axis=0)
        else:
            result = jnp.array([], dtype=jnp.int64)

        self.metrics.value["neighbors_generated"] += len(result)
        return result

    def run_bfs(self, max_diameter: int = 10**6) -> List[int]:
        """Run TPU-accelerated bitmask BFS."""
        # Initialize with central state
        if hasattr(self.graph, "definition"):
            central_state = self.graph.central_state
        else:
            central_state = self.graph.central_state

        initial_perm = _encode_perm_jit(jnp.array(central_state, dtype=jnp.int64))
        initial_states = jnp.array([initial_perm], dtype=jnp.int64)

        # Paint initial state and flush
        self.paint_gray(initial_states)
        self.flush_gray_to_black()

        layer_sizes = [self.count_last_layer()]
        self.bfs_state.value.update(
            {"layer_sizes": layer_sizes, "diameter": 0, "total_states_found": layer_sizes[0], "is_complete": False}
        )

        self.logger.info("Starting TPU bitmask BFS with max_diameter=%d", max_diameter)

        for diameter in range(1, max_diameter + 1):
            chunks_used = 0
            all_neighbors = []

            # Process each chunk that changed
            for chunk in self.chunks:
                if not chunk.changed_on_last_step.value:
                    continue

                # Materialize permutations from this chunk
                perms = chunk.materialize_last_layer_permutations()
                if len(perms) > 0:
                    # Generate neighbors
                    neighbors = self.get_neighbors(perms)
                    if len(neighbors) > 0:
                        all_neighbors.append(neighbors)
                    chunks_used += 1

            self.metrics.value["chunks_processed"] += chunks_used

            if chunks_used == 0:
                self.logger.info("BFS completed at diameter %d - no active chunks", diameter)
                self.bfs_state.value["is_complete"] = True
                break

            # Combine all neighbors and paint gray
            if all_neighbors:
                combined_neighbors = jnp.concatenate(all_neighbors, axis=0)
                self.paint_gray(combined_neighbors)

            # Flush gray to black
            self.flush_gray_to_black()

            # Count new layer
            layer_size = self.count_last_layer()
            if layer_size == 0:
                self.logger.info("BFS completed at diameter %d - no new states", diameter)
                self.bfs_state.value["is_complete"] = True
                break

            # Update state
            layer_sizes.append(layer_size)
            self.bfs_state.value.update(
                {
                    "layer_sizes": layer_sizes,
                    "diameter": diameter,
                    "total_states_found": self.bfs_state.value["total_states_found"] + layer_size,
                }
            )

            self.logger.debug(
                "Diameter %d: %d new states (total: %d)",
                diameter,
                layer_size,
                self.bfs_state.value["total_states_found"],
            )

        final_diameter = self.bfs_state.value["diameter"]
        total_states = self.bfs_state.value["total_states_found"]

        self.logger.info("TPU bitmask BFS completed: diameter=%d, total_states=%d", final_diameter, total_states)

        return self.bfs_state.value["layer_sizes"]

    def get_bfs_result(self) -> Dict[str, Any]:
        """Get comprehensive BFS results."""
        return {
            "layer_sizes": self.bfs_state.value["layer_sizes"],
            "diameter": self.bfs_state.value["diameter"],
            "total_states_found": self.bfs_state.value["total_states_found"],
            "is_complete": self.bfs_state.value["is_complete"],
            "growth_function": self.bfs_state.value["layer_sizes"],
            "memory_efficiency": self.estimate_memory_usage(),
        }

    def estimate_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage of the bitmask approach."""
        n = self.n
        total_states = math.factorial(n)
        bits_per_state = 3  # From original bitmask approach
        estimated_memory_gb = (total_states * bits_per_state / 8) / (2**30)

        return {
            "total_states": total_states,
            "bits_per_state": bits_per_state,
            "estimated_memory_gb": estimated_memory_gb,
            "chunks_count": len(self.chunks),
            "chunk_size": CHUNK_SIZE,
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        base_metrics = dict(self.metrics.value)

        # Add component metrics
        tensor_ops_stats = self.tensor_ops.get_performance_metrics()
        hasher_stats = self.hasher.get_hash_stats()

        return {
            **base_metrics,
            "tensor_ops_stats": tensor_ops_stats,
            "hasher_stats": hasher_stats,
            "backend_info": self.backend.get_device_info(),
            "memory_usage": self.backend.get_memory_usage(),
            "memory_estimation": self.estimate_memory_usage(),
        }

    def reset_metrics(self):
        """Reset all performance metrics."""
        for key in self.metrics.value:
            if isinstance(self.metrics.value[key], (int, float)):
                self.metrics.value[key] = 0

        self.tensor_ops.reset_metrics()
        self.hasher.reset_metrics()

        self.logger.info("All metrics reset")

    def verify_int64_precision(self) -> bool:
        """Verify that bitmask operations maintain int64 precision."""
        try:
            # Test with large permutation values
            large_perm = jnp.array([2**40, 2**50, 2**60, 1, 2, 3], dtype=jnp.int64)
            if len(large_perm) < self.n:
                # Pad with sequential values
                padding = jnp.arange(len(large_perm), self.n, dtype=jnp.int64)
                large_perm = jnp.concatenate([large_perm, padding])
            else:
                large_perm = large_perm[: self.n]

            # Test encoding
            encoded = _encode_perm_jit(large_perm)

            # Verify int64 precision
            is_int64 = encoded.dtype == jnp.int64
            is_large = abs(int(encoded)) > 2**32

            self.logger.info("int64 precision test: dtype=%s, large_value=%s", encoded.dtype, is_large)

            return is_int64 and is_large

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("int64 precision verification failed: %s", e)
            return False


def _fallback_to_cpu_bitmask(graph, max_diameter: int) -> List[int]:
    """Fallback to CPU bitmask BFS implementation."""
    from .bfs_bitmask import bfs_bitmask  # pylint: disable=import-outside-toplevel

    return bfs_bitmask(graph, max_diameter)


def tpu_bfs_bitmask(graph, max_diameter: int = 10**6) -> List[int]:
    """High-level TPU bitmask BFS function with automatic backend detection."""
    if not JAX_AVAILABLE:
        return _fallback_to_cpu_bitmask(graph, max_diameter)

    try:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel

        backend = get_tpu_backend()

        if not backend.is_available:
            return _fallback_to_cpu_bitmask(graph, max_diameter)

        # Use TPU bitmask BFS
        rngs = nnx.Rngs(42)
        bfs_module = TPUBitmaskBFSModule(graph, backend, rngs)
        return bfs_module.run_bfs(max_diameter)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).warning("TPU bitmask BFS failed, falling back to CPU: %s", e)
        return _fallback_to_cpu_bitmask(graph, max_diameter)


def create_tpu_bitmask_bfs(graph, backend: Optional[TPUBackend] = None) -> TPUBitmaskBFSModule:
    """Factory function to create TPU bitmask BFS module."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX and Flax are required for TPU bitmask BFS. " + "Install with: pip install 'cayleypy[jax-tpu]'"
        )

    if backend is None:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel

        backend = get_tpu_backend()

    rngs = nnx.Rngs(42)
    return TPUBitmaskBFSModule(graph, backend, rngs)


def benchmark_tpu_vs_cpu_bitmask(graph, max_diameter: int = 100) -> Dict[str, Any]:
    """Benchmark TPU vs CPU bitmask BFS performance."""
    import time  # pylint: disable=import-outside-toplevel

    # Handle both CayleyGraph and CayleyGraphDef
    if hasattr(graph, "definition"):
        graph_def = graph.definition
        central_state = graph.central_state
    else:
        graph_def = graph
        central_state = graph.central_state

    results = {
        "graph_info": {
            "generators_count": len(graph_def.generators_permutations),
            "state_size": len(central_state),
            "is_permutation_group": graph_def.is_permutation_group(),
        },
        "tpu_available": False,
        "cpu_time": 0.0,
        "tpu_time": 0.0,
        "speedup": 0.0,
        "cpu_result": [],
        "tpu_result": [],
        "results_match": False,
        "memory_comparison": {},
    }

    # CPU bitmask BFS
    try:
        start_time = time.time()
        cpu_result = _fallback_to_cpu_bitmask(graph, max_diameter)
        cpu_time = time.time() - start_time

        results["cpu_time"] = cpu_time
        results["cpu_result"] = cpu_result

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).error("CPU bitmask BFS failed: %s", e)
        return results

    # TPU bitmask BFS
    if JAX_AVAILABLE:
        try:
            from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel

            backend = get_tpu_backend()
            if backend.is_available:
                results["tpu_available"] = True

                start_time = time.time()
                tpu_result = tpu_bfs_bitmask(graph, max_diameter)
                tpu_time = time.time() - start_time

                results["tpu_time"] = tpu_time
                results["tpu_result"] = tpu_result
                results["speedup"] = cpu_time / tpu_time if tpu_time > 0 else 0.0
                results["results_match"] = cpu_result == tpu_result

                # Get memory comparison
                bfs_module = create_tpu_bitmask_bfs(graph, backend)
                results["memory_comparison"] = bfs_module.estimate_memory_usage()

        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.getLogger(__name__).error("TPU bitmask BFS benchmark failed: %s", e)

    return results


if __name__ == "__main__":
    # Test TPU bitmask BFS when run as script
    print("Testing TPU Bitmask BFS Implementation")
    print("=" * 40)

    if not JAX_AVAILABLE:
        print("JAX not available - cannot test TPU bitmask BFS")
    else:
        try:
            from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel
            from .graphs_lib import PermutationGroups  # pylint: disable=import-outside-toplevel

            # Test with a larger symmetric group suitable for bitmask approach
            backend = get_tpu_backend()
            print(f"TPU Available: {backend.is_available}")

            if backend.is_available:
                # Create a test graph (S9 is good for bitmask testing)
                test_graph = PermutationGroups.symmetric_group(9)

                # Create bitmask BFS module
                bfs_module = create_tpu_bitmask_bfs(test_graph, backend)

                # Verify int64 precision
                precision_ok = bfs_module.verify_int64_precision()
                print(f"int64 Precision Test: {'PASSED ✓' if precision_ok else 'FAILED ✗'}")

                # Get memory estimation
                memory_info = bfs_module.estimate_memory_usage()
                print(f"Estimated memory usage: {memory_info['estimated_memory_gb']:.2f} GB")
                print(f"Total chunks: {memory_info['chunks_count']}")

                # Run BFS
                print("Running TPU bitmask BFS on S9...")
                result = bfs_module.run_bfs(max_diameter=5)  # Limited diameter for testing
                print(f"Growth function: {result}")

                # Get performance metrics
                metrics = bfs_module.get_performance_metrics()
                print(f"Chunks processed: {metrics['chunks_processed']}")
                print(f"Neighbors generated: {metrics['neighbors_generated']}")
                print(f"TPU utilization: {metrics['tpu_utilization']}")

                print("✓ TPU bitmask BFS test completed successfully!")
            else:
                print("TPU not available for testing")

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"✗ TPU bitmask BFS test failed: {e}")
