"""
TPU BFS Performance Optimization - Phase 1 Implementation Specification

This module defines the implementation specification for Phase 1 optimizations:
- Compilation optimization with kernel caching
- Memory layout optimization for TPU
- Kernel fusion for BFS operations
"""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import hashlib

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


@dataclass
class KernelSignature:
    """Signature for TPU kernel caching."""

    state_size: int
    generators_count: int
    max_diameter: int
    use_bitmask: bool
    precision: str = "int64"

    def to_hash(self) -> str:
        """Generate hash for kernel caching."""
        signature_str = (
            f"{self.state_size}_{self.generators_count}_{self.max_diameter}_{self.use_bitmask}_{self.precision}"
        )
        return hashlib.md5(signature_str.encode()).hexdigest()


class TPUKernelCache:
    """Persistent kernel cache for TPU BFS operations."""

    def __init__(self):
        self.compiled_kernels: Dict[str, Any] = {}
        self.kernel_metadata: Dict[str, KernelSignature] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get_or_compile_bfs_kernel(self, signature: KernelSignature):
        """Get cached kernel or compile new one."""
        cache_key = signature.to_hash()

        if cache_key in self.compiled_kernels:
            self.cache_hits += 1
            return self.compiled_kernels[cache_key]

        # Compile new kernel
        self.cache_misses += 1
        kernel = self._compile_bfs_kernel(signature)

        # Cache the compiled kernel
        self.compiled_kernels[cache_key] = kernel
        self.kernel_metadata[cache_key] = signature

        return kernel

    def _compile_bfs_kernel(self, signature: KernelSignature):
        """Compile optimized BFS kernel for given signature."""

        @jax.jit
        def optimized_bfs_kernel(states, generators, visited_hashes):
            """Optimized BFS kernel with fused operations."""
            # Phase 1.3: Fused BFS step
            return fused_bfs_step_optimized(states, generators, visited_hashes, signature)

        return optimized_bfs_kernel

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(total_requests, 1)

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cached_kernels": len(self.compiled_kernels),
        }


class TPUMemoryOptimizer:
    """Memory layout optimization for TPU operations."""

    @staticmethod
    def pad_to_tpu_optimal(shape: Tuple[int, ...], target_multiple: int = 128) -> Tuple[int, ...]:
        """Pad shape to TPU-optimal dimensions (multiples of 128)."""
        optimized_shape = []
        for dim in shape:
            # Pad to next multiple of target_multiple
            padded_dim = ((dim + target_multiple - 1) // target_multiple) * target_multiple
            optimized_shape.append(padded_dim)
        return tuple(optimized_shape)

    @staticmethod
    def optimize_state_layout(states: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Optimize state array layout for TPU memory hierarchy."""
        original_shape = states.shape

        # Phase 1.2: Memory layout optimization
        optimal_shape = TPUMemoryOptimizer.pad_to_tpu_optimal(original_shape)

        # Pad the array to optimal shape
        pad_widths = [(0, opt_dim - orig_dim) for orig_dim, opt_dim in zip(original_shape, optimal_shape)]
        optimized_states = jnp.pad(states, pad_widths, mode="constant", constant_values=0)

        # Reshape for optimal memory access patterns
        if len(optimal_shape) == 2:
            # Ensure dimensions are multiples of 128 for systolic array efficiency
            optimized_states = optimized_states.reshape(optimal_shape)

        metadata = {
            "original_shape": original_shape,
            "optimized_shape": optimal_shape,
            "padding_applied": pad_widths,
            "memory_efficiency": (states.size / optimized_states.size) * 100,
        }

        return optimized_states, metadata

    @staticmethod
    def unpad_results(padded_results: jnp.ndarray, original_shape: Tuple[int, ...]) -> jnp.ndarray:
        """Remove padding from results to restore original shape."""
        # Extract original data from padded array
        slices = tuple(slice(0, dim) for dim in original_shape)
        return padded_results[slices]


def fused_bfs_step_optimized(
    states: jnp.ndarray, generators: jnp.ndarray, visited_hashes: jnp.ndarray, _signature: KernelSignature
) -> jnp.ndarray:
    """Fused BFS step with all operations in single kernel."""

    # Phase 1.3: Kernel fusion implementation

    # Step 1: Vectorized expansion (optimized for TPU)
    expanded_states = expand_layer_vectorized_optimized(states, generators)

    # Step 2: Parallel hashing (fused with expansion)
    state_hashes = hash_batch_fused(expanded_states)

    # Step 3: Deduplication and filtering (single pass)
    unique_states = deduplicate_and_filter_fused(expanded_states, state_hashes, visited_hashes)

    return unique_states


@jax.jit
def expand_layer_vectorized_optimized(states: jnp.ndarray, generators: jnp.ndarray) -> jnp.ndarray:
    """Optimized vectorized layer expansion for TPU."""

    def apply_single_generator(gen_idx):
        """Apply single generator to all states."""
        generator = generators[gen_idx]
        # Vectorized permutation application
        return jax.vmap(lambda state: state[generator])(states)

    # Apply all generators in parallel
    generator_indices = jnp.arange(len(generators))
    expanded_per_gen = jax.vmap(apply_single_generator)(generator_indices)

    # Reshape to flat list of expanded states
    total_expanded = expanded_per_gen.reshape(-1, states.shape[-1])

    return total_expanded


@jax.jit
def hash_batch_fused(states: jnp.ndarray) -> jnp.ndarray:
    """Fused batch hashing optimized for TPU."""

    # Use TPU-optimized hash function
    def tpu_hash_function(state):
        """TPU-optimized hash function using systolic arrays."""
        # Polynomial hash with TPU-friendly operations
        hash_base = jnp.int64(1000000007)  # Large prime
        hash_val = jnp.int64(0)

        # Vectorized hash computation
        powers = jnp.power(hash_base, jnp.arange(len(state), dtype=jnp.int64))
        hash_val = jnp.sum(state.astype(jnp.int64) * powers)

        return hash_val

    # Parallel hashing across all states
    return jax.vmap(tpu_hash_function)(states)


@jax.jit
def deduplicate_and_filter_fused(states: jnp.ndarray, hashes: jnp.ndarray, visited_hashes: jnp.ndarray) -> jnp.ndarray:
    """Fused deduplication and filtering in single pass."""

    # Step 1: Remove duplicates within current batch
    unique_hashes, unique_indices = jnp.unique(hashes, return_index=True, size=len(hashes))

    # Filter out fill values from jnp.unique
    fill_value = jnp.iinfo(hashes.dtype).min
    valid_mask = unique_hashes != fill_value
    valid_indices = unique_indices[valid_mask]
    valid_hashes = unique_hashes[valid_mask]

    if len(valid_indices) == 0:
        return jnp.array([], dtype=states.dtype).reshape(0, states.shape[-1])

    unique_states = states[valid_indices]

    # Step 2: Filter out previously visited states
    def is_not_visited(hash_val):
        return ~jnp.any(visited_hashes == hash_val)

    not_visited_mask = jax.vmap(is_not_visited)(valid_hashes)
    new_states = unique_states[not_visited_mask]

    return new_states


class OptimizedTPUBFSModule(nnx.Module):
    """Phase 1 optimized TPU BFS module."""

    def __init__(self, graph, backend, _rngs: nnx.Rngs):
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax required for optimized TPU BFS")

        self.graph = graph
        self.backend = backend

        # Initialize optimization components
        self.kernel_cache = TPUKernelCache()
        self.memory_optimizer = TPUMemoryOptimizer()

        # Graph properties
        if hasattr(graph, "definition"):
            graph_def = graph.definition
            central_state = graph.central_state
        else:
            graph_def = graph
            central_state = graph.central_state

        self.state_size = len(central_state)
        self.generators = nnx.Param(jnp.array(graph_def.generators_permutations, dtype=jnp.int64))

        # Performance tracking
        self.optimization_metrics = nnx.Variable(
            {
                "kernel_cache_hits": 0,
                "memory_optimizations": 0,
                "fused_operations": 0,
                "compilation_time_saved": 0.0,
                "memory_efficiency": 0.0,
            }
        )

    def optimized_bfs_step(
        self, current_layer: jnp.ndarray, visited_hashes: List[jnp.ndarray], max_diameter: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Optimized BFS step using Phase 1 optimizations."""

        # Create kernel signature
        signature = KernelSignature(
            state_size=self.state_size,
            generators_count=len(self.generators.value),
            max_diameter=max_diameter,
            use_bitmask=False,
        )

        # Phase 1.1: Get cached or compile kernel
        bfs_kernel = self.kernel_cache.get_or_compile_bfs_kernel(signature)

        # Phase 1.2: Optimize memory layout
        optimized_states, memory_metadata = self.memory_optimizer.optimize_state_layout(current_layer)

        # Combine visited hashes for efficient filtering
        if visited_hashes:
            combined_visited = jnp.concatenate(visited_hashes)
        else:
            combined_visited = jnp.array([], dtype=jnp.int64)

        # Phase 1.3: Execute fused kernel
        new_states = bfs_kernel(optimized_states, self.generators.value, combined_visited)

        # Unpad results if necessary
        if new_states.shape != current_layer.shape:
            new_states = self.memory_optimizer.unpad_results(new_states, current_layer.shape)

        # Generate hashes for new states
        new_hashes = hash_batch_fused(new_states)

        # Update metrics
        self.optimization_metrics.value.update(
            {
                "kernel_cache_hits": self.kernel_cache.cache_hits,
                "memory_optimizations": self.optimization_metrics.value["memory_optimizations"] + 1,
                "fused_operations": self.optimization_metrics.value["fused_operations"] + 1,
                "memory_efficiency": memory_metadata["memory_efficiency"],
            }
        )

        return new_states, new_hashes

    def run_optimized_bfs(self, max_diameter: int = 1000) -> List[int]:
        """Run BFS with Phase 1 optimizations."""

        # Initialize with central state
        if hasattr(self.graph, "definition"):
            central_state = self.graph.central_state
        else:
            central_state = self.graph.central_state

        start_state = jnp.array([central_state], dtype=jnp.int64)
        start_hash = hash_batch_fused(start_state)

        current_layer = start_state
        visited_hashes = [start_hash]
        layer_sizes = [1]

        for _ in range(max_diameter):
            # Optimized BFS step
            new_layer, new_hashes = self.optimized_bfs_step(current_layer, visited_hashes, max_diameter)

            if len(new_layer) == 0:
                break

            # Update state
            current_layer = new_layer
            visited_hashes.append(new_hashes)
            layer_sizes.append(len(new_layer))

        return layer_sizes

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get Phase 1 optimization metrics."""
        base_metrics = dict(self.optimization_metrics.value)
        cache_stats = self.kernel_cache.get_cache_stats()

        return {**base_metrics, "cache_stats": cache_stats, "phase": "Phase 1 - Foundation Optimizations"}


def create_optimized_tpu_bfs(graph, backend=None) -> OptimizedTPUBFSModule:
    """Factory function for Phase 1 optimized TPU BFS."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX and Flax required for optimized TPU BFS")

    if backend is None:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel

        backend = get_tpu_backend()

    rngs = nnx.Rngs(42)
    return OptimizedTPUBFSModule(graph, backend, rngs)


# Performance testing utilities for Phase 1
def benchmark_phase1_optimizations(graph, max_diameter: int = 20) -> Dict[str, Any]:
    """Benchmark Phase 1 optimizations against baseline."""

    if not JAX_AVAILABLE:
        return {"error": "JAX not available"}

    try:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel
        from .tpu_bfs import create_tpu_bfs  # pylint: disable=import-outside-toplevel

        backend = get_tpu_backend()
        if not backend.is_available:
            return {"error": "TPU not available"}

        import time  # pylint: disable=import-outside-toplevel

        # Baseline implementation
        baseline_module = create_tpu_bfs(graph, backend)

        start_time = time.time()
        baseline_result = baseline_module.run_bfs(max_diameter)
        baseline_time = time.time() - start_time

        # Optimized implementation
        optimized_module = create_optimized_tpu_bfs(graph, backend)

        start_time = time.time()
        optimized_result = optimized_module.run_optimized_bfs(max_diameter)
        optimized_time = time.time() - start_time

        # Calculate improvements
        speedup = baseline_time / optimized_time if optimized_time > 0 else 0
        results_identical = baseline_result == optimized_result

        # Get optimization metrics
        opt_metrics = optimized_module.get_optimization_metrics()

        return {
            "baseline_time": baseline_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
            "results_identical": results_identical,
            "baseline_result": baseline_result,
            "optimized_result": optimized_result,
            "optimization_metrics": opt_metrics,
            "success": speedup >= 1.5 and results_identical,  # Phase 1 target: 1.5x speedup
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test Phase 1 optimizations
    print("TPU BFS Phase 1 Optimization Test")
    print("=" * 50)

    if not JAX_AVAILABLE:
        print("❌ JAX not available")
        import sys

        sys.exit(1)

    try:
        from .graphs_lib import PermutationGroups  # pylint: disable=import-outside-toplevel
        from .cayley_graph import CayleyGraph  # pylint: disable=import-outside-toplevel

        # Test with medium-sized graph
        test_graph = CayleyGraph(PermutationGroups.coxeter(6))

        print("Testing on graph: S6 Coxeter")
        print(f"State size: {len(test_graph.central_state)}")
        print(f"Generators: {len(test_graph.definition.generators_permutations)}")

        # Run benchmark
        results = benchmark_phase1_optimizations(test_graph, max_diameter=15)

        if "error" in results:
            print(f"❌ Test failed: {results['error']}")
            import sys

            sys.exit(1)

        print("\n📊 PHASE 1 OPTIMIZATION RESULTS:")
        print(f"   Baseline time: {results['baseline_time']:.3f}s")
        print(f"   Optimized time: {results['optimized_time']:.3f}s")
        print(f"   Speedup: {results['speedup']:.2f}x")
        print(f"   Results identical: {'✅' if results['results_identical'] else '❌'}")

        if results["success"]:
            print("   🎉 Phase 1 optimization SUCCESS!")
        else:
            print(f"   ⚠️  Phase 1 optimization needs improvement")

        # Show optimization metrics
        opt_metrics = results["optimization_metrics"]
        print(f"\n🔧 OPTIMIZATION METRICS:")
        print(f"   Cache hit rate: {opt_metrics['cache_stats']['hit_rate']:.1%}")
        print(f"   Memory efficiency: {opt_metrics['memory_efficiency']:.1f}%")
        print(f"   Fused operations: {opt_metrics['fused_operations']}")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import sys

        sys.exit(1)
