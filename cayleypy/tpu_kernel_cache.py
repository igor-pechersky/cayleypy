"""TPU Kernel Caching System for CayleyPy.

This module provides a sophisticated kernel caching system to eliminate compilation overhead
in TPU BFS operations, implementing persistent storage and comprehensive metrics tracking.
"""

import hashlib
import logging
import os
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Callable

try:
    import jax
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    nnx = None  # type: ignore

from .tpu_backend import TPUBackend


@dataclass
class KernelSignature:
    """Unique signature for TPU kernel identification and caching."""

    # Core graph characteristics
    state_size: int
    generator_count: int
    generator_hash: str  # Hash of generator permutations

    # Operation characteristics
    operation_type: str  # 'bfs_step', 'expand_layer', 'hash_batch', etc.
    batch_size: Optional[int] = None
    max_states: Optional[int] = None

    # TPU-specific parameters
    dtype: str = "int64"
    use_systolic_array: bool = True
    memory_layout: str = "optimized"

    # Compilation parameters
    jit_options: Optional[Dict[str, Any]] = None
    backend_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure consistent signature generation."""
        if self.jit_options is None:
            self.jit_options = {}
        if self.backend_config is None:
            self.backend_config = {}

    def to_cache_key(self) -> str:
        """Generate unique cache key from signature."""
        # Create deterministic hash from all signature components
        signature_dict = asdict(self)

        # Sort dictionary for consistent hashing
        sorted_items = sorted(signature_dict.items())
        signature_str = str(sorted_items)

        # Generate SHA-256 hash for unique identification
        return hashlib.sha256(signature_str.encode()).hexdigest()

    @classmethod
    def from_graph_and_operation(
        cls, graph, operation_type: str, batch_size: Optional[int] = None, **kwargs
    ) -> "KernelSignature":
        """Create signature from graph and operation parameters."""
        # Handle None graph (for graph-independent operations)
        if graph is None:
            return cls(
                state_size=kwargs.get("state_size", 4),
                generator_count=0,
                generator_hash="none",
                operation_type=operation_type,
                batch_size=batch_size,
                **{k: v for k, v in kwargs.items() if k != "state_size"},
            )

        # Handle both CayleyGraph and CayleyGraphDef
        if hasattr(graph, "definition"):
            graph_def = graph.definition
            central_state = graph.central_state
        elif hasattr(graph, "generators_permutations"):
            graph_def = graph
            central_state = getattr(graph, "central_state", [0, 1, 2, 3])
        else:
            raise ValueError(f"Invalid graph object: {type(graph)}")

        # Generate hash of generator permutations for uniqueness
        generators_str = str(graph_def.generators_permutations)
        generator_hash = hashlib.md5(generators_str.encode()).hexdigest()

        return cls(
            state_size=len(central_state),
            generator_count=len(graph_def.generators_permutations),
            generator_hash=generator_hash,
            operation_type=operation_type,
            batch_size=batch_size,
            **kwargs,
        )

    def __str__(self) -> str:
        """Human-readable signature representation."""
        return (
            f"KernelSignature(op={self.operation_type}, "
            f"state_size={self.state_size}, "
            f"gen_count={self.generator_count}, "
            f"batch_size={self.batch_size})"
        )


@dataclass
class CachedKernel:
    """Container for cached compiled kernel with metadata."""

    compiled_function: Callable
    signature: KernelSignature
    compilation_time: float
    cache_timestamp: float
    hit_count: int = 0
    last_used: float = 0.0

    def __post_init__(self):
        """Initialize timestamps."""
        if self.last_used == 0.0:
            self.last_used = time.time()

    def record_hit(self):
        """Record cache hit and update usage statistics."""
        self.hit_count += 1
        self.last_used = time.time()

    def get_age_seconds(self) -> float:
        """Get age of cached kernel in seconds."""
        return time.time() - self.cache_timestamp


class TPUKernelCache(nnx.Module):
    """Persistent TPU kernel cache with comprehensive metrics and monitoring."""

    def __init__(
        self,
        tpu_backend: TPUBackend,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 1000,
        enable_persistence: bool = True,
        rngs: Optional[nnx.Rngs] = None,
    ):
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for TPU kernel cache")

        self.backend = tpu_backend
        self.max_cache_size = max_cache_size
        self.enable_persistence = enable_persistence

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cayleypy/tpu_kernel_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.memory_cache: nnx.Variable[Dict[str, CachedKernel]] = nnx.Variable({})

        # Cache metrics
        self.metrics = nnx.Variable(
            {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "compilations": 0,
                "total_compilation_time": 0.0,
                "average_compilation_time": 0.0,
                "cache_size": 0,
                "persistent_loads": 0,
                "persistent_saves": 0,
                "evictions": 0,
                "hit_rate": 0.0,
                "compilation_time_saved": 0.0,
                "memory_usage_mb": 0.0,
            }
        )

        # Performance tracking
        self.operation_stats: nnx.Variable[Dict[str, Dict[str, Any]]] = nnx.Variable({})

        # Initialize RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(42)
        self.rngs = rngs

        self.logger = logging.getLogger(__name__)

        # Load persistent cache on initialization
        if self.enable_persistence:
            self._load_persistent_cache()

        self.logger.info(
            "TPU Kernel Cache initialized: cache_dir=%s, max_size=%d, persistence=%s",
            self.cache_dir,
            self.max_cache_size,
            self.enable_persistence,
        )

    def get_or_compile_kernel(
        self, kernel_signature: KernelSignature, compile_fn: Callable[[], Callable], force_recompile: bool = False
    ) -> Callable:
        """Get cached kernel or compile new one with automatic caching."""
        self.metrics.value["total_requests"] += 1

        cache_key = kernel_signature.to_cache_key()

        # Check memory cache first (unless forced recompilation)
        if not force_recompile and cache_key in self.memory_cache.value:
            cached_kernel = self.memory_cache.value[cache_key]
            cached_kernel.record_hit()

            self.metrics.value["cache_hits"] += 1
            self._update_hit_rate()

            # Estimate compilation time saved
            avg_compilation_time = self.metrics.value.get("average_compilation_time", 1.0)
            self.metrics.value["compilation_time_saved"] += avg_compilation_time

            self.logger.debug("Cache hit for %s (hits: %d)", kernel_signature, cached_kernel.hit_count)
            return cached_kernel.compiled_function

        # Cache miss - need to compile
        self.metrics.value["cache_misses"] += 1
        self._update_hit_rate()

        self.logger.info("Cache miss for %s - compiling kernel", kernel_signature)

        # Compile the kernel with timing
        compile_start_time = time.time()
        try:
            compiled_function = compile_fn()
        except Exception as e:
            self.logger.error("Kernel compilation failed for %s: %s", kernel_signature, e)
            raise

        compilation_time = time.time() - compile_start_time

        # Create cached kernel
        cached_kernel = CachedKernel(
            compiled_function=compiled_function,
            signature=kernel_signature,
            compilation_time=compilation_time,
            cache_timestamp=time.time(),
        )

        # Add to memory cache
        self._add_to_cache(cache_key, cached_kernel)

        # Update metrics
        self.metrics.value["compilations"] += 1
        self.metrics.value["total_compilation_time"] += compilation_time
        self.metrics.value["average_compilation_time"] = (
            self.metrics.value["total_compilation_time"] / self.metrics.value["compilations"]
        )

        # Track operation-specific statistics
        op_type = kernel_signature.operation_type
        if op_type not in self.operation_stats.value:
            self.operation_stats.value[op_type] = {
                "count": 0,
                "total_compilation_time": 0.0,
                "average_compilation_time": 0.0,
            }

        op_stats = self.operation_stats.value[op_type]
        op_stats["count"] += 1
        op_stats["total_compilation_time"] += compilation_time
        op_stats["average_compilation_time"] = op_stats["total_compilation_time"] / op_stats["count"]

        # Save to persistent cache
        if self.enable_persistence:
            self._save_to_persistent_cache(cache_key, cached_kernel)

        self.logger.info("Kernel compiled and cached: %s (%.3fs)", kernel_signature, compilation_time)

        return compiled_function

    def _add_to_cache(self, cache_key: str, cached_kernel: CachedKernel):
        """Add kernel to memory cache with eviction if necessary."""
        # Check if cache is full
        if len(self.memory_cache.value) >= self.max_cache_size:
            self._evict_oldest_kernel()

        self.memory_cache.value[cache_key] = cached_kernel
        self.metrics.value["cache_size"] = len(self.memory_cache.value)

        # Update memory usage estimate
        self._update_memory_usage()

    def _evict_oldest_kernel(self):
        """Evict least recently used kernel from cache."""
        if not self.memory_cache.value:
            return

        # Find kernel with oldest last_used timestamp
        oldest_key = min(self.memory_cache.value.keys(), key=lambda k: self.memory_cache.value[k].last_used)

        evicted_kernel = self.memory_cache.value.pop(oldest_key)
        self.metrics.value["evictions"] += 1
        self.metrics.value["cache_size"] = len(self.memory_cache.value)

        self.logger.debug(
            "Evicted kernel: %s (age: %.1fs, hits: %d)",
            evicted_kernel.signature,
            evicted_kernel.get_age_seconds(),
            evicted_kernel.hit_count,
        )

    def _update_hit_rate(self):
        """Update cache hit rate metric."""
        total = self.metrics.value["total_requests"]
        hits = self.metrics.value["cache_hits"]
        self.metrics.value["hit_rate"] = hits / max(1, total)

    def _update_memory_usage(self):
        """Update estimated memory usage."""
        # Rough estimate: each cached kernel uses ~1MB
        estimated_mb = len(self.memory_cache.value) * 1.0
        self.metrics.value["memory_usage_mb"] = estimated_mb

    def _get_persistent_cache_path(self, cache_key: str) -> Path:
        """Get path for persistent cache file."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _save_to_persistent_cache(self, cache_key: str, cached_kernel: CachedKernel):
        """Save kernel to persistent cache."""
        try:
            cache_path = self._get_persistent_cache_path(cache_key)

            # Create cache data (exclude the compiled function for serialization)
            cache_data = {
                "signature": cached_kernel.signature,
                "compilation_time": cached_kernel.compilation_time,
                "cache_timestamp": cached_kernel.cache_timestamp,
                "hit_count": cached_kernel.hit_count,
                "last_used": cached_kernel.last_used,
            }

            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)

            self.metrics.value["persistent_saves"] += 1
            self.logger.debug("Saved kernel to persistent cache: %s", cache_key[:8])

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.warning("Failed to save kernel to persistent cache: %s", e)

    def _load_persistent_cache(self):
        """Load kernels from persistent cache."""
        if not self.cache_dir.exists():
            return

        loaded_count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    _ = pickle.load(f)  # Load for validation but don't use

                # Note: We can't restore the compiled function from persistent cache
                # This is just for metadata and signature tracking
                # The kernel will need to be recompiled on first use

                self.logger.debug("Found persistent cache entry: %s", cache_file.stem[:8])
                loaded_count += 1

            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.warning("Failed to load cache file %s: %s", cache_file, e)
                # Remove corrupted cache file
                try:
                    cache_file.unlink()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

        if loaded_count > 0:
            self.metrics.value["persistent_loads"] = loaded_count
            self.logger.info("Loaded %d entries from persistent cache", loaded_count)

    def clear_cache(self, clear_persistent: bool = False):
        """Clear memory cache and optionally persistent cache."""
        self.memory_cache.value.clear()
        self.metrics.value["cache_size"] = 0

        if clear_persistent and self.enable_persistence:
            try:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                self.logger.info("Cleared persistent cache")
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.warning("Failed to clear persistent cache: %s", e)

        self.logger.info("Cache cleared (persistent: %s)", clear_persistent)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "metrics": dict(self.metrics.value),
            "operation_stats": dict(self.operation_stats.value),
            "cache_entries": [
                {
                    "signature": str(kernel.signature),
                    "hit_count": kernel.hit_count,
                    "age_seconds": kernel.get_age_seconds(),
                    "compilation_time": kernel.compilation_time,
                }
                for kernel in self.memory_cache.value.values()
            ],
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        metrics = self.metrics.value

        return {
            "hit_rate": metrics["hit_rate"],
            "total_requests": metrics["total_requests"],
            "compilation_time_saved": metrics["compilation_time_saved"],
            "average_compilation_time": metrics["average_compilation_time"],
            "cache_size": metrics["cache_size"],
            "memory_usage_mb": metrics["memory_usage_mb"],
            "top_operations": self._get_top_operations(),
        }

    def _get_top_operations(self) -> Dict[str, Any]:
        """Get statistics for most frequently cached operations."""
        op_stats = self.operation_stats.value

        # Sort by compilation count
        sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["count"], reverse=True)

        return dict(sorted_ops[:5])  # Top 5 operations

    def optimize_cache_size(self, target_hit_rate: float = 0.8) -> int:
        """Optimize cache size based on hit rate target."""
        current_hit_rate = self.metrics.value["hit_rate"]
        current_size = self.max_cache_size

        if current_hit_rate < target_hit_rate:
            # Increase cache size
            new_size = min(current_size * 2, 2000)  # Cap at 2000
        elif current_hit_rate > 0.95 and current_size > 100:
            # Decrease cache size if hit rate is very high
            new_size = max(current_size // 2, 100)  # Minimum 100
        else:
            new_size = current_size

        if new_size != current_size:
            self.max_cache_size = new_size
            self.logger.info(
                "Optimized cache size: %d -> %d (hit_rate: %.3f)", current_size, new_size, current_hit_rate
            )

        return new_size

    def reset_metrics(self):
        """Reset all cache metrics."""
        for key in self.metrics.value:
            if isinstance(self.metrics.value[key], (int, float)):
                self.metrics.value[key] = 0

        self.operation_stats.value.clear()
        self.logger.info("Cache metrics reset")


def create_kernel_signature(graph, operation_type: str, batch_size: Optional[int] = None, **kwargs) -> KernelSignature:
    """Factory function to create kernel signature."""
    return KernelSignature.from_graph_and_operation(graph, operation_type, batch_size, **kwargs)


def create_tpu_kernel_cache(
    tpu_backend: Optional[TPUBackend] = None,
    cache_dir: Optional[str] = None,
    max_cache_size: int = 1000,
    enable_persistence: bool = True,
) -> TPUKernelCache:
    """Factory function to create TPU kernel cache."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX and Flax are required for TPU kernel cache. " + "Install with: pip install 'cayleypy[jax-tpu]'"
        )

    if tpu_backend is None:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel

        tpu_backend = get_tpu_backend()

    return TPUKernelCache(
        backend=tpu_backend, cache_dir=cache_dir, max_cache_size=max_cache_size, enable_persistence=enable_persistence
    )


if __name__ == "__main__":
    # Test kernel caching system
    print("Testing TPU Kernel Caching System")
    print("=" * 40)

    if not JAX_AVAILABLE:
        print("JAX not available - cannot test kernel cache")
    else:
        try:
            from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel

            # Test with TPU backend
            test_backend = get_tpu_backend()
            print(f"TPU Available: {test_backend.is_available}")

            if test_backend.is_available:
                # Create simple mock graph for testing
                class MockGraph:
                    """Mock graph for testing."""

                    def __init__(self):
                        self.central_state = [0, 1, 2, 3]
                        self.definition = self
                        self.generators_permutations = [[1, 0, 2, 3], [0, 2, 1, 3]]

                test_graph = MockGraph()

                # Create kernel cache
                cache = create_tpu_kernel_cache(test_backend, max_cache_size=10)

                # Create test signature
                test_signature = create_kernel_signature(test_graph, "test_operation", batch_size=100)
                print(f"Test signature: {test_signature}")
                print(f"Cache key: {test_signature.to_cache_key()[:16]}...")

                # Test compilation and caching
                def dummy_compile_fn():
                    @jax.jit
                    def dummy_kernel(x):
                        return x + 1

                    return dummy_kernel

                # First call - should compile
                print("\nFirst call (should compile):")
                first_start_time = time.time()
                kernel1 = cache.get_or_compile_kernel(test_signature, dummy_compile_fn)
                time1 = time.time() - first_start_time
                print("Time: {:.4f}s".format(time1))

                # Second call - should hit cache
                print("\nSecond call (should hit cache):")
                second_start_time = time.time()
                kernel2 = cache.get_or_compile_kernel(test_signature, dummy_compile_fn)
                time2 = time.time() - second_start_time
                print(f"Time: {time2:.4f}s")
                print(f"Speedup: {time1/time2:.1f}x")

                # Verify same kernel returned
                print(f"Same kernel returned: {kernel1 is kernel2}")

                # Get cache statistics
                stats = cache.get_cache_stats()
                print("\nCache Statistics:")
                print(f"Hit rate: {stats['metrics']['hit_rate']:.3f}")
                print(f"Total requests: {stats['metrics']['total_requests']}")
                print(f"Cache hits: {stats['metrics']['cache_hits']}")
                print(f"Cache misses: {stats['metrics']['cache_misses']}")
                print(f"Compilation time saved: {stats['metrics']['compilation_time_saved']:.4f}s")

                print("\n✓ TPU Kernel Cache test completed successfully!")
            else:
                print("TPU not available for testing")

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"✗ TPU Kernel Cache test failed: {e}")
