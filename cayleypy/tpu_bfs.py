"""TPU BFS Implementation for CayleyPy.

This module provides TPU-accelerated breadth-first search with native int64 support,
optimized for TPU v6e (Trillium) architecture with NNX state management.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

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


# JIT-compiled helper functions for TPU BFS operations
@jax.jit
def _expand_layer_jit(current_layer: jnp.ndarray, generators: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled layer expansion using TPU v6e's systolic array."""
    def apply_all_generators(state):
        # Use TPU's vectorization for generator application
        return jax.vmap(lambda gen: state[gen])(generators)
    
    # Apply generators to all states in current layer
    # Leverage TPU v6e's 256x256 systolic array
    expanded = jax.vmap(apply_all_generators)(current_layer)
    
    # Flatten to get all new states
    return expanded.reshape(-1, current_layer.shape[-1])


@jax.jit
def _filter_visited_states_jit(new_hashes: jnp.ndarray, visited_hashes: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled filtering of already visited states."""
    # Use TPU-optimized membership testing
    is_visited = jax.vmap(lambda h: jnp.any(visited_hashes == h))(new_hashes)
    return ~is_visited


class TPUBFSModule(nnx.Module):
    """NNX module for TPU-accelerated breadth-first search with int64 support."""

    def __init__(self, graph, backend: TPUBackend, rngs: nnx.Rngs):
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for TPU BFS")

        self.graph = graph
        self.backend = backend
        
        # Handle both CayleyGraph and CayleyGraphDef
        if hasattr(graph, 'definition'):
            # CayleyGraph object
            graph_def = graph.definition
            central_state = graph.central_state
        else:
            # CayleyGraphDef object
            graph_def = graph
            central_state = graph.central_state
        
        # Store generators as NNX parameters for optimization
        self.generators = nnx.Param(
            jnp.array(graph_def.generators_permutations, dtype=jnp.int64)
        )
        
        # Initialize hasher and tensor ops
        state_size = len(central_state)
        self.hasher = TPUHasherModule(state_size, backend, rngs)
        self.tensor_ops = TPUTensorOpsModule(backend, rngs)
        
        # BFS state tracking
        self.bfs_state = nnx.Variable({
            'current_layer': None,
            'visited_hashes': None,
            'layer_sizes': [],
            'diameter': 0,
            'total_states_found': 0,
            'is_complete': False
        })
        
        # Performance metrics
        self.metrics = nnx.Variable({
            'states_processed': 0,
            'hash_operations': 0,
            'memory_peak_mb': 0.0,
            'tpu_utilization': 0.0,
            'layer_expansions': 0,
            'deduplication_operations': 0,
            'visited_checks': 0,
            'systolic_array_operations': 0
        })
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("TPU BFS Module initialized for graph with %d generators", 
                        len(graph_def.generators_permutations))

    def expand_layer(self, current_layer: jnp.ndarray) -> jnp.ndarray:
        """Expand current layer by applying all generators using TPU v6e."""
        # Use JIT-compiled expansion
        expanded_states = _expand_layer_jit(current_layer, self.generators.value)
        
        # Update metrics
        self.metrics.value['layer_expansions'] += 1
        self.metrics.value['systolic_array_operations'] += 1
        self.metrics.value['states_processed'] += len(expanded_states)
        
        return expanded_states

    def get_neighbors(self, states: jnp.ndarray) -> jnp.ndarray:
        """Get all neighbors of given states by applying all generators."""
        # Apply all generators to all states
        def apply_generator(gen_idx):
            return jax.vmap(lambda state: state[self.generators.value[gen_idx]])(states)
        
        # Get neighbors for each generator
        all_neighbors = []
        for i in range(len(self.generators.value)):
            neighbors = apply_generator(i)
            all_neighbors.append(neighbors)
        
        # Concatenate all neighbors
        return jnp.concatenate(all_neighbors, axis=0)

    def get_unique_states(self, states: jnp.ndarray, hashes: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Remove duplicates from states and return unique states with their hashes."""
        if hashes is None:
            hashes = self.hasher.hash_batch(states)
        
        # Get unique hashes and corresponding indices
        unique_hashes, unique_indices = self.tensor_ops.unique_with_indices(hashes)
        
        # Filter out fill values
        fill_value = jnp.iinfo(hashes.dtype).min
        valid_mask = unique_hashes != fill_value
        valid_indices = unique_indices[valid_mask]
        valid_hashes = unique_hashes[valid_mask]
        
        if len(valid_indices) > 0:
            unique_states = states[valid_indices]
        else:
            unique_states = jnp.array([], dtype=jnp.int64).reshape(0, states.shape[1])
            valid_hashes = jnp.array([], dtype=jnp.int64)
        
        return unique_states, valid_hashes

    def bfs_step(self, current_layer: jnp.ndarray, 
                 seen_states_hashes: List[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single BFS step following the reference implementation."""
        # 1. Get all neighbors of current layer
        neighbors = self.get_neighbors(current_layer)
        self.metrics.value['states_processed'] += len(neighbors)
        
        # 2. Get unique neighbors (remove duplicates within this expansion)
        unique_neighbors, neighbor_hashes = self.get_unique_states(neighbors)
        self.metrics.value['deduplication_operations'] += 1
        self.metrics.value['hash_operations'] += len(neighbors)
        
        # 3. Remove states that have been seen in previous layers
        is_new = jnp.ones(len(neighbor_hashes), dtype=bool)
        
        for seen_hashes in seen_states_hashes:
            if len(seen_hashes) > 0:
                is_not_in_seen = ~self.tensor_ops.isin(neighbor_hashes, seen_hashes)
                is_new = is_new & is_not_in_seen
        
        # 4. Filter to get truly new states
        new_states = unique_neighbors[is_new]
        new_hashes = neighbor_hashes[is_new]
        
        self.metrics.value['visited_checks'] += 1
        self.metrics.value['tpu_utilization'] += 1.0
        
        return new_states, new_hashes

    def initialize_bfs(self):
        """Initialize BFS state with proper int64 support."""
        # Convert central state to int64 array
        if hasattr(self.graph, 'definition'):
            central_state = self.graph.central_state
        else:
            central_state = self.graph.central_state
        start_state = jnp.array([central_state], dtype=jnp.int64)
        start_hash = self.hasher.hash_batch(start_state)
        
        self.bfs_state.value.update({
            'current_layer': start_state,
            'visited_hashes': start_hash,
            'layer_sizes': [1],
            'diameter': 0,
            'total_states_found': 1,
            'is_complete': False
        })
        
        # Reset metrics
        for key in self.metrics.value:
            if isinstance(self.metrics.value[key], (int, float)):
                self.metrics.value[key] = 0
        
        self.logger.info("BFS initialized with start state: %s", central_state)

    def run_bfs(self, max_diameter: int = 1000) -> List[int]:
        """Run complete BFS leveraging TPU v6e's capabilities."""
        self.initialize_bfs()
        
        self.logger.info("Starting TPU BFS with max_diameter=%d", max_diameter)
        
        # Initialize seen states list with the starting layer
        current_layer = self.bfs_state.value['current_layer']
        current_layer_hashes = self.bfs_state.value['visited_hashes']
        seen_states_hashes = [current_layer_hashes]
        
        for diameter in range(max_diameter):
            # Perform BFS step with all previously seen states
            new_layer, new_layer_hashes = self.bfs_step(current_layer, seen_states_hashes)
            
            # Check if we found any new states
            if len(new_layer) == 0:
                self.logger.info("BFS completed at diameter %d - no new states found", diameter)
                self.bfs_state.value['is_complete'] = True
                break
                
            # Add new layer hashes to seen states
            seen_states_hashes.append(new_layer_hashes)
            
            # Update state
            self.bfs_state.value.update({
                'current_layer': new_layer,
                'visited_hashes': jnp.concatenate(seen_states_hashes),  # All seen hashes
                'diameter': diameter + 1,
                'total_states_found': self.bfs_state.value['total_states_found'] + len(new_layer)
            })
            
            self.bfs_state.value['layer_sizes'].append(len(new_layer))
            
            self.logger.debug("Diameter %d: found %d new states (total: %d)", 
                            diameter + 1, len(new_layer), 
                            self.bfs_state.value['total_states_found'])
            
            # Update current layer for next iteration
            current_layer = new_layer
        
        final_diameter = self.bfs_state.value['diameter']
        total_states = self.bfs_state.value['total_states_found']
        
        self.logger.info("BFS completed: diameter=%d, total_states=%d", 
                        final_diameter, total_states)
        
        return self.bfs_state.value['layer_sizes']

    def get_bfs_result(self) -> Dict[str, Any]:
        """Get comprehensive BFS results."""
        return {
            'layer_sizes': self.bfs_state.value['layer_sizes'],
            'diameter': self.bfs_state.value['diameter'],
            'total_states_found': self.bfs_state.value['total_states_found'],
            'is_complete': self.bfs_state.value['is_complete'],
            'growth_function': self.bfs_state.value['layer_sizes']
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get BFS performance metrics."""
        base_metrics = dict(self.metrics.value)
        
        # Add hasher and tensor ops metrics
        hasher_stats = self.hasher.get_hash_stats()
        tensor_ops_stats = self.tensor_ops.get_performance_metrics()
        
        return {
            **base_metrics,
            'hasher_stats': hasher_stats,
            'tensor_ops_stats': tensor_ops_stats,
            'backend_info': self.backend.get_device_info(),
            'memory_usage': self.backend.get_memory_usage()
        }

    def reset_metrics(self):
        """Reset all performance metrics."""
        for key in self.metrics.value:
            if isinstance(self.metrics.value[key], (int, float)):
                self.metrics.value[key] = 0
        
        self.hasher.reset_metrics()
        self.tensor_ops.reset_metrics()
        
        self.logger.info("All metrics reset")

    def get_current_layer_info(self) -> Dict[str, Any]:
        """Get information about the current BFS layer."""
        current_layer = self.bfs_state.value.get('current_layer')
        if current_layer is None:
            return {'layer_exists': False}
        
        return {
            'layer_exists': True,
            'layer_size': len(current_layer),
            'current_diameter': self.bfs_state.value['diameter'],
            'total_states_found': self.bfs_state.value['total_states_found'],
            'is_complete': self.bfs_state.value['is_complete']
        }

    def verify_int64_precision(self) -> bool:
        """Verify that BFS operations maintain int64 precision."""
        try:
            # Create test states with large int64 values
            large_state = jnp.array([[2**40, 2**50, 2**60]], dtype=jnp.int64)
            
            # Test expansion
            if len(self.generators.value) > 0:
                expanded = self.expand_layer(large_state)
                
                # Verify expanded states maintain int64 precision
                if expanded.dtype != jnp.int64:
                    return False
                
                # Check that values are still large (indicating no precision loss)
                max_val = jnp.max(expanded)
                if int(max_val) < 2**32:  # Should be much larger than int32 range
                    return False
            
            # Test hashing with large values
            hash_result = self.hasher.hash_batch(large_state)
            if hash_result.dtype != jnp.int64:
                return False
            
            self.logger.info("int64 precision verification passed")
            return True
            
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("int64 precision verification failed: %s", e)
            return False


def _fallback_to_cpu_bfs(graph, max_diameter: int) -> List[int]:
    """Fallback to CPU BFS implementation."""
    # Handle both CayleyGraph and CayleyGraphDef
    if hasattr(graph, 'definition'):
        # Already a CayleyGraph object
        cayley_graph = graph
    else:
        # CayleyGraphDef - need to create CayleyGraph
        from .cayley_graph import CayleyGraph  # pylint: disable=import-outside-toplevel
        cayley_graph = CayleyGraph(graph)
    
    from .bfs_numpy import bfs_numpy  # pylint: disable=import-outside-toplevel
    return bfs_numpy(cayley_graph, max_diameter)


def tpu_bfs(graph, max_diameter: int = 1000) -> List[int]:
    """High-level TPU BFS function with automatic backend detection."""
    if not JAX_AVAILABLE:
        # Fallback to CPU implementation
        return _fallback_to_cpu_bfs(graph, max_diameter)
    
    try:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel
        
        backend = get_tpu_backend()
        
        if not backend.is_available:
            # Fallback to CPU implementation
            return _fallback_to_cpu_bfs(graph, max_diameter)
        
        # Use TPU BFS with native int64 support
        rngs = nnx.Rngs(42)
        bfs_module = TPUBFSModule(graph, backend, rngs)
        return bfs_module.run_bfs(max_diameter)
        
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).warning("TPU BFS failed, falling back to CPU: %s", e)
        # Fallback to CPU implementation
        return _fallback_to_cpu_bfs(graph, max_diameter)


def create_tpu_bfs(graph, backend: Optional[TPUBackend] = None) -> TPUBFSModule:
    """Factory function to create TPU BFS module with error handling."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX and Flax are required for TPU BFS. " +
            "Install with: pip install 'cayleypy[jax-tpu]'"
        )
    
    if backend is None:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel
        backend = get_tpu_backend()
    
    rngs = nnx.Rngs(42)
    return TPUBFSModule(graph, backend, rngs)


def benchmark_tpu_vs_cpu_bfs(graph, max_diameter: int = 100) -> Dict[str, Any]:
    """Benchmark TPU vs CPU BFS performance."""
    import time  # pylint: disable=import-outside-toplevel
    
    # Handle both CayleyGraph and CayleyGraphDef
    if hasattr(graph, 'definition'):
        graph_def = graph.definition
        central_state = graph.central_state
    else:
        graph_def = graph
        central_state = graph.central_state
    
    results = {
        'graph_info': {
            'generators_count': len(graph_def.generators_permutations),
            'state_size': len(central_state)
        },
        'tpu_available': False,
        'cpu_time': 0.0,
        'tpu_time': 0.0,
        'speedup': 0.0,
        'cpu_result': [],
        'tpu_result': [],
        'results_match': False
    }
    
    # CPU BFS
    try:
        start_time = time.time()
        cpu_result = _fallback_to_cpu_bfs(graph, max_diameter)
        cpu_time = time.time() - start_time
        
        results['cpu_time'] = cpu_time
        results['cpu_result'] = cpu_result
        
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).error("CPU BFS failed: %s", e)
        return results
    
    # TPU BFS
    if JAX_AVAILABLE:
        try:
            from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel
            
            backend = get_tpu_backend()
            if backend.is_available:
                results['tpu_available'] = True
                
                start_time = time.time()
                tpu_result = tpu_bfs(graph, max_diameter)
                tpu_time = time.time() - start_time
                
                results['tpu_time'] = tpu_time
                results['tpu_result'] = tpu_result
                results['speedup'] = cpu_time / tpu_time if tpu_time > 0 else 0.0
                results['results_match'] = cpu_result == tpu_result
                
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.getLogger(__name__).error("TPU BFS benchmark failed: %s", e)
    
    return results


if __name__ == "__main__":
    # Test TPU BFS when run as script
    print("Testing TPU BFS Implementation")
    print("=" * 40)
    
    if not JAX_AVAILABLE:
        print("JAX not available - cannot test TPU BFS")
    else:
        try:
            from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel
            from .graphs_lib import PermutationGroups  # pylint: disable=import-outside-toplevel
            
            # Test with a small symmetric group
            backend = get_tpu_backend()
            print(f"TPU Available: {backend.is_available}")
            
            if backend.is_available:
                # Create a small test graph
                test_graph = PermutationGroups.symmetric_group(4)
                
                # Create BFS module
                bfs_module = create_tpu_bfs(test_graph, backend)
                
                # Verify int64 precision
                precision_ok = bfs_module.verify_int64_precision()
                print(f"int64 Precision Test: {'PASSED ✓' if precision_ok else 'FAILED ✗'}")
                
                # Run BFS
                print("Running TPU BFS on S4...")
                result = bfs_module.run_bfs(max_diameter=10)
                print(f"Growth function: {result}")
                
                # Get performance metrics
                metrics = bfs_module.get_performance_metrics()
                print(f"States processed: {metrics['states_processed']}")
                print(f"Hash operations: {metrics['hash_operations']}")
                print(f"TPU utilization: {metrics['tpu_utilization']}")
                
                print("✓ TPU BFS test completed successfully!")
            else:
                print("TPU not available for testing")
                
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"✗ TPU BFS test failed: {e}")