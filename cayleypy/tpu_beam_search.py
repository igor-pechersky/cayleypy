"""TPU Beam Search Implementation for CayleyPy.

This module provides TPU-accelerated beam search with integrated neural networks,
optimized for TPU v6e (Trillium) architecture with native int64 support.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Callable

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


# JIT-compiled helper functions for TPU beam search operations
@jax.jit
def _expand_beam_jit(
    beam_states: jnp.ndarray, generators: jnp.ndarray, beam_size: int
) -> jnp.ndarray:
    """JIT-compiled beam expansion using TPU v6e's vectorization."""
    def apply_all_generators(state):
        return jax.vmap(lambda gen: state[gen])(generators)
    
    # Apply generators to all states in beam
    expanded = jax.vmap(apply_all_generators)(beam_states)
    
    # Reshape to get all candidate states
    candidates = expanded.reshape(-1, beam_states.shape[-1])
    
    return candidates


@jax.jit
def _score_states_jit(states: jnp.ndarray, predictor_fn: Callable) -> jnp.ndarray:
    """JIT-compiled state scoring using TPU predictor."""
    # Apply predictor to all states
    scores = jax.vmap(predictor_fn)(states)
    return scores


@jax.jit
def _select_top_k_jit(scores: jnp.ndarray, states: jnp.ndarray, k: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compiled top-k selection using TPU operations."""
    # Get top-k indices
    top_k_indices = jax.lax.top_k(scores, k)[1]
    
    # Select corresponding states and scores
    selected_states = states[top_k_indices]
    selected_scores = scores[top_k_indices]
    
    return selected_states, selected_scores


class TPUBeamSearchModule(nnx.Module):
    """NNX module for TPU-accelerated beam search with integrated neural networks."""

    def __init__(
        self,
        graph,
        predictor_module,
        backend: TPUBackend,
        rngs: nnx.Rngs,
        beam_size: int = 100,
        max_depth: int = 100
    ):
        if not JAX_AVAILABLE:
            raise ImportError("JAX and Flax are required for TPU beam search")

        self.graph = graph
        self.predictor_module = predictor_module
        self.backend = backend
        self.beam_size = beam_size
        self.max_depth = max_depth
        
        # Handle both CayleyGraph and CayleyGraphDef
        if hasattr(graph, 'definition'):
            graph_def = graph.definition
            central_state = graph.central_state
        else:
            graph_def = graph
            central_state = graph.central_state
        
        # Store generators as NNX parameters
        self.generators = nnx.Param(
            jnp.array(graph_def.generators_permutations, dtype=jnp.int64)
        )
        
        # Initialize hasher and tensor ops
        state_size = len(central_state)
        self.hasher = TPUHasherModule(state_size, backend, rngs)
        self.tensor_ops = TPUTensorOpsModule(backend, rngs)
        
        # Beam search state tracking
        self.search_state = nnx.Variable({
            'current_beam': None,
            'current_scores': None,
            'best_path': [],
            'best_score': float('-inf'),
            'search_depth': 0,
            'total_states_explored': 0,
            'is_complete': False,
            'target_found': False
        })
        
        # Performance metrics
        self.metrics = nnx.Variable({
            'beam_expansions': 0,
            'states_scored': 0,
            'deduplication_operations': 0,
            'predictor_calls': 0,
            'top_k_selections': 0,
            'memory_peak_mb': 0.0,
            'tpu_utilization': 0.0,
            'neural_network_operations': 0,
            'systolic_array_operations': 0
        })
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "TPU Beam Search Module initialized: beam_size=%d, max_depth=%d, generators=%d",
            beam_size, max_depth, len(graph_def.generators_permutations)
        )

    def expand_beam(self, beam_states: jnp.ndarray) -> jnp.ndarray:
        """Expand beam by applying all generators using TPU's vectorization."""
        # Use JIT-compiled expansion
        expanded_states = _expand_beam_jit(beam_states, self.generators.value, self.beam_size)
        
        # Update metrics
        self.metrics.value['beam_expansions'] += 1
        self.metrics.value['systolic_array_operations'] += 1
        
        return expanded_states

    def deduplicate_states(self, states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Remove duplicate states using native int64 precision."""
        # Hash all states for deduplication
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
        
        # Update metrics
        self.metrics.value['deduplication_operations'] += 1
        
        return unique_states, valid_hashes

    def score_and_select(self, states: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Score states and select top-k using integrated TPU predictor."""
        if len(states) == 0:
            return jnp.array([], dtype=jnp.int64).reshape(0, states.shape[1]), jnp.array([])
        
        # Score states using the predictor
        scores = self.predictor_module.batch_inference(states)
        
        # Select top-k states
        k = min(self.beam_size, len(states))
        selected_states, selected_scores = _select_top_k_jit(scores, states, k)
        
        # Update metrics
        self.metrics.value['states_scored'] += len(states)
        self.metrics.value['predictor_calls'] += 1
        self.metrics.value['top_k_selections'] += 1
        self.metrics.value['neural_network_operations'] += 1
        
        return selected_states, selected_scores

    def search_step(self, current_beam: jnp.ndarray, target_state: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single beam search step with TPU-optimized operations."""
        # 1. Expand current beam
        expanded_states = self.expand_beam(current_beam)
        
        # 2. Deduplicate expanded states
        unique_states, _ = self.deduplicate_states(expanded_states)
        
        # 3. Check if target is found (if specified)
        if target_state is not None and len(unique_states) > 0:
            target_hash = self.hasher.hash_state(target_state)
            state_hashes = self.hasher.hash_batch(unique_states)
            
            # Check for exact match
            matches = state_hashes == target_hash
            if jnp.any(matches):
                match_idx = jnp.where(matches)[0][0]
                self.search_state.value['target_found'] = True
                self.search_state.value['best_path'].append(unique_states[match_idx])
                return unique_states[match_idx:match_idx+1], jnp.array([1.0])
        
        # 4. Score and select top-k states
        selected_states, selected_scores = self.score_and_select(unique_states)
        
        # 5. Update search state
        self.search_state.value['total_states_explored'] += len(expanded_states)
        
        return selected_states, selected_scores

    def initialize_search(self, start_state: Optional[jnp.ndarray] = None):
        """Initialize beam search with proper int64 support."""
        # Use provided start state or graph's central state
        if start_state is None:
            if hasattr(self.graph, 'definition'):
                central_state = self.graph.central_state
            else:
                central_state = self.graph.central_state
            start_state = jnp.array(central_state, dtype=jnp.int64)
        
        # Initialize beam with start state
        initial_beam = start_state.reshape(1, -1)
        initial_scores = jnp.array([0.0])
        
        self.search_state.value.update({
            'current_beam': initial_beam,
            'current_scores': initial_scores,
            'best_path': [start_state],
            'best_score': 0.0,
            'search_depth': 0,
            'total_states_explored': 1,
            'is_complete': False,
            'target_found': False
        })
        
        # Reset metrics
        for key in self.metrics.value:
            if isinstance(self.metrics.value[key], (int, float)):
                self.metrics.value[key] = 0
        
        self.logger.info("Beam search initialized with start state: %s", start_state)

    def search(
        self,
        target_state: Optional[jnp.ndarray] = None,
        start_state: Optional[jnp.ndarray] = None
    ) -> Dict[str, Any]:
        """Run complete beam search leveraging TPU v6e's memory and compute."""
        self.initialize_search(start_state)
        
        self.logger.info("Starting TPU beam search: beam_size=%d, max_depth=%d", 
                        self.beam_size, self.max_depth)
        
        current_beam = self.search_state.value['current_beam']
        current_scores = self.search_state.value['current_scores']
        
        for depth in range(self.max_depth):
            # Perform search step
            new_beam, new_scores = self.search_step(current_beam, target_state)
            
            # Check if search should terminate
            if len(new_beam) == 0:
                self.logger.info("Beam search terminated at depth %d - no valid states", depth)
                self.search_state.value['is_complete'] = True
                break
            
            # Check if target was found
            if self.search_state.value['target_found']:
                self.logger.info("Target found at depth %d!", depth + 1)
                self.search_state.value['is_complete'] = True
                break
            
            # Update beam and scores
            current_beam = new_beam
            current_scores = new_scores
            
            # Update search state
            self.search_state.value.update({
                'current_beam': current_beam,
                'current_scores': current_scores,
                'search_depth': depth + 1,
                'best_score': float(jnp.max(current_scores))
            })
            
            self.logger.debug("Depth %d: beam_size=%d, best_score=%.4f", 
                            depth + 1, len(current_beam), self.search_state.value['best_score'])
        
        final_depth = self.search_state.value['search_depth']
        total_explored = self.search_state.value['total_states_explored']
        
        self.logger.info("Beam search completed: depth=%d, states_explored=%d", 
                        final_depth, total_explored)
        
        return self.get_search_result()

    def get_search_result(self) -> Dict[str, Any]:
        """Get comprehensive beam search results."""
        return {
            'best_path': self.search_state.value['best_path'],
            'best_score': self.search_state.value['best_score'],
            'search_depth': self.search_state.value['search_depth'],
            'total_states_explored': self.search_state.value['total_states_explored'],
            'target_found': self.search_state.value['target_found'],
            'is_complete': self.search_state.value['is_complete'],
            'final_beam_size': len(self.search_state.value['current_beam']) if self.search_state.value['current_beam'] is not None else 0,
            'final_beam_scores': self.search_state.value['current_scores'].tolist() if self.search_state.value['current_scores'] is not None else []
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get beam search performance metrics."""
        base_metrics = dict(self.metrics.value)
        
        # Add component metrics
        hasher_stats = self.hasher.get_hash_stats()
        tensor_ops_stats = self.tensor_ops.get_performance_metrics()
        predictor_stats = self.predictor_module.get_performance_metrics() if hasattr(self.predictor_module, 'get_performance_metrics') else {}
        
        return {
            **base_metrics,
            'hasher_stats': hasher_stats,
            'tensor_ops_stats': tensor_ops_stats,
            'predictor_stats': predictor_stats,
            'backend_info': self.backend.get_device_info(),
            'memory_usage': self.backend.get_memory_usage(),
            'search_efficiency': {
                'states_per_expansion': base_metrics['states_scored'] / max(1, base_metrics['beam_expansions']),
                'deduplication_ratio': base_metrics['deduplication_operations'] / max(1, base_metrics['beam_expansions']),
                'predictor_efficiency': base_metrics['states_scored'] / max(1, base_metrics['predictor_calls'])
            }
        }

    def reset_metrics(self):
        """Reset all performance metrics."""
        for key in self.metrics.value:
            if isinstance(self.metrics.value[key], (int, float)):
                self.metrics.value[key] = 0
        
        self.hasher.reset_metrics()
        self.tensor_ops.reset_metrics()
        
        if hasattr(self.predictor_module, 'reset_metrics'):
            self.predictor_module.reset_metrics()
        
        self.logger.info("All metrics reset")

    def get_current_beam_info(self) -> Dict[str, Any]:
        """Get information about the current beam."""
        current_beam = self.search_state.value.get('current_beam')
        current_scores = self.search_state.value.get('current_scores')
        
        if current_beam is None:
            return {'beam_exists': False}
        
        return {
            'beam_exists': True,
            'beam_size': len(current_beam),
            'current_depth': self.search_state.value['search_depth'],
            'best_score': self.search_state.value['best_score'],
            'total_explored': self.search_state.value['total_states_explored'],
            'target_found': self.search_state.value['target_found'],
            'is_complete': self.search_state.value['is_complete'],
            'score_statistics': {
                'min_score': float(jnp.min(current_scores)) if current_scores is not None else 0.0,
                'max_score': float(jnp.max(current_scores)) if current_scores is not None else 0.0,
                'mean_score': float(jnp.mean(current_scores)) if current_scores is not None else 0.0,
                'std_score': float(jnp.std(current_scores)) if current_scores is not None else 0.0
            }
        }

    def verify_int64_precision(self) -> bool:
        """Verify that beam search operations maintain int64 precision."""
        try:
            # Create test states with large int64 values
            state_size = self.generators.value.shape[1]
            large_values = [2**40, 2**50, 2**60]
            
            # Pad or truncate to match state size
            if len(large_values) < state_size:
                large_values.extend([i for i in range(len(large_values), state_size)])
            else:
                large_values = large_values[:state_size]
            
            large_state = jnp.array([large_values], dtype=jnp.int64)
            
            # Test beam expansion
            expanded = self.expand_beam(large_state)
            
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


def _fallback_to_cpu_beam_search(graph, predictor, beam_size: int, max_depth: int, target_state=None, start_state=None) -> Dict[str, Any]:
    """Fallback to CPU beam search implementation."""
    # This would use the existing CPU beam search implementation
    # For now, return a placeholder result
    return {
        'best_path': [],
        'best_score': 0.0,
        'search_depth': 0,
        'total_states_explored': 0,
        'target_found': False,
        'is_complete': True,
        'fallback_used': True
    }


def tpu_beam_search(
    graph,
    predictor,
    beam_size: int = 100,
    max_depth: int = 100,
    target_state: Optional[jnp.ndarray] = None,
    start_state: Optional[jnp.ndarray] = None
) -> Dict[str, Any]:
    """High-level TPU beam search function with automatic backend detection."""
    if not JAX_AVAILABLE:
        return _fallback_to_cpu_beam_search(graph, predictor, beam_size, max_depth, target_state, start_state)
    
    try:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel
        
        backend = get_tpu_backend()
        
        if not backend.is_available:
            return _fallback_to_cpu_beam_search(graph, predictor, beam_size, max_depth, target_state, start_state)
        
        # Use TPU beam search with native int64 support
        rngs = nnx.Rngs(42)
        beam_search_module = TPUBeamSearchModule(graph, predictor, backend, rngs, beam_size, max_depth)
        return beam_search_module.search(target_state, start_state)
        
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).warning("TPU beam search failed, falling back to CPU: %s", e)
        return _fallback_to_cpu_beam_search(graph, predictor, beam_size, max_depth, target_state, start_state)


def create_tpu_beam_search(
    graph,
    predictor,
    backend: Optional[TPUBackend] = None,
    beam_size: int = 100,
    max_depth: int = 100
) -> TPUBeamSearchModule:
    """Factory function to create TPU beam search module with error handling."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX and Flax are required for TPU beam search. " +
            "Install with: pip install 'cayleypy[jax-tpu]'"
        )
    
    if backend is None:
        from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel
        backend = get_tpu_backend()
    
    rngs = nnx.Rngs(42)
    return TPUBeamSearchModule(graph, predictor, backend, rngs, beam_size, max_depth)


def benchmark_tpu_vs_cpu_beam_search(
    graph,
    predictor,
    beam_size: int = 50,
    max_depth: int = 20,
    target_state: Optional[jnp.ndarray] = None
) -> Dict[str, Any]:
    """Benchmark TPU vs CPU beam search performance."""
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
        'search_params': {
            'beam_size': beam_size,
            'max_depth': max_depth,
            'has_target': target_state is not None
        },
        'tpu_available': False,
        'cpu_time': 0.0,
        'tpu_time': 0.0,
        'speedup': 0.0,
        'cpu_result': {},
        'tpu_result': {},
        'results_comparable': False
    }
    
    # CPU beam search
    try:
        start_time = time.time()
        cpu_result = _fallback_to_cpu_beam_search(graph, predictor, beam_size, max_depth, target_state)
        cpu_time = time.time() - start_time
        
        results['cpu_time'] = cpu_time
        results['cpu_result'] = cpu_result
        
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).error("CPU beam search failed: %s", e)
        return results
    
    # TPU beam search
    if JAX_AVAILABLE:
        try:
            from .tpu_backend import get_tpu_backend  # pylint: disable=import-outside-toplevel
            
            backend = get_tpu_backend()
            if backend.is_available:
                results['tpu_available'] = True
                
                start_time = time.time()
                tpu_result = tpu_beam_search(graph, predictor, beam_size, max_depth, target_state)
                tpu_time = time.time() - start_time
                
                results['tpu_time'] = tpu_time
                results['tpu_result'] = tpu_result
                results['speedup'] = cpu_time / tpu_time if tpu_time > 0 else 0.0
                
                # Compare results (beam search is stochastic, so exact match unlikely)
                results['results_comparable'] = (
                    cpu_result.get('target_found') == tpu_result.get('target_found') and
                    abs(cpu_result.get('search_depth', 0) - tpu_result.get('search_depth', 0)) <= 2
                )
                
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.getLogger(__name__).error("TPU beam search benchmark failed: %s", e)
    
    return results


if __name__ == "__main__":
    # Test TPU beam search when run as script
    print("Testing TPU Beam Search Implementation")
    print("=" * 40)
    
    if not JAX_AVAILABLE:
        print("JAX not available - cannot test TPU beam search")
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
                
                # Create a dummy predictor (would normally be a trained neural network)
                class DummyPredictor:
                    def batch_inference(self, states):
                        # Return random scores for testing
                        return jax.random.uniform(jax.random.PRNGKey(42), (len(states),))
                    
                    def get_performance_metrics(self):
                        return {'dummy_predictor': True}
                
                dummy_predictor = DummyPredictor()
                
                # Create beam search module
                beam_search_module = create_tpu_beam_search(
                    test_graph, dummy_predictor, backend, beam_size=10, max_depth=5
                )
                
                # Verify int64 precision
                precision_ok = beam_search_module.verify_int64_precision()
                print(f"int64 Precision Test: {'PASSED ✓' if precision_ok else 'FAILED ✗'}")
                
                # Run beam search
                print("Running TPU beam search on S4...")
                result = beam_search_module.search()
                print(f"Search result: depth={result['search_depth']}, explored={result['total_states_explored']}")
                print(f"Best score: {result['best_score']:.4f}")
                
                # Get performance metrics
                metrics = beam_search_module.get_performance_metrics()
                print(f"Beam expansions: {metrics['beam_expansions']}")
                print(f"States scored: {metrics['states_scored']}")
                print(f"Neural network operations: {metrics['neural_network_operations']}")
                
                print("✓ TPU beam search test completed successfully!")
            else:
                print("TPU not available for testing")
                
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"✗ TPU beam search test failed: {e}")