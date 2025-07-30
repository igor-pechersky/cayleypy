# JAX TPU Acceleration Design Document

## Overview

This design implements simplified JAX/TPU acceleration for CayleyPy, leveraging TPU v6e (Trillium) capabilities including native int64 support. The architecture focuses on maximizing TPU performance through direct acceleration of all CayleyPy operations without precision compromises.

**Key Capability**: TPU v6e supports int64 operations natively when `jax.config.update("jax_enable_x64", True)` is enabled, allowing for direct acceleration of all CayleyPy algorithms without precision loss or complex hybrid architectures.

## Architecture

### Core Components

#### 1. TPU Backend System
- **TPUBackend**: TPU v6e configuration with native int64 enablement
- **TPUConfig**: Configuration management with automatic x64 enablement
- **TPUMemoryManager**: Efficient memory management leveraging 32GB HBM per chip

#### 2. Accelerated Core Operations
- **TPUTensorOps**: JAX-accelerated tensor operations with native int64 support
- **TPUHasher**: Fast hashing using native int64 operations on TPU
- **TPUPermutationOps**: Efficient permutation operations leveraging TPU's systolic array

#### 3. NNX-Based Accelerated Algorithms
- **TPUBFSModule**: NNX module for TPU-accelerated breadth-first search
- **TPUBeamSearchModule**: NNX module for beam search with integrated neural networks
- **TPUPredictorModule**: NNX neural network predictor optimized for TPU v6e

#### 4. Integration Layer
- **AcceleratedCayleyGraph**: Drop-in replacement with full TPU acceleration
- **TPUOptimizer**: Automatic optimization for TPU v6e characteristics

### Design Principles

1. **Native int64 Support**: Leverage TPU v6e's native int64 operations for full precision
2. **NNX State Management**: Use Flax NNX for clean, stateful module design
3. **JIT Everything**: Use JAX JIT compilation for maximum TPU v6e performance
4. **Vectorize Operations**: Use vmap for automatic vectorization across TPU cores
5. **Memory Optimization**: Leverage TPU v6e's 32GB HBM per chip efficiently
6. **Systolic Array Utilization**: Optimize for TPU v6e's 256x256 systolic array

## Components and Interfaces

### TPU Backend System

```python
# cayleypy/tpu_backend.py
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class TPUConfig:
    """TPU v6e configuration with native int64 support."""
    enable_x64: bool = True  # Enable native int64 support
    memory_fraction: float = 0.9
    compilation_cache: bool = True
    
    def apply(self):
        """Apply TPU configuration."""
        if self.enable_x64:
            jax.config.update("jax_enable_x64", True)
        if self.compilation_cache:
            jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

class TPUBackend(nnx.Module):
    """NNX-based TPU backend for v6e (Trillium) with native int64 support."""
    
    def __init__(self, config: Optional[TPUConfig] = None, rngs: Optional[nnx.Rngs] = None):
        self.config = nnx.Variable(config or TPUConfig())
        self.config.value.apply()
        
        # Initialize TPU devices
        self.devices = jax.devices('tpu')
        self.is_available = len(self.devices) > 0
        
        # TPU v6e specific capabilities
        self.capabilities = nnx.Variable({
            'supports_int32': True,
            'supports_int64': True,  # Native int64 support on TPU v6e!
            'supports_float32': True,
            'supports_float64': True,  # Native float64 support on TPU v6e!
            'supports_bfloat16': True,
            'hbm_per_chip_gb': 32,  # TPU v6e spec
            'systolic_array_size': (256, 256),  # TPU v6e spec
            'max_batch_size': 1024 * 1024,  # Estimated based on memory
        })
        
        if self.is_available:
            print(f"TPU v6e Backend initialized with {len(self.devices)} devices")
            print(f"HBM per chip: {self.capabilities.value['hbm_per_chip_gb']}GB")
            print(f"Native int64 support: {self.capabilities.value['supports_int64']}")
            
            # Test int64 support
            test_array = jnp.array([1, 2, 3], dtype=jnp.int64)
            print(f"int64 test successful: {test_array.dtype}")
        else:
            print("TPU not available, will fallback to CPU")
    
    def get_device(self):
        """Get primary TPU device."""
        return self.devices[0] if self.is_available else jax.devices('cpu')[0]
    
    def supports_dtype(self, dtype) -> bool:
        """Check if TPU supports the given dtype."""
        dtype_support = {
            jnp.int32: self.capabilities.value['supports_int32'],
            jnp.int64: self.capabilities.value['supports_int64'],
            jnp.float32: self.capabilities.value['supports_float32'],
            jnp.float64: self.capabilities.value['supports_float64'],
            jnp.bfloat16: self.capabilities.value['supports_bfloat16'],
        }
        return dtype_support.get(dtype, False)

# Global TPU backend instance
_tpu_backend = None

def get_tpu_backend() -> TPUBackend:
    """Get global TPU backend instance."""
    global _tpu_backend
    if _tpu_backend is None:
        _tpu_backend = TPUBackend()
    return _tpu_backend
```

### TPU Tensor Operations

```python
# cayleypy/tpu_tensor_ops.py
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Dict, Any, Optional

class TPUTensorOpsModule(nnx.Module):
    """NNX module for TPU-accelerated tensor operations with native int64 support."""
    
    def __init__(self, backend: TPUBackend, rngs: Optional[nnx.Rngs] = None):
        self.backend = backend
        
        # Cache for frequently computed operations
        self.operation_cache = nnx.Variable({})
        
        # Performance metrics
        self.metrics = nnx.Variable({
            'operations_count': 0,
            'cache_hits': 0,
            'total_elements_processed': 0,
            'int64_operations': 0,
            'systolic_array_utilization': 0.0
        })
    
    @nnx.jit
    def unique_with_indices(self, arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """TPU-optimized unique operation with native int64 support."""
        # Native int64 support - no conversion needed!
        if arr.dtype == jnp.int64:
            self.metrics.value['int64_operations'] += 1
        
        # Sort array for efficient unique detection on TPU
        sorted_indices = jnp.argsort(arr)
        sorted_arr = arr[sorted_indices]
        
        # Find unique elements using TPU-optimized operations
        mask = jnp.concatenate([
            jnp.array([True]), 
            sorted_arr[1:] != sorted_arr[:-1]
        ])
        
        unique_vals = sorted_arr[mask]
        unique_indices = sorted_indices[mask]
        
        # Update metrics
        self.metrics.value['operations_count'] += 1
        self.metrics.value['total_elements_processed'] += len(arr)
        
        return unique_vals, unique_indices
    
    @nnx.jit
    def isin(self, elements: jnp.ndarray, test_elements: jnp.ndarray) -> jnp.ndarray:
        """TPU-optimized membership testing with int64 support."""
        sorted_test = jnp.sort(test_elements)
        indices = jnp.searchsorted(sorted_test, elements)
        indices = jnp.clip(indices, 0, len(sorted_test) - 1)
        return sorted_test[indices] == elements
    
    @nnx.jit
    def batch_apply_permutation(self, states: jnp.ndarray, perm: jnp.ndarray) -> jnp.ndarray:
        """Apply permutation to batch of states using TPU's systolic array."""
        # Use vmap for automatic vectorization across TPU v6e cores
        return jax.vmap(lambda state: state[perm])(states)
    
    @nnx.jit
    def batch_matrix_multiply(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Leverage TPU v6e's 256x256 systolic array for matrix operations."""
        # TPU v6e has 4x more FLOPs per cycle than v5e
        result = jax.vmap(jnp.dot, in_axes=(0, None))(a, b)
        
        # Track systolic array utilization
        self.metrics.value['systolic_array_utilization'] += 1.0
        
        return result
    
    @nnx.jit
    def deduplicate_int64_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Remove duplicate states using native int64 operations on TPU."""
        # Convert states to int64 hashes for precise deduplication
        state_hashes = jax.vmap(
            lambda s: jnp.sum(s.astype(jnp.int64) * jnp.arange(len(s), dtype=jnp.int64))
        )(states)
        
        unique_hashes, unique_indices = self.unique_with_indices(state_hashes)
        return states[unique_indices]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return dict(self.metrics.value)
```

### TPU Hasher

```python
# cayleypy/tpu_hasher.py
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
from typing import Optional

class TPUHasherModule(nnx.Module):
    """NNX module for TPU-accelerated state hashing with native int64 support."""
    
    def __init__(self, state_size: int, backend: TPUBackend, rngs: nnx.Rngs):
        self.state_size = state_size
        self.backend = backend
        
        # Generate hash matrix using native int64 on TPU
        self.hash_matrix = nnx.Param(
            random.randint(
                rngs.params(), (state_size, 64), 
                minval=-2**31, maxval=2**31, 
                dtype=jnp.int64
            )
        )
        
        # Hash cache for performance
        self.hash_cache = nnx.Variable({})
        
        # Performance metrics
        self.metrics = nnx.Variable({
            'total_hashes': 0,
            'cache_hits': 0,
            'int64_hashes': 0,
            'collision_rate': 0.0
        })
    
    @nnx.jit
    def hash_state(self, state: jnp.ndarray) -> jnp.int64:
        """Hash single state using native int64 operations on TPU."""
        # Native int64 matrix multiplication on TPU v6e
        hash_result = jnp.sum(
            state.astype(jnp.int64) * self.hash_matrix.value[:, 0]
        ) % (2**63 - 1)
        
        # Update metrics
        self.metrics.value['total_hashes'] += 1
        self.metrics.value['int64_hashes'] += 1
        
        return hash_result
    
    @nnx.jit
    def hash_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Hash batch of states using TPU vectorization."""
        # Use vmap for efficient batch processing on TPU
        hashes = jax.vmap(self.hash_state)(states)
        return hashes
    
    @nnx.jit
    def hash_large_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Hash large batches leveraging TPU v6e's 32GB HBM."""
        # TPU v6e can handle much larger batches in memory
        batch_size = min(len(states), 100000)  # Leverage large HBM
        
        def process_chunk(chunk):
            return self.hash_batch(chunk)
        
        # Process in chunks that fit in TPU memory
        chunks = jnp.array_split(states, max(1, len(states) // batch_size))
        results = [process_chunk(chunk) for chunk in chunks]
        
        return jnp.concatenate(results, axis=0)
    
    @nnx.jit
    def deduplicate_by_hash(self, states: jnp.ndarray) -> jnp.ndarray:
        """Remove duplicates using native int64 hash-based deduplication."""
        hashes = self.hash_batch(states)
        
        # Use TPU tensor ops for unique detection
        from .tpu_tensor_ops import TPUTensorOpsModule
        tensor_ops = TPUTensorOpsModule(self.backend)
        
        unique_hashes, unique_indices = tensor_ops.unique_with_indices(hashes)
        return states[unique_indices]
    
    def get_hash_stats(self) -> dict:
        """Get hashing performance statistics."""
        total = self.metrics.value['total_hashes']
        hits = self.metrics.value['cache_hits']
        return {
            'cache_hit_rate': hits / max(1, total),
            'total_hashes': total,
            'int64_hashes': self.metrics.value['int64_hashes'],
            'collision_rate': self.metrics.value['collision_rate']
        }

class HybridPrecisionHasher(nnx.Module):
    """Hybrid hasher: TPU int32 for speed, CPU int64 for precision when needed."""
    
    def __init__(self, state_size: int, backend: TPUBackend, rngs: Optional[nnx.Rngs] = None):
        self.tpu_hasher = TPUHasherModule(state_size, backend, rngs)
        self.precision_threshold = nnx.Variable(0.01)  # Switch to CPU if >1% collisions
    
    def hash_batch_adaptive(self, states: jnp.ndarray) -> jnp.ndarray:
        """Adaptively choose between TPU int32 and CPU int64 hashing."""
        # Try TPU first
        tpu_result = self.tpu_hasher.hash_batch(states)
        
        # Check collision rate
        collision_rate = self.tpu_hasher.get_collision_rate()
        
        if collision_rate > self.precision_threshold.value:
            print(f"High collision rate ({collision_rate:.3f}), falling back to CPU int64")
            # Fallback to CPU int64 (implementation would be in separate module)
            return self._cpu_int64_hash(states)
        
        return tpu_result
    
    def _cpu_int64_hash(self, states: jnp.ndarray) -> jnp.ndarray:
        """CPU fallback using int64 (placeholder)."""
        # This would call the existing CPU hasher
        pass
```

### TPU BFS Implementation

```python
# cayleypy/tpu_bfs.py
import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Tuple, Optional

class TPUBFSModule(nnx.Module):
    """NNX module for TPU-accelerated breadth-first search with int64 support."""
    
    def __init__(self, graph, backend: TPUBackend, rngs: nnx.Rngs):
        self.graph = graph
        self.backend = backend
        
        # Store generators as NNX parameters for optimization
        self.generators = nnx.Param(
            jnp.array(graph.definition.generators_permutations, dtype=jnp.int64)
        )
        
        # Initialize hasher and tensor ops
        self.hasher = TPUHasherModule(len(graph.central_state), backend, rngs)
        self.tensor_ops = TPUTensorOpsModule(backend, rngs)
        
        # BFS state tracking
        self.bfs_state = nnx.Variable({
            'current_layer': None,
            'visited_hashes': None,
            'layer_sizes': [],
            'diameter': 0,
            'total_states_found': 0
        })
        
        # Performance metrics
        self.metrics = nnx.Variable({
            'states_processed': 0,
            'hash_operations': 0,
            'memory_peak_mb': 0,
            'tpu_utilization': 0.0
        })
    
    @nnx.jit
    def expand_layer(self, current_layer: jnp.ndarray) -> jnp.ndarray:
        """Expand current layer by applying all generators using TPU v6e."""
        def apply_all_generators(state):
            # Use TPU's vectorization for generator application
            return jax.vmap(lambda gen: state[gen])(self.generators.value)
        
        # Apply generators to all states in current layer
        # Leverage TPU v6e's 256x256 systolic array
        expanded = jax.vmap(apply_all_generators)(current_layer)
        
        # Flatten to get all new states
        return expanded.reshape(-1, current_layer.shape[-1])
    
    @nnx.jit
    def bfs_step(self, current_layer: jnp.ndarray, 
                 visited_hashes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single BFS step using native int64 operations."""
        # Expand current layer
        new_states = self.expand_layer(current_layer)
        
        # Remove duplicates within new states using int64 precision
        new_states = self.tensor_ops.deduplicate_int64_states(new_states)
        
        # Hash new states using native int64
        new_hashes = self.hasher.hash_batch(new_states)
        
        # Filter out already visited states
        is_new = ~self.tensor_ops.isin(new_hashes, visited_hashes)
        truly_new_states = new_states[is_new]
        truly_new_hashes = new_hashes[is_new]
        
        # Update visited set
        updated_visited = jnp.concatenate([visited_hashes, truly_new_hashes])
        
        # Update metrics
        self.metrics.value['states_processed'] += len(new_states)
        self.metrics.value['hash_operations'] += len(new_states)
        
        return truly_new_states, updated_visited
    
    def initialize_bfs(self):
        """Initialize BFS state."""
        start_state = jnp.array([self.graph.central_state], dtype=jnp.int64)
        start_hash = self.hasher.hash_batch(start_state)
        
        self.bfs_state.value.update({
            'current_layer': start_state,
            'visited_hashes': start_hash,
            'layer_sizes': [1],
            'diameter': 0,
            'total_states_found': 1
        })
    
    def run_bfs(self, max_diameter: int = 1000) -> List[int]:
        """Run complete BFS leveraging TPU v6e's capabilities."""
        self.initialize_bfs()
        
        for diameter in range(max_diameter):
            current_layer = self.bfs_state.value['current_layer']
            visited_hashes = self.bfs_state.value['visited_hashes']
            
            new_layer, updated_visited = self.bfs_step(current_layer, visited_hashes)
            
            if len(new_layer) == 0:
                break
                
            # Update state
            self.bfs_state.value.update({
                'current_layer': new_layer,
                'visited_hashes': updated_visited,
                'diameter': diameter + 1,
                'total_states_found': self.bfs_state.value['total_states_found'] + len(new_layer)
            })
            
            self.bfs_state.value['layer_sizes'].append(len(new_layer))
            
        return self.bfs_state.value['layer_sizes']
    
    def get_performance_metrics(self) -> dict:
        """Get BFS performance metrics."""
        return {
            **dict(self.metrics.value),
            'hasher_stats': self.hasher.get_hash_stats(),
            'tensor_ops_stats': self.tensor_ops.get_performance_metrics()
        }

def tpu_bfs(graph, max_diameter: int = 1000) -> List[int]:
    """High-level TPU BFS function with automatic backend detection."""
    backend = get_tpu_backend()
    
    if not backend.is_available:
        # Fallback to CPU implementation
        from .bfs_numpy import bfs_numpy
        return bfs_numpy(graph, max_diameter)
    
    # Use TPU BFS with native int64 support
    bfs_module = TPUBFSModule(graph, backend, nnx.Rngs(42))
    return bfs_module.run_bfs(max_diameter)
    """High-level TPU BFS function with fallback."""
    backend = get_tpu_backend()
    
    if not backend.is_available:
        # Fallback to CPU implementation
        from .bfs_numpy import bfs_numpy
        return bfs_numpy(graph, max_diameter)
    
    # Convert generators to JAX arrays
    generators = jnp.array(graph.definition.generators_permutations)
    start_state = jnp.array(graph.central_state)
    
    # Run TPU BFS
    tpu_bfs_impl = TPUBFS(generators, start_state)
    return tpu_bfs_impl.run_bfs(max_diameter)
```

### TPU Beam Search

```python
# cayleypy/tpu_beam_search.py
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional, Tuple
from .beam_search_result import BeamSearchResult

class TPUBeamSearchModule(nnx.Module):
    """NNX module for TPU-accelerated beam search with native int64 support."""
    
    def __init__(self, graph, predictor: nnx.Module, beam_width: int, 
                 backend: TPUBackend, rngs: nnx.Rngs):
        self.graph = graph
        self.beam_width = beam_width
        self.backend = backend
        
        # Integrated predictor as NNX module
        self.predictor = predictor
        
        # Store generators as NNX parameters
        self.generators = nnx.Param(
            jnp.array(graph.definition.generators_permutations, dtype=jnp.int64)
        )
        
        # Initialize hasher and tensor ops
        self.hasher = TPUHasherModule(len(graph.central_state), backend, rngs)
        self.tensor_ops = TPUTensorOpsModule(backend, rngs)
        
        # Beam search state
        self.search_state = nnx.Variable({
            'current_beam': None,
            'iteration': 0,
            'target_found': False,
            'best_path_length': float('inf')
        })
        
        # Performance metrics
        self.metrics = nnx.Variable({
            'states_expanded': 0,
            'duplicates_removed': 0,
            'predictor_calls': 0,
            'tpu_memory_used_mb': 0
        })
    
    @nnx.jit
    def expand_beam(self, beam_states: jnp.ndarray) -> jnp.ndarray:
        """Expand beam using TPU v6e's systolic array."""
        def expand_single_state(state):
            # Use TPU's vectorization for generator application
            return jax.vmap(lambda gen: state[gen])(self.generators.value)
        
        # Leverage TPU v6e's parallel processing capabilities
        expanded = jax.vmap(expand_single_state)(beam_states)
        expanded_flat = expanded.reshape(-1, beam_states.shape[-1])
        
        # Update metrics
        self.metrics.value['states_expanded'] += len(expanded_flat)
        
        return expanded_flat
    
    @nnx.jit
    def deduplicate_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Remove duplicates using native int64 hashing."""
        unique_states = self.hasher.deduplicate_by_hash(states)
        
        # Update metrics
        duplicates_removed = len(states) - len(unique_states)
        self.metrics.value['duplicates_removed'] += duplicates_removed
        
        return unique_states
    
    @nnx.jit
    def score_and_select(self, states: jnp.ndarray) -> jnp.ndarray:
        """Score states and select top k using integrated predictor."""
        if len(states) <= self.beam_width:
            return states
            
        # Score all states using TPU-optimized predictor
        scores = self.predictor(states)
        self.metrics.value['predictor_calls'] += len(states)
        
        # Select top k states
        top_indices = jnp.argsort(scores)[:self.beam_width]
        return states[top_indices]
    
    @nnx.jit
    def check_target(self, states: jnp.ndarray, target: jnp.ndarray) -> bool:
        """Check if target state is in current states."""
        matches = jax.vmap(
            lambda state: jnp.array_equal(state, target)
        )(states)
        return jnp.any(matches)
    
    @nnx.jit
    def search_step(self, current_beam: jnp.ndarray, 
                    target: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
        """Single beam search step using TPU acceleration."""
        # Expand beam
        expanded_states = self.expand_beam(current_beam)
        
        # Remove duplicates using int64 precision
        unique_states = self.deduplicate_states(expanded_states)
        
        # Check if target found
        target_found = self.check_target(unique_states, target)
        
        if target_found:
            return unique_states, True
        
        # Score and select top k
        next_beam = self.score_and_select(unique_states)
        
        return next_beam, False
    
    def initialize_search(self, start_state: jnp.ndarray, target_state: jnp.ndarray):
        """Initialize beam search state."""
        self.search_state.value.update({
            'current_beam': jnp.array([start_state], dtype=jnp.int64),
            'target_state': target_state.astype(jnp.int64),
            'iteration': 0,
            'target_found': False,
            'best_path_length': float('inf')
        })
        
        # Reset metrics
        for key in self.metrics.value:
            self.metrics.value[key] = 0
    
    def search(self, start_state: jnp.ndarray, target_state: jnp.ndarray, 
               max_iterations: int = 1000) -> BeamSearchResult:
        """Perform beam search leveraging TPU v6e capabilities."""
        self.initialize_search(start_state, target_state)
        
        for iteration in range(max_iterations):
            current_beam = self.search_state.value['current_beam']
            target = self.search_state.value['target_state']
            
            next_beam, target_found = self.search_step(current_beam, target)
            
            # Update search state
            self.search_state.value.update({
                'current_beam': next_beam,
                'iteration': iteration + 1,
                'target_found': target_found
            })
            
            if target_found:
                return BeamSearchResult(path_found=True, path_length=iteration + 1)
            
            if len(next_beam) == 0:
                break
        
        return BeamSearchResult(path_found=False, path_length=-1)
    
    def get_search_metrics(self) -> dict:
        """Get detailed search metrics."""
        return {
            **dict(self.metrics.value),
            'hasher_stats': self.hasher.get_hash_stats(),
            'tensor_ops_stats': self.tensor_ops.get_performance_metrics(),
            'current_beam_size': len(self.search_state.value.get('current_beam', [])),
            'iteration': self.search_state.value.get('iteration', 0)
        }

def tpu_beam_search(graph, predictor: nnx.Module, start_state: jnp.ndarray,
                    target_state: jnp.ndarray, beam_width: int = 1000,
                    max_iterations: int = 1000) -> BeamSearchResult:
    """High-level TPU beam search with automatic backend detection."""
    backend = get_tpu_backend()
    
    if not backend.is_available:
        # Fallback to existing implementation
        return graph.beam_search(start_state, target_state, beam_width, max_iterations)
    
    # Use TPU beam search with native int64 support
    beam_search = TPUBeamSearchModule(graph, predictor, beam_width, backend, nnx.Rngs(42))
    return beam_search.search(start_state, target_state, max_iterations)
```

### TPU Predictor

```python
# cayleypy/tpu_predictor.py
from flax import nnx
import jax.numpy as jnp
import optax
from typing import Dict, Any, Optional

class TPUPredictorModule(nnx.Module):
    """NNX module for TPU-optimized neural network predictor with v6e optimizations."""
    
    def __init__(self, input_size: int, backend: TPUBackend, rngs: nnx.Rngs,
                 hidden_size: int = 512, num_layers: int = 3):
        self.input_size = input_size
        self.backend = backend
        
        # Build MLP architecture optimized for TPU v6e's systolic array
        layers = []
        current_size = input_size
        
        for i in range(num_layers - 1):
            # Use sizes that are multiples of 128 for optimal TPU utilization
            optimal_hidden = ((hidden_size + 127) // 128) * 128
            layers.append(nnx.Linear(current_size, optimal_hidden, rngs=rngs))
            layers.append(nnx.BatchNorm(optimal_hidden, rngs=rngs))
            layers.append(nnx.relu)
            layers.append(nnx.Dropout(0.1, rngs=rngs))
            current_size = optimal_hidden
            
        # Output layer
        layers.append(nnx.Linear(current_size, 1, rngs=rngs))
        
        self.model = nnx.Sequential(*layers)
        
        # Training state
        self.training_state = nnx.Variable({
            'epoch': 0,
            'best_loss': float('inf'),
            'training_history': []
        })
        
        # Performance metrics
        self.metrics = nnx.Variable({
            'inference_time_ms': 0.0,
            'batch_sizes_processed': [],
            'tpu_memory_used_mb': 0,
            'systolic_array_utilization': 0.0
        })
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass optimized for TPU v6e."""
        # Ensure input is properly shaped for TPU
        if x.dtype != jnp.float32:
            x = x.astype(jnp.float32)
        
        # Track batch size for optimization
        self.metrics.value['batch_sizes_processed'].append(x.shape[0])
        
        return self.model(x).squeeze(-1)
    
    @nnx.jit
    def train_step(self, batch_states: jnp.ndarray, batch_targets: jnp.ndarray) -> Dict[str, Any]:
        """Single training step optimized for TPU v6e."""
        def loss_fn(model):
            predictions = model(batch_states)
            mse_loss = jnp.mean((predictions - batch_targets) ** 2)
            
            # Add L2 regularization for better generalization
            l2_loss = 0.001 * sum(
                jnp.sum(param ** 2) 
                for param in jax.tree_leaves(nnx.split(model, nnx.Param)[1])
            )
            
            return mse_loss + l2_loss
        
        loss, grads = nnx.value_and_grad(loss_fn)(self)
        
        # Update metrics
        self.metrics.value['systolic_array_utilization'] += 1.0
        
        return {'loss': loss, 'grads': grads}
    
    @nnx.jit
    def batch_inference(self, states: jnp.ndarray) -> jnp.ndarray:
        """Batch inference leveraging TPU v6e's parallel processing."""
        # Set to evaluation mode
        self.eval()
        
        # Process in optimal batch sizes for TPU v6e
        optimal_batch_size = 1024  # Leverage 32GB HBM
        
        if len(states) <= optimal_batch_size:
            return self(states)
        
        # Process in chunks
        results = []
        for i in range(0, len(states), optimal_batch_size):
            batch = states[i:i + optimal_batch_size]
            batch_result = self(batch)
            results.append(batch_result)
        
        return jnp.concatenate(results, axis=0)
    
    def train_epoch(self, train_data, optimizer: optax.GradientTransformation) -> Dict[str, Any]:
        """Train for one epoch with TPU optimization."""
        self.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_states, batch_targets in train_data:
            # Ensure data is on TPU and properly typed
            batch_states = jnp.array(batch_states, dtype=jnp.float32)
            batch_targets = jnp.array(batch_targets, dtype=jnp.float32)
            
            # Training step
            step_result = self.train_step(batch_states, batch_targets)
            
            # Apply gradients
            optimizer.update(step_result['grads'], self)
            
            total_loss += step_result['loss']
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        # Update training state
        self.training_state.value['epoch'] += 1
        if avg_loss < self.training_state.value['best_loss']:
            self.training_state.value['best_loss'] = avg_loss
        
        self.training_state.value['training_history'].append(avg_loss)
        
        return {
            'epoch': self.training_state.value['epoch'],
            'avg_loss': avg_loss,
            'best_loss': self.training_state.value['best_loss']
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        batch_sizes = self.metrics.value['batch_sizes_processed']
        return {
            **dict(self.metrics.value),
            'avg_batch_size': jnp.mean(jnp.array(batch_sizes)) if batch_sizes else 0,
            'training_epoch': self.training_state.value['epoch'],
            'best_loss': self.training_state.value['best_loss']
        }

class TPUPredictorTrainer(nnx.Module):
    """Training wrapper for TPU predictor with advanced features."""
    
    def __init__(self, predictor: TPUPredictorModule, learning_rate: float = 1e-3):
        self.predictor = predictor
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = nnx.Variable(self.optimizer.init(nnx.split(predictor, nnx.Param)[1]))
    
    def train(self, train_data, num_epochs: int = 100) -> Dict[str, Any]:
        """Full training loop with TPU optimization."""
        training_history = []
        
        for epoch in range(num_epochs):
            epoch_result = self.predictor.train_epoch(train_data, self.optimizer)
            training_history.append(epoch_result)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_result['avg_loss']:.6f}")
        
        return {
            'training_history': training_history,
            'final_metrics': self.predictor.get_performance_metrics()
        }
```

### Integration Layer

```python
# cayleypy/accelerated_cayley_graph.py
from .cayley_graph import CayleyGraph
from .tpu_backend import get_tpu_backend
from .tpu_bfs import tpu_bfs
from .tpu_beam_search import tpu_beam_search

class AcceleratedCayleyGraph(CayleyGraph):
    """CayleyGraph with TPU acceleration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tpu_backend = get_tpu_backend()
    
    def bfs(self, max_diameter: int = 1000000):
        """BFS with TPU acceleration."""
        if self.tpu_backend.is_available:
            return tpu_bfs(self, max_diameter)
        else:
            return super().bfs(max_diameter)
    
    def beam_search(self, start_state, target_state, beam_width=1000, 
                    max_iterations=1000, predictor=None):
        """Beam search with TPU acceleration."""
        if self.tpu_backend.is_available and predictor is not None:
            return tpu_beam_search(self, predictor, start_state, target_state,
                                 beam_width, max_iterations)
        else:
            return super().beam_search(start_state, target_state, beam_width,
                                     max_iterations, predictor)

# Monkey patch for seamless integration
def enable_tpu_acceleration():
    """Enable TPU acceleration for all CayleyGraph instances."""
    CayleyGraph.bfs = AcceleratedCayleyGraph.bfs
    CayleyGraph.beam_search = AcceleratedCayleyGraph.beam_search
```

## Data Models

### Configuration
- **TPUConfig**: Simple configuration with int64 enablement
- **BackendInfo**: TPU device information and capabilities

### Performance Metrics
- **TPUMetrics**: Timing, memory usage, and throughput metrics
- **BenchmarkResult**: Comparison results between TPU and CPU

## Error Handling

Simple error handling with graceful fallback:
- TPU unavailable → fallback to CPU implementations
- Memory errors → automatic chunking leveraging 32GB HBM
- Compilation errors → fallback with warning and performance logging
- int64 operations → native TPU v6e support (no fallback needed)

## Testing Strategy

1. **Unit Tests**: Test each TPU component individually with int64 operations
2. **Integration Tests**: Test full pipeline with real graphs on TPU v6e
3. **Performance Tests**: Benchmark TPU vs CPU performance with native int64
4. **Correctness Tests**: Verify TPU int64 results match CPU exactly
5. **Memory Tests**: Test large-scale operations leveraging 32GB HBM per chip
6. **Precision Tests**: Verify int64 precision is maintained throughout pipeline