# JAX GPU/TPU Acceleration Design Document

## Overview

This design document outlines the implementation of JAX support for GPU/TPU acceleration in CayleyPy. The solution will provide seamless integration of JAX's high-performance computing capabilities while maintaining backward compatibility with existing CPU-based implementations. The design leverages JAX's JIT compilation, vectorization, and hardware-specific optimizations to achieve significant performance improvements for large-scale graph operations.

## Architecture

### Core Components

#### 1. NNX-Centric Backend System
- **NNXBackend**: Central configuration using Flax NNX's device management
- **NNX Device Mesh**: Leverage NNX's distributed computing capabilities
- **NNX State Management**: Use NNX's automatic state handling for all operations

#### 2. NNX-Accelerated Core Operations
- **NNX Functional Modules**: Replace raw JAX operations with NNX functional modules
- **NNX Hash Modules**: Stateful hash functions implemented as NNX modules
- **NNX BFS Modules**: BFS algorithms as composable NNX modules
- **NNX Beam Search**: Beam search implemented as stateful NNX modules

#### 3. Unified NNX Neural Network Integration
- **NNX Predictor Ecosystem**: All predictors as NNX modules with shared infrastructure
- **NNX Training Pipeline**: Unified training system using NNX transforms
- **NNX State Persistence**: Automatic checkpointing and state management

#### 4. NNX Memory and Optimization Management
- **NNX Sharding**: Use NNX's built-in sharding for memory management
- **NNX Transforms**: Leverage NNX's jit, vmap, and other transforms
- **NNX Checkpointing**: Built-in gradient checkpointing and memory optimization

## Components and Interfaces

### NNX Backend System

```python
from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

class NNXBackend(nnx.Module):
    """NNX-based backend for hardware acceleration and state management."""
    
    def __init__(self, preferred_device: str = "auto", rngs: nnx.Rngs = None):
        self.device_type = self._detect_device(preferred_device)
        self.mesh = self._create_device_mesh()
        self.sharding = self._setup_sharding()
        self.config = nnx.Variable(self._setup_optimization_flags())
    
    def _detect_device(self, preferred: str) -> str:
        """Detect and configure optimal device using NNX patterns."""
        
    def _create_device_mesh(self) -> Mesh:
        """Create device mesh for distributed computation."""
        devices = jax.devices(self.device_type)
        return Mesh(devices, axis_names=['devices'])
    
    def _setup_sharding(self) -> NamedSharding:
        """Setup sharding strategy for tensors."""
        return NamedSharding(self.mesh, PartitionSpec('devices'))
        
    def _setup_optimization_flags(self) -> dict:
        """Configure XLA and JAX optimization flags."""
        
    def is_available(self) -> bool:
        """Check if NNX acceleration is available."""
```

### NNX Tensor Operations Module

```python
# cayleypy/nnx_tensor_ops.py
from flax import nnx
import jax.numpy as jnp
from typing import Tuple, Optional

class TensorOpsModule(nnx.Module):
    """NNX module for tensor operations with automatic state management."""
    
    def __init__(self, backend: NNXBackend, rngs: nnx.Rngs):
        self.backend = backend
        self.cache = nnx.Variable({})  # Cache for frequently used operations
    
    @nnx.jit
    def unique_with_indices(self, arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """NNX-optimized unique operation with indices and caching."""
        cache_key = f"unique_{arr.shape}_{hash(arr.tobytes())}"
        if cache_key in self.cache.value:
            return self.cache.value[cache_key]
        
        result = self._compute_unique(arr)
        self.cache.value[cache_key] = result
        return result
    
    def _compute_unique(self, arr: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Internal unique computation."""
        sorted_arr = jnp.sort(arr)
        mask = jnp.concatenate([jnp.array([True]), sorted_arr[1:] != sorted_arr[:-1]])
        unique_vals = sorted_arr[mask]
        indices = jnp.searchsorted(sorted_arr, unique_vals)
        return unique_vals, indices
    
    @nnx.jit  
    def isin_via_searchsorted(self, elements: jnp.ndarray, test_elements: jnp.ndarray) -> jnp.ndarray:
        """Fast membership testing using searchsorted with NNX optimization."""
        sorted_test = jnp.sort(test_elements)
        indices = jnp.searchsorted(sorted_test, elements)
        indices = jnp.clip(indices, 0, len(sorted_test) - 1)
        return sorted_test[indices] == elements
    
    @nnx.vmap
    def batch_matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Vectorized matrix multiplication using NNX vmap."""
        return jnp.dot(a, b)
    
    @nnx.vmap
    def vectorized_element_wise_equal(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Vectorized element-wise equality comparison."""
        return jnp.array_equal(a, b)
```

### NNX Hash Functions

```python
# cayleypy/nnx_hasher.py
from flax import nnx
import jax.numpy as jnp
from jax import random
from .hasher import StateHasher

class NNXStateHasher(nnx.Module, StateHasher):
    """NNX-based state hashing with automatic state management."""
    
    def __init__(self, state_size: int, backend: NNXBackend, rngs: nnx.Rngs):
        self.state_size = state_size
        self.backend = backend
        
        # Use NNX Parameter for hash matrix - automatically managed
        hash_key = rngs.params()
        self.hash_matrix = nnx.Param(
            random.randint(hash_key, (state_size, 64), 0, 2**31)
        )
        
        # NNX Variable for caching hash results
        self.hash_cache = nnx.Variable({})
        
        # Statistics tracking as NNX Variables
        self.stats = nnx.Variable({
            'total_hashes': 0,
            'cache_hits': 0,
            'batch_sizes': []
        })
    
    @nnx.jit
    def hash_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Hash a single state vector with caching."""
        state_key = hash(state.tobytes())
        
        if state_key in self.hash_cache.value:
            self.stats.value['cache_hits'] += 1
            return self.hash_cache.value[state_key]
        
        # Compute hash using matrix multiplication
        hash_result = jnp.sum(state * self.hash_matrix.value, axis=0) % (2**31 - 1)
        
        # Update cache and stats
        self.hash_cache.value[state_key] = hash_result
        self.stats.value['total_hashes'] += 1
        
        return hash_result
    
    @nnx.vmap
    def hash_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Hash a batch of states using NNX vectorization."""
        self.stats.value['batch_sizes'].append(states.shape[0])
        return self.hash_state(states)
    
    @nnx.jit
    def hash_large_batch(self, states: jnp.ndarray, chunk_size: int = 10000) -> jnp.ndarray:
        """Process large batches with automatic chunking and memory management."""
        def process_chunk(chunk):
            return self.hash_batch(chunk)
        
        # Use NNX's scan for efficient chunked processing
        chunks = jnp.array_split(states, max(1, states.shape[0] // chunk_size))
        results = nnx.scan(lambda _, chunk: (None, process_chunk(chunk)), 
                          None, chunks)[1]
        
        return jnp.concatenate(results, axis=0)
    
    def get_cache_stats(self) -> dict:
        """Get caching statistics."""
        total = self.stats.value['total_hashes']
        hits = self.stats.value['cache_hits']
        return {
            'cache_hit_rate': hits / max(1, total),
            'total_hashes': total,
            'avg_batch_size': jnp.mean(jnp.array(self.stats.value['batch_sizes'])) if self.stats.value['batch_sizes'] else 0
        }

class OptimizedNNXStateHasher(NNXStateHasher):
    """Memory-optimized version with advanced NNX features."""
    
    def __init__(self, state_size: int, backend: NNXBackend, rngs: nnx.Rngs):
        super().__init__(state_size, backend, rngs)
        
        # Add memory-efficient hash matrix using NNX sharding
        self.hash_matrix = nnx.Param(
            self.hash_matrix.value,
            sharding=backend.sharding
        )
    
    @nnx.remat  # Use NNX rematerialization for memory efficiency
    def hash_large_batch(self, states: jnp.ndarray, chunk_size: int = 10000) -> jnp.ndarray:
        """Memory-efficient large batch processing with rematerialization."""
        return super().hash_large_batch(states, chunk_size)
```

### NNX BFS Implementation

```python
# cayleypy/nnx_bfs.py
from flax import nnx
import jax.numpy as jnp
from typing import Tuple, List
from .cayley_graph import CayleyGraph
from .nnx_tensor_ops import TensorOpsModule

class NNXBFSModule(nnx.Module):
    """NNX-based BFS implementation with state management and optimization."""
    
    def __init__(self, graph: CayleyGraph, backend: NNXBackend, rngs: nnx.Rngs):
        self.graph = graph
        self.backend = backend
        self.tensor_ops = TensorOpsModule(backend, rngs)
        
        # Store generators as NNX Parameters for optimization
        self.generators = nnx.Param(
            jnp.array(graph.definition.generators_permutations),
            sharding=backend.sharding
        )
        
        # BFS state tracking
        self.bfs_state = nnx.Variable({
            'current_layer': None,
            'visited': None,
            'layer_sizes': [],
            'diameter': 0
        })
        
        # Performance metrics
        self.metrics = nnx.Variable({
            'total_states_processed': 0,
            'memory_usage': 0,
            'computation_time': 0.0
        })
    
    @nnx.jit
    def _apply_generators(self, states: jnp.ndarray) -> jnp.ndarray:
        """Apply all generators to all states using NNX vectorization."""
        def apply_single_generator(gen, state):
            return state[gen]  # Permutation application
        
        # Use NNX vmap for efficient generator application
        expanded = nnx.vmap(
            nnx.vmap(apply_single_generator, in_axes=(None, 0)),
            in_axes=(0, None)
        )(self.generators.value, states)
        
        return expanded.reshape(-1, states.shape[-1])
    
    @nnx.jit
    def _bfs_step(self, current_layer: jnp.ndarray, visited: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single BFS step using NNX operations."""
        
        # Apply all generators to current layer
        next_states = self._apply_generators(current_layer)
        
        # Remove duplicates and visited states using tensor ops module
        unique_states, _ = self.tensor_ops.unique_with_indices(next_states)
        is_visited = self.tensor_ops.isin_via_searchsorted(unique_states, visited)
        new_states = unique_states[~is_visited]
        
        # Update visited set
        updated_visited = jnp.concatenate([visited, new_states])
        
        # Update metrics
        self.metrics.value['total_states_processed'] += len(new_states)
        
        return new_states, updated_visited
    
    def initialize_bfs(self):
        """Initialize BFS state."""
        start_state = jnp.array([self.graph.central_state])
        self.bfs_state.value.update({
            'current_layer': start_state,
            'visited': start_state.copy(),
            'layer_sizes': [1],
            'diameter': 0
        })
    
    @nnx.jit
    def step(self) -> bool:
        """Perform one BFS step. Returns True if more layers exist."""
        current_layer = self.bfs_state.value['current_layer']
        visited = self.bfs_state.value['visited']
        
        new_layer, updated_visited = self._bfs_step(current_layer, visited)
        
        # Update state
        self.bfs_state.value.update({
            'current_layer': new_layer,
            'visited': updated_visited,
            'diameter': self.bfs_state.value['diameter'] + 1
        })
        
        if len(new_layer) > 0:
            self.bfs_state.value['layer_sizes'].append(len(new_layer))
            return True
        return False
    
    def run_full_bfs(self, max_diameter: int = 1000000) -> List[int]:
        """Run complete BFS and return layer sizes."""
        self.initialize_bfs()
        
        for _ in range(max_diameter):
            if not self.step():
                break
        
        return self.bfs_state.value['layer_sizes']
    
    def get_metrics(self) -> dict:
        """Get BFS performance metrics."""
        return dict(self.metrics.value)

def nnx_bfs(graph: CayleyGraph, max_diameter: int = 1000000) -> List[int]:
    """NNX-accelerated BFS implementation with automatic fallback."""
    
    try:
        # Try NNX implementation
        backend = NNXBackend()
        if backend.is_available():
            bfs_module = NNXBFSModule(graph, backend, nnx.Rngs(42))
            return bfs_module.run_full_bfs(max_diameter)
    except Exception as e:
        print(f"NNX BFS failed: {e}. Falling back to CPU implementation.")
    
    # Fallback to CPU implementation
    from .bfs_numpy import bfs_numpy
    return bfs_numpy(graph, max_diameter)
```

### NNX Beam Search

```python
# cayleypy/nnx_beam_search.py
from flax import nnx
import jax.numpy as jnp
from typing import Optional
from .beam_search_result import BeamSearchResult
from .nnx_hasher import NNXStateHasher
from .nnx_tensor_ops import TensorOpsModule

class NNXBeamSearchModule(nnx.Module):
    """NNX-based beam search with integrated state management and neural network predictor."""
    
    def __init__(self, graph, predictor_model: nnx.Module, beam_width: int, backend: NNXBackend, rngs: nnx.Rngs):
        self.graph = graph
        self.beam_width = beam_width
        self.backend = backend
        
        # Integrated predictor as NNX module
        self.predictor = predictor_model
        
        # Hash module for deduplication
        self.hasher = NNXStateHasher(len(graph.central_state), backend, rngs)
        
        # Tensor operations module
        self.tensor_ops = TensorOpsModule(backend, rngs)
        
        # Generators as NNX parameters
        self.generators = nnx.Param(
            jnp.array(graph.definition.generators_permutations),
            sharding=backend.sharding
        )
        
        # Beam search state
        self.search_state = nnx.Variable({
            'current_beam': None,
            'iteration': 0,
            'best_scores': None,
            'path_history': [],
            'target_found': False
        })
        
        # Search metrics
        self.metrics = nnx.Variable({
            'states_expanded': 0,
            'duplicates_removed': 0,
            'predictor_calls': 0,
            'memory_peak': 0
        })
    
    @nnx.jit
    def _expand_beam(self, beam_states: jnp.ndarray) -> jnp.ndarray:
        """Expand beam by applying all generators using NNX vectorization."""
        def apply_generator_to_state(state, gen):
            return state[gen]  # Permutation application
        
        # Use nested NNX vmap for efficient expansion
        expanded = nnx.vmap(
            nnx.vmap(apply_generator_to_state, in_axes=(0, None)),
            in_axes=(None, 0)
        )(beam_states, self.generators.value)
        
        # Reshape to flat array of states
        expanded_flat = expanded.reshape(-1, beam_states.shape[-1])
        
        # Update metrics
        self.metrics.value['states_expanded'] += len(expanded_flat)
        
        return expanded_flat
    
    @nnx.jit
    def _deduplicate_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Remove duplicate states using hash-based deduplication."""
        # Hash all states
        hashes = self.hasher.hash_batch(states)
        
        # Find unique hashes
        unique_hashes, unique_indices = self.tensor_ops.unique_with_indices(hashes)
        unique_states = states[unique_indices]
        
        # Update metrics
        duplicates_removed = len(states) - len(unique_states)
        self.metrics.value['duplicates_removed'] += duplicates_removed
        
        return unique_states
    
    @nnx.jit
    def _score_and_select(self, states: jnp.ndarray) -> jnp.ndarray:
        """Score states and select top k using integrated predictor."""
        # Score states using the integrated predictor
        scores = self.predictor(states)
        self.metrics.value['predictor_calls'] += len(states)
        
        # Select top k states
        if len(states) <= self.beam_width:
            return states
        
        top_indices = jnp.argsort(scores)[:self.beam_width]
        return states[top_indices]
    
    @nnx.jit
    def _check_target(self, states: jnp.ndarray, target: jnp.ndarray) -> bool:
        """Check if target state is in the current beam."""
        matches = nnx.vmap(
            lambda state: jnp.array_equal(state, target)
        )(states)
        return jnp.any(matches)
    
    def initialize_search(self, start_state: jnp.ndarray, target_state: jnp.ndarray):
        """Initialize beam search state."""
        self.search_state.value.update({
            'current_beam': jnp.array([start_state]),
            'target_state': target_state,
            'iteration': 0,
            'target_found': False,
            'path_history': [start_state]
        })
        
        # Reset metrics
        self.metrics.value.update({
            'states_expanded': 0,
            'duplicates_removed': 0,
            'predictor_calls': 0,
            'memory_peak': 0
        })
    
    @nnx.jit
    def step(self) -> bool:
        """Perform one beam search step. Returns True if search should continue."""
        current_beam = self.search_state.value['current_beam']
        target_state = self.search_state.value['target_state']
        
        # Expand current beam
        expanded_states = self._expand_beam(current_beam)
        
        # Remove duplicates
        unique_states = self._deduplicate_states(expanded_states)
        
        # Check if target is found
        if self._check_target(unique_states, target_state):
            self.search_state.value['target_found'] = True
            return False
        
        # Score and select top k states
        next_beam = self._score_and_select(unique_states)
        
        # Update search state
        self.search_state.value.update({
            'current_beam': next_beam,
            'iteration': self.search_state.value['iteration'] + 1
        })
        
        # Continue if beam is not empty
        return len(next_beam) > 0
    
    def search(self, start_state: jnp.ndarray, target_state: jnp.ndarray, 
               max_iterations: int = 1000) -> BeamSearchResult:
        """Perform complete beam search."""
        self.initialize_search(start_state, target_state)
        
        for _ in range(max_iterations):
            if not self.step():
                break
        
        # Create result
        if self.search_state.value['target_found']:
            return BeamSearchResult(
                path_found=True, 
                path_length=self.search_state.value['iteration']
            )
        else:
            return BeamSearchResult(path_found=False, path_length=-1)
    
    def get_search_metrics(self) -> dict:
        """Get detailed search metrics."""
        return {
            **dict(self.metrics.value),
            'hash_stats': self.hasher.get_cache_stats(),
            'current_beam_size': len(self.search_state.value.get('current_beam', [])),
            'iteration': self.search_state.value.get('iteration', 0)
        }

def nnx_beam_search(graph, predictor_model: nnx.Module, start_state: jnp.ndarray, 
                    target_state: jnp.ndarray, beam_width: int = 1000, 
                    max_iterations: int = 1000) -> BeamSearchResult:
    """High-level NNX beam search function with automatic fallback."""
    
    try:
        backend = NNXBackend()
        if backend.is_available():
            beam_search = NNXBeamSearchModule(
                graph, predictor_model, beam_width, backend, nnx.Rngs(42)
            )
            return beam_search.search(start_state, target_state, max_iterations)
    except Exception as e:
        print(f"NNX beam search failed: {e}. Falling back to CPU implementation.")
        # Fallback to existing beam search implementation
        return graph.beam_search(start_state, target_state, beam_width, max_iterations)
```

### Unified NNX Predictor System

```python
# cayleypy/nnx_predictor.py
from flax import nnx
import jax.numpy as jnp
import optax
from typing import Dict, List, Optional
from .predictor import Predictor

class NNXPredictorModule(nnx.Module):
    """Unified NNX predictor with multiple architecture support."""
    
    def __init__(self, input_size: int, architecture: str, config: Dict, 
                 backend: NNXBackend, rngs: nnx.Rngs):
        self.input_size = input_size
        self.architecture = architecture
        self.backend = backend
        
        # Create model based on architecture
        if architecture == "resmlp":
            self.model = self._create_resmlp(config, rngs)
        elif architecture == "transformer":
            self.model = self._create_transformer(config, rngs)
        elif architecture == "cnn":
            self.model = self._create_cnn(config, rngs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Training state
        self.training_state = nnx.Variable({
            'epoch': 0,
            'best_loss': float('inf'),
            'training_history': []
        })
        
        # Performance metrics
        self.metrics = nnx.Variable({
            'inference_time': 0.0,
            'batch_sizes_processed': [],
            'memory_usage': 0
        })
    
    def _create_resmlp(self, config: Dict, rngs: nnx.Rngs) -> nnx.Module:
        """Create ResidualMLP architecture."""
        class ResidualBlock(nnx.Module):
            def __init__(self, size: int, dropout_rate: float, rngs: nnx.Rngs):
                self.linear1 = nnx.Linear(size, size, rngs=rngs)
                self.linear2 = nnx.Linear(size, size, rngs=rngs)
                self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
                self.norm = nnx.LayerNorm(size, rngs=rngs)
            
            def __call__(self, x):
                residual = x
                x = self.norm(x)
                x = nnx.relu(self.linear1(x))
                x = self.dropout(x)
                x = self.linear2(x)
                return x + residual
        
        class ResMLP(nnx.Module):
            def __init__(self, input_size: int, hidden_sizes: List[int], 
                         num_residual_blocks: int, dropout_rate: float, rngs: nnx.Rngs):
                # Input projection
                self.input_proj = nnx.Linear(input_size, hidden_sizes[0], rngs=rngs)
                
                # Hidden layers
                self.hidden_layers = []
                for i in range(len(hidden_sizes) - 1):
                    self.hidden_layers.append(
                        nnx.Linear(hidden_sizes[i], hidden_sizes[i+1], rngs=rngs)
                    )
                
                # Residual blocks
                self.residual_blocks = []
                for _ in range(num_residual_blocks):
                    self.residual_blocks.append(
                        ResidualBlock(hidden_sizes[-1], dropout_rate, rngs)
                    )
                
                # Output layer
                self.output_layer = nnx.Linear(hidden_sizes[-1], 1, rngs=rngs)
                self.final_dropout = nnx.Dropout(dropout_rate, rngs=rngs)
            
            def __call__(self, x):
                # Input projection
                x = nnx.relu(self.input_proj(x))
                
                # Hidden layers
                for layer in self.hidden_layers:
                    x = nnx.relu(layer(x))
                
                # Residual blocks
                for block in self.residual_blocks:
                    x = block(x)
                
                # Output
                x = self.final_dropout(x)
                return self.output_layer(x)
        
        return ResMLP(
            input_size=self.input_size,
            hidden_sizes=config.get('hidden_sizes', [512, 256, 128]),
            num_residual_blocks=config.get('num_residual_blocks', 3),
            dropout_rate=config.get('dropout_rate', 0.1),
            rngs=rngs
        )
    
    def _create_transformer(self, config: Dict, rngs: nnx.Rngs) -> nnx.Module:
        """Create Transformer architecture for sequence modeling."""
        class TransformerBlock(nnx.Module):
            def __init__(self, d_model: int, num_heads: int, d_ff: int, rngs: nnx.Rngs):
                self.attention = nnx.MultiHeadAttention(
                    num_heads=num_heads,
                    in_features=d_model,
                    rngs=rngs
                )
                self.feed_forward = nnx.Sequential(
                    nnx.Linear(d_model, d_ff, rngs=rngs),
                    nnx.relu,
                    nnx.Linear(d_ff, d_model, rngs=rngs)
                )
                self.norm1 = nnx.LayerNorm(d_model, rngs=rngs)
                self.norm2 = nnx.LayerNorm(d_model, rngs=rngs)
            
            def __call__(self, x):
                # Self-attention with residual connection
                attn_out = self.attention(x)
                x = self.norm1(x + attn_out)
                
                # Feed-forward with residual connection
                ff_out = self.feed_forward(x)
                x = self.norm2(x + ff_out)
                
                return x
        
        class TransformerPredictor(nnx.Module):
            def __init__(self, input_size: int, d_model: int, num_layers: int, 
                         num_heads: int, d_ff: int, rngs: nnx.Rngs):
                self.input_proj = nnx.Linear(input_size, d_model, rngs=rngs)
                self.pos_encoding = nnx.Param(
                    jnp.zeros((1, input_size, d_model))  # Learnable positional encoding
                )
                
                self.transformer_blocks = []
                for _ in range(num_layers):
                    self.transformer_blocks.append(
                        TransformerBlock(d_model, num_heads, d_ff, rngs)
                    )
                
                self.output_layer = nnx.Linear(d_model, 1, rngs=rngs)
            
            def __call__(self, x):
                # Reshape input for sequence processing
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1, 1)  # Treat each element as a token
                
                # Project to model dimension
                x = self.input_proj(x)
                
                # Add positional encoding
                x = x + self.pos_encoding.value[:, :x.shape[1], :]
                
                # Apply transformer blocks
                for block in self.transformer_blocks:
                    x = block(x)
                
                # Global average pooling and output
                x = jnp.mean(x, axis=1)
                return self.output_layer(x)
        
        return TransformerPredictor(
            input_size=self.input_size,
            d_model=config.get('d_model', 256),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            d_ff=config.get('d_ff', 1024),
            rngs=rngs
        )
    
    def _create_cnn(self, config: Dict, rngs: nnx.Rngs) -> nnx.Module:
        """Create CNN architecture for permutation patterns."""
        class CNNPredictor(nnx.Module):
            def __init__(self, input_size: int, num_filters: List[int], 
                         kernel_sizes: List[int], rngs: nnx.Rngs):
                self.conv_layers = []
                in_channels = 1
                
                for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
                    self.conv_layers.append(
                        nnx.Conv(
                            in_features=in_channels,
                            out_features=filters,
                            kernel_size=(kernel_size,),
                            rngs=rngs
                        )
                    )
                    in_channels = filters
                
                # Calculate output size after convolutions
                conv_output_size = self._calculate_conv_output_size(input_size, kernel_sizes)
                final_features = num_filters[-1] * conv_output_size
                
                self.fc_layers = nnx.Sequential(
                    nnx.Linear(final_features, 256, rngs=rngs),
                    nnx.relu,
                    nnx.Linear(256, 64, rngs=rngs),
                    nnx.relu,
                    nnx.Linear(64, 1, rngs=rngs)
                )
            
            def _calculate_conv_output_size(self, input_size: int, kernel_sizes: List[int]) -> int:
                size = input_size
                for kernel_size in kernel_sizes:
                    size = size - kernel_size + 1  # Valid padding
                return max(1, size)
            
            def __call__(self, x):
                # Reshape for 1D convolution
                batch_size = x.shape[0]
                x = x.reshape(batch_size, -1, 1)  # (batch, length, channels)
                
                # Apply convolution layers
                for conv_layer in self.conv_layers:
                    x = nnx.relu(conv_layer(x))
                
                # Flatten and apply fully connected layers
                x = x.reshape(batch_size, -1)
                return self.fc_layers(x)
        
        return CNNPredictor(
            input_size=self.input_size,
            num_filters=config.get('num_filters', [32, 64, 128]),
            kernel_sizes=config.get('kernel_sizes', [3, 3, 3]),
            rngs=rngs
        )
    
    @nnx.jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with performance tracking."""
        import time
        start_time = time.time()
        
        result = self.model(x)
        
        # Update metrics
        self.metrics.value['inference_time'] += time.time() - start_time
        self.metrics.value['batch_sizes_processed'].append(x.shape[0])
        
        return result.squeeze(-1)
    
    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics."""
        batch_sizes = self.metrics.value['batch_sizes_processed']
        return {
            'total_inference_time': self.metrics.value['inference_time'],
            'average_batch_size': jnp.mean(jnp.array(batch_sizes)) if batch_sizes else 0,
            'total_batches_processed': len(batch_sizes),
            'memory_usage': self.metrics.value['memory_usage']
        }

class NNXPredictor(Predictor):
    """Unified NNX-based predictor with multiple architectures and training."""
    
    def __init__(self, graph, model_config: Dict):
        super().__init__(graph, model_config)
        
        # Initialize backend
        self.backend = NNXBackend()
        
        # Create predictor module
        self.predictor_module = NNXPredictorModule(
            input_size=len(graph.central_state),
            architecture=model_config.get('architecture', 'resmlp'),
            config=model_config,
            backend=self.backend,
            rngs=nnx.Rngs(model_config.get('seed', 42))
        )
        
        # Initialize optimizer
        optimizer_config = model_config.get('optimizer', {})
        learning_rate = optimizer_config.get('learning_rate', 1e-3)
        self.optimizer = nnx.Optimizer(
            self.predictor_module, 
            optax.adam(learning_rate)
        )
    
    @nnx.jit
    def predict_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Predict distances for a batch of states."""
        return self.predictor_module(states)
    
    @nnx.jit
    def train_step(self, states: jnp.ndarray, targets: jnp.ndarray) -> float:
        """Single training step with automatic differentiation."""
        def loss_fn(predictor):
            predictions = predictor(states)
            return jnp.mean((predictions - targets) ** 2)
        
        loss, grads = nnx.value_and_grad(loss_fn)(self.predictor_module)
        self.optimizer.update(grads)
        
        # Update training state
        self.predictor_module.training_state.value['training_history'].append(float(loss))
        
        return loss
    
    def train_epoch(self, train_data: List[tuple], validation_data: Optional[List[tuple]] = None) -> Dict:
        """Train for one epoch with validation."""
        epoch_losses = []
        
        for states, targets in train_data:
            loss = self.train_step(states, targets)
            epoch_losses.append(float(loss))
        
        # Validation
        val_loss = None
        if validation_data:
            val_losses = []
            for states, targets in validation_data:
                predictions = self.predict_batch(states)
                val_loss_batch = jnp.mean((predictions - targets) ** 2)
                val_losses.append(float(val_loss_batch))
            val_loss = jnp.mean(jnp.array(val_losses))
        
        # Update training state
        epoch = self.predictor_module.training_state.value['epoch'] + 1
        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        
        self.predictor_module.training_state.value.update({
            'epoch': epoch,
            'best_loss': min(self.predictor_module.training_state.value['best_loss'], avg_train_loss)
        })
        
        return {
            'epoch': epoch,
            'train_loss': float(avg_train_loss),
            'val_loss': float(val_loss) if val_loss is not None else None,
            'performance_metrics': self.predictor_module.get_performance_metrics()
        }
```

## Data Models

### NNX Configuration

```python
@dataclass
class NNXConfig:
    """Configuration for NNX backend with comprehensive settings."""
    device_type: str = "auto"  # "cpu", "gpu", "tpu", "auto"
    enable_jit: bool = True
    enable_x64: bool = False
    memory_fraction: float = 0.9
    optimization_flags: dict = field(default_factory=dict)
    chunk_size: int = 10000
    
    # NNX-specific configurations
    sharding_strategy: str = "auto"  # "auto", "data_parallel", "model_parallel", "pipeline_parallel"
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
    compilation_cache: bool = True
    
    # Training configurations
    default_optimizer: str = "adam"
    default_learning_rate: float = 1e-3
    gradient_clipping: Optional[float] = None
    
    # Memory management
    enable_memory_profiling: bool = False
    memory_efficient_attention: bool = True
    activation_checkpointing: bool = False
    
    def to_env_vars(self) -> dict:
        """Convert to environment variables for XLA configuration."""
        env_vars = {}
        if self.optimization_flags:
            xla_flags = " ".join(f"--{k}={v}" for k, v in self.optimization_flags.items())
            env_vars["XLA_FLAGS"] = xla_flags
        
        # Add NNX-specific environment variables
        if self.mixed_precision:
            env_vars["JAX_ENABLE_X64"] = "false"
        if self.compilation_cache:
            env_vars["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
            
        return env_vars
    
    def get_sharding_config(self) -> dict:
        """Get sharding configuration for distributed training."""
        return {
            'strategy': self.sharding_strategy,
            'gradient_checkpointing': self.gradient_checkpointing,
            'memory_efficient_attention': self.memory_efficient_attention
        }
```

### Performance Metrics

```python
@dataclass
class NNXPerformanceMetrics:
    """Comprehensive performance metrics for NNX operations."""
    operation_name: str
    cpu_time: float
    nnx_time: float
    speedup: float
    memory_usage: int
    device_type: str
    batch_size: int
    
    # NNX-specific metrics
    compilation_time: float = 0.0
    gradient_computation_time: float = 0.0
    parameter_update_time: float = 0.0
    state_management_overhead: float = 0.0
    
    # Memory breakdown
    parameter_memory: int = 0
    activation_memory: int = 0
    gradient_memory: int = 0
    cache_memory: int = 0
    
    # Training metrics
    forward_pass_time: float = 0.0
    backward_pass_time: float = 0.0
    optimizer_step_time: float = 0.0
    
    @property
    def efficiency_ratio(self) -> float:
        """Calculate efficiency ratio (speedup per GB)."""
        return self.speedup / (self.memory_usage / 1e9)
    
    @property
    def training_efficiency(self) -> float:
        """Calculate training efficiency (forward+backward speedup)."""
        total_training_time = self.forward_pass_time + self.backward_pass_time
        return self.speedup / max(total_training_time, 1e-6)
    
    @property
    def memory_breakdown(self) -> dict:
        """Get detailed memory usage breakdown."""
        total_memory = max(self.memory_usage, 1)
        return {
            'parameters': self.parameter_memory / total_memory,
            'activations': self.activation_memory / total_memory,
            'gradients': self.gradient_memory / total_memory,
            'cache': self.cache_memory / total_memory,
            'other': max(0, 1 - (self.parameter_memory + self.activation_memory + 
                               self.gradient_memory + self.cache_memory) / total_memory)
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            'operation_name': self.operation_name,
            'performance': {
                'cpu_time': self.cpu_time,
                'nnx_time': self.nnx_time,
                'speedup': self.speedup,
                'efficiency_ratio': self.efficiency_ratio,
                'training_efficiency': self.training_efficiency
            },
            'memory': {
                'total_usage': self.memory_usage,
                'breakdown': self.memory_breakdown
            },
            'timing_breakdown': {
                'compilation': self.compilation_time,
                'forward_pass': self.forward_pass_time,
                'backward_pass': self.backward_pass_time,
                'gradient_computation': self.gradient_computation_time,
                'parameter_update': self.parameter_update_time,
                'optimizer_step': self.optimizer_step_time,
                'state_management_overhead': self.state_management_overhead
            },
            'system': {
                'device_type': self.device_type,
                'batch_size': self.batch_size
            }
        }
```

## Error Handling

### NNX-Specific Error Handling

```python
class NNXAccelerationError(Exception):
    """Base exception for NNX acceleration errors."""
    pass

class NNXDeviceError(NNXAccelerationError):
    """Raised when NNX device operations fail."""
    pass

class NNXMemoryError(NNXAccelerationError):
    """Raised when NNX operations exceed memory limits."""
    pass

class NNXCompilationError(NNXAccelerationError):
    """Raised when NNX JIT compilation fails."""
    pass

class NNXStateError(NNXAccelerationError):
    """Raised when NNX state management encounters issues."""
    pass

class NNXShardingError(NNXAccelerationError):
    """Raised when NNX sharding configuration is invalid."""
    pass

def with_nnx_fallback(cpu_func):
    """Decorator to provide CPU fallback for NNX operations with comprehensive error handling."""
    def decorator(nnx_func):
        def wrapper(*args, **kwargs):
            try:
                backend = NNXBackend()
                if backend.is_available():
                    return nnx_func(*args, **kwargs)
                else:
                    logger.info("NNX backend not available. Using CPU implementation.")
                    return cpu_func(*args, **kwargs)
            except (NNXDeviceError, NNXMemoryError, NNXCompilationError, NNXStateError, NNXShardingError) as e:
                logger.warning(f"NNX operation failed: {e}. Falling back to CPU.")
                return cpu_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Unexpected error in NNX operation: {e}. Falling back to CPU.")
                return cpu_func(*args, **kwargs)
        return wrapper
    return decorator

class NNXErrorHandler:
    """Centralized error handling for NNX operations."""
    
    def __init__(self):
        self.error_counts = {}
        self.fallback_counts = {}
    
    def handle_error(self, operation_name: str, error: Exception, fallback_func=None):
        """Handle NNX errors with logging and optional fallback."""
        self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
        
        if isinstance(error, NNXMemoryError):
            logger.warning(f"Memory error in {operation_name}: {error}")
            if fallback_func:
                logger.info(f"Attempting memory-efficient fallback for {operation_name}")
                return self._try_fallback(operation_name, fallback_func)
        
        elif isinstance(error, NNXCompilationError):
            logger.warning(f"Compilation error in {operation_name}: {error}")
            if fallback_func:
                logger.info(f"Using non-JIT fallback for {operation_name}")
                return self._try_fallback(operation_name, fallback_func)
        
        elif isinstance(error, NNXStateError):
            logger.error(f"State management error in {operation_name}: {error}")
            # State errors are more serious, may need to reset module state
            raise error
        
        else:
            logger.error(f"Unexpected error in {operation_name}: {error}")
            if fallback_func:
                return self._try_fallback(operation_name, fallback_func)
            raise error
    
    def _try_fallback(self, operation_name: str, fallback_func):
        """Try fallback function with error tracking."""
        try:
            self.fallback_counts[operation_name] = self.fallback_counts.get(operation_name, 0) + 1
            return fallback_func()
        except Exception as fallback_error:
            logger.error(f"Fallback also failed for {operation_name}: {fallback_error}")
            raise fallback_error
    
    def get_error_summary(self) -> dict:
        """Get summary of errors and fallbacks."""
        return {
            'error_counts': dict(self.error_counts),
            'fallback_counts': dict(self.fallback_counts),
            'total_errors': sum(self.error_counts.values()),
            'total_fallbacks': sum(self.fallback_counts.values())
        }

# Global error handler instance
nnx_error_handler = NNXErrorHandler()
```

## Testing Strategy

### Unit Tests
- **JAX Operation Tests**: Verify correctness of JAX implementations against NumPy equivalents
- **Device Detection Tests**: Test automatic device selection and fallback mechanisms
- **Memory Management Tests**: Validate chunked processing and memory efficiency
- **Performance Tests**: Benchmark JAX vs CPU implementations

### Integration Tests
- **End-to-End BFS Tests**: Compare JAX BFS results with existing implementations
- **Beam Search Integration**: Test JAX beam search with neural network predictors
- **Multi-Device Tests**: Validate behavior across different hardware configurations

### Performance Benchmarks
- **Scalability Tests**: Measure performance across different problem sizes
- **Memory Usage Tests**: Monitor memory consumption patterns
- **Hardware-Specific Tests**: Validate optimizations on GPU and TPU hardware

### Test Infrastructure

```python
# tests/test_nnx_acceleration.py
import pytest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from cayleypy.nnx_tensor_ops import TensorOpsModule
from cayleypy.nnx_bfs import nnx_bfs, NNXBFSModule
from cayleypy.nnx_predictor import NNXPredictor
from cayleypy.bfs_numpy import bfs_numpy

class TestNNXAcceleration:
    
    @pytest.fixture
    def nnx_backend(self):
        """Create NNX backend for testing."""
        return NNXBackend()
    
    @pytest.fixture
    def tensor_ops_module(self, nnx_backend):
        """Create tensor operations module for testing."""
        return TensorOpsModule(nnx_backend, nnx.Rngs(42))
    
    @pytest.mark.skipif(not jax.devices("gpu"), reason="GPU not available")
    def test_nnx_tensor_ops_correctness(self, tensor_ops_module):
        """Test NNX tensor operations against NumPy."""
        arr = jnp.array(np.random.randint(0, 100, 1000))
        
        # Test unique operation
        nnx_unique, _ = tensor_ops_module.unique_with_indices(arr)
        numpy_unique = np.unique(arr)
        np.testing.assert_array_equal(nnx_unique, numpy_unique)
        
        # Test isin operation
        test_elements = jnp.array([1, 5, 10, 50, 99])
        nnx_isin = tensor_ops_module.isin_via_searchsorted(test_elements, arr)
        numpy_isin = np.isin(test_elements, arr)
        np.testing.assert_array_equal(nnx_isin, numpy_isin)
    
    def test_nnx_bfs_correctness(self):
        """Test NNX BFS against NumPy BFS."""
        from cayleypy import PermutationGroups, CayleyGraph
        
        graph = CayleyGraph(PermutationGroups.lrx(8))
        nnx_result = nnx_bfs(graph)
        numpy_result = bfs_numpy(graph)
        
        assert nnx_result == numpy_result
    
    def test_nnx_bfs_module_state_management(self, nnx_backend):
        """Test NNX BFS module state management."""
        from cayleypy import PermutationGroups, CayleyGraph
        
        graph = CayleyGraph(PermutationGroups.lrx(6))
        bfs_module = NNXBFSModule(graph, nnx_backend, nnx.Rngs(42))
        
        # Test initialization
        bfs_module.initialize_bfs()
        assert bfs_module.bfs_state.value['diameter'] == 0
        assert len(bfs_module.bfs_state.value['layer_sizes']) == 1
        
        # Test stepping
        has_more = bfs_module.step()
        assert has_more
        assert bfs_module.bfs_state.value['diameter'] == 1
        
        # Test metrics
        metrics = bfs_module.get_metrics()
        assert 'total_states_processed' in metrics
        assert metrics['total_states_processed'] > 0
    
    def test_nnx_predictor_architectures(self, nnx_backend):
        """Test different NNX predictor architectures."""
        from cayleypy import PermutationGroups, CayleyGraph
        
        graph = CayleyGraph(PermutationGroups.lrx(6))
        
        # Test ResMLParchitecture
        resmlp_config = {
            'architecture': 'resmlp',
            'hidden_sizes': [64, 32],
            'num_residual_blocks': 2
        }
        resmlp_predictor = NNXPredictor(graph, resmlp_config)
        
        # Test prediction
        test_states = jnp.array([graph.central_state, graph.central_state])
        predictions = resmlp_predictor.predict_batch(test_states)
        assert predictions.shape == (2,)
        
        # Test training step
        targets = jnp.array([0.0, 1.0])
        loss = resmlp_predictor.train_step(test_states, targets)
        assert isinstance(loss, (float, jnp.ndarray))
    
    def test_nnx_beam_search_integration(self, nnx_backend):
        """Test NNX beam search with predictor integration."""
        from cayleypy import PermutationGroups, CayleyGraph
        from cayleypy.nnx_beam_search import NNXBeamSearchModule
        
        graph = CayleyGraph(PermutationGroups.lrx(6))
        
        # Create simple predictor
        predictor_config = {
            'architecture': 'resmlp',
            'hidden_sizes': [32, 16],
            'num_residual_blocks': 1
        }
        predictor = NNXPredictor(graph, predictor_config).predictor_module
        
        # Create beam search module
        beam_search = NNXBeamSearchModule(
            graph, predictor, beam_width=10, backend=nnx_backend, rngs=nnx.Rngs(42)
        )
        
        # Test search
        start_state = jnp.array(graph.central_state)
        target_state = jnp.array([1, 0, 2, 3, 4, 5])  # Simple permutation
        
        result = beam_search.search(start_state, target_state, max_iterations=5)
        assert hasattr(result, 'path_found')
        
        # Test metrics
        metrics = beam_search.get_search_metrics()
        assert 'states_expanded' in metrics
        assert 'hash_stats' in metrics
    
    @pytest.mark.performance
    def test_nnx_performance_improvement(self, nnx_backend):
        """Benchmark NNX vs NumPy performance."""
        from cayleypy import PermutationGroups, CayleyGraph
        import time
        
        graph = CayleyGraph(PermutationGroups.lrx(10))
        
        # Benchmark BFS
        start_time = time.time()
        numpy_result = bfs_numpy(graph, max_diameter=5)
        numpy_time = time.time() - start_time
        
        start_time = time.time()
        nnx_result = nnx_bfs(graph, max_diameter=5)
        nnx_time = time.time() - start_time
        
        # Verify correctness
        assert numpy_result == nnx_result
        
        # Check for performance improvement (may not always be faster for small graphs)
        speedup = numpy_time / max(nnx_time, 1e-6)
        print(f"NNX speedup: {speedup:.2f}x")
    
    @pytest.mark.memory
    def test_nnx_memory_efficiency(self, nnx_backend):
        """Test NNX memory efficiency features."""
        from cayleypy.nnx_hasher import OptimizedNNXStateHasher
        
        # Test chunked processing
        hasher = OptimizedNNXStateHasher(10, nnx_backend, nnx.Rngs(42))
        
        # Create large batch
        large_batch = jnp.ones((50000, 10))
        
        # Test chunked hashing
        hashes = hasher.hash_large_batch(large_batch, chunk_size=1000)
        assert hashes.shape == (50000,)
        
        # Test cache statistics
        stats = hasher.get_cache_stats()
        assert 'cache_hit_rate' in stats
        assert 'total_hashes' in stats
    
    def test_nnx_error_handling(self, nnx_backend):
        """Test NNX error handling and fallback mechanisms."""
        from cayleypy.nnx_bfs import nnx_bfs
        from cayleypy import PermutationGroups, CayleyGraph
        
        # Test with invalid graph (should fallback gracefully)
        graph = CayleyGraph(PermutationGroups.lrx(8))
        
        # Mock a failure scenario
        original_available = nnx_backend.is_available
        nnx_backend.is_available = lambda: False
        
        try:
            result = nnx_bfs(graph, max_diameter=3)
            # Should still work via fallback
            assert isinstance(result, list)
            assert len(result) > 0
        finally:
            nnx_backend.is_available = original_available
    
    @pytest.mark.distributed
    @pytest.mark.skipif(jax.device_count() < 2, reason="Multiple devices required")
    def test_nnx_distributed_operations(self, nnx_backend):
        """Test NNX distributed operations across multiple devices."""
        # Test sharding and distributed computation
        if jax.device_count() >= 2:
            # Test distributed BFS or beam search
            pass  # Implementation depends on multi-device setup

class TestNNXIntegration:
    """Integration tests for NNX components working together."""
    
    def test_end_to_end_training_and_search(self):
        """Test complete pipeline: training predictor -> beam search."""
        from cayleypy import PermutationGroups, CayleyGraph
        from cayleypy.nnx_beam_search import nnx_beam_search
        
        graph = CayleyGraph(PermutationGroups.lrx(6))
        
        # Create and train predictor
        predictor_config = {
            'architecture': 'resmlp',
            'hidden_sizes': [32, 16],
            'num_residual_blocks': 1
        }
        predictor = NNXPredictor(graph, predictor_config)
        
        # Generate some training data
        states = jnp.array([graph.central_state] * 10)
        targets = jnp.array([0.0] * 10)
        
        # Train for a few steps
        for _ in range(5):
            loss = predictor.train_step(states, targets)
        
        # Use trained predictor in beam search
        start_state = jnp.array(graph.central_state)
        target_state = jnp.array([1, 0, 2, 3, 4, 5])
        
        result = nnx_beam_search(
            graph, predictor.predictor_module, start_state, target_state,
            beam_width=5, max_iterations=10
        )
        
        assert hasattr(result, 'path_found')
```

## Implementation Phases

### Phase 1: Core NNX Infrastructure
1. Implement NNXBackend configuration system with device mesh and sharding
2. Create NNX tensor operations module with state management
3. Implement NNX hash functions with caching and performance tracking
4. Add device detection and comprehensive fallback mechanisms

### Phase 2: NNX BFS Acceleration
1. Implement NNXBFSModule with stateful BFS algorithms
2. Add memory-efficient chunked processing using NNX transforms
3. Create performance benchmarks comparing NNX vs CPU implementations
4. Integrate with existing CayleyGraph API using fallback decorators

### Phase 3: NNX Beam Search Acceleration
1. Implement NNXBeamSearchModule with integrated predictor support
2. Add hash-based duplicate detection using NNX state management
3. Optimize memory usage for large beams with NNX sharding
4. Create comprehensive beam search benchmarks

### Phase 4: Unified NNX Neural Network System
1. Implement NNXPredictorModule with multiple architecture support (ResMLPTransformer, CNN)
2. Create unified training pipeline using NNX transforms and optimizers
3. Add automatic state management and checkpointing
4. Implement end-to-end training examples with performance monitoring

### Phase 5: Advanced NNX Features and Optimization
1. Add distributed training support using NNX sharding strategies
2. Implement gradient checkpointing and memory optimization
3. Create comprehensive benchmarking suite with detailed metrics
4. Add advanced error handling and recovery mechanisms
5. Implement mixed precision training and hardware-specific optimizations

### Phase 6: Documentation and Integration
1. Create comprehensive documentation with NNX-specific examples
2. Add integration guides for existing CayleyPy workflows
3. Implement migration tools from existing predictors to NNX
4. Create performance tuning guides for different hardware configurations

This revised design leverages Flax NNX comprehensively throughout the system, providing better state management, more intuitive APIs, and seamless integration between different components. The NNX-centric approach simplifies the codebase while providing more powerful features for neural network integration and distributed computing.