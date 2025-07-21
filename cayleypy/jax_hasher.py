"""JAX-based state hashing system for CayleyPy.

This module provides JAX implementations of state hashing functionality,
optimized for TPU/GPU computation with vectorized operations and JIT compilation.
"""

import math
import random
from typing import Callable, Optional, TYPE_CHECKING, Union, Any

if TYPE_CHECKING:
    import jax.numpy as jnp
    JaxArray = jnp.ndarray
else:
    JaxArray = Any

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax
    import jax.random as jrandom
    from jax.experimental import pjit
    from jax.experimental.pjit import PartitionSpec as P
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    jrandom = None
    lax = None
    pjit = None
    P = None
    # Create dummy decorators when JAX is not available
    def jit(func):
        return func
    def vmap(func):
        return func

from .jax_tensor_ops import (
    tensor_split, concatenate_arrays, stack_arrays, 
    full_like, arange, chunked_operation
)

if TYPE_CHECKING:
    from cayleypy import JAXCayleyGraph

MAX_INT = 2**62


@jit
def _splitmix64_jax(x: JaxArray) -> JaxArray:
    """JAX implementation of SplitMix64 hash function.
    
    This is a high-quality pseudorandom number generator that's commonly used
    for hash functions. Ported from the PyTorch version for compatibility.
    
    Args:
        x: Input values to hash
        
    Returns:
        Hashed values
    """
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    x = x ^ (x >> 31)
    return x


class JAXStateHasher:
    """JAX-based helper class to hash states efficiently.
    
    This class provides vectorized hashing operations optimized for TPU/GPU
    computation, with support for different state encodings and chunked processing
    for memory efficiency.
    """

    def __init__(self, state_size: int, random_seed: Optional[int] = None, 
                 chunk_size: int = 2**18, use_string_encoder: bool = False):
        """Initialize the state hasher.
        
        Args:
            state_size: Size of encoded states
            random_seed: Random seed for hash function
            chunk_size: Chunk size for memory-efficient processing
            use_string_encoder: Whether states use string encoding (bit-packed)
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not available. Install with: pip install jax[tpu] or pip install jax[cuda]"
            )
        
        self.state_size = state_size
        self.chunk_size = chunk_size
        self.use_string_encoder = use_string_encoder
        
        # If states are single int64, use identity function
        self.is_identity = (state_size == 1)
        
        if self.is_identity:
            self.make_hashes = self._identity_hash
            return
        
        # Initialize random seed
        self.seed = random_seed if random_seed is not None else random.randint(-MAX_INT, MAX_INT)
        
        # Choose hash function based on encoding type
        if use_string_encoder:
            # For bit-encoded states, use SplitMix64 to avoid collisions
            self.make_hashes = self._hash_splitmix64
        else:
            # For regular states, use dot product with random vector
            self._initialize_vector_hasher()
            self.make_hashes = self._hash_dot_product

    def _initialize_vector_hasher(self) -> None:
        """Initialize random vector for dot product hashing."""
        # Create PRNG key from seed
        key = jrandom.PRNGKey(self.seed)
        
        # Generate random vector for hashing
        self.vec_hasher = jrandom.randint(
            key, 
            shape=(self.state_size, 1), 
            minval=-MAX_INT, 
            maxval=MAX_INT,
            dtype=jnp.int64
        )

    @jit
    def _identity_hash(self, states: JaxArray) -> JaxArray:
        """Identity hash function for single-element states.
        
        Args:
            states: Input states
            
        Returns:
            Flattened states (identity transformation)
        """
        return states.reshape(-1)

    def _hash_dot_product(self, states: JaxArray) -> JaxArray:
        """Hash states using dot product with random vector.
        
        Args:
            states: Input states of shape (batch_size, state_size)
            
        Returns:
            Hash values of shape (batch_size,)
        """
        if states.shape[0] <= self.chunk_size:
            return self._hash_dot_product_chunk(states)
        else:
            # Process in chunks for memory efficiency
            return chunked_operation(
                states, 
                self._hash_dot_product_chunk, 
                self.chunk_size
            )

    @jit
    def _hash_dot_product_chunk(self, states: JaxArray) -> JaxArray:
        """Hash a chunk of states using dot product.
        
        Optimized with vectorized operations for TPU efficiency.
        
        Args:
            states: Chunk of states
            
        Returns:
            Hash values for the chunk
        """
        # Use vectorized dot product for better TPU utilization
        vectorized_dot = vmap(lambda state: jnp.dot(state, self.vec_hasher.flatten()))
        return vectorized_dot(states)

    def _hash_splitmix64(self, states: JaxArray) -> JaxArray:
        """Hash states using SplitMix64 algorithm.
        
        This is used for bit-encoded states to avoid hash collisions.
        
        Args:
            states: Input states of shape (batch_size, state_size)
            
        Returns:
            Hash values of shape (batch_size,)
        """
        if states.shape[0] <= self.chunk_size:
            return self._hash_splitmix64_chunk(states)
        else:
            # Process in chunks for memory efficiency
            return chunked_operation(
                states,
                self._hash_splitmix64_chunk,
                self.chunk_size
            )

    @jit
    def _hash_splitmix64_chunk(self, states: JaxArray) -> JaxArray:
        """Hash a chunk of states using SplitMix64.
        
        Optimized with lax.scan for better TPU performance.
        
        Args:
            states: Chunk of states of shape (batch_size, state_size)
            
        Returns:
            Hash values for the chunk
        """
        batch_size, state_size = states.shape
        
        # Use lax.scan for efficient iteration over state elements
        def scan_fn(h_carry, i):
            h_new = h_carry ^ _splitmix64_jax(states[:, i])
            h_new = h_new * 0x85EBCA6B
            return h_new, None
        
        # Initialize hash with seed
        h_init = jnp.full((batch_size,), self.seed, dtype=jnp.int64)
        final_h, _ = lax.scan(scan_fn, h_init, jnp.arange(state_size))
        
        return final_h

    def hash_states(self, states: JaxArray) -> JaxArray:
        """Hash a batch of states with TPU optimizations.
        
        Args:
            states: States to hash, shape (batch_size, state_size)
            
        Returns:
            Hash values, shape (batch_size,)
        """
        # Ensure states have correct shape
        if states.ndim == 1:
            states = states.reshape(1, -1)
        elif states.ndim > 2:
            states = states.reshape(-1, self.state_size)
        
        # Use vectorized hashing for better TPU performance
        if states.shape[0] > 1:
            return vectorized_hash_states(states, self)
        else:
            return self.make_hashes(states)

    def hash_single_state(self, state: JaxArray) -> int:
        """Hash a single state.
        
        Args:
            state: Single state to hash
            
        Returns:
            Hash value as Python int
        """
        if state.ndim == 0:
            state = state.reshape(1)
        elif state.ndim > 1:
            state = state.flatten()
        
        state_batch = state.reshape(1, -1)
        hash_result = self.make_hashes(state_batch)
        return int(hash_result[0])


class JAXBatchHasher:
    """Batch hasher for processing multiple state batches efficiently.
    
    This class is optimized for scenarios where you need to hash many
    batches of states, with automatic memory management and vectorization.
    """

    def __init__(self, hasher: JAXStateHasher, max_batch_size: int = 2**20):
        """Initialize batch hasher.
        
        Args:
            hasher: Base state hasher
            max_batch_size: Maximum batch size for processing
        """
        self.hasher = hasher
        self.max_batch_size = max_batch_size

    def hash_multiple_batches(self, state_batches: list) -> list:
        """Hash multiple batches of states.
        
        Args:
            state_batches: List of state arrays to hash
            
        Returns:
            List of hash arrays
        """
        results = []
        
        for batch in state_batches:
            if len(batch) > self.max_batch_size:
                # Split large batches
                num_chunks = (len(batch) + self.max_batch_size - 1) // self.max_batch_size
                chunks = jnp.array_split(batch, num_chunks, axis=0)
                chunk_hashes = [self.hasher.hash_states(chunk) for chunk in chunks]
                batch_hashes = concatenate_arrays(chunk_hashes, axis=0)
            else:
                batch_hashes = self.hasher.hash_states(batch)
            
            results.append(batch_hashes)
        
        return results

    def hash_and_concatenate(self, state_batches: list) -> JaxArray:
        """Hash multiple batches and concatenate results.
        
        Args:
            state_batches: List of state arrays to hash
            
        Returns:
            Concatenated hash array
        """
        hash_batches = self.hash_multiple_batches(state_batches)
        return concatenate_arrays(hash_batches, axis=0)


# Vectorized hashing functions using vmap
@jit
def vectorized_hash_states(states: JaxArray, hasher: JAXStateHasher) -> JaxArray:
    """Vectorized state hashing using vmap for maximum TPU efficiency.
    
    Args:
        states: Batch of states, shape (batch_size, state_size)
        hasher: JAXStateHasher instance
        
    Returns:
        Hash values, shape (batch_size,)
    """
    _check_jax_available()
    
    def single_state_hash(state):
        """Hash a single state vector."""
        if hasher.is_identity:
            return state[0] if len(state) > 0 else 0
        elif hasher.use_string_encoder:
            return _vectorized_splitmix64_single(state, hasher.seed)
        else:
            return jnp.dot(state, hasher.vec_hasher.flatten())
    
    # Use vmap for efficient vectorization across batch dimension
    vectorized_fn = vmap(single_state_hash, in_axes=0, out_axes=0)
    return vectorized_fn(states)


@jit
def _vectorized_splitmix64_single(state: JaxArray, seed: int) -> int:
    """Vectorized SplitMix64 hash for a single state."""
    h = seed
    
    def hash_element(carry, x):
        h = carry ^ _splitmix64_jax(x)
        h = h * 0x85EBCA6B
        return h, None
    
    final_h, _ = lax.scan(hash_element, h, state)
    return final_h


# TPU-optimized distributed hashing
@pjit(
    in_axis_resources=(P('batch', None), P()),
    out_axis_resources=P('batch')
)
def distributed_hash_states(states: JaxArray, hasher: JAXStateHasher) -> JaxArray:
    """Distributed state hashing across TPU cores.
    
    Args:
        states: Batch of states sharded across devices
        hasher: JAXStateHasher instance
        
    Returns:
        Hash values distributed across devices
    """
    if not JAX_AVAILABLE or pjit is None:
        return hasher.hash_states(states)
    
    return vectorized_hash_states(states, hasher)


# Memory-efficient batch processing
@jit
def memory_efficient_batch_hash(states: JaxArray, hasher: JAXStateHasher, 
                               max_batch_size: int = 2**16) -> JaxArray:
    """Memory-efficient batch hashing with automatic chunking.
    
    Args:
        states: Large batch of states
        hasher: JAXStateHasher instance
        max_batch_size: Maximum batch size per chunk
        
    Returns:
        Hash values for all states
    """
    _check_jax_available()
    
    def process_chunk(chunk):
        return vectorized_hash_states(chunk, hasher)
    
    batch_size = states.shape[0]
    
    if batch_size <= max_batch_size:
        return process_chunk(states)
    else:
        # Use lax.map for efficient chunked processing
        num_chunks = (batch_size + max_batch_size - 1) // max_batch_size
        chunks = jnp.array_split(states, num_chunks, axis=0)
        chunk_array = jnp.stack([jnp.pad(chunk, ((0, max_batch_size - chunk.shape[0]), (0, 0))) 
                                for chunk in chunks])
        
        results = lax.map(process_chunk, chunk_array)
        
        # Remove padding and concatenate
        valid_results = []
        for i, chunk in enumerate(chunks):
            valid_results.append(results[i][:chunk.shape[0]])
        
        return jnp.concatenate(valid_results, axis=0)


# Utility functions for hash management
def create_hash_function(state_size: int, encoding_type: str = "regular", 
                        random_seed: Optional[int] = None) -> JAXStateHasher:
    """Create a hash function for given state configuration.
    
    Args:
        state_size: Size of state vectors
        encoding_type: Type of encoding ("regular" or "string")
        random_seed: Random seed for reproducibility
        
    Returns:
        Configured JAXStateHasher
    """
    use_string_encoder = (encoding_type == "string")
    return JAXStateHasher(
        state_size=state_size,
        random_seed=random_seed,
        use_string_encoder=use_string_encoder
    )


def hash_state_collection(states: Union[JaxArray, list], 
                         hasher: JAXStateHasher) -> JaxArray:
    """Hash a collection of states.
    
    Args:
        states: States to hash (array or list of arrays)
        hasher: Hasher to use
        
    Returns:
        Hash values
    """
    if isinstance(states, list):
        # Handle list of state arrays
        all_hashes = []
        for state_batch in states:
            batch_hashes = hasher.hash_states(jnp.array(state_batch))
            all_hashes.append(batch_hashes)
        return concatenate_arrays(all_hashes, axis=0)
    else:
        # Handle single array
        return hasher.hash_states(jnp.array(states))


# Performance optimization utilities
@jit
def fast_hash_comparison(hashes1: JaxArray, hashes2: JaxArray) -> JaxArray:
    """Fast comparison of hash arrays.
    
    Args:
        hashes1: First hash array
        hashes2: Second hash array
        
    Returns:
        Boolean array of equality comparisons
    """
    return hashes1 == hashes2


@jit
def find_hash_duplicates(hashes: JaxArray) -> tuple:
    """Find duplicate hashes in an array.
    
    Args:
        hashes: Array of hash values
        
    Returns:
        Tuple of (unique_hashes, inverse_indices, counts)
    """
    from .jax_tensor_ops import unique_with_indices
    return unique_with_indices(hashes, return_inverse=True, return_counts=True)


def benchmark_hash_performance(hasher: JAXStateHasher, 
                              test_states: JaxArray, 
                              num_iterations: int = 10,
                              use_distributed: bool = False) -> dict:
    """Benchmark hash performance with TPU optimization options.
    
    Args:
        hasher: Hasher to benchmark
        test_states: Test states for benchmarking
        num_iterations: Number of iterations to run
        use_distributed: Whether to use distributed TPU hashing
        
    Returns:
        Performance statistics
    """
    import time
    
    # Choose hashing function based on options
    if use_distributed and JAX_AVAILABLE and pjit is not None:
        hash_fn = lambda states: distributed_hash_states(states, hasher)
    else:
        hash_fn = lambda states: vectorized_hash_states(states, hasher)
    
    # Warm up JIT compilation
    _ = hash_fn(test_states[:100])
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        _ = hash_fn(test_states)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "states_per_second": len(test_states) / (sum(times) / len(times)),
        "method": "distributed" if use_distributed else "vectorized"
    }


# Advanced TPU optimization utilities
class TPUOptimizedHasher:
    """TPU-optimized wrapper for JAXStateHasher with advanced features."""
    
    def __init__(self, base_hasher: JAXStateHasher, enable_sharding: bool = True):
        """Initialize TPU-optimized hasher.
        
        Args:
            base_hasher: Base JAXStateHasher instance
            enable_sharding: Whether to enable automatic sharding
        """
        self.base_hasher = base_hasher
        self.enable_sharding = enable_sharding and JAX_AVAILABLE and pjit is not None
        
        # Pre-compile hash functions for different batch sizes
        self._compile_hash_functions()
    
    def _compile_hash_functions(self):
        """Pre-compile hash functions for common batch sizes."""
        if not JAX_AVAILABLE:
            return
        
        # Common batch sizes for pre-compilation
        batch_sizes = [1, 32, 128, 512, 2048, 8192]
        
        for batch_size in batch_sizes:
            dummy_states = jnp.zeros((batch_size, self.base_hasher.state_size))
            # Trigger JIT compilation
            _ = self.hash_states(dummy_states)
    
    @jit
    def hash_states(self, states: JaxArray) -> JaxArray:
        """Hash states with TPU optimizations.
        
        Args:
            states: States to hash
            
        Returns:
            Hash values
        """
        if self.enable_sharding:
            return distributed_hash_states(states, self.base_hasher)
        else:
            return vectorized_hash_states(states, self.base_hasher)
    
    @jit
    def hash_states_chunked(self, states: JaxArray, chunk_size: int = 2**16) -> JaxArray:
        """Hash states with memory-efficient chunking.
        
        Args:
            states: States to hash
            chunk_size: Size of processing chunks
            
        Returns:
            Hash values
        """
        return memory_efficient_batch_hash(states, self.base_hasher, chunk_size)


# Gradient-based hash optimization (for advanced use cases)
@jit
def hash_gradient_checkpoint(states: JaxArray, hasher: JAXStateHasher) -> JaxArray:
    """Hash states with gradient checkpointing for memory efficiency.
    
    Args:
        states: States to hash
        hasher: JAXStateHasher instance
        
    Returns:
        Hash values with gradient checkpointing
    """
    if not JAX_AVAILABLE:
        return hasher.hash_states(states)
    
    # Use gradient checkpointing for memory-intensive operations
    from jax.experimental import checkpoint
    
    @checkpoint
    def checkpointed_hash(states_chunk):
        return vectorized_hash_states(states_chunk, hasher)
    
    return checkpointed_hash(states)
#Advanced JAX/TPU optimizations for hashing

@jit
def _vectorized_splitmix64_single(state: JaxArray, seed: int) -> JaxArray:
    """Vectorized SplitMix64 for a single state using lax.scan."""
    h = jnp.array(seed, dtype=jnp.int64)
    
    def scan_fn(h_carry, x_i):
        h_new = h_carry ^ _splitmix64_jax(x_i)
        h_new = h_new * 0x85EBCA6B
        return h_new, None
    
    final_h, _ = lax.scan(scan_fn, h, state)
    return final_h


# TPU sharding support for large-scale hashing
# Define the functions without decorators first
def _distributed_hash_states_impl(states: JaxArray, hasher: JAXStateHasher) -> JaxArray:
    """Implementation of distributed state hashing across TPU cores."""
    return hasher.hash_states(states)

def _distributed_vectorized_hash_states_impl(states: JaxArray, hasher: JAXStateHasher) -> JaxArray:
    """Implementation of distributed vectorized hashing across TPU cores."""
    return vectorized_hash_states(states, hasher)

def _distributed_batch_hash_with_params_impl(states: JaxArray, vec_hasher: JaxArray, seed: int) -> JaxArray:
    """Implementation of distributed batch hashing with explicit parameters."""
    def hash_fn(state):
        return jnp.dot(state, vec_hasher.flatten())
    
    vectorized_fn = vmap(hash_fn, in_axes=0, out_axes=0)
    return vectorized_fn(states)

# Apply decorators only if JAX and pjit are available
if JAX_AVAILABLE and pjit is not None and P is not None:
    try:
        # Try to create decorated versions
        distributed_hash_states = pjit(
            _distributed_hash_states_impl,
            in_axis_resources=(P('batch', None),),
            out_axis_resources=P('batch')
        )
        
        distributed_vectorized_hash_states = pjit(
            _distributed_vectorized_hash_states_impl,
            in_axis_resources=(P('batch', None), None),
            out_axis_resources=P('batch')
        )
        
        distributed_batch_hash_with_params = pjit(
            _distributed_batch_hash_with_params_impl,
            in_axis_resources=(P('batch', None), None, None),
            out_axis_resources=P('batch')
        )
    except Exception as e:
        # If decorating fails, fall back to undecorated versions
        print(f"Warning: Could not create pjit-decorated functions: {e}")
        distributed_hash_states = _distributed_hash_states_impl
        distributed_vectorized_hash_states = _distributed_vectorized_hash_states_impl
        distributed_batch_hash_with_params = _distributed_batch_hash_with_params_impl
else:
    # Fallback implementations when pjit is not available
    distributed_hash_states = _distributed_hash_states_impl
    distributed_vectorized_hash_states = _distributed_vectorized_hash_states_impl
    distributed_batch_hash_with_params = _distributed_batch_hash_with_params_impl


class OptimizedJAXStateHasher(JAXStateHasher):
    """Enhanced JAX state hasher with advanced TPU optimizations."""
    
    def __init__(self, state_size: int, random_seed: Optional[int] = None, 
                 chunk_size: int = 2**18, use_string_encoder: bool = False,
                 enable_sharding: bool = True, use_scan: bool = True):
        """Initialize optimized hasher with TPU-specific features."""
        super().__init__(state_size, random_seed, chunk_size, use_string_encoder)
        self.enable_sharding = enable_sharding and JAX_AVAILABLE and pjit is not None
        self.use_scan = use_scan
    
    @jit
    def _optimized_hash_dot_product_chunk(self, states: JaxArray) -> JaxArray:
        """Optimized dot product hashing using lax operations."""
        if self.use_scan:
            def scan_fn(carry, state):
                hash_val = jnp.dot(state, self.vec_hasher.flatten())
                return carry, hash_val
            
            _, hash_values = lax.scan(scan_fn, None, states)
            return hash_values
        else:
            return (states @ self.vec_hasher).reshape(-1)
    
    @jit
    def _optimized_hash_splitmix64_chunk(self, states: JaxArray) -> JaxArray:
        """Optimized SplitMix64 hashing using vectorized operations."""
        batch_size, state_size = states.shape
        
        # Vectorized implementation using vmap
        def hash_single_state(state):
            return _vectorized_splitmix64_single(state, self.seed)
        
        vectorized_hash = vmap(hash_single_state, in_axes=0, out_axes=0)
        return vectorized_hash(states)
    
    def hash_states_optimized(self, states: JaxArray) -> JaxArray:
        """Optimized state hashing with TPU sharding support."""
        if states.ndim == 1:
            states = states.reshape(1, -1)
        elif states.ndim > 2:
            states = states.reshape(-1, self.state_size)
        
        if self.enable_sharding:
            return distributed_hash_states(states, self)
        else:
            return self.hash_states(states)
    
    def batch_hash_with_vectorization(self, state_batches: list) -> JaxArray:
        """Batch hash multiple arrays with full vectorization."""
        # Combine all batches into a single array for maximum vectorization
        combined_states = concatenate_arrays(state_batches, axis=0)
        
        if self.enable_sharding:
            return distributed_vectorized_hash_states(combined_states, self)
        else:
            return vectorized_hash_states(combined_states, self)


# Memory-efficient hashing for very large state spaces
def memory_efficient_hash_large_batch(states: JaxArray, hasher: JAXStateHasher, 
                                     max_memory_gb: float = 4.0) -> JaxArray:
    """Memory-efficient hashing for very large batches."""
    _check_jax_available()
    
    # Estimate memory usage
    element_size = states.dtype.itemsize
    batch_size_gb = (states.size * element_size) / (1024**3)
    
    if batch_size_gb <= max_memory_gb:
        return hasher.hash_states(states)
    
    # Process in chunks
    chunk_size = int(max_memory_gb * (1024**3) / (states.shape[1] * element_size))
    chunk_size = max(1, min(chunk_size, states.shape[0]))
    
    # Use lax.scan for memory-efficient processing
    def scan_fn(carry, chunk):
        chunk_hashes = hasher.hash_states(chunk)
        return carry, chunk_hashes
    
    chunks = jnp.array_split(states, max(1, states.shape[0] // chunk_size), axis=0)
    _, hash_results = lax.scan(scan_fn, None, jnp.stack(chunks))
    
    return jnp.concatenate(hash_results, axis=0)


# Gradient checkpointing for memory efficiency
if JAX_AVAILABLE:
    try:
        from jax.experimental import remat
        
        @remat
        def memory_efficient_vectorized_hash(states: JaxArray, hasher: JAXStateHasher) -> JaxArray:
            """Memory-efficient vectorized hashing with gradient checkpointing."""
            return vectorized_hash_states(states, hasher)
        
        @remat
        def memory_efficient_batch_hash(state_batches: list, hasher: JAXStateHasher) -> JaxArray:
            """Memory-efficient batch hashing with gradient checkpointing."""
            combined_states = concatenate_arrays(state_batches, axis=0)
            return hasher.hash_states(combined_states)
    except ImportError:
        # Fallback if remat is not available
        def memory_efficient_vectorized_hash(states: JaxArray, hasher: JAXStateHasher) -> JaxArray:
            return vectorized_hash_states(states, hasher)
        
        def memory_efficient_batch_hash(state_batches: list, hasher: JAXStateHasher) -> JaxArray:
            combined_states = concatenate_arrays(state_batches, axis=0)
            return hasher.hash_states(combined_states)


# Performance benchmarking with TPU-specific metrics
def benchmark_hash_performance_advanced(hasher: JAXStateHasher, 
                                       test_states: JaxArray, 
                                       num_iterations: int = 10,
                                       test_sharding: bool = True) -> dict:
    """Advanced benchmark with TPU-specific performance metrics."""
    import time
    
    results = {}
    
    # Warm up JIT compilation
    _ = hasher.hash_states(test_states[:100])
    if hasattr(hasher, 'hash_states_optimized'):
        _ = hasher.hash_states_optimized(test_states[:100])
    
    # Benchmark standard hashing
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        _ = hasher.hash_states(test_states)
        end_time = time.time()
        times.append(end_time - start_time)
    
    results['standard'] = {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "states_per_second": len(test_states) / (sum(times) / len(times))
    }
    
    # Benchmark vectorized hashing
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        _ = vectorized_hash_states(test_states, hasher)
        end_time = time.time()
        times.append(end_time - start_time)
    
    results['vectorized'] = {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "states_per_second": len(test_states) / (sum(times) / len(times))
    }
    
    # Benchmark sharded hashing if available
    if test_sharding and JAX_AVAILABLE and pjit is not None:
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            _ = distributed_hash_states(test_states, hasher)
            end_time = time.time()
            times.append(end_time - start_time)
        
        results['sharded'] = {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "states_per_second": len(test_states) / (sum(times) / len(times))
        }
    
    return results