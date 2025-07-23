"""JAX-based state hashing system for CayleyPy.

This module provides JAX implementations of state hashing functionality,
optimized for TPU/GPU computation with vectorized operations and JIT compilation.
"""

import random
import time
from typing import Optional, TYPE_CHECKING, Union

try:
    import jax.numpy as jnp
    from jax import jit
    import jax.random as jrandom

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    jrandom = None

from .jax_tensor_ops import concatenate_arrays, chunked_operation, unique_with_indices

if TYPE_CHECKING:
    from cayleypy import JAXCayleyGraph

MAX_INT = 2**62


@jit
def _identity_hash_static(states: jnp.ndarray) -> jnp.ndarray:
    """Static JIT-compiled identity hash function for single-element states.

    Args:
        states: Input states

    Returns:
        Flattened states (identity transformation)
    """
    return states.reshape(-1)


@jit
def _dot_product_hash_static(states: jnp.ndarray, vec_hasher: jnp.ndarray) -> jnp.ndarray:
    """Static JIT-compiled dot product hash function.

    Args:
        states: Chunk of states
        vec_hasher: Random vector for hashing

    Returns:
        Hash values for the chunk
    """
    return (states @ vec_hasher).reshape(-1)


@jit
def _splitmix64_hash_static(states: jnp.ndarray, seed: int) -> jnp.ndarray:
    """Static JIT-compiled SplitMix64 hash function.

    Args:
        states: Chunk of states of shape (batch_size, state_size)
        seed: Random seed for hashing

    Returns:
        Hash values for the chunk
    """
    batch_size, state_size = states.shape

    # Initialize hash with seed
    h = jnp.full((batch_size,), seed, dtype=jnp.int32)

    # Process each element of the state vector
    for i in range(state_size):
        h = h ^ _splitmix64_jax(states[:, i])
        h = h * 0x85EBCA6B

    return h


@jit
def _splitmix64_jax(x: jnp.ndarray) -> jnp.ndarray:
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

    def __init__(
        self,
        state_size: int,
        random_seed: Optional[int] = None,
        chunk_size: int = 2**18,
        use_string_encoder: bool = False,
    ):
        """Initialize the state hasher.

        Args:
            state_size: Size of encoded states
            random_seed: Random seed for hash function
            chunk_size: Chunk size for memory-efficient processing
            use_string_encoder: Whether states use string encoding (bit-packed)
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available. Install with: pip install jax[tpu] or pip install jax[cuda]")

        self.state_size = state_size
        self.chunk_size = chunk_size
        self.use_string_encoder = use_string_encoder

        # If states are single int64, use identity function
        self.is_identity = state_size == 1

        if self.is_identity:
            self.make_hashes = _identity_hash_static
            return

        # Initialize random seed
        self.seed = random_seed if random_seed is not None else random.randint(-(2**31), 2**31 - 1)

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

        # Generate random vector for hashing with safe integer range
        self.vec_hasher = jrandom.randint(
            key, shape=(self.state_size, 1), minval=-(2**31), maxval=2**31 - 1, dtype=jnp.int32
        )

    def _hash_dot_product(self, states: jnp.ndarray) -> jnp.ndarray:
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
            return chunked_operation(states, self._hash_dot_product_chunk, self.chunk_size)

    def _hash_dot_product_chunk(self, states: jnp.ndarray) -> jnp.ndarray:
        """Hash a chunk of states using dot product.

        Args:
            states: Chunk of states

        Returns:
            Hash values for the chunk
        """
        return _dot_product_hash_static(states, self.vec_hasher)

    def _hash_splitmix64(self, states: jnp.ndarray) -> jnp.ndarray:
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
            return chunked_operation(states, self._hash_splitmix64_chunk, self.chunk_size)

    def _hash_splitmix64_chunk(self, states: jnp.ndarray) -> jnp.ndarray:
        """Hash a chunk of states using SplitMix64.

        Args:
            states: Chunk of states of shape (batch_size, state_size)

        Returns:
            Hash values for the chunk
        """
        return _splitmix64_hash_static(states, self.seed)

    def hash_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Hash a batch of states.

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

        return self.make_hashes(states)

    def hash_single_state(self, state: jnp.ndarray) -> int:
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

    def hash_and_concatenate(self, state_batches: list) -> jnp.ndarray:
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
def vectorized_hash_states(states: jnp.ndarray, hasher_params: dict) -> jnp.ndarray:
    """Vectorized state hashing using vmap.

    Args:
        states: Batch of states
        hasher_params: Parameters for hashing (unused for now)

    Returns:
        Hash values
    """
    # This would be implemented with vmap for maximum vectorization
    # For now, we'll use the batch processing approach
    # TODO: Implement actual vectorized hashing
    _ = hasher_params  # Suppress unused argument warning
    return jnp.zeros(states.shape[0], dtype=jnp.int64)


# Utility functions for hash management
def create_hash_function(
    state_size: int, encoding_type: str = "regular", random_seed: Optional[int] = None
) -> JAXStateHasher:
    """Create a hash function for given state configuration.

    Args:
        state_size: Size of state vectors
        encoding_type: Type of encoding ("regular" or "string")
        random_seed: Random seed for reproducibility

    Returns:
        Configured JAXStateHasher
    """
    use_string_encoder = encoding_type == "string"
    return JAXStateHasher(state_size=state_size, random_seed=random_seed, use_string_encoder=use_string_encoder)


def hash_state_collection(states: Union[jnp.ndarray, list], hasher: JAXStateHasher) -> jnp.ndarray:
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
def fast_hash_comparison(hashes1: jnp.ndarray, hashes2: jnp.ndarray) -> jnp.ndarray:
    """Fast comparison of hash arrays.

    Args:
        hashes1: First hash array
        hashes2: Second hash array

    Returns:
        Boolean array of equality comparisons
    """
    return hashes1 == hashes2


@jit
def find_hash_duplicates(hashes: jnp.ndarray) -> tuple:
    """Find duplicate hashes in an array.

    Args:
        hashes: Array of hash values

    Returns:
        Tuple of (unique_hashes, inverse_indices, counts)
    """
    return unique_with_indices(hashes, return_inverse=True, return_counts=True)


def benchmark_hash_performance(hasher: JAXStateHasher, test_states: jnp.ndarray, num_iterations: int = 10) -> dict:
    """Benchmark hash performance.

    Args:
        hasher: Hasher to benchmark
        test_states: Test states for benchmarking
        num_iterations: Number of iterations to run

    Returns:
        Performance statistics
    """
    # Warm up JIT compilation
    _ = hasher.hash_states(test_states[:100])

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start_time = time.time()
        _ = hasher.hash_states(test_states)
        end_time = time.time()
        times.append(end_time - start_time)

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "states_per_second": len(test_states) / (sum(times) / len(times)),
    }
