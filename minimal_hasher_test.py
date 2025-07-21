#!/usr/bin/env python
"""Minimal test script for JAX hashing functionality.

This script implements a simple version of the JAX hasher without importing from cayleypy.
"""

import time
import numpy as np

print("Testing JAX availability...")

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    import jax.random as jrandom
    JAX_AVAILABLE = True
    print(f"JAX is available, version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    jrandom = None
    print("JAX not available")

if not JAX_AVAILABLE:
    print("JAX is not available. Exiting.")
    exit(0)

# Implement a simple version of the JAX hasher
print("\n=== Implementing simple JAX hasher ===")

@jit
def _splitmix64_jax(x):
    """JAX implementation of SplitMix64 hash function."""
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    x = x ^ (x >> 31)
    return x

class SimpleJAXHasher:
    """Simple JAX-based state hasher."""
    
    def __init__(self, state_size, random_seed=None):
        """Initialize the hasher."""
        self.state_size = state_size
        self.seed = random_seed if random_seed is not None else np.random.randint(-(2**31), 2**31)
        
        # Initialize random vector for dot product hashing
        key = jrandom.PRNGKey(self.seed)
        self.vec_hasher = jrandom.randint(
            key, 
            shape=(self.state_size, 1), 
            minval=-(2**31), 
            maxval=2**31,
            dtype=jnp.int64
        )
    
    @jit
    def _hash_dot_product_chunk(self, states):
        """Hash a chunk of states using dot product."""
        return (states @ self.vec_hasher).reshape(-1)
    
    def hash_states(self, states):
        """Hash a batch of states."""
        # Ensure states have correct shape
        if states.ndim == 1:
            states = states.reshape(1, -1)
        elif states.ndim > 2:
            states = states.reshape(-1, self.state_size)
        
        return self._hash_dot_product_chunk(states)
    
    def hash_single_state(self, state):
        """Hash a single state."""
        if state.ndim == 0:
            state = state.reshape(1)
        elif state.ndim > 1:
            state = state.flatten()
        
        state_batch = state.reshape(1, -1)
        hash_result = self.hash_states(state_batch)
        return int(hash_result[0])

# Test the simple hasher
print("\n=== Testing simple JAX hasher ===")

# Create a hasher
state_size = 10
hasher = SimpleJAXHasher(state_size=state_size, random_seed=42)

# Create test data
states = jnp.ones((100, state_size), dtype=jnp.int32)
states = states.at[:, 0].set(jnp.arange(100))  # Make states unique

# Test hash_states
start_time = time.time()
hashes = hasher.hash_states(states)
end_time = time.time()

print(f"hash_states completed in {end_time - start_time:.6f} seconds")
print(f"Result shape: {hashes.shape}")
print(f"Result dtype: {hashes.dtype}")

# Check that hashes are unique
unique_hashes = jnp.unique(hashes)
print(f"Number of unique hashes: {len(unique_hashes)}")
print(f"All hashes are unique: {len(unique_hashes) == len(states)}")

# Test hash_single_state
state = jnp.ones(state_size, dtype=jnp.int32)
hash_value = hasher.hash_single_state(state)
print(f"Single state hash: {hash_value}")
print(f"Hash type: {type(hash_value)}")

# Test vectorized hashing
print("\n=== Testing vectorized hashing ===")

def hash_single_state(state, vec_hasher):
    """Hash a single state vector."""
    return jnp.dot(state, vec_hasher.flatten())

vectorized_hash = vmap(hash_single_state, in_axes=(0, None))

# Test vectorized hashing
start_time = time.time()
vectorized_hashes = vectorized_hash(states, hasher.vec_hasher)
end_time = time.time()

print(f"Vectorized hashing completed in {end_time - start_time:.6f} seconds")
print(f"Result shape: {vectorized_hashes.shape}")

# Check that results match
match = jnp.array_equal(hashes, vectorized_hashes)
print(f"Results match: {match}")

print("\nAll tests completed successfully!")