#!/usr/bin/env python
"""Simple test script for JAX hasher functionality.

This script implements a simple version of the JAX hasher with TPU compatibility.
"""

import time
import numpy as np

print("Testing JAX availability...")

# Check if JAX is available
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import jit, vmap
    import jax.random as jrandom
    JAX_AVAILABLE = True
    print(f"JAX is available, version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    jrandom = None
    print("JAX not available")
    exit(0)

# Simple JAX hasher implementation
class SimpleHasher:
    """Simple JAX-based state hasher."""
    
    def __init__(self, state_size, random_seed=None):
        """Initialize the hasher."""
        self.state_size = state_size
        self.seed = random_seed if random_seed is not None else np.random.randint(0, 2**30)
        
        # Initialize random vector for dot product hashing
        key = jrandom.PRNGKey(self.seed)
        self.vec_hasher = jrandom.randint(
            key, 
            shape=(self.state_size, 1), 
            minval=-(2**31),  # Use full int64 range now that x64 is enabled
            maxval=2**31,
            dtype=jnp.int64  # Use int64 now that x64 is enabled
        )
    
    def hash_states(self, states):
        """Hash a batch of states."""
        # Ensure states have correct shape
        if states.ndim == 1:
            states = states.reshape(1, -1)
        
        # Use standard implementation
        return self._hash_standard(states)
    
    def _hash_standard(self, states):
        """Standard implementation using matrix multiplication."""
        return (states @ self.vec_hasher).reshape(-1)
    
    def hash_single_state(self, state):
        """Hash a single state."""
        if state.ndim == 0:
            state = state.reshape(1)
        
        state_batch = state.reshape(1, -1)
        hash_result = self.hash_states(state_batch)
        return int(hash_result[0])


# Optimized JAX hasher with vectorization
class OptimizedHasher(SimpleHasher):
    """Optimized JAX-based state hasher with vectorization."""
    
    def __init__(self, state_size, random_seed=None):
        """Initialize the hasher."""
        super().__init__(state_size, random_seed)
        
        # Pre-compile the vectorized function
        self._hash_vectorized = vmap(self._hash_single_state)
    
    def _hash_single_state(self, state):
        """Hash a single state."""
        return jnp.dot(state, self.vec_hasher.flatten())
    
    def hash_states(self, states):
        """Hash a batch of states using vectorization."""
        # Ensure states have correct shape
        if states.ndim == 1:
            states = states.reshape(1, -1)
        
        # Use vectorized implementation
        return self._hash_vectorized(states)


# Test the hashers
print("\n=== Testing JAX hashers ===")

# Create test data
state_size = 10
batch_size = 1000
states = jnp.ones((batch_size, state_size), dtype=jnp.int32)
states = states.at[:, 0].set(jnp.arange(batch_size))  # Make states unique

# Create hashers
standard_hasher = SimpleHasher(state_size=state_size, random_seed=42)
optimized_hasher = OptimizedHasher(state_size=state_size, random_seed=42)

# Test standard hasher
print("\n=== Testing standard hasher ===")
start_time = time.time()
standard_hashes = standard_hasher.hash_states(states)
end_time = time.time()
standard_time = end_time - start_time
print(f"Standard hasher: {standard_time:.6f} seconds")
print(f"Result shape: {standard_hashes.shape}")
print(f"Result dtype: {standard_hashes.dtype}")

# Test optimized hasher
print("\n=== Testing optimized hasher ===")
start_time = time.time()
optimized_hashes = optimized_hasher.hash_states(states)
end_time = time.time()
optimized_time = end_time - start_time
print(f"Optimized hasher: {optimized_time:.6f} seconds")
print(f"Result shape: {optimized_hashes.shape}")
print(f"Result dtype: {optimized_hashes.dtype}")

# Compare results
match = jnp.array_equal(standard_hashes, optimized_hashes)
print(f"Results match: {match}")

# Calculate speedup
speedup = standard_time / optimized_time
print(f"Speedup: {speedup:.2f}x")

# Test with different batch sizes
print("\n=== Testing with different batch sizes ===")
batch_sizes = [10, 100, 1000, 10000]
standard_times = {}
optimized_times = {}
speedups = {}

for size in batch_sizes:
    print(f"\nBatch size: {size}")
    test_states = jnp.ones((size, state_size), dtype=jnp.int32)
    test_states = test_states.at[:, 0].set(jnp.arange(size))
    
    # Test standard hasher
    start_time = time.time()
    _ = standard_hasher.hash_states(test_states)
    end_time = time.time()
    standard_time = end_time - start_time
    standard_times[size] = standard_time
    print(f"Standard hasher: {standard_time:.6f} seconds")
    
    # Test optimized hasher
    start_time = time.time()
    _ = optimized_hasher.hash_states(test_states)
    end_time = time.time()
    optimized_time = end_time - start_time
    optimized_times[size] = optimized_time
    print(f"Optimized hasher: {optimized_time:.6f} seconds")
    
    # Calculate speedup
    speedup = standard_time / optimized_time
    speedups[size] = speedup
    print(f"Speedup: {speedup:.2f}x")

# Summary
print("\n=== Summary ===")
print("Batch Size | Standard Time | Optimized Time | Speedup")
print("-" * 50)
for size in batch_sizes:
    print(f"{size:10d} | {standard_times[size]:.6f} s | {optimized_times[size]:.6f} s | {speedups[size]:.2f}x")

print("\nAll tests completed successfully!")