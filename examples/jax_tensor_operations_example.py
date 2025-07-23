#!/usr/bin/env python3
"""Example demonstrating JAX tensor operations in CayleyPy.

This example shows how to use the JAX tensor operations and hashing system
for high-performance computation with automatic TPU/GPU/CPU optimization.
"""

import numpy as np

try:
    import jax.numpy as jnp
    from cayleypy import JAXDeviceManager, JAXStateHasher, JAXHashSet
    from cayleypy.jax_tensor_ops import (
        unique_with_indices,
        gather_along_axis,
        isin_via_searchsorted,
        sort_with_indices,
        concatenate_arrays,
        batch_matmul,
        chunked_operation,
    )

    JAX_AVAILABLE = True
except ImportError:
    print("JAX not available. Install with: pip install cayleypy[jax]")
    JAX_AVAILABLE = False
    exit(1)


def demonstrate_unique_operations():
    """Demonstrate unique tensor operations."""
    print("=== Unique Operations Demo ===")

    # Create array with duplicates
    data = jnp.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
    print(f"Original array: {data}")

    # Basic unique
    unique_vals = unique_with_indices(data)
    print(f"Unique values: {unique_vals}")

    # Unique with inverse indices
    unique_vals, inverse = unique_with_indices(data, return_inverse=True)
    print(f"Inverse indices: {inverse}")
    print(f"Reconstructed: {unique_vals[inverse]}")

    # Unique with counts
    unique_vals, counts = unique_with_indices(data, return_counts=True)
    print(f"Counts: {counts}")

    print()


def demonstrate_gather_operations():
    """Demonstrate gather operations."""
    print("=== Gather Operations Demo ===")

    # Create 2D array for gathering
    source = jnp.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]])
    print(f"Source array:\n{source}")

    # Gather specific indices
    indices = jnp.array([[0, 3, 1], [2, 1, 3], [1, 0, 2]])
    print(f"Indices:\n{indices}")

    result = gather_along_axis(source, indices, axis=1)
    print(f"Gathered result:\n{result}")

    print()


def demonstrate_membership_testing():
    """Demonstrate membership testing operations."""
    print("=== Membership Testing Demo ===")

    # Create test data
    elements = jnp.array([1, 5, 10, 15, 20, 25, 30])
    test_set = jnp.array([5, 15, 25, 35, 45])

    print(f"Elements: {elements}")
    print(f"Test set: {test_set}")

    # Test membership
    membership = isin_via_searchsorted(elements, test_set)
    print(f"Membership: {membership}")
    print(f"Elements in set: {elements[membership]}")

    print()


def demonstrate_sorting_operations():
    """Demonstrate sorting operations."""
    print("=== Sorting Operations Demo ===")

    # Create random-like data
    data = jnp.array([64, 34, 25, 12, 22, 11, 90])
    print(f"Original: {data}")

    # Sort with indices
    sorted_vals, indices = sort_with_indices(data)
    print(f"Sorted values: {sorted_vals}")
    print(f"Sort indices: {indices}")
    print(f"Verification: {data[indices]}")

    print()


def demonstrate_array_operations():
    """Demonstrate array manipulation operations."""
    print("=== Array Operations Demo ===")

    # Create multiple arrays
    arrays = [jnp.array([1, 2, 3]), jnp.array([4, 5, 6]), jnp.array([7, 8, 9])]
    print(f"Arrays to concatenate: {arrays}")

    # Concatenate
    concatenated = concatenate_arrays(arrays, axis=0)
    print(f"Concatenated: {concatenated}")

    # Batch matrix multiplication
    batch_a = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    batch_b = jnp.array([[[2, 0], [0, 2]], [[1, 1], [1, 1]]])

    print(f"Batch A shape: {batch_a.shape}")
    print(f"Batch B shape: {batch_b.shape}")

    result = batch_matmul(batch_a, batch_b)
    print(f"Batch matmul result:\n{result}")

    print()


def demonstrate_chunked_processing():
    """Demonstrate chunked processing for large arrays."""
    print("=== Chunked Processing Demo ===")

    # Create large array
    large_array = jnp.arange(1000).reshape(100, 10)
    print(f"Large array shape: {large_array.shape}")

    # Define operation to apply
    def sum_operation(chunk):
        return jnp.sum(chunk, axis=1)

    # Process in chunks
    result = chunked_operation(large_array, sum_operation, chunk_size=30)
    print(f"Chunked result shape: {result.shape}")
    print(f"First 5 results: {result[:5]}")

    # Verify against direct computation
    direct_result = jnp.sum(large_array, axis=1)
    print(f"Results match: {jnp.array_equal(result, direct_result)}")

    print()


def demonstrate_state_hashing():
    """Demonstrate state hashing system."""
    print("=== State Hashing Demo ===")

    # Create state hasher
    hasher = JAXStateHasher(state_size=5, random_seed=42)
    print(f"Hasher identity: {hasher.is_identity}")
    print(f"Hasher uses string encoder: {hasher.use_string_encoder}")

    # Create some states
    states = jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])  # Duplicate
    print(f"States shape: {states.shape}")

    # Hash the states
    hashes = hasher.hash_states(states)
    print(f"Hashes: {hashes}")

    # Check for duplicates
    unique_hashes, inverse, counts = unique_with_indices(hashes, return_inverse=True, return_counts=True)
    print(f"Unique hashes: {len(unique_hashes)}")
    print(f"Duplicate states at indices: {jnp.where(counts[inverse] > 1)[0]}")

    print()


def demonstrate_hash_set():
    """Demonstrate JAX hash set functionality."""
    print("=== Hash Set Demo ===")

    # Create hash set
    hash_set = JAXHashSet()
    print(f"Initial hash set empty: {hash_set.is_empty()}")

    # Add some hashes
    hashes1 = jnp.array([10, 20, 30, 40])
    hash_set.add_sorted_hashes(hashes1)
    print(f"Added hashes: {hashes1}")
    print(f"Hash set size: {hash_set.size()}")

    # Add more hashes
    hashes2 = jnp.array([25, 35, 45, 55])
    hash_set.add_sorted_hashes(hashes2)
    print(f"Added more hashes: {hashes2}")
    print(f"Hash set size: {hash_set.size()}")

    # Test membership
    test_values = jnp.array([15, 20, 25, 30, 50, 55, 60])
    membership = hash_set.contains(test_values)
    print(f"Test values: {test_values}")
    print(f"In hash set: {membership}")

    # Get mask for unseen values
    mask = hash_set.get_mask_to_remove_seen_hashes(test_values)
    unseen_values = test_values[mask]
    print(f"Unseen values: {unseen_values}")

    print()


def demonstrate_performance_comparison():
    """Demonstrate performance with different array sizes."""
    print("=== Performance Comparison Demo ===")

    # Set up device manager
    device_manager = JAXDeviceManager(device="auto")
    print(f"Using device: {device_manager.device_type}")

    # Test different array sizes
    sizes = [100, 1000, 10000]

    for size in sizes:
        print(f"\nTesting with {size} elements:")

        # Create test data
        data = device_manager.put_on_device(jnp.arange(size * 3).reshape(size, 3))

        # Test unique operation
        unique_vals = unique_with_indices(data.flatten())
        print(f"  Unique values found: {len(unique_vals)}")

        # Test hashing
        hasher = JAXStateHasher(state_size=3, random_seed=42)
        hashes = hasher.hash_states(data)
        unique_hashes = unique_with_indices(hashes)
        print(f"  Unique hashes: {len(unique_hashes)}")

        # Test hash set operations
        hash_set = JAXHashSet()
        hash_set.add_sorted_hashes(jnp.sort(hashes))
        print(f"  Hash set size: {hash_set.size()}")

    print()


def main():
    """Run all demonstrations."""
    if not JAX_AVAILABLE:
        return

    print("CayleyPy JAX Tensor Operations Demo")
    print("=" * 40)
    print()

    try:
        demonstrate_unique_operations()
        demonstrate_gather_operations()
        demonstrate_membership_testing()
        demonstrate_sorting_operations()
        demonstrate_array_operations()
        demonstrate_chunked_processing()
        demonstrate_state_hashing()
        demonstrate_hash_set()
        demonstrate_performance_comparison()

        print("All demonstrations completed successfully!")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("This may be due to JAX/device configuration issues.")


if __name__ == "__main__":
    main()
