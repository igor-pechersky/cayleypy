#!/usr/bin/env python3
"""Example demonstrating JAX device management in CayleyPy.

This example shows how to use the JAX device management system for
high-performance computation with automatic TPU/GPU/CPU detection.
"""

import numpy as np

try:
    import jax.numpy as jnp
    from cayleypy import JAXDeviceManager, DeviceFallbackHandler
    JAX_AVAILABLE = True
except ImportError:
    print("JAX not available. Install with: pip install cayleypy[jax]")
    JAX_AVAILABLE = False
    exit(1)


def demonstrate_device_selection():
    """Demonstrate automatic device selection."""
    print("=== Device Selection Demo ===")
    
    # Automatic device selection (TPU > GPU > CPU preference)
    manager = JAXDeviceManager(device="auto")
    print(f"Auto-selected device: {manager.device_type}")
    print(f"Primary device: {manager.primary_device}")
    print(f"Total devices: {manager.get_device_count()}")
    print(f"Is TPU: {manager.is_tpu()}")
    print(f"Is GPU: {manager.is_gpu()}")
    print(f"Is CPU: {manager.is_cpu()}")
    print()


def demonstrate_memory_management():
    """Demonstrate memory information and management."""
    print("=== Memory Management Demo ===")
    
    manager = JAXDeviceManager(device="auto")
    
    # Get memory information
    memory_info = manager.get_memory_info()
    for device_str, info in memory_info.items():
        print(f"Device: {device_str}")
        if 'error' in info:
            print(f"  Error: {info['error']}")
        else:
            print(f"  Platform: {info['platform']}")
            print(f"  Total Memory: {info['total_memory_gb']:.1f} GB")
            print(f"  Free Memory: {info['free_memory_gb']:.1f} GB")
        print()
    
    # Clear cache
    manager.clear_cache()
    print("Cache cleared successfully")
    print()


def demonstrate_array_operations():
    """Demonstrate array placement and operations."""
    print("=== Array Operations Demo ===")
    
    manager = JAXDeviceManager(device="auto")
    
    # Place different types of arrays on device
    numpy_array = np.array([1, 2, 3, 4, 5])
    python_list = [10, 20, 30, 40, 50]
    
    jax_array1 = manager.put_on_device(numpy_array)
    jax_array2 = manager.put_on_device(python_list)
    
    print(f"Original NumPy array: {numpy_array}")
    print(f"JAX array on device: {jax_array1}")
    print(f"Array sum: {jnp.sum(jax_array1)}")
    print()
    
    print(f"Original Python list: {python_list}")
    print(f"JAX array on device: {jax_array2}")
    print(f"Array product: {jnp.prod(jax_array2)}")
    print()
    
    # Matrix operations
    matrix_a = manager.put_on_device([[1, 2], [3, 4]])
    matrix_b = manager.put_on_device([[5, 6], [7, 8]])
    
    result = jnp.matmul(matrix_a, matrix_b)
    print(f"Matrix A:\n{matrix_a}")
    print(f"Matrix B:\n{matrix_b}")
    print(f"A @ B:\n{result}")
    print()


def demonstrate_device_fallback():
    """Demonstrate device fallback functionality."""
    print("=== Device Fallback Demo ===")
    
    # Create fallback handler with custom hierarchy
    handler = DeviceFallbackHandler(["tpu", "gpu", "cpu"])
    
    def compute_function(device_manager, data):
        """Example computation function."""
        array = device_manager.put_on_device(data)
        return jnp.sum(array ** 2)
    
    # Execute with automatic fallback
    test_data = [1, 2, 3, 4, 5]
    result = handler.execute_with_fallback(compute_function, test_data)
    print(f"Computation result: {result}")
    print(f"Expected: {sum(x**2 for x in test_data)}")
    
    # Get best available device manager
    best_manager = handler.get_best_device_manager()
    print(f"Best available device: {best_manager.device_type}")
    print()


def demonstrate_large_array_handling():
    """Demonstrate handling of large arrays with potential sharding."""
    print("=== Large Array Handling Demo ===")
    
    manager = JAXDeviceManager(device="auto")
    
    # Create a moderately large array
    large_data = np.random.rand(1000, 100)
    print(f"Large array shape: {large_data.shape}")
    print(f"Large array size: {large_data.nbytes / 1e6:.1f} MB")
    
    # Place on device (may be sharded if multiple devices available)
    jax_large = manager.put_on_device(large_data)
    print(f"JAX array shape: {jax_large.shape}")
    
    # Perform computation
    column_sums = jnp.sum(jax_large, axis=0)
    print(f"Column sums shape: {column_sums.shape}")
    print(f"First 5 column sums: {column_sums[:5]}")
    print()


def main():
    """Run all demonstrations."""
    if not JAX_AVAILABLE:
        return
    
    print("CayleyPy JAX Device Management Demo")
    print("=" * 40)
    print()
    
    try:
        demonstrate_device_selection()
        demonstrate_memory_management()
        demonstrate_array_operations()
        demonstrate_device_fallback()
        demonstrate_large_array_handling()
        
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("This may be due to JAX/device configuration issues.")


if __name__ == "__main__":
    main()