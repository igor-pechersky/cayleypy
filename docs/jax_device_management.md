# JAX Device Management in CayleyPy

CayleyPy includes a comprehensive JAX device management system that provides automatic TPU/GPU/CPU detection, device placement utilities, and memory management functions for high-performance graph computation.

## Overview

The JAX device management system consists of two main components:

1. **JAXDeviceManager**: Core device management with automatic device selection
2. **DeviceFallbackHandler**: Graceful fallback between devices when operations fail

## Installation

To use JAX features in CayleyPy, install with JAX support:

```bash
# For TPU support (recommended for Google Colab/Cloud)
pip install cayleypy[jax]

# Alternative installations:
pip install jax[tpu]  # TPU support
pip install jax[cuda]  # GPU support  
pip install jax[cpu]   # CPU only
```

## Quick Start

```python
from cayleypy import JAXDeviceManager
import jax.numpy as jnp

# Automatic device selection (TPU > GPU > CPU preference)
manager = JAXDeviceManager(device="auto")

# Place arrays on device
data = [1, 2, 3, 4, 5]
jax_array = manager.put_on_device(data)

# Perform computations
result = jnp.sum(jax_array ** 2)
print(f"Result: {result}")
```

## Device Selection

### Automatic Selection
```python
# Automatic selection with TPU > GPU > CPU preference
manager = JAXDeviceManager(device="auto")
print(f"Selected: {manager.device_type}")
```

### Manual Selection
```python
# Specific device types
tpu_manager = JAXDeviceManager(device="tpu")
gpu_manager = JAXDeviceManager(device="gpu") 
cpu_manager = JAXDeviceManager(device="cpu")
```

### Device Information
```python
manager = JAXDeviceManager(device="auto")

# Check device type
print(f"Is TPU: {manager.is_tpu()}")
print(f"Is GPU: {manager.is_gpu()}")
print(f"Is CPU: {manager.is_cpu()}")

# Get device count
print(f"Device count: {manager.get_device_count()}")
```

## Memory Management

### Memory Information
```python
manager = JAXDeviceManager(device="auto")

# Get detailed memory info for all devices
memory_info = manager.get_memory_info()
for device_str, info in memory_info.items():
    print(f"Device: {device_str}")
    print(f"  Platform: {info['platform']}")
    print(f"  Total Memory: {info['total_memory_gb']:.1f} GB")
    print(f"  Free Memory: {info['free_memory_gb']:.1f} GB")
```

### Cache Management
```python
# Clear JAX compilation cache
manager.clear_cache()
```

### Memory Configuration
```python
# Disable memory preallocation
manager = JAXDeviceManager(
    device="auto", 
    enable_memory_preallocation=False
)
```

## Array Operations

### Basic Array Placement
```python
import numpy as np

manager = JAXDeviceManager(device="auto")

# From NumPy arrays
numpy_array = np.array([1, 2, 3, 4])
jax_array = manager.put_on_device(numpy_array)

# From Python lists
python_list = [5, 6, 7, 8]
jax_array = manager.put_on_device(python_list)
```

### Large Array Handling
```python
# Large arrays are automatically sharded across multiple devices
large_array = np.random.rand(10000, 1000)  # ~80MB array
jax_array = manager.put_on_device(large_array)

# The manager automatically determines if sharding is needed
```

### Matrix Operations
```python
manager = JAXDeviceManager(device="auto")

# Create matrices on device
A = manager.put_on_device([[1, 2], [3, 4]])
B = manager.put_on_device([[5, 6], [7, 8]])

# Perform operations
C = jnp.matmul(A, B)
print(f"Result:\n{C}")
```

## Device Fallback

The `DeviceFallbackHandler` provides automatic fallback between devices when operations fail:

```python
from cayleypy import DeviceFallbackHandler

# Create handler with custom device preference
handler = DeviceFallbackHandler(["tpu", "gpu", "cpu"])

def my_computation(device_manager, data):
    array = device_manager.put_on_device(data)
    return jnp.sum(array ** 2)

# Execute with automatic fallback
result = handler.execute_with_fallback(my_computation, [1, 2, 3, 4, 5])

# Get best available device manager
best_manager = handler.get_best_device_manager()
```

## Error Handling

The system provides specific exceptions for different error conditions:

```python
from cayleypy.jax_device_manager import DeviceNotFoundError, OutOfMemoryError

try:
    manager = JAXDeviceManager(device="tpu")
except DeviceNotFoundError:
    print("TPU not available, falling back to GPU/CPU")
    manager = JAXDeviceManager(device="auto")
```

## Environment Configuration

### TPU Configuration
```python
import os

# Set TPU environment variables before importing JAX
os.environ["TPU_NAME"] = "your-tpu-name"
os.environ["JAX_PLATFORM_NAME"] = "tpu"

# Then create device manager
manager = JAXDeviceManager(device="tpu")
```

### Memory Configuration
```python
import os

# Configure memory allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
```

## Integration with CayleyPy

The JAX device management system is designed to integrate seamlessly with future JAX-based CayleyGraph implementations:

```python
# Future usage pattern (when JAX CayleyGraph is implemented)
from cayleypy import JAXDeviceManager, JAXCayleyGraph

manager = JAXDeviceManager(device="auto")
graph_def = ...  # Your graph definition

# JAX-based graph with automatic device management
jax_graph = JAXCayleyGraph(graph_def, device_manager=manager)
```

## Performance Tips

1. **TPU Usage**: TPUs excel at large matrix operations and benefit from XLA compilation
2. **Memory Management**: Use `enable_memory_preallocation=False` for dynamic workloads
3. **Array Sharding**: Large arrays are automatically sharded across multiple devices
4. **Cache Management**: Clear caches periodically for long-running computations

## Troubleshooting

### Common Issues

1. **JAX Not Available**
   ```
   ImportError: JAX is not available
   ```
   Solution: Install JAX with `pip install jax[tpu]` or `pip install jax[cuda]`

2. **No Devices Found**
   ```
   DeviceNotFoundError: No compatible devices found
   ```
   Solution: Check JAX installation and device availability with `jax.devices()`

3. **TPU Not Detected**
   ```
   DeviceNotFoundError: Requested device 'tpu' not available
   ```
   Solution: Ensure TPU is properly configured and `TPU_NAME` environment variable is set

### Debugging

```python
import jax

# Check available devices
print("Available devices:", jax.devices())

# Check platform
print("Default platform:", jax.default_backend())

# Test basic operation
test_array = jax.numpy.array([1, 2, 3])
print("JAX working:", test_array.sum())
```

## Examples

See `examples/jax_device_example.py` for a comprehensive demonstration of all features.

## API Reference

### JAXDeviceManager

- `__init__(device="auto", enable_memory_preallocation=True)`
- `put_on_device(array, device=None) -> jnp.ndarray`
- `get_memory_info() -> Dict[str, Dict[str, float]]`
- `clear_cache() -> None`
- `get_device_count() -> int`
- `is_tpu() -> bool`
- `is_gpu() -> bool`
- `is_cpu() -> bool`

### DeviceFallbackHandler

- `__init__(preferred_devices=None)`
- `execute_with_fallback(func, *args, **kwargs)`
- `get_best_device_manager() -> JAXDeviceManager`