---
inclusion: fileMatch
fileMatchPattern: '*jax*|*nnx*'
---

# JAX/NNX Development Guidelines for CayleyPy

## Import Handling

When working with JAX/NNX code, follow these patterns to avoid import and type checking issues:

### Optional JAX Imports
```python
try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from flax import nnx
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    Mesh = None  # type: ignore
    NamedSharding = None  # type: ignore
    PartitionSpec = None  # type: ignore
    nnx = None  # type: ignore
```

### Type Annotations for Optional Dependencies
- Use `Optional[Any]` for parameters that might be JAX/NNX types when JAX is not available
- Add `# type: ignore` comments for None assignments to imported modules
- Use explicit type annotations for variables that might have different types based on JAX availability

## Exception Handling

### Broad Exception Catching
When catching broad exceptions for graceful fallback behavior, add pylint disable comments:

```python
try:
    # JAX device detection or other hardware-specific operations
    devices = jax.devices("gpu")
except Exception:  # pylint: disable=broad-exception-caught
    # Graceful fallback for hardware detection
    pass
```

Use broad exception catching appropriately for:
- Hardware device detection
- Optional feature initialization
- Graceful degradation scenarios
- Cross-platform compatibility

### Global State Management
When using global variables for singleton patterns, add pylint disable comments:

```python
def get_global_backend():
    global _global_backend  # pylint: disable=global-statement
    # Implementation
```

## NNX Variable Handling

### Variable Assignment
When working with NNX Variables, be explicit about types:

```python
# Correct way to handle NNX Variable assignment
metrics_dict = {
    "device_count": len(self.devices),
    "memory_allocated": 0.0,
    "compilation_cache_size": 0,
    "operations_count": 0,
}
self.metrics = nnx.Variable(metrics_dict)
```

### Accessing Variable Values
Always check for JAX availability when accessing Variable values:

```python
if JAX_AVAILABLE and hasattr(self.metrics, "value"):
    metrics: Dict[str, Any] = dict(self.metrics.value)
else:
    metrics = dict(self.metrics)
```

## Logging Best Practices

### Lazy Formatting
Always use lazy % formatting in logging functions:

```python
# Correct
self.logger.info("Backend initialized with %d %s device(s)", len(devices), device_type)

# Incorrect
self.logger.info(f"Backend initialized with {len(devices)} {device_type} device(s)")
```

## Testing Patterns

### JAX Availability Checks
Always check JAX availability in tests:

```python
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXFeatures:
    def test_feature(self):
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
        # Test implementation
```

### Local Imports in Tests
For test-specific imports, use local imports with pylint disable:

```python
def test_jax_feature(self):
    # Import JAX numpy locally since it's only used in this test
    import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
```

## Hardware Detection

### Subprocess Usage
When using subprocess for hardware detection, always specify check parameter:

```python
result = subprocess.run(
    ["nvidia-smi"], 
    capture_output=True, 
    text=True, 
    timeout=5, 
    check=False  # Explicitly set check parameter
)
```

## Memory Management

### Return Type Annotations
Use flexible return types for functions that may return different types based on JAX availability:

```python
def get_memory_usage(self) -> Dict[str, Any]:  # Use Any instead of specific types
    # Implementation that may return different value types
```

## Configuration Management

### Optional Fields in Dataclasses
Use Optional types for fields that may be None:

```python
@dataclass
class NNXConfig:
    xla_flags: Optional[Dict[str, Any]] = None
    mesh_shape: Optional[Tuple[int, ...]] = None
```

### Null Checking
Always check for None before indexing optional fields:

```python
if mesh_shape is not None and len(self.devices) != mesh_shape[0]:
    # Safe to index mesh_shape[0]
```