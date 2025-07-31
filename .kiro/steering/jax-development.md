---
inclusion: fileMatch
fileMatchPattern: '*jax*|*nnx*|*tpu*'
---

# JAX/NNX Development Guidelines for CayleyPy

## MANDATORY: Use Flax NNX for All Neural Network Components

**ALL neural network implementations MUST use Flax NNX (Next Generation) instead of legacy Flax or other frameworks.**

### Required NNX Patterns
- Use `nnx.Module` as base class for all neural network components
- Use `nnx.Variable` for mutable state management
- Use `nnx.Param` for trainable parameters
- Use `nnx.jit` for compilation instead of raw `jax.jit`
- Use `nnx.Rngs` for random number generation

### Example NNX Module Structure
```python
class TPUTensorOpsModule(nnx.Module):
    """NNX module for TPU-accelerated tensor operations."""
    
    def __init__(self, backend: TPUBackend, rngs: Optional[nnx.Rngs] = None):
        self.backend = backend
        self.metrics = nnx.Variable({
            'operations_count': 0,
            'total_elements_processed': 0
        })
    
    @nnx.jit
    def operation(self, x: jnp.ndarray) -> jnp.ndarray:
        # Implementation using NNX patterns
        pass
```

## MANDATORY: Consult MCP Servers for Implementation

**ALWAYS consult these MCP servers before implementing JAX/NNX code:**

### Required MCP Server Consultations
1. **context7**: For up-to-date JAX and Flax NNX documentation
   - Query for latest NNX patterns and best practices
   - Verify API compatibility and method signatures
   - Check for recent changes in NNX architecture

2. **web-search-serper**: For current JAX/NNX examples and community practices
   - Search for recent NNX implementation examples
   - Find TPU v6e optimization techniques
   - Locate performance benchmarking approaches

### Consultation Workflow
```python
# 1. First consult context7 for official documentation
# Query: "Flax NNX Module implementation patterns"
# Query: "JAX TPU v6e int64 support"

# 2. Then consult web-search-serper for community examples
# Search: "Flax NNX TPU implementation examples 2024"
# Search: "JAX TPU v6e native int64 operations"

# 3. Implement based on gathered information
```

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

### Unused Parameter Handling
When parameters are required by the interface but not used in the implementation, explicitly delete them:

```python
def __init__(self, backend: TPUBackend, rngs: Optional[nnx.Rngs] = None):
    del rngs  # Unused parameter - explicitly delete to silence warnings
    # Implementation continues...
```

### Import Cleanup
Remove unused imports to avoid linting warnings:

```python
# Only import what you actually use
from typing import Optional, Dict, Any  # Remove Tuple if not used

# If imports are needed for type annotations but not runtime, use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple  # Only imported for type checking
```

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

### Exception Chaining
Always use `from e` when re-raising exceptions to maintain the exception chain:

```python
# Correct - maintains exception chain
try:
    backend = TPUBackend()
except Exception as e:
    raise RuntimeError(f"TPU backend initialization failed: {e}") from e

# Incorrect - breaks exception chain
try:
    backend = TPUBackend()
except Exception as e:
    raise RuntimeError(f"TPU backend initialization failed: {e}")
```

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

### Type Annotations for NNX Variables
Always provide explicit type annotations for NNX Variables to avoid mypy errors:

```python
# Correct - explicit type annotation
self.operation_cache: nnx.Variable[Dict[str, Any]] = nnx.Variable({})

# Incorrect - missing type annotation (causes mypy error)
self.operation_cache = nnx.Variable({})
```

### Accessing Variable Values
Always check for JAX availability when accessing Variable values:

```python
if JAX_AVAILABLE and hasattr(self.metrics, "value"):
    metrics: Dict[str, Any] = dict(self.metrics.value)
else:
    metrics = dict(self.metrics)
```

### Type Casting for Dictionary Values
When accessing dictionary values that might be objects, cast them appropriately:

```python
# Correct - explicit type annotation for dictionary value
hbm_per_chip: Any = self.capabilities.value["hbm_per_chip_gb"]
total_memory_gb = len(self.devices) * float(hbm_per_chip)

# Incorrect - direct float conversion of object (causes mypy error)
total_memory_gb = len(self.devices) * float(self.capabilities.value["hbm_per_chip_gb"])
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

### Logger Initialization
Always use the full method name for logger initialization:

```python
# Correct
self.logger = logging.getLogger(__name__)

# Incorrect (will cause mypy/pylint errors)
self.logger = logging.getL  # Truncated method name
```

## String Handling

### Avoid Implicit String Concatenation
Use explicit string concatenation or multi-line strings:

```python
# Correct - explicit concatenation
raise ImportError(
    "JAX and Flax are required for TPU backend. "
    "Install with: pip install 'cayleypy[jax-tpu]'"
)

# Incorrect - implicit concatenation
raise ImportError(
    "JAX and Flax are required for TPU backend. " "Install with: pip install 'cayleypy[jax-tpu]'"
)
```

### F-string Usage
Only use f-strings when there are actual interpolated variables:

```python
# Correct - has interpolated variables
print(f"TPU v6e Backend initialized with {len(self.devices)} devices")

# Incorrect - no interpolated variables
print(f"Native int64 support: VERIFIED ✓")  # Should be regular string

# Correct alternative
print("Native int64 support: VERIFIED ✓")
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

### Unused Variable Handling in Tests
When variables are created for testing but not used in assertions, explicitly delete them:

```python
def test_backend_initialization(self):
    """Test backend can be initialized."""
    backend = TPUBackend()
    assert backend.is_available
    del backend  # Silence unused variable warning
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

## TPU v6e Native int64 Support

### Direct int64 Operations
TPU v6e (Trillium) supports int64 operations natively when x64 is enabled:

```python
class TPUBackend:
    def __init__(self):
        # Enable native int64 support on TPU v6e
        jax.config.update("jax_enable_x64", True)
        
        # Test int64 support
        test_array = jnp.array([1, 2, 3], dtype=jnp.int64)
        assert test_array.dtype == jnp.int64
```

### Native int64 Hashing
Use direct int64 operations on TPU without conversion:

```python
class TPUHasher:
    def hash_state(self, state: jnp.ndarray) -> jnp.int64:
        """Hash single state using native int64 operations on TPU."""
        return jnp.sum(
            state.astype(jnp.int64) * self.hash_matrix[:, 0]
        ) % (2**63 - 1)
```

### Simplified Error Handling
With native int64 support, error handling is simplified:

```python
try:
    # TPU operation with native int64 support
    result = tpu_operation(data.astype(jnp.int64))
except Exception as e:  # pylint: disable=broad-exception-caught
    # Fallback to CPU only if TPU is unavailable
    if "TPU not available" in str(e):
        result = cpu_operation(data)
    else:
        raise
```