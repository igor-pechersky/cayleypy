# Design Document

## Overview

This design document outlines the conversion of CayleyPy from PyTorch/CUDA to JAX/TPU architecture. The conversion will maintain full API compatibility while leveraging JAX's functional programming paradigm, advanced compilation features, and superior TPU performance. The design focuses on systematic replacement of PyTorch components with JAX equivalents while introducing TPU-specific optimizations.

## Architecture

### High-Level Architecture Changes

The conversion follows a layered approach:

1. **Tensor Backend Layer**: Replace PyTorch tensors with JAX arrays
2. **Device Management Layer**: Replace CUDA device management with JAX device abstraction
3. **Computation Layer**: Convert imperative PyTorch operations to functional JAX operations
4. **Compilation Layer**: Add JIT compilation and optimization passes
5. **Memory Management Layer**: Implement JAX-specific memory optimization strategies

### Core Components Migration

```
PyTorch Components → JAX Equivalents
├── torch.Tensor → jax.Array
├── torch.device → jax.devices()
├── torch.cuda → jax.device_put/device_get
├── torch.unique → jax.numpy.unique (with custom implementation)
├── torch.gather → advanced indexing with jax.numpy
├── torch.sort → jax.numpy.sort
├── torch.randint → jax.random with explicit keys
└── torch.vstack/hstack → jax.numpy.vstack/hstack
```

## Components and Interfaces

### 1. Device Management (`jax_device_manager.py`)

**Purpose**: Abstract device selection and management across CPU, GPU, and TPU

```python
class JAXDeviceManager:
    def __init__(self, device: str = "auto"):
        self.device_type = self._select_device(device)
        self.devices = jax.devices(self.device_type)
        self.primary_device = self.devices[0]
    
    def _select_device(self, device: str) -> str:
        # Auto-detection logic: TPU > GPU > CPU
        
    def put_on_device(self, array: jax.Array) -> jax.Array:
        # Device placement with sharding for large arrays
        
    def get_memory_info(self) -> dict:
        # Device memory information
```

### 2. Tensor Operations (`jax_tensor_ops.py`)

**Purpose**: Provide PyTorch-compatible tensor operations using JAX

```python
class JAXTensorOps:
    @staticmethod
    @jax.jit
    def unique_with_indices(arr: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Custom implementation of torch.unique functionality
        
    @staticmethod
    @jax.jit
    def gather_along_axis(arr: jax.Array, indices: jax.Array, axis: int) -> jax.Array:
        # Equivalent to torch.gather
        
    @staticmethod
    @jax.jit
    def searchsorted_mask(sorted_arr: jax.Array, values: jax.Array) -> jax.Array:
        # Equivalent to isin_via_searchsorted
```

### 3. State Encoding (`jax_string_encoder.py`)

**Purpose**: Convert string encoder from PyTorch to JAX with vectorization

```python
class JAXStringEncoder:
    def __init__(self, code_width: int, n: int):
        self.code_width = code_width
        self.n = n
        self.encoded_length = self._compute_encoded_length()
    
    @partial(jax.jit, static_argnums=(0,))
    def encode(self, states: jax.Array) -> jax.Array:
        # Vectorized encoding using JAX operations
        
    @partial(jax.jit, static_argnums=(0,))
    def decode(self, encoded_states: jax.Array) -> jax.Array:
        # Vectorized decoding using JAX operations
```

### 4. Hashing System (`jax_hasher.py`)

**Purpose**: Convert state hashing to JAX with TPU optimization

```python
class JAXStateHasher:
    def __init__(self, graph, random_seed: Optional[int], chunk_size: int):
        self.rng_key = jax.random.PRNGKey(random_seed or 42)
        self.chunk_size = chunk_size
        self.hash_params = self._initialize_hash_params()
    
    @partial(jax.jit, static_argnums=(0,))
    def make_hashes(self, states: jax.Array) -> jax.Array:
        # Vectorized hashing optimized for TPU
        
    @partial(jax.jit, static_argnums=(0,))
    def make_hashes_chunked(self, states: jax.Array) -> jax.Array:
        # Memory-efficient chunked hashing
```

### 5. Core CayleyGraph (`jax_cayley_graph.py`)

**Purpose**: Main graph class with JAX backend

```python
class JAXCayleyGraph:
    def __init__(self, definition: CayleyGraphDef, **kwargs):
        self.device_manager = JAXDeviceManager(kwargs.get('device', 'auto'))
        self.tensor_ops = JAXTensorOps()
        self.hasher = JAXStateHasher(self, kwargs.get('random_seed'), kwargs.get('hash_chunk_size', 2**25))
        # Initialize other components
    
    @partial(jax.jit, static_argnums=(0, 1))
    def _apply_generator_batched(self, gen_idx: int, states: jax.Array) -> jax.Array:
        # JIT-compiled generator application
        
    @partial(jax.jit, static_argnums=(0,))
    def get_neighbors(self, states: jax.Array) -> jax.Array:
        # Vectorized neighbor generation
        
    def bfs(self, **kwargs) -> BfsResult:
        # BFS with TPU-optimized batching and compilation
```

### 6. TPU Optimization Layer (`tpu_optimizations.py`)

**Purpose**: TPU-specific optimizations and compilation strategies

```python
class TPUOptimizer:
    @staticmethod
    def optimize_for_tpu(func):
        # Decorator for TPU-specific optimizations
        
    @staticmethod
    def shard_large_arrays(arr: jax.Array, num_devices: int) -> jax.Array:
        # Automatic sharding for large state spaces
        
    @staticmethod
    def compile_bfs_kernel(graph_def: CayleyGraphDef):
        # Pre-compile BFS kernels for specific graph types
```

## Data Models

### JAX Array Representations

**State Representation**:
- Internal: `jax.Array` with shape `(batch_size, state_size)` and dtype `jnp.int64`
- Encoded: `jax.Array` with shape `(batch_size, encoded_size)` for bit-packed states
- Hashed: `jax.Array` with shape `(batch_size,)` for hash values

**Generator Representation**:
- Permutations: `jax.Array` with shape `(n_generators, state_size)` and dtype `jnp.int64`
- Matrices: List of `jax.Array` matrices for matrix groups
- Encoded generators: Pre-compiled functions for bit-packed operations

### Memory Layout Optimization

**TPU Memory Hierarchy**:
```
HBM (High Bandwidth Memory) ← Large state spaces, BFS layers
VMEM (Vector Memory) ← Active computations, generators
SMEM (Scalar Memory) ← Control flow, small arrays
```

**Sharding Strategy**:
- Automatic sharding for arrays > 1GB
- Replicated small arrays (generators, central state)
- Partitioned large arrays (BFS layers, neighbor computations)

## Error Handling

### Device Fallback Strategy

```python
class DeviceFallbackHandler:
    def __init__(self, preferred_devices: list[str]):
        self.device_hierarchy = preferred_devices
        
    def execute_with_fallback(self, func, *args, **kwargs):
        for device in self.device_hierarchy:
            try:
                return self._execute_on_device(func, device, *args, **kwargs)
            except (OutOfMemoryError, DeviceNotFoundError) as e:
                self._log_fallback(device, str(e))
                continue
        raise RuntimeError("All devices failed")
```

### Compilation Error Handling

```python
class CompilationErrorHandler:
    @staticmethod
    def safe_jit(func, static_argnums=None, device=None):
        try:
            return jax.jit(func, static_argnums=static_argnums, device=device)
        except JAXCompilationError as e:
            # Fallback to non-compiled version with warning
            warnings.warn(f"JIT compilation failed: {e}. Using non-compiled version.")
            return func
```

### Numerical Stability

```python
class NumericalStabilityHandler:
    @staticmethod
    def safe_unique(arr: jax.Array, tolerance: float = 1e-10) -> jax.Array:
        # Handle floating-point precision issues in unique operations
        
    @staticmethod
    def safe_hash(states: jax.Array) -> jax.Array:
        # Ensure hash consistency across different precisions
```

## Testing Strategy

### Compatibility Testing

**Numerical Equivalence Tests**:
```python
class CompatibilityTester:
    def test_operation_equivalence(self, pytorch_op, jax_op, test_inputs):
        pytorch_result = pytorch_op(*test_inputs)
        jax_result = jax_op(*test_inputs)
        assert jnp.allclose(pytorch_result, jax_result, rtol=1e-10)
        
    def test_bfs_equivalence(self, graph_def):
        pytorch_graph = PyTorchCayleyGraph(graph_def)
        jax_graph = JAXCayleyGraph(graph_def)
        
        pytorch_bfs = pytorch_graph.bfs(max_diameter=5)
        jax_bfs = jax_graph.bfs(max_diameter=5)
        
        assert pytorch_bfs.layer_sizes == jax_bfs.layer_sizes
        # Compare layer contents with numerical tolerance
```

**Performance Benchmarking**:
```python
class PerformanceBenchmark:
    def benchmark_bfs_performance(self, graph_def, device_types):
        results = {}
        for device in device_types:
            graph = JAXCayleyGraph(graph_def, device=device)
            start_time = time.time()
            result = graph.bfs(max_diameter=10)
            end_time = time.time()
            results[device] = {
                'time': end_time - start_time,
                'memory_peak': self._get_peak_memory(device),
                'layer_sizes': result.layer_sizes
            }
        return results
```

### TPU-Specific Testing

**Multi-Device Testing**:
```python
class TPUTester:
    def test_multi_core_scaling(self, graph_def):
        for num_cores in [1, 2, 4, 8]:
            graph = JAXCayleyGraph(graph_def, device='tpu')
            # Test scaling efficiency
            
    def test_memory_sharding(self, large_graph_def):
        # Test automatic sharding for large state spaces
        
    def test_compilation_caching(self, graph_def):
        # Test JIT compilation caching across runs
```

## Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)
- Device management system
- Basic tensor operations
- Memory management utilities
- Error handling framework

### Phase 2: State Management (Weeks 3-4)
- State encoding/decoding
- Hashing system
- Generator application
- Basic neighbor computation

### Phase 3: Graph Operations (Weeks 5-6)
- BFS algorithm conversion
- Random walks
- Path finding utilities
- Beam search

### Phase 4: Optimization (Weeks 7-8)
- JIT compilation integration
- TPU-specific optimizations
- Memory sharding
- Performance tuning

### Phase 5: Testing & Validation (Weeks 9-10)
- Comprehensive test suite
- Performance benchmarking
- Documentation updates
- Migration guide creation

## Performance Considerations

### TPU Optimization Strategies

**Vectorization**:
- Use `jax.vmap` for batch operations
- Leverage TPU's matrix multiplication units
- Minimize scalar operations in inner loops

**Memory Access Patterns**:
- Coalesce memory accesses
- Use appropriate data layouts for TPU
- Minimize host-device transfers

**Compilation Optimization**:
- Pre-compile frequently used kernels
- Use static arguments for shape specialization
- Cache compiled functions across runs

### Scalability Targets

**Performance Goals**:
- 2x speedup on TPU vs current GPU implementation
- Support for state spaces up to 10^12 states
- Linear scaling with number of TPU cores
- Memory usage within 80% of available TPU memory

**Scalability Metrics**:
- BFS performance: states/second
- Memory efficiency: states/GB
- Compilation overhead: < 10% of total runtime
- Multi-device scaling efficiency: > 80%

## Dependency Management Strategy

### Intelligent JAX Installation

**Optional Dependencies Structure**:
```python
# pyproject.toml
[project.optional-dependencies]
jax-cpu = [
    "jax[cpu]>=0.4.0",
]
jax-cuda = [
    "jax[cuda12]>=0.4.0",  # or cuda11 based on user needs
]
jax-tpu = [
    "jax[tpu]>=0.4.0",
]
# Convenience meta-packages
jax-gpu = ["cayleypy[jax-cuda]"]
jax-all = ["cayleypy[jax-cpu,jax-cuda,jax-tpu]"]
```

**Runtime Dependency Detection**:
```python
class DependencyManager:
    def __init__(self):
        self.available_backends = self._detect_available_backends()
        self.recommended_install = self._get_install_recommendation()
    
    def _detect_available_backends(self) -> dict[str, bool]:
        backends = {
            'pytorch': self._check_pytorch(),
            'jax-cpu': self._check_jax_cpu(),
            'jax-gpu': self._check_jax_gpu(), 
            'jax-tpu': self._check_jax_tpu()
        }
        return backends
    
    def _check_jax_gpu(self) -> bool:
        try:
            import jax
            return len(jax.devices('gpu')) > 0
        except (ImportError, RuntimeError):
            return False
    
    def _check_jax_tpu(self) -> bool:
        try:
            import jax
            return len(jax.devices('tpu')) > 0
        except (ImportError, RuntimeError):
            return False
    
    def get_install_command(self, detected_hardware: str) -> str:
        commands = {
            'tpu': "pip install 'cayleypy[jax-tpu]'",
            'gpu': "pip install 'cayleypy[jax-cuda]'", 
            'cpu': "pip install 'cayleypy[jax-cpu]'"
        }
        return commands.get(detected_hardware, commands['cpu'])
```

### Environment-Aware Testing Framework

**Test Environment Detection**:
```python
class TestEnvironment:
    def __init__(self):
        self.has_pytorch = self._check_pytorch()
        self.has_jax = self._check_jax()
        self.available_devices = self._detect_devices()
        self.test_markers = self._generate_markers()
    
    def _detect_devices(self) -> dict[str, bool]:
        devices = {'cpu': True}  # CPU always available
        
        if self.has_jax:
            try:
                import jax
                devices['jax-gpu'] = len(jax.devices('gpu')) > 0
                devices['jax-tpu'] = len(jax.devices('tpu')) > 0
            except RuntimeError:
                pass
                
        if self.has_pytorch:
            try:
                import torch
                devices['pytorch-gpu'] = torch.cuda.is_available()
            except RuntimeError:
                pass
                
        return devices
    
    def _generate_markers(self) -> list[str]:
        markers = []
        for device, available in self.available_devices.items():
            if not available:
                markers.append(f"skip_{device.replace('-', '_')}")
        return markers
```

**Pytest Configuration**:
```python
# conftest.py
import pytest
from cayleypy.testing import TestEnvironment

test_env = TestEnvironment()

def pytest_configure(config):
    # Register custom markers
    config.addinivalue_line("markers", "requires_pytorch: mark test as requiring PyTorch")
    config.addinivalue_line("markers", "requires_jax: mark test as requiring JAX")
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "requires_tpu: mark test as requiring TPU")

def pytest_collection_modifyitems(config, items):
    for item in items:
        # Skip tests based on available hardware/software
        if "requires_pytorch" in item.keywords and not test_env.has_pytorch:
            item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
        if "requires_jax" in item.keywords and not test_env.has_jax:
            item.add_marker(pytest.mark.skip(reason="JAX not available"))
        if "requires_gpu" in item.keywords and not any(test_env.available_devices[k] for k in ['jax-gpu', 'pytorch-gpu']):
            item.add_marker(pytest.mark.skip(reason="GPU not available"))
        if "requires_tpu" in item.keywords and not test_env.available_devices.get('jax-tpu', False):
            item.add_marker(pytest.mark.skip(reason="TPU not available"))
```

**Smart Test Execution**:
```python
class EnvironmentAwareTest:
    @pytest.mark.requires_jax
    @pytest.mark.requires_gpu
    def test_jax_gpu_performance(self):
        """Test JAX GPU performance - skipped if JAX or GPU unavailable"""
        pass
    
    @pytest.mark.requires_tpu
    def test_tpu_scaling(self):
        """Test TPU scaling - skipped if TPU unavailable"""
        pass
    
    def test_cpu_fallback(self):
        """Test CPU fallback - always runs"""
        pass
    
    @pytest.mark.parametrize("backend", ["pytorch", "jax"])
    def test_backend_equivalence(self, backend):
        """Test backend equivalence - skips unavailable backends"""
        if backend == "pytorch" and not test_env.has_pytorch:
            pytest.skip("PyTorch not available")
        if backend == "jax" and not test_env.has_jax:
            pytest.skip("JAX not available")
        # Test implementation
```

## Migration Strategy

### Backward Compatibility

**API Preservation**:
- Maintain identical public method signatures
- Preserve return types and data structures
- Support existing parameter names and defaults
- Provide deprecation warnings for removed features

**Gradual Migration Path**:
1. Install JAX alongside PyTorch using optional dependencies
2. Use feature flags to enable JAX backend
3. Run parallel validation during transition
4. Deprecate PyTorch backend after validation
5. Remove PyTorch dependencies in future version

### Configuration Management

```python
class BackendConfig:
    def __init__(self):
        self.dependency_manager = DependencyManager()
        self.backend = self._select_backend()
        self.device = os.environ.get('CAYLEYPY_DEVICE', 'auto')
        self.enable_jit = os.environ.get('CAYLEYPY_JIT', 'true').lower() == 'true'
    
    def _select_backend(self) -> str:
        # Priority: environment variable > available backends > error
        env_backend = os.environ.get('CAYLEYPY_BACKEND')
        if env_backend:
            if env_backend in self.dependency_manager.available_backends:
                return env_backend
            else:
                raise RuntimeError(f"Requested backend '{env_backend}' not available. "
                                 f"Install with: {self.dependency_manager.get_install_command('auto')}")
        
        # Auto-select best available backend
        if self.dependency_manager.available_backends.get('jax-tpu'):
            return 'jax'
        elif self.dependency_manager.available_backends.get('jax-gpu'):
            return 'jax'
        elif self.dependency_manager.available_backends.get('pytorch'):
            return 'pytorch'
        elif self.dependency_manager.available_backends.get('jax-cpu'):
            return 'jax'
        else:
            raise RuntimeError("No compatible backend found. Install JAX or PyTorch.")
        
    def get_graph_class(self):
        if self.backend == 'pytorch':
            return PyTorchCayleyGraph
        elif self.backend == 'jax':
            return JAXCayleyGraph
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
```

This design provides a comprehensive roadmap for converting CayleyPy to JAX/TPU while maintaining compatibility and achieving superior performance on TPU hardware.