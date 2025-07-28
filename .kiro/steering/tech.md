# CayleyPy Technical Stack

## Build System
- **Build backend**: setuptools
- **Package manager**: pip with optional dependencies
- **Python version**: >=3.9

## Core Dependencies
- **numpy**: Numerical computations and array operations
- **scipy**: Scientific computing utilities
- **numba**: JIT compilation for performance-critical code
- **h5py**: HDF5 file format support for datasets
- **kagglehub**: Integration with Kaggle datasets and models

## Optional Dependencies
- **torch**: PyTorch for neural network models (optional to avoid Kaggle installation delays)
- **networkx**: Graph analysis and visualization (dev dependency)
- **JAX**: GPU/TPU acceleration support with hardware-specific installations:
  - `cayleypy[jax]`: CPU-only JAX >=0.4.20
  - `cayleypy[jax-gpu]`: GPU JAX with CUDA 12 >=0.4.20
  - `cayleypy[jax-tpu]`: TPU JAX >=0.7.0

## Development Tools
- **Black**: Code formatting (line length: 120)
- **pylint**: Code linting with custom configuration
- **mypy**: Static type checking
- **pytest**: Testing framework with benchmarking support
- **coverage**: Test coverage analysis

## Documentation
- **Sphinx**: Documentation generation
- **sphinx_rtd_theme**: Read the Docs theme
- **sphinx_markdown_parser**: Markdown support in Sphinx

## Common Commands

### Development Setup
```bash
git clone https://github.com/cayleypy/cayleypy.git
cd cayleypy
pip install -e .[torch,jax,lint,test,dev,docs]  # CPU JAX
# OR for GPU: pip install -e .[torch,jax-gpu,lint,test,dev,docs]
# OR for TPU: pip install -e .[torch,jax-tpu,lint,test,dev,docs]
```

### Code Quality
```bash
./lint.sh                    # Run all linting checks
black .                      # Format code
pylint ./cayleypy           # Lint code
mypy ./cayleypy             # Type checking
```

### Testing
```bash
pytest                       # Run tests
RUN_SLOW_TESTS=1 pytest    # Run all tests including slow ones
coverage run -m pytest && coverage html  # Generate coverage report
```

### Documentation
```bash
./docs/build_docs.sh        # Build documentation locally
```

## Performance Considerations
- Use numba JIT compilation for performance-critical loops
- Leverage JAX for GPU/TPU acceleration when available
- Implement efficient bit manipulation for large state spaces
- Use hash functions for duplicate removal in beam search
## JAX
/NNX Development Guidelines

### Code Quality Standards
- Use `# pylint: disable=broad-exception-caught` for hardware detection and graceful fallback scenarios
- Use `# pylint: disable=global-statement` for singleton pattern implementations
- Use `# type: ignore` for optional JAX imports when JAX is not available
- Always use lazy % formatting in logging functions instead of f-strings

### Import Patterns
```python
try:
    import jax
    from flax import nnx
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    nnx = None  # type: ignore
```

### Testing Patterns
```python
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXFeatures:
    def test_feature(self):
        # Test implementation
```

### Type Annotations
- Use `Optional[Dict[str, Any]]` for dataclass fields that may be None
- Use `Dict[str, Any]` for return types that may contain mixed value types
- Always check for None before indexing optional tuple/list fields