# Requirements Document

## Introduction

This specification outlines the conversion of CayleyPy from PyTorch/CUDA to JAX/TPU architecture. CayleyPy is a Python library for analyzing extremely large graphs using mathematical group theory and AI/ML methods, focusing on state-space graphs too large for memory storage. The conversion aims to leverage JAX's superior performance on TPUs, functional programming paradigm, and advanced compilation capabilities while maintaining full API compatibility and mathematical correctness.

## Requirements

### Requirement 1: Core Architecture Migration

**User Story:** As a CayleyPy developer, I want the library to use JAX as the primary tensor computation backend instead of PyTorch, so that I can leverage TPU acceleration and JAX's advanced compilation features.

#### Acceptance Criteria

1. WHEN the library is imported THEN JAX SHALL be the primary tensor computation backend
2. WHEN tensor operations are performed THEN they SHALL use JAX arrays instead of PyTorch tensors
3. WHEN device selection occurs THEN the system SHALL support CPU, GPU, and TPU devices through JAX
4. WHEN memory management is needed THEN JAX-specific memory optimization techniques SHALL be used
5. IF TPU is available THEN the system SHALL automatically detect and utilize TPU resources
6. WHEN compilation is beneficial THEN JAX JIT compilation SHALL be applied to performance-critical functions

### Requirement 2: API Compatibility Preservation

**User Story:** As a CayleyPy user, I want the public API to remain unchanged after the JAX conversion, so that my existing code continues to work without modifications.

#### Acceptance Criteria

1. WHEN existing CayleyPy code is run THEN all public methods SHALL maintain identical signatures
2. WHEN CayleyGraph objects are created THEN the constructor parameters SHALL remain the same
3. WHEN BFS operations are performed THEN the return types and data structures SHALL be identical
4. WHEN state encoding/decoding occurs THEN the input/output formats SHALL be preserved
5. WHEN beam search is executed THEN the API SHALL remain functionally equivalent
6. IF backward compatibility breaks THEN deprecation warnings SHALL be provided for at least one major version

### Requirement 3: Performance Optimization for TPU

**User Story:** As a researcher analyzing large symmetric groups, I want CayleyPy to achieve superior performance on TPUs compared to the current GPU implementation, so that I can analyze larger state spaces efficiently.

#### Acceptance Criteria

1. WHEN BFS operations run on TPU THEN performance SHALL exceed current GPU implementation by at least 2x
2. WHEN large matrix operations are performed THEN TPU vectorization SHALL be fully utilized
3. WHEN memory-intensive operations occur THEN TPU memory hierarchy SHALL be optimized
4. WHEN batch processing happens THEN TPU-specific batch sizes SHALL be automatically determined
5. IF operations are TPU-incompatible THEN graceful fallback to CPU/GPU SHALL occur
6. WHEN compilation overhead exists THEN JIT compilation SHALL provide net performance benefits

### Requirement 4: Functional Programming Paradigm

**User Story:** As a CayleyPy maintainer, I want the internal implementation to follow JAX's functional programming paradigm, so that the code is more maintainable and benefits from JAX's optimization features.

#### Acceptance Criteria

1. WHEN internal functions are implemented THEN they SHALL be pure functions without side effects
2. WHEN state mutations are needed THEN functional updates SHALL be used instead of in-place modifications
3. WHEN random number generation occurs THEN JAX's PRNG system SHALL be used with explicit keys
4. WHEN control flow is implemented THEN JAX-compatible constructs (lax.cond, lax.while_loop) SHALL be used
5. IF imperative patterns exist THEN they SHALL be refactored to functional equivalents
6. WHEN debugging is needed THEN JAX debugging tools SHALL be integrated

### Requirement 5: Advanced Compilation Features

**User Story:** As a performance-conscious user, I want CayleyPy to leverage JAX's advanced compilation features like XLA and automatic differentiation, so that I can achieve maximum computational efficiency.

#### Acceptance Criteria

1. WHEN performance-critical functions are called THEN JIT compilation SHALL be automatically applied
2. WHEN vectorized operations are possible THEN vmap SHALL be used for automatic vectorization
3. WHEN parallel computation is beneficial THEN pmap SHALL be used for multi-device parallelization
4. WHEN gradient computation is needed THEN JAX's automatic differentiation SHALL be available
5. IF compilation fails THEN informative error messages SHALL guide users to fix issues
6. WHEN XLA optimization is possible THEN it SHALL be enabled by default for supported operations

### Requirement 6: Intelligent Dependency Management

**User Story:** As a CayleyPy user, I want JAX dependencies to be managed intelligently with optional dependency sections (similar to the existing `torch` section) so that I can install and use the library on both TPU (like v3-8) and non-TPU (like L4, A100) devices without dependency conflicts.

#### Acceptance Criteria

1. WHEN installing with `pip install cayleypy[jax-cpu]` THEN JAX with CPU-only support SHALL be installed
2. WHEN installing with `pip install cayleypy[jax-cuda]` THEN JAX with CUDA/GPU support SHALL be installed for devices like L4 and A100
3. WHEN installing with `pip install cayleypy[jax-tpu]` THEN JAX with TPU support SHALL be installed for devices like v3-8
4. WHEN installing with `pip install cayleypy[jax]` THEN the most appropriate JAX variant SHALL be automatically selected based on detected hardware
5. WHEN JAX dependencies are missing THEN informative error messages SHALL guide users to install appropriate extras (e.g., "Install with pip install cayleypy[jax-cuda] for GPU support")
6. WHEN multiple JAX variants are available THEN the system SHALL automatically select the optimal backend (TPU > GPU > CPU priority)
7. WHEN dependency conflicts occur THEN clear resolution instructions SHALL be provided with specific pip install commands
8. WHEN PyTorch dependencies exist alongside JAX THEN both SHALL coexist without conflicts for gradual migration
9. IF hardware detection fails THEN fallback installation instructions SHALL be provided for manual selection

### Requirement 7: Device Management and Scalability

**User Story:** As a user with access to multiple compute devices, I want CayleyPy to intelligently manage and scale across available TPU, GPU, and CPU resources, so that I can maximize computational throughput.

#### Acceptance Criteria

1. WHEN multiple TPU cores are available THEN computation SHALL be distributed across all cores
2. WHEN device memory is insufficient THEN automatic sharding SHALL be implemented
3. WHEN mixed-precision computation is beneficial THEN it SHALL be automatically applied
4. WHEN device selection occurs THEN the system SHALL choose the optimal device based on workload
5. IF device failures occur THEN graceful error handling and recovery SHALL be implemented
6. WHEN scaling to larger problems THEN memory usage SHALL scale efficiently with problem size

### Requirement 8: Environment-Aware Testing Framework

**User Story:** As a CayleyPy contributor, I want comprehensive tests that work across all environments (CPU, GPU/PyTorch, GPU/JAX, TPU/JAX) and automatically skip irrelevant tests, so that I can maintain code quality across different deployment scenarios.

#### Acceptance Criteria

1. WHEN tests run on CPU-only environments THEN GPU and TPU specific tests SHALL be automatically skipped
2. WHEN tests run on GPU environments THEN TPU-specific tests SHALL be skipped and GPU tests SHALL execute
3. WHEN tests run on TPU environments THEN all relevant tests SHALL execute including TPU-specific optimizations
4. WHEN PyTorch is unavailable THEN PyTorch-specific tests SHALL be skipped with appropriate markers
5. WHEN JAX is unavailable THEN JAX-specific tests SHALL be skipped with informative messages
6. WHEN mathematical operations are performed THEN results SHALL be numerically identical across available backends
7. WHEN performance benchmarks are run THEN they SHALL adapt to available hardware and skip unsupported configurations
8. WHEN continuous integration runs THEN test results SHALL clearly indicate which environment configurations were tested

### Requirement 9: Documentation and Migration Guide

**User Story:** As a CayleyPy user transitioning to the JAX version, I want clear documentation and migration guidance, so that I can understand the benefits and any necessary changes to my workflow.

#### Acceptance Criteria

1. WHEN users access documentation THEN JAX-specific features SHALL be clearly documented
2. WHEN migration is needed THEN step-by-step migration guides SHALL be provided
3. WHEN performance comparisons are made THEN benchmarks SHALL be documented with specific hardware configurations
4. WHEN troubleshooting is needed THEN common JAX-related issues SHALL be documented with solutions
5. IF breaking changes exist THEN they SHALL be clearly highlighted with workarounds
6. WHEN examples are provided THEN they SHALL demonstrate JAX-specific optimizations and best practices