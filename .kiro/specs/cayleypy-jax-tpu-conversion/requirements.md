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

### Requirement 6: Device Management and Scalability

**User Story:** As a user with access to multiple compute devices, I want CayleyPy to intelligently manage and scale across available TPU, GPU, and CPU resources, so that I can maximize computational throughput.

#### Acceptance Criteria

1. WHEN multiple TPU cores are available THEN computation SHALL be distributed across all cores
2. WHEN device memory is insufficient THEN automatic sharding SHALL be implemented
3. WHEN mixed-precision computation is beneficial THEN it SHALL be automatically applied
4. WHEN device selection occurs THEN the system SHALL choose the optimal device based on workload
5. IF device failures occur THEN graceful error handling and recovery SHALL be implemented
6. WHEN scaling to larger problems THEN memory usage SHALL scale efficiently with problem size

### Requirement 7: Testing and Validation Framework

**User Story:** As a CayleyPy contributor, I want comprehensive tests that validate the JAX conversion maintains mathematical correctness and performance, so that I can confidently deploy the new implementation.

#### Acceptance Criteria

1. WHEN mathematical operations are performed THEN results SHALL be numerically identical to PyTorch implementation
2. WHEN performance benchmarks are run THEN JAX implementation SHALL meet or exceed PyTorch performance
3. WHEN edge cases are tested THEN all existing test cases SHALL pass without modification
4. WHEN new JAX-specific features are added THEN corresponding tests SHALL be implemented
5. IF numerical differences exist THEN they SHALL be within acceptable floating-point precision tolerances
6. WHEN continuous integration runs THEN all tests SHALL pass on CPU, GPU, and TPU environments

### Requirement 8: Documentation and Migration Guide

**User Story:** As a CayleyPy user transitioning to the JAX version, I want clear documentation and migration guidance, so that I can understand the benefits and any necessary changes to my workflow.

#### Acceptance Criteria

1. WHEN users access documentation THEN JAX-specific features SHALL be clearly documented
2. WHEN migration is needed THEN step-by-step migration guides SHALL be provided
3. WHEN performance comparisons are made THEN benchmarks SHALL be documented with specific hardware configurations
4. WHEN troubleshooting is needed THEN common JAX-related issues SHALL be documented with solutions
5. IF breaking changes exist THEN they SHALL be clearly highlighted with workarounds
6. WHEN examples are provided THEN they SHALL demonstrate JAX-specific optimizations and best practices