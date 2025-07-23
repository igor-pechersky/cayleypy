# Implementation Plan

- [x] 1. Set up JAX infrastructure and device management
  - Create JAX device management system with automatic TPU/GPU/CPU detection
  - Implement device placement utilities and memory management functions
  - Write unit tests for device selection and memory allocation
  - _Requirements: 1.1, 1.3, 1.4, 6.1, 6.4_

- [x] 2. Implement core JAX tensor operations
  - [x] 2.1 Create JAX equivalents for PyTorch tensor operations
    - Implement `unique_with_indices` function to replace `torch.unique`
    - Create `gather_along_axis` function to replace `torch.gather`
    - Implement `searchsorted_mask` function to replace `isin_via_searchsorted`
    - Write comprehensive unit tests for all tensor operations
    - _Requirements: 1.2, 7.1, 7.3_

  - [x] 2.2 Implement JAX-based state hashing system
    - Convert `StateHasher` class from PyTorch to JAX with vectorized operations
    - Implement chunked hashing for memory efficiency on TPU
    - Add JIT compilation for hash computation functions
    - Create unit tests comparing hash outputs with PyTorch implementation
    - _Requirements: 1.2, 3.2, 4.1, 5.1_

- [x] 3. Convert state encoding and decoding systems
  - [x] 3.1 Implement JAX string encoder
    - Convert `StringEncoder` class to use JAX arrays and operations
    - Add vectorized encoding/decoding with `jax.vmap`
    - Implement JIT compilation for encoding functions
    - Write unit tests to verify encoding/decoding equivalence with PyTorch
    - _Requirements: 1.2, 2.3, 4.2, 5.2_

  - [x] 3.2 Create permutation and matrix generator systems
    - Implement JAX-based permutation application using advanced indexing
    - Convert matrix generator operations to JAX with batch processing
    - Add JIT compilation for generator application functions
    - Write unit tests for generator operations across different group types
    - _Requirements: 1.2, 2.3, 4.2, 5.1_

- [x] 4. Implement core CayleyGraph functionality
  - [x] 4.1 Create JAX CayleyGraph class structure
    - Implement `JAXCayleyGraph` class with identical public API to PyTorch version
    - Set up device management and memory allocation in constructor
    - Implement state encoding/decoding methods using JAX backend
    - Write unit tests for basic graph initialization and state operations
    - _Requirements: 2.1, 2.2, 2.3, 1.1_

  - [x] 4.2 Implement neighbor generation and path operations
    - Convert `get_neighbors` method to use JAX with vectorized operations
    - Implement `apply_path` method with JAX generator applications
    - Add JIT compilation for neighbor computation functions
    - Write unit tests comparing neighbor generation with PyTorch implementation
    - _Requirements: 2.3, 4.2, 5.1, 7.1_

- [x] 5. Convert BFS algorithm to JAX
  - [x] 5.1 Implement core BFS loop with JAX operations
    - Convert main BFS iteration loop to use JAX arrays and operations
    - Implement unique state detection using JAX tensor operations
    - Add memory-efficient batching for large state spaces
    - Write unit tests for BFS layer generation and state deduplication
    - _Requirements: 2.3, 3.1, 3.2, 7.1_

  - [x] 5.2 Add TPU-specific BFS optimizations
    - Implement automatic sharding for large BFS layers across TPU cores
    - Add JIT compilation for BFS inner loops and state processing
    - Optimize memory access patterns for TPU architecture
    - Write performance tests comparing BFS speed on TPU vs GPU
    - _Requirements: 3.1, 3.2, 3.3, 5.1, 6.2_

- [ ] 6. Refactor and eliminate code duplication
  - [ ] 6.1 Create shared base classes and utilities
    - Extract common functionality between PyTorch and JAX implementations into abstract base classes
    - Create shared utility functions for common operations (state encoding, hashing, etc.)
    - Implement backend-agnostic interfaces for tensor operations
    - Write unit tests for shared components to ensure backend compatibility
    - _Requirements: 2.1, 2.2, 4.1, 4.2_

  - [ ] 6.2 Consolidate string encoder implementations
    - Create unified string encoder interface with backend-specific implementations
    - Extract common permutation logic into shared utility functions
    - Implement factory pattern for creating appropriate encoder based on backend
    - Write unit tests comparing encoder outputs across backends
    - _Requirements: 2.3, 4.1, 4.2_

  - [ ] 6.3 Unify BFS algorithm implementations
    - Extract core BFS logic into backend-agnostic algorithm class
    - Create backend-specific tensor operation adapters
    - Implement shared path reconstruction and result formatting
    - Write integration tests ensuring identical BFS results across backends
    - _Requirements: 2.3, 3.1, 7.1, 7.3_

  - [ ] 6.4 Consolidate hash computation systems
    - Create unified hashing interface with backend-specific implementations
    - Extract common hash computation algorithms into shared utilities
    - Implement consistent hash output across PyTorch and JAX backends
    - Write unit tests verifying hash consistency and performance
    - _Requirements: 3.2, 4.1, 5.1, 7.5_

- [ ] 7. Implement advanced graph algorithms
  - [ ] 7.1 Convert random walks to JAX
    - Implement `random_walks` method using JAX random number generation
    - Convert both "classic" and "bfs" random walk modes to JAX
    - Add proper PRNG key management for reproducible random walks
    - Write unit tests for random walk generation and state distribution
    - _Requirements: 2.3, 4.3, 7.1_

  - [ ] 7.2 Convert beam search algorithm
    - Implement `beam_search` method using JAX arrays and operations
    - Add JIT compilation for beam search scoring and selection
    - Implement path reconstruction using JAX operations
    - Write unit tests for beam search pathfinding and optimization
    - _Requirements: 2.3, 4.2, 5.1, 7.1_

- [ ] 8. Add compilation and optimization features
  - [ ] 8.1 Implement JIT compilation system
    - Add automatic JIT compilation for performance-critical functions
    - Implement compilation error handling with graceful fallbacks
    - Create compilation caching system for repeated operations
    - Write unit tests for JIT compilation and performance validation
    - _Requirements: 1.6, 5.1, 5.5, 3.6_

  - [ ] 8.2 Add advanced JAX features
    - Implement `vmap` vectorization for batch operations
    - Add `pmap` parallelization for multi-device computation
    - Integrate automatic differentiation capabilities where applicable
    - Write unit tests for vectorization and parallelization features
    - _Requirements: 5.2, 5.3, 5.4, 6.1, 6.3_

- [ ] 9. Implement error handling and device management
  - [ ] 9.1 Create comprehensive error handling system
    - Implement device fallback logic for TPU/GPU/CPU
    - Add graceful handling of out-of-memory conditions
    - Create informative error messages for compilation failures
    - Write unit tests for error handling and recovery scenarios
    - _Requirements: 1.5, 6.5, 5.5_

  - [ ] 9.2 Add numerical stability and precision handling
    - Implement safe operations for floating-point precision issues
    - Add tolerance handling for numerical comparisons
    - Create consistent hash computation across different precisions
    - Write unit tests for numerical stability and precision handling
    - _Requirements: 7.1, 7.5_

- [ ] 10. Create comprehensive test suite
  - [ ] 10.1 Implement compatibility tests
    - Create test suite comparing JAX and PyTorch implementations
    - Add numerical equivalence tests for all major operations
    - Implement BFS result comparison tests across backends
    - Write performance benchmark tests for different device types
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ] 10.2 Add TPU-specific tests
    - Create multi-core scaling tests for TPU parallelization
    - Implement memory sharding tests for large state spaces
    - Add compilation caching and performance tests
    - Write device-specific integration tests
    - _Requirements: 3.1, 6.1, 6.2, 7.2_

- [x] 11. Implement intelligent dependency management system
  - [x] 11.1 Create dependency detection and management utilities
    - Implement `DependencyManager` class with runtime backend detection
    - Add hardware detection for CPU, GPU (CUDA), and TPU environments
    - Create installation recommendation system based on detected hardware
    - Write unit tests for dependency detection across different environments
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 11.2 Update pyproject.toml with optional JAX dependencies
    - Create separate optional dependency groups for jax-cpu, jax-cuda, jax-tpu
    - Add convenience meta-packages for common installation patterns
    - Implement proper version constraints and compatibility requirements
    - Write installation tests for different dependency combinations
    - _Requirements: 6.1, 6.2, 6.3, 6.5_

- [ ] 12. Create environment-aware testing framework
  - [ ] 12.1 Implement test environment detection system
    - Create `TestEnvironment` class for runtime hardware/software detection
    - Implement pytest markers for different backend and hardware requirements
    - Add automatic test skipping based on available dependencies
    - Write tests for test environment detection and marker generation
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 12.2 Update existing tests with environment awareness
    - Add appropriate pytest markers to all existing tests
    - Implement parametrized tests that adapt to available backends
    - Create backend equivalence tests that skip unavailable backends
    - Update CI configuration to report tested environment combinations
    - _Requirements: 8.1, 8.2, 8.3, 8.6, 8.7, 8.8_

- [ ] 13. Integrate backend selection and configuration
  - [ ] 13.1 Create intelligent backend configuration system
    - Implement enhanced `BackendConfig` with dependency-aware backend selection
    - Add automatic backend selection with priority system (TPU > GPU > CPU)
    - Create informative error messages with installation instructions
    - Write unit tests for configuration management and backend selection
    - _Requirements: 2.6, 1.1, 1.3, 6.4, 6.5, 6.6_

  - [ ] 13.2 Implement backward compatibility layer
    - Create compatibility shims for deprecated PyTorch features
    - Add deprecation warnings for removed functionality
    - Implement gradual migration path with parallel validation
    - Write integration tests for backward compatibility
    - _Requirements: 2.1, 2.2, 2.4, 2.6_

- [ ] 14. Update documentation and create migration guide
  - [ ] 14.1 Update API documentation
    - Update docstrings to reflect JAX backend and TPU capabilities
    - Add performance notes and TPU-specific optimization guidance
    - Create examples demonstrating JAX-specific features
    - Write troubleshooting guide for common JAX/TPU issues and dependency management
    - Document installation instructions for different environments (TPU, GPU, CPU)
    - _Requirements: 9.1, 9.2, 9.4, 9.5, 6.4, 6.6_

  - [ ] 14.2 Create comprehensive migration guide
    - Write step-by-step migration instructions from PyTorch to JAX
    - Document performance comparisons and benchmarking results
    - Create examples showing before/after code patterns
    - Add best practices guide for TPU optimization
    - Include dependency management and installation guide for different environments
    - _Requirements: 9.2, 9.3, 9.6, 6.1, 6.2, 6.3_

- [ ] 15. Final integration and validation
  - [ ] 15.1 Perform end-to-end integration testing
    - Run full test suite on CPU, GPU, and TPU environments with environment-aware skipping
    - Validate performance targets and scalability requirements across available hardware
    - Test all public API methods for backward compatibility
    - Perform memory usage and performance profiling
    - Validate dependency management and installation across different environments
    - _Requirements: 3.1, 7.6, 8.1, 8.2, 8.6, 8.7, 8.8_

  - [ ] 15.2 Optimize and finalize implementation
    - Profile and optimize critical performance bottlenecks
    - Fine-tune TPU memory usage and sharding strategies
    - Validate numerical accuracy across all test cases
    - Prepare final release with comprehensive documentation
    - Validate dependency management works correctly in production environments
    - _Requirements: 3.1, 3.3, 7.6, 8.5, 6.5, 6.6_