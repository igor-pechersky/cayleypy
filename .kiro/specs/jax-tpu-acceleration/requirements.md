# Requirements Document

## Introduction

This feature adds simplified JAX/TPU acceleration to CayleyPy, leveraging modern TPU capabilities including native int64 support. The design assumes TPU hardware is always available and focuses on maximizing performance through direct TPU acceleration without complex device orchestration.

## Requirements

### Requirement 1

**User Story:** As a researcher working with large Cayley graphs, I want TPU-accelerated BFS operations with native int64 support, so that I can compute growth functions for massive graphs in reasonable time.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL automatically enable int64 support via jax.config.update("jax_enable_x64", True)
2. WHEN running BFS on TPU THEN the system SHALL achieve at least 10x speedup compared to CPU implementations for graphs with >10^5 states
3. WHEN processing large state spaces THEN the system SHALL handle int64 permutation operations natively on TPU v6e
4. WHEN memory limits are approached THEN the system SHALL leverage TPU v6e's 32GB HBM per chip efficiently

### Requirement 2

**User Story:** As a machine learning practitioner using beam search, I want TPU-accelerated neural network inference with full int64 precision, so that I can process larger beam widths efficiently.

#### Acceptance Criteria

1. WHEN performing beam search THEN the system SHALL support beam widths up to 10^6 states on TPU v6e
2. WHEN computing hash functions THEN TPU implementations SHALL provide at least 5x speedup over CPU using native int64 operations
3. WHEN running neural network inference THEN the system SHALL utilize TPU v6e's 256x256 systolic array optimally
4. WHEN deduplicating states THEN the system SHALL use TPU-native int64 operations for precise set operations

### Requirement 3

**User Story:** As a developer integrating CayleyPy, I want seamless TPU acceleration, so that existing code works without modification while gaining performance benefits.

#### Acceptance Criteria

1. WHEN TPU acceleration is available THEN the system SHALL automatically use it for supported operations
2. WHEN calling existing CayleyGraph methods THEN the API SHALL remain unchanged
3. WHEN TPU operations fail THEN the system SHALL log errors and fall back to CPU gracefully
4. WHEN running tests THEN all functionality SHALL work identically on TPU and CPU

### Requirement 4

**User Story:** As a performance-conscious user, I want optimized TPU implementations with full precision, so that I can achieve maximum computational throughput.

#### Acceptance Criteria

1. WHEN performing tensor operations THEN the system SHALL use JAX's JIT compilation with native int64 support
2. WHEN processing batches THEN the system SHALL use vmap for automatic vectorization across TPU v6e cores
3. WHEN managing memory THEN the system SHALL leverage TPU v6e's 32GB HBM per chip efficiently
4. WHEN running computations THEN the system SHALL maintain full int64 precision equivalent to CPU versions

### Requirement 5

**User Story:** As a researcher analyzing performance, I want simple benchmarking capabilities, so that I can measure TPU acceleration benefits.

#### Acceptance Criteria

1. WHEN running benchmarks THEN the system SHALL provide timing comparisons between TPU and CPU
2. WHEN measuring throughput THEN the system SHALL report states processed per second
3. WHEN testing correctness THEN benchmarks SHALL verify TPU results match CPU results exactly
4. WHEN profiling memory THEN the system SHALL report peak memory usage during operations