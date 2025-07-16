---
inclusion: always
---

# CayleyPy Development Guidelines

CayleyPy is a Python library for analyzing extremely large graphs using mathematical group theory and AI/ML methods. Focus on state-space graphs too large for memory storage.

## Code Standards

- Black formatter with 120-character lines
- Type hints required (mypy configured)
- Docstrings mandatory for public APIs
- Descriptive variable names for mathematical concepts
- PEP 8 compliance

## Architecture Rules

### Graph Definitions

- Return `CayleyGraphDef` objects for new Cayley graphs
- Permutation graphs → `PermutationGroups` class in `graphs_lib.py`
- Matrix graphs → `MatrixGroups` class in `graphs_lib.py`
- Puzzle graphs → `Puzzles` class in `puzzles/puzzles.py`
- Complex puzzle logic → separate files in `puzzles/`
- Hardcoded moves → `puzzles/moves.py`

### Design Patterns

- Abstract base classes for core interfaces (e.g., `Puzzle`)
- Hashable state representations (tuples, strings)
- Separate puzzle logic from solving algorithms
- Support BFS and A\* search strategies

### Performance

- NumPy arrays for numerical work
- Numba for performance-critical loops
- JAX/TPU support for large-scale computation
- Efficient bit manipulation for states
- Vectorized operations preferred

## Testing

- pytest with unit tests for all new functionality
- pytest-benchmark for performance testing
- Test edge cases and boundaries
- Use `RUN_SLOW_TESTS=1` for comprehensive tests
- Maintain coverage with `coverage` tool

## Mathematical Conventions

- Consistent state representation (e.g., 0 for blank tiles)
- Admissible heuristics for optimal A\* search
- Descriptive generator names following domain conventions
- Mathematical references in docstrings (Wikipedia, arXiv)

## File Organization

- Data files in `cayleypy/data/` with descriptive CSV names
- Test files with `_test.py` suffix
- Clean separation between core library and examples

## Dependencies

- Python 3.9+ support required
- Core: numpy, scipy, numba, h5py
- Optional dependencies grouped by functionality
- Graceful handling of missing optional dependencies

## Error Handling

- Clear error messages for invalid definitions
- Graceful TPU/JAX dependency handling
- Input validation for parameters and states
- Appropriate exception types for different conditions
