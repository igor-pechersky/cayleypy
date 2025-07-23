"""JAX implementation of permutation and matrix generator systems.

This module provides JAX-based implementations for applying permutations and matrix
generators to states, optimized for TPU/GPU computation with JIT compilation and
vectorization.
"""

from typing import List, Union

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from functools import partial
    import numpy as np

    # Enable 64-bit precision for JAX
    jax.config.update("jax_enable_x64", True)

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore


def _check_jax_available():
    """Check if JAX is available and raise error if not."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is not available. Install with: pip install 'cayleypy[jax-cpu]', "
            "'cayleypy[jax-cuda]', or 'cayleypy[jax-tpu]'"
        )


class JAXPermutationGenerator:
    """JAX-based permutation generator for efficient permutation application.

    This class provides JIT-compiled functions for applying permutations to states
    using JAX's advanced indexing capabilities.
    """

    def __init__(self, permutation: List[int]):
        """Initialize permutation generator.

        Args:
            permutation: List representing the permutation
        """
        _check_jax_available()

        self.permutation = permutation
        self.n = len(permutation)
        self._validate_permutation()

        # Convert to JAX array for efficient indexing
        self.perm_array = jnp.array(permutation, dtype=jnp.int32)

    def _validate_permutation(self):
        """Validate that the input is a valid permutation."""
        sorted_perm = sorted(self.permutation)
        expected = list(range(self.n))
        assert sorted_perm == expected, f"Invalid permutation: {self.permutation}"

    @partial(jit, static_argnums=(0,))
    def apply_single(self, state: jnp.ndarray) -> jnp.ndarray:
        """Apply permutation to a single state.

        Args:
            state: 1D array representing the state

        Returns:
            Permuted state
        """
        return state[self.perm_array]

    @partial(jit, static_argnums=(0,))
    def apply_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Apply permutation to a batch of states.

        Args:
            states: 2D array where each row is a state

        Returns:
            Batch of permuted states
        """
        return states[:, self.perm_array]

    def get_inverse_generator(self) -> "JAXPermutationGenerator":
        """Get the inverse permutation generator.

        Returns:
            JAXPermutationGenerator for the inverse permutation
        """
        inverse_perm = [0] * self.n
        for i, p in enumerate(self.permutation):
            inverse_perm[p] = i
        return JAXPermutationGenerator(inverse_perm)


class JAXMatrixGenerator:
    """JAX-based matrix generator for efficient matrix operations.

    This class provides JIT-compiled functions for applying matrix generators to states
    with support for modular arithmetic and batch processing.
    """

    def __init__(self, matrix: Union[List[List[int]], np.ndarray, jnp.ndarray], modulo: int = 0):
        """Initialize matrix generator.

        Args:
            matrix: Square matrix as 2D array
            modulo: Modulo for arithmetic (0 means no modulo)
        """
        _check_jax_available()

        # Convert to JAX array
        if isinstance(matrix, (list, np.ndarray)):
            self.matrix = jnp.array(matrix, dtype=jnp.int64)
        else:
            self.matrix = matrix

        self.modulo = modulo
        self.n = self.matrix.shape[0]

        self._validate_matrix()

        # Apply modulo if specified
        if self.modulo > 0:
            self.matrix = self.matrix % self.modulo

    def _validate_matrix(self):
        """Validate that the matrix is square and has correct properties."""
        assert len(self.matrix.shape) == 2, "Matrix must be 2D"
        assert self.matrix.shape[0] == self.matrix.shape[1], "Matrix must be square"

        if self.modulo != 0:
            assert 2 <= self.modulo <= 2**31, "Invalid modulo value"

    @partial(jit, static_argnums=(0,))
    def apply_single(self, state: jnp.ndarray) -> jnp.ndarray:
        """Apply matrix to a single state.

        Args:
            state: 2D array representing the state (n x m matrix)

        Returns:
            Result of matrix multiplication
        """
        result = self.matrix @ state
        if self.modulo > 0:
            result = result % self.modulo
        return result

    @partial(jit, static_argnums=(0,))
    def apply_batch(self, states: jnp.ndarray) -> jnp.ndarray:
        """Apply matrix to a batch of states.

        Args:
            states: 3D array where each element along first axis is a state (batch_size x n x m)
                   or 2D array for single matrix states (n x m)

        Returns:
            Batch of results from matrix multiplication
        """
        if len(states.shape) == 2:
            # Single state case - treat as batch of size 1
            states = states[None, :, :]
            result = jnp.einsum("ij,bjk->bik", self.matrix, states)
            result = result[0]  # Remove batch dimension
        else:
            # Batch case
            result = jnp.einsum("ij,bjk->bik", self.matrix, states)

        if self.modulo > 0:
            result = result % self.modulo
        return result

    def get_inverse_generator(self) -> "JAXMatrixGenerator":
        """Get the inverse matrix generator.

        Returns:
            JAXMatrixGenerator for the inverse matrix

        Raises:
            ValueError: If matrix is not invertible
        """
        if self.modulo > 0:
            # For modular arithmetic, we need modular inverse
            # This is a simplified implementation - full modular inverse is complex
            raise NotImplementedError("Modular matrix inverse not implemented")

        try:
            # Use numpy for inverse calculation, then convert to JAX
            matrix_np = np.array(self.matrix)
            inv_matrix_np = np.linalg.inv(matrix_np)
            inv_matrix_int = np.round(inv_matrix_np).astype(np.int64)

            # Verify the inverse is correct
            identity = np.eye(self.n, dtype=np.int64)
            if not np.allclose(matrix_np @ inv_matrix_int, identity):
                raise ValueError("Matrix is not invertible over integers")

            return JAXMatrixGenerator(inv_matrix_int, self.modulo)

        except np.linalg.LinAlgError as exc:
            raise ValueError("Matrix is not invertible") from exc

    def is_inverse_to(self, other: "JAXMatrixGenerator") -> bool:
        """Check if this matrix is inverse to another matrix.

        Args:
            other: Another matrix generator

        Returns:
            True if matrices are inverses of each other
        """
        if self.modulo != other.modulo or self.n != other.n:
            return False

        identity = jnp.eye(self.n, dtype=jnp.int64)

        # Check both directions
        prod1 = self.matrix @ other.matrix
        prod2 = other.matrix @ self.matrix

        if self.modulo > 0:
            prod1 = prod1 % self.modulo
            prod2 = prod2 % self.modulo

        return bool(jnp.array_equal(prod1, identity) and jnp.array_equal(prod2, identity))


class JAXGeneratorSystem:
    """Unified system for managing both permutation and matrix generators.

    This class provides a unified interface for working with different types of
    generators in JAX, with automatic batching and JIT compilation.
    """

    def __init__(self, generators: List[Union[JAXPermutationGenerator, JAXMatrixGenerator]]):
        """Initialize generator system.

        Args:
            generators: List of generators (all must be same type)
        """
        _check_jax_available()

        if not generators:
            raise ValueError("At least one generator is required")

        self.generators = generators
        self.n_generators = len(generators)

        # Determine generator type
        first_gen = generators[0]
        if isinstance(first_gen, JAXPermutationGenerator):
            self.generator_type = "permutation"
            self.state_size = first_gen.n
            # Verify all generators are permutations of same size
            for gen in generators:
                assert isinstance(gen, JAXPermutationGenerator), "All generators must be same type"
                assert gen.n == self.state_size, "All permutation generators must have same size"
        elif isinstance(first_gen, JAXMatrixGenerator):
            self.generator_type = "matrix"
            self.state_size = first_gen.n
            # Verify all generators are matrices of same size
            for gen in generators:
                assert isinstance(gen, JAXMatrixGenerator), "All generators must be same type"
                assert gen.n == self.state_size, "All matrix generators must have same size"
        else:
            raise ValueError(f"Unsupported generator type: {type(first_gen)}")

    def apply_generator(self, generator_idx: int, state: jnp.ndarray) -> jnp.ndarray:
        """Apply a specific generator to a state.

        Args:
            generator_idx: Index of generator to apply
            state: State to transform

        Returns:
            Transformed state
        """
        assert 0 <= generator_idx < self.n_generators, f"Invalid generator index: {generator_idx}"

        generator = self.generators[generator_idx]

        if self.generator_type == "permutation":
            # For permutations: 1D = single state, 2D = batch
            if len(state.shape) == 1:
                return generator.apply_single(state)
            else:
                return generator.apply_batch(state)
        else:
            # For matrices: 2D = single state, 3D = batch
            if len(state.shape) == 2:
                return generator.apply_single(state)
            else:
                return generator.apply_batch(state)

    def apply_generator_sequence(self, generator_indices: List[int], state: jnp.ndarray) -> jnp.ndarray:
        """Apply a sequence of generators to a state.

        Args:
            generator_indices: List of generator indices to apply in order
            state: Initial state

        Returns:
            Final transformed state
        """
        current_state = state
        for gen_idx in generator_indices:
            current_state = self.apply_generator(gen_idx, current_state)
        return current_state

    def get_inverse_generators(self) -> "JAXGeneratorSystem":
        """Get system with inverse generators.

        Returns:
            JAXGeneratorSystem with inverse generators in same order
        """
        inverse_gens = [gen.get_inverse_generator() for gen in self.generators]
        return JAXGeneratorSystem(inverse_gens)

    def is_inverse_closed(self) -> bool:
        """Check if the generator set is inverse-closed.

        Returns:
            True if for each generator, its inverse is also in the set
        """
        for gen in self.generators:
            inverse_gen = gen.get_inverse_generator()

            # Check if inverse is in the set
            found_inverse = False
            for other_gen in self.generators:
                if self.generator_type == "permutation":
                    # Both are permutation generators
                    if hasattr(other_gen, "permutation") and hasattr(inverse_gen, "permutation"):
                        if other_gen.permutation == inverse_gen.permutation:
                            found_inverse = True
                            break
                else:  # matrix
                    # Both are matrix generators
                    if isinstance(other_gen, JAXMatrixGenerator) and isinstance(gen, JAXMatrixGenerator):
                        if other_gen.is_inverse_to(gen):
                            found_inverse = True
                            break

            if not found_inverse:
                return False

        return True


# Utility functions for creating generators from existing data structures
def create_permutation_generators_from_lists(permutations: List[List[int]]) -> List[JAXPermutationGenerator]:
    """Create JAX permutation generators from list of permutations.

    Args:
        permutations: List of permutation lists

    Returns:
        List of JAXPermutationGenerator objects
    """
    return [JAXPermutationGenerator(perm) for perm in permutations]


def create_matrix_generators_from_arrays(
    matrices: List[Union[List[List[int]], np.ndarray]], modulo: int = 0
) -> List[JAXMatrixGenerator]:
    """Create JAX matrix generators from list of matrices.

    Args:
        matrices: List of matrix arrays
        modulo: Modulo for arithmetic operations

    Returns:
        List of JAXMatrixGenerator objects
    """
    return [JAXMatrixGenerator(matrix, modulo) for matrix in matrices]


def create_generator_system_from_cayley_def(cayley_def) -> JAXGeneratorSystem:
    """Create JAX generator system from CayleyGraphDef.

    Args:
        cayley_def: CayleyGraphDef object

    Returns:
        JAXGeneratorSystem for the given definition
    """
    generators: List[Union[JAXPermutationGenerator, JAXMatrixGenerator]]
    if cayley_def.is_permutation_group():
        perm_generators = create_permutation_generators_from_lists(cayley_def.generators_permutations)
        generators = list(perm_generators)  # type: ignore  # Convert to satisfy type checker
    else:
        # Convert MatrixGenerator objects to JAX matrix generators
        generators = []
        for mat_gen in cayley_def.generators_matrices:
            jax_gen = JAXMatrixGenerator(mat_gen.matrix, mat_gen.modulo)
            generators.append(jax_gen)

    return JAXGeneratorSystem(generators)
