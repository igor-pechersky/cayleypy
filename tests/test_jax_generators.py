"""Tests for JAX generator systems."""

import pytest
import numpy as np
from typing import List

try:
    import jax
    import jax.numpy as jnp
    from cayleypy.jax_generators import (
        JAXPermutationGenerator,
        JAXMatrixGenerator,
        JAXGeneratorSystem,
        create_permutation_generators_from_lists,
        create_matrix_generators_from_arrays,
        create_generator_system_from_cayley_def,
    )
    from cayleypy.cayley_graph_def import CayleyGraphDef, MatrixGenerator
    from cayleypy.graphs_lib import PermutationGroups, MatrixGroups

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXPermutationGenerator:
    """Test JAX permutation generator functionality."""

    def test_init_valid_permutation(self):
        """Test initialization with valid permutation."""
        perm = [1, 0, 2]
        gen = JAXPermutationGenerator(perm)
        assert gen.permutation == perm
        assert gen.n == 3
        assert jnp.array_equal(gen.perm_array, jnp.array([1, 0, 2]))

    def test_init_invalid_permutation(self):
        """Test initialization with invalid permutation raises error."""
        with pytest.raises(AssertionError):
            JAXPermutationGenerator([0, 2, 2])  # Not a valid permutation

        with pytest.raises(AssertionError):
            JAXPermutationGenerator([0, 1, 3])  # Missing element 2

    def test_apply_single_state(self):
        """Test applying permutation to single state."""
        perm = [2, 0, 1]  # Cycle (0 2 1)
        gen = JAXPermutationGenerator(perm)

        state = jnp.array([10, 20, 30])
        result = gen.apply_single(state)
        expected = jnp.array([30, 10, 20])  # state[perm] = [state[2], state[0], state[1]]

        assert jnp.array_equal(result, expected)

    def test_apply_batch_states(self):
        """Test applying permutation to batch of states."""
        perm = [1, 2, 0]  # Cycle (0 1 2)
        gen = JAXPermutationGenerator(perm)

        states = jnp.array([[10, 20, 30], [40, 50, 60]])
        result = gen.apply_batch(states)
        expected = jnp.array([[20, 30, 10], [50, 60, 40]])  # [state[1], state[2], state[0]]

        assert jnp.array_equal(result, expected)

    def test_get_inverse_generator(self):
        """Test getting inverse permutation generator."""
        perm = [2, 0, 1]  # Cycle (0 2 1)
        gen = JAXPermutationGenerator(perm)
        inv_gen = gen.get_inverse_generator()

        # Inverse of (0 2 1) should be (0 1 2) = [1, 2, 0]
        expected_inv = [1, 2, 0]
        assert inv_gen.permutation == expected_inv

        # Test that applying both gives identity
        state = jnp.array([10, 20, 30])
        result = inv_gen.apply_single(gen.apply_single(state))
        assert jnp.array_equal(result, state)

    def test_identity_permutation(self):
        """Test identity permutation."""
        perm = [0, 1, 2]
        gen = JAXPermutationGenerator(perm)

        state = jnp.array([10, 20, 30])
        result = gen.apply_single(state)
        assert jnp.array_equal(result, state)

    def test_transposition(self):
        """Test simple transposition."""
        perm = [1, 0, 2]  # Swap first two elements
        gen = JAXPermutationGenerator(perm)

        state = jnp.array([10, 20, 30])
        result = gen.apply_single(state)
        expected = jnp.array([20, 10, 30])
        assert jnp.array_equal(result, expected)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXMatrixGenerator:
    """Test JAX matrix generator functionality."""

    def test_init_valid_matrix(self):
        """Test initialization with valid matrix."""
        matrix = [[1, 2], [3, 4]]
        gen = JAXMatrixGenerator(matrix)

        assert gen.n == 2
        assert gen.modulo == 0
        assert jnp.array_equal(gen.matrix, jnp.array([[1, 2], [3, 4]]))

    def test_init_with_modulo(self):
        """Test initialization with modulo arithmetic."""
        matrix = [[5, 7], [2, 3]]
        modulo = 3
        gen = JAXMatrixGenerator(matrix, modulo)

        expected_matrix = jnp.array([[2, 1], [2, 0]])  # 5%3=2, 7%3=1, etc.
        assert jnp.array_equal(gen.matrix, expected_matrix)
        assert gen.modulo == 3

    def test_init_invalid_matrix(self):
        """Test initialization with invalid matrix raises error."""
        with pytest.raises(AssertionError):
            JAXMatrixGenerator([[1, 2, 3], [4, 5, 6]])  # Not square

    def test_apply_single_state(self):
        """Test applying matrix to single state."""
        matrix = [[2, 1], [1, 2]]
        gen = JAXMatrixGenerator(matrix)

        state = jnp.array([[3], [4]])  # Column vector
        result = gen.apply_single(state)
        expected = jnp.array([[10], [11]])  # [[2*3 + 1*4], [1*3 + 2*4]]

        assert jnp.array_equal(result, expected)

    def test_apply_single_state_2d(self):
        """Test applying matrix to 2D state."""
        matrix = [[2, 0], [0, 3]]
        gen = JAXMatrixGenerator(matrix)

        state = jnp.array([[1, 2], [3, 4]])
        result = gen.apply_single(state)
        expected = jnp.array([[2, 4], [9, 12]])  # Scale rows by 2 and 3

        assert jnp.array_equal(result, expected)

    def test_apply_batch_states(self):
        """Test applying matrix to batch of states."""
        matrix = [[1, 1], [0, 1]]
        gen = JAXMatrixGenerator(matrix)

        states = jnp.array([[[1], [2]], [[3], [4]]])
        result = gen.apply_batch(states)
        expected = jnp.array([[[3], [2]], [[7], [4]]])  # [[1+2], [2]]  # [[3+4], [4]]

        assert jnp.array_equal(result, expected)

    def test_modular_arithmetic(self):
        """Test modular arithmetic in matrix operations."""
        matrix = [[2, 3], [1, 4]]
        modulo = 5
        gen = JAXMatrixGenerator(matrix, modulo)

        state = jnp.array([[3], [2]])
        result = gen.apply_single(state)
        # [[2*3 + 3*2], [1*3 + 4*2]] = [[12], [11]] mod 5 = [[2], [1]]
        expected = jnp.array([[2], [1]])

        assert jnp.array_equal(result, expected)

    def test_get_inverse_generator_integer(self):
        """Test getting inverse for integer matrix."""
        # Use a simple 2x2 matrix with integer inverse
        matrix = [[1, 1], [0, 1]]  # Upper triangular with det=1
        gen = JAXMatrixGenerator(matrix)

        inv_gen = gen.get_inverse_generator()

        # Verify it's actually the inverse
        identity = jnp.eye(2, dtype=jnp.int64)
        product = gen.matrix @ inv_gen.matrix
        assert jnp.array_equal(product, identity)

    def test_get_inverse_generator_not_invertible(self):
        """Test getting inverse for non-invertible matrix raises error."""
        matrix = [[1, 1], [1, 1]]  # Singular matrix
        gen = JAXMatrixGenerator(matrix)

        with pytest.raises(ValueError, match="not invertible"):
            gen.get_inverse_generator()

    def test_get_inverse_generator_modular_not_implemented(self):
        """Test that modular inverse is not implemented."""
        matrix = [[1, 1], [0, 1]]
        gen = JAXMatrixGenerator(matrix, modulo=5)

        with pytest.raises(NotImplementedError):
            gen.get_inverse_generator()

    def test_is_inverse_to(self):
        """Test checking if two matrices are inverses."""
        matrix1 = [[1, 1], [0, 1]]
        matrix2 = [[1, -1], [0, 1]]  # Inverse of matrix1

        gen1 = JAXMatrixGenerator(matrix1)
        gen2 = JAXMatrixGenerator(matrix2)

        assert gen1.is_inverse_to(gen2)
        assert gen2.is_inverse_to(gen1)

    def test_is_inverse_to_different_modulo(self):
        """Test that matrices with different modulos are not inverses."""
        matrix = [[1, 1], [0, 1]]
        gen1 = JAXMatrixGenerator(matrix, modulo=3)
        gen2 = JAXMatrixGenerator(matrix, modulo=5)

        assert not gen1.is_inverse_to(gen2)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXGeneratorSystem:
    """Test JAX generator system functionality."""

    def test_init_permutation_system(self):
        """Test initialization with permutation generators."""
        perms = [[1, 0, 2], [0, 2, 1]]
        generators = [JAXPermutationGenerator(perm) for perm in perms]
        system = JAXGeneratorSystem(generators)

        assert system.generator_type == "permutation"
        assert system.n_generators == 2
        assert system.state_size == 3

    def test_init_matrix_system(self):
        """Test initialization with matrix generators."""
        matrices = [[[1, 1], [0, 1]], [[1, 0], [1, 1]]]
        generators = [JAXMatrixGenerator(matrix) for matrix in matrices]
        system = JAXGeneratorSystem(generators)

        assert system.generator_type == "matrix"
        assert system.n_generators == 2
        assert system.state_size == 2

    def test_init_mixed_generators_error(self):
        """Test that mixing generator types raises error."""
        perm_gen = JAXPermutationGenerator([1, 0, 2])
        matrix_gen = JAXMatrixGenerator([[1, 0], [0, 1]])

        with pytest.raises(AssertionError):
            JAXGeneratorSystem([perm_gen, matrix_gen])

    def test_init_empty_generators_error(self):
        """Test that empty generator list raises error."""
        with pytest.raises(ValueError):
            JAXGeneratorSystem([])

    def test_apply_generator_permutation(self):
        """Test applying specific generator in permutation system."""
        perms = [[1, 0, 2], [0, 2, 1]]
        generators = [JAXPermutationGenerator(perm) for perm in perms]
        system = JAXGeneratorSystem(generators)

        state = jnp.array([10, 20, 30])

        # Apply first generator
        result0 = system.apply_generator(0, state)
        expected0 = jnp.array([20, 10, 30])
        assert jnp.array_equal(result0, expected0)

        # Apply second generator
        result1 = system.apply_generator(1, state)
        expected1 = jnp.array([10, 30, 20])
        assert jnp.array_equal(result1, expected1)

    def test_apply_generator_matrix(self):
        """Test applying specific generator in matrix system."""
        matrices = [[[2, 0], [0, 1]], [[1, 1], [0, 1]]]
        generators = [JAXMatrixGenerator(matrix) for matrix in matrices]
        system = JAXGeneratorSystem(generators)

        state = jnp.array([[3], [4]])

        # Apply first generator (scale first component by 2)
        result0 = system.apply_generator(0, state)
        expected0 = jnp.array([[6], [4]])
        assert jnp.array_equal(result0, expected0)

        # Apply second generator (add first to second)
        result1 = system.apply_generator(1, state)
        expected1 = jnp.array([[7], [4]])
        assert jnp.array_equal(result1, expected1)

    def test_apply_generator_invalid_index(self):
        """Test applying generator with invalid index raises error."""
        perm_gen = JAXPermutationGenerator([1, 0, 2])
        system = JAXGeneratorSystem([perm_gen])

        state = jnp.array([10, 20, 30])

        with pytest.raises(AssertionError):
            system.apply_generator(1, state)  # Only index 0 is valid

        with pytest.raises(AssertionError):
            system.apply_generator(-1, state)

    def test_apply_generator_sequence(self):
        """Test applying sequence of generators."""
        perms = [[1, 0, 2], [0, 2, 1]]
        generators = [JAXPermutationGenerator(perm) for perm in perms]
        system = JAXGeneratorSystem(generators)

        state = jnp.array([10, 20, 30])

        # Apply generators 0, 1, 0 in sequence
        result = system.apply_generator_sequence([0, 1, 0], state)

        # Manual calculation:
        # Start: [10, 20, 30]
        # Apply 0: [20, 10, 30] (swap first two)
        # Apply 1: [20, 30, 10] (cycle last two)
        # Apply 0: [30, 20, 10] (swap first two)
        expected = jnp.array([30, 20, 10])
        assert jnp.array_equal(result, expected)

    def test_get_inverse_generators(self):
        """Test getting system with inverse generators."""
        perms = [[1, 0, 2], [0, 2, 1]]
        generators = [JAXPermutationGenerator(perm) for perm in perms]
        system = JAXGeneratorSystem(generators)

        inv_system = system.get_inverse_generators()

        assert inv_system.generator_type == "permutation"
        assert inv_system.n_generators == 2

        # Test that applying generator and its inverse gives identity
        state = jnp.array([10, 20, 30])
        for i in range(2):
            transformed = system.apply_generator(i, state)
            restored = inv_system.apply_generator(i, transformed)
            assert jnp.array_equal(restored, state)

    def test_is_inverse_closed_true(self):
        """Test inverse closure check for closed set."""
        # Use generators that are their own inverses (involutions)
        perms = [[1, 0, 2], [0, 2, 1, 3], [2, 1, 0, 3]]  # Different sizes for variety
        # Actually, let's use same size
        perms = [[1, 0, 2], [2, 1, 0]]  # Both are involutions
        generators = [JAXPermutationGenerator(perm) for perm in perms]
        system = JAXGeneratorSystem(generators)

        # These are both involutions (self-inverse), so set should be inverse-closed
        assert system.is_inverse_closed()

    def test_is_inverse_closed_false(self):
        """Test inverse closure check for non-closed set."""
        # Use a 3-cycle which is not its own inverse
        perms = [[1, 2, 0]]  # 3-cycle, inverse is [2, 0, 1]
        generators = [JAXPermutationGenerator(perm) for perm in perms]
        system = JAXGeneratorSystem(generators)

        # The inverse [2, 0, 1] is not in the set, so not inverse-closed
        assert not system.is_inverse_closed()


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestUtilityFunctions:
    """Test utility functions for creating generators."""

    def test_create_permutation_generators_from_lists(self):
        """Test creating permutation generators from lists."""
        perms = [[1, 0, 2], [0, 2, 1]]
        generators = create_permutation_generators_from_lists(perms)

        assert len(generators) == 2
        assert all(isinstance(gen, JAXPermutationGenerator) for gen in generators)
        assert generators[0].permutation == [1, 0, 2]
        assert generators[1].permutation == [0, 2, 1]

    def test_create_matrix_generators_from_arrays(self):
        """Test creating matrix generators from arrays."""
        matrices = [[[1, 1], [0, 1]], [[1, 0], [1, 1]]]
        generators = create_matrix_generators_from_arrays(matrices, modulo=5)

        assert len(generators) == 2
        assert all(isinstance(gen, JAXMatrixGenerator) for gen in generators)
        assert all(gen.modulo == 5 for gen in generators)

    def test_create_generator_system_from_cayley_def_permutation(self):
        """Test creating generator system from permutation CayleyGraphDef."""
        # Use a simple permutation group (Coxeter generators for S3)
        cayley_def = PermutationGroups.coxeter(3)
        system = create_generator_system_from_cayley_def(cayley_def)

        assert system.generator_type == "permutation"
        assert system.state_size == 3
        assert system.n_generators == len(cayley_def.generators_permutations)

    def test_create_generator_system_from_cayley_def_matrix(self):
        """Test creating generator system from matrix CayleyGraphDef."""
        # Use the Heisenberg group
        cayley_def = MatrixGroups.heisenberg(modulo=3)
        system = create_generator_system_from_cayley_def(cayley_def)

        assert system.generator_type == "matrix"
        assert system.state_size == 3  # 3x3 matrices
        assert system.n_generators == len(cayley_def.generators_matrices)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXGeneratorsIntegration:
    """Integration tests for JAX generators with real group examples."""

    def test_symmetric_group_s3(self):
        """Test with symmetric group S3."""
        # S3 generators: (1 2) and (1 2 3)
        perms = [[1, 0, 2], [1, 2, 0]]
        generators = [JAXPermutationGenerator(perm) for perm in perms]
        system = JAXGeneratorSystem(generators)

        # Test that we can generate all 6 elements of S3
        identity = jnp.array([0, 1, 2])
        visited_states = set()

        # BFS-like exploration
        queue = [identity]
        visited_states.add(tuple(identity.tolist()))  # Convert to Python list first

        while queue:
            current_state = queue.pop(0)
            for gen_idx in range(system.n_generators):
                new_state = system.apply_generator(gen_idx, current_state)
                state_tuple = tuple(new_state.tolist())  # Convert to Python list first
                if state_tuple not in visited_states:
                    visited_states.add(state_tuple)
                    queue.append(new_state)

        # S3 has 6 elements
        assert len(visited_states) == 6

    def test_matrix_group_sl2_mod3(self):
        """Test with SL2(Z/3Z) matrix group."""
        # Simple generators for SL2(Z/3Z)
        matrices = [[[1, 1], [0, 1]], [[1, 0], [1, 1]]]  # Upper triangular  # Lower triangular
        generators = [JAXMatrixGenerator(matrix, modulo=3) for matrix in matrices]
        system = JAXGeneratorSystem(generators)

        # Test basic operations
        identity = jnp.array([[1, 0], [0, 1]])

        # Apply first generator
        result1 = system.apply_generator(0, identity)
        expected1 = jnp.array([[1, 1], [0, 1]])
        assert jnp.array_equal(result1, expected1)

        # Apply second generator
        result2 = system.apply_generator(1, identity)
        expected2 = jnp.array([[1, 0], [1, 1]])
        assert jnp.array_equal(result2, expected2)

    def test_performance_batch_operations(self):
        """Test performance with batch operations."""
        # Create a larger permutation for performance testing
        n = 10
        perm = list(range(1, n)) + [0]  # n-cycle
        gen = JAXPermutationGenerator(perm)

        # Create batch of random states
        batch_size = 1000
        key = jax.random.PRNGKey(42)
        states = jax.random.permutation(key, jnp.arange(n), axis=0, independent=True)
        states = jnp.tile(states, (batch_size, 1))

        # Apply generator to batch
        result = gen.apply_batch(states)

        # Verify shape and that each state is properly permuted
        assert result.shape == (batch_size, n)

        # Check that first state is correctly permuted
        expected_first = states[0][gen.perm_array]
        assert jnp.array_equal(result[0], expected_first)


if __name__ == "__main__":
    pytest.main([__file__])
