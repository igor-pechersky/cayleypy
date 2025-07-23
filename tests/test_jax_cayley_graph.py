"""Tests for JAX CayleyGraph implementation."""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from cayleypy.cayley_graph_def import CayleyGraphDef, MatrixGenerator
from cayleypy.jax_cayley_graph import JAXCayleyGraph
from cayleypy.jax_device_manager import JAXDeviceManager, DeviceNotFoundError


def get_available_device():
    """Get the first available device for testing."""
    try:
        # Try auto-selection first
        manager = JAXDeviceManager(device="auto")
        return manager.device_type
    except DeviceNotFoundError:
        # Fallback to checking available platforms
        if JAX_AVAILABLE:
            available_platforms = jax.local_devices()
            platform_types = {d.platform for d in available_platforms}
            # Prefer CPU for tests, then GPU, then TPU
            for device in ["cpu", "gpu", "tpu"]:
                if device in platform_types:
                    return device
        return "cpu"  # Default fallback


# Get available device for all tests
TEST_DEVICE = get_available_device() if JAX_AVAILABLE else "cpu"


@pytest.mark.requires_jax
class TestJAXCayleyGraphBasic:
    """Test basic JAX CayleyGraph functionality."""

    def test_init_permutation_group(self):
        """Test initialization with permutation group."""
        # Simple 3-element permutation group
        generators = [[1, 2, 0], [1, 0, 2]]  # 3-cycle and swap
        definition = CayleyGraphDef.create(generators)

        graph = JAXCayleyGraph(definition, device=TEST_DEVICE, verbose=0)

        assert graph.definition == definition
        assert graph.encoded_state_size == 3
        assert graph.string_encoder is None  # No encoding for small groups
        assert jnp.array_equal(graph.central_state, jnp.array([0, 1, 2]))

    def test_init_with_bit_encoding(self):
        """Test initialization with bit encoding."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)

        graph = JAXCayleyGraph(definition, bit_encoding_width=2, device=TEST_DEVICE)

        assert graph.string_encoder is not None
        assert graph.string_encoder.get_code_width() == 2
        assert graph.encoded_state_size == graph.string_encoder.get_encoded_length()

    def test_init_matrix_group(self):
        """Test initialization with matrix group."""
        # 2x2 matrix group with modulo 3
        gen1 = MatrixGenerator.create([[1, 1], [0, 1]], modulo=3)
        gen2 = MatrixGenerator.create([[1, 0], [1, 1]], modulo=3)

        definition = CayleyGraphDef.for_matrix_group(
            generators=[gen1, gen2], central_state=[[1, 0], [0, 1]]  # Identity matrix
        )

        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        assert graph.definition == definition
        assert graph.encoded_state_size == 4  # 2x2 matrix flattened

    def test_device_selection(self):
        """Test device selection."""
        generators = [[1, 0]]  # Simple 2-element group
        definition = CayleyGraphDef.create(generators)

        # Test available device
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)
        if TEST_DEVICE == "cpu":
            assert graph.device_manager.is_cpu()
        elif TEST_DEVICE == "gpu":
            assert graph.device_manager.is_gpu()
        elif TEST_DEVICE == "tpu":
            assert graph.device_manager.is_tpu()

    def test_encode_decode_states(self):
        """Test state encoding and decoding."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test with single state
        state = [0, 1, 2]
        encoded = graph.encode_states(state)
        decoded = graph.decode_states(encoded)

        assert jnp.array_equal(decoded.reshape(-1), jnp.array(state))

        # Test with multiple states
        states = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        encoded = graph.encode_states(states)
        decoded = graph.decode_states(encoded)

        np.testing.assert_array_equal(decoded, states)

    def test_encode_decode_with_string_encoder(self):
        """Test encoding/decoding with string encoder."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, bit_encoding_width=2, device=TEST_DEVICE)

        states = [[0, 1, 2], [1, 2, 0]]
        encoded = graph.encode_states(states)
        decoded = graph.decode_states(encoded)

        # Should match original states
        np.testing.assert_array_equal(decoded, states)

        # Encoded should be different shape
        assert encoded.shape[1] == graph.string_encoder.get_encoded_length()


@pytest.mark.requires_jax
class TestJAXCayleyGraphOperations:
    """Test CayleyGraph operations."""

    def test_apply_single_generator(self):
        """Test applying single generator."""
        generators = [[1, 2, 0], [1, 0, 2]]  # 3-cycle and swap
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Apply first generator to identity
        state = jnp.array([0, 1, 2])
        result = graph.apply_generator(state, 0)
        expected = jnp.array([1, 2, 0])  # 3-cycle

        assert jnp.array_equal(result, expected)

        # Apply second generator
        result = graph.apply_generator(state, 1)
        expected = jnp.array([1, 0, 2])  # swap first two

        assert jnp.array_equal(result, expected)

    def test_apply_generator_batch(self):
        """Test applying generator to batch of states."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Batch of states
        states = jnp.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

        # Apply first generator to all states
        results = graph.apply_generator(states, 0)
        expected = jnp.array([[1, 2, 0], [2, 0, 1], [0, 1, 2]])

        np.testing.assert_array_equal(results, expected)

    def test_apply_generator_matrix_group(self):
        """Test applying generator to matrix group."""
        # 2x2 matrix group with modulo 3
        gen1 = MatrixGenerator.create([[1, 1], [0, 1]], modulo=3)
        gen2 = MatrixGenerator.create([[1, 0], [1, 1]], modulo=3)

        definition = CayleyGraphDef.for_matrix_group(generators=[gen1, gen2], central_state=[[1, 0], [0, 1]])

        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Identity matrix as starting state (2D format)
        identity = jnp.array([[1, 0], [0, 1]])

        # Apply first generator
        result = graph.apply_generator(identity, 0)
        expected = jnp.array([[1, 1], [0, 1]])  # Expected result matrix

        assert jnp.array_equal(result, expected)

    def test_state_validation(self):
        """Test state validation."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Valid state
        valid_state = [0, 1, 2]
        assert graph._validate_state(valid_state)

        # Invalid states
        assert not graph._validate_state([0, 1])  # Wrong length
        assert not graph._validate_state([0, 1, 3])  # Invalid element
        assert not graph._validate_state([0, 0, 1])  # Duplicate element

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Invalid generator index
        state = jnp.array([0, 1, 2])
        with pytest.raises(IndexError):
            graph.apply_generator(state, 5)

        # Invalid state shape for encoding
        with pytest.raises(ValueError):
            graph.encode_states([[0, 1]])  # Wrong length

    def test_memory_efficiency(self):
        """Test memory efficiency with larger groups."""
        # Create a slightly larger permutation group
        generators = [[1, 2, 3, 0], [1, 0, 3, 2]]  # 4-element group
        definition = CayleyGraphDef.create(generators)

        # Test with string encoding for efficiency
        graph = JAXCayleyGraph(definition, bit_encoding_width=2, device=TEST_DEVICE)

        # Should use string encoder for larger groups
        assert graph.string_encoder is not None

        # Test batch operations
        states = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1]]
        encoded = graph.encode_states(states)
        decoded = graph.decode_states(encoded)

        np.testing.assert_array_equal(decoded, states)

    def test_get_neighbors_permutation_group(self):
        """Test get_neighbors for permutation groups."""
        generators = [[1, 2, 0], [1, 0, 2]]  # 3-cycle and swap
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test with single state
        state = jnp.array([[0, 1, 2]])  # Batch format
        neighbors = graph.get_neighbors(state)

        # Should have 2 neighbors (one for each generator)
        assert neighbors.shape[0] == 2
        assert neighbors.shape[1] == 3

        # First neighbor: apply generator 0 ([1,2,0]) to [0,1,2] -> [1,2,0]
        expected_neighbor_0 = jnp.array([1, 2, 0])
        assert jnp.array_equal(neighbors[0], expected_neighbor_0)

        # Second neighbor: apply generator 1 ([1,0,2]) to [0,1,2] -> [1,0,2]
        expected_neighbor_1 = jnp.array([1, 0, 2])
        assert jnp.array_equal(neighbors[1], expected_neighbor_1)

    def test_get_neighbors_matrix_group(self):
        """Test get_neighbors for matrix groups."""
        # 2x2 matrix group with modulo 3
        gen1 = MatrixGenerator.create([[1, 1], [0, 1]], modulo=3)
        gen2 = MatrixGenerator.create([[1, 0], [1, 1]], modulo=3)

        definition = CayleyGraphDef.for_matrix_group(generators=[gen1, gen2], central_state=[[1, 0], [0, 1]])
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test with identity matrix (flattened)
        identity_flat = jnp.array([[1, 0, 0, 1]])  # Batch format
        neighbors = graph.get_neighbors(identity_flat)

        # Should have 2 neighbors (one for each generator)
        assert neighbors.shape[0] == 2
        assert neighbors.shape[1] == 4

        # First neighbor: [[1,1],[0,1]] * [[1,0],[0,1]] = [[1,1],[0,1]]
        expected_neighbor_0 = jnp.array([1, 1, 0, 1])
        assert jnp.array_equal(neighbors[0], expected_neighbor_0)

        # Second neighbor: [[1,0],[1,1]] * [[1,0],[0,1]] = [[1,0],[1,1]]
        expected_neighbor_1 = jnp.array([1, 0, 1, 1])
        assert jnp.array_equal(neighbors[1], expected_neighbor_1)

    def test_get_neighbors_decoded(self):
        """Test get_neighbors_decoded method."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test with decoded state
        state = jnp.array([0, 1, 2])  # Single state, not batch
        neighbors = graph.get_neighbors_decoded(state)

        # Should return decoded neighbors
        assert neighbors.shape[0] == 2
        assert neighbors.shape[1] == 3

        expected_neighbors = jnp.array([[1, 2, 0], [1, 0, 2]])
        np.testing.assert_array_equal(neighbors, expected_neighbors)

    def test_apply_path_single_generator(self):
        """Test apply_path with single generator."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Apply single generator
        state = jnp.array([0, 1, 2])
        result = graph.apply_path(state, [0])
        expected = jnp.array([1, 2, 0])

        assert jnp.array_equal(result, expected)

    def test_apply_path_multiple_generators(self):
        """Test apply_path with multiple generators."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Apply sequence of generators
        state = jnp.array([0, 1, 2])
        result = graph.apply_path(state, [0, 1])  # First apply gen 0, then gen 1

        # Step 1: [0,1,2] -> [1,2,0] (apply gen 0)
        # Step 2: [1,2,0] -> [2,1,0] (apply gen 1)
        expected = jnp.array([2, 1, 0])

        assert jnp.array_equal(result, expected)

    def test_apply_path_matrix_group(self):
        """Test apply_path with matrix group."""
        # 2x2 matrix group with modulo 3
        gen1 = MatrixGenerator.create([[1, 1], [0, 1]], modulo=3)
        gen2 = MatrixGenerator.create([[1, 0], [1, 1]], modulo=3)

        definition = CayleyGraphDef.for_matrix_group(generators=[gen1, gen2], central_state=[[1, 0], [0, 1]])
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Start with identity matrix
        identity = jnp.array([[1, 0], [0, 1]])

        # Apply gen1 then gen2
        result = graph.apply_path(identity, [0, 1])

        # Step 1: [[1,1],[0,1]] * [[1,0],[0,1]] = [[1,1],[0,1]]
        # Step 2: [[1,0],[1,1]] * [[1,1],[0,1]] = [[1,1],[1,2]] = [[1,1],[1,2]] (mod 3)
        expected = jnp.array([[1, 1], [1, 2]])

        assert jnp.array_equal(result, expected)

    def test_neighbors_batch_processing(self):
        """Test neighbor generation with batch of states."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Batch of states
        states = jnp.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        neighbors = graph.get_neighbors(states)

        # Should have 6 neighbors total (3 states * 2 generators)
        assert neighbors.shape[0] == 6
        assert neighbors.shape[1] == 3

        # Check first state's neighbors
        # State [0,1,2] with gen 0 -> [1,2,0]
        assert jnp.array_equal(neighbors[0], jnp.array([1, 2, 0]))
        # State [0,1,2] with gen 1 -> [1,0,2]
        assert jnp.array_equal(neighbors[3], jnp.array([1, 0, 2]))

    def test_jit_compilation_performance(self):
        """Test that JIT compilation works without errors."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test multiple calls to ensure JIT compilation works
        state = jnp.array([[0, 1, 2]])

        # First call (compilation)
        neighbors1 = graph.get_neighbors(state)

        # Second call (should use compiled version)
        neighbors2 = graph.get_neighbors(state)

        # Results should be identical
        assert jnp.array_equal(neighbors1, neighbors2)

        # Test apply_path JIT compilation
        path_result1 = graph.apply_path([0, 1, 2], [0, 1])
        path_result2 = graph.apply_path([0, 1, 2], [0, 1])

        assert jnp.array_equal(path_result1, path_result2)


@pytest.mark.requires_jax
class TestJAXCayleyGraphCompatibility:
    """Test compatibility with existing CayleyGraph interface."""

    def test_api_compatibility(self):
        """Test that JAX implementation has same API as original."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)

        # Create both implementations
        jax_graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test that key attributes exist
        assert hasattr(jax_graph, "definition")
        assert hasattr(jax_graph, "central_state")
        assert hasattr(jax_graph, "encoded_state_size")
        assert hasattr(jax_graph, "encode_states")
        assert hasattr(jax_graph, "decode_states")
        assert hasattr(jax_graph, "apply_generator")

    def test_state_representation_consistency(self):
        """Test that state representations are consistent."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Central state should be identity permutation
        expected_central = jnp.array([0, 1, 2])
        assert jnp.array_equal(graph.central_state, expected_central)

        # Encoding/decoding should preserve state
        test_states = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        encoded = graph.encode_states(test_states)
        decoded = graph.decode_states(encoded)

        np.testing.assert_array_equal(decoded, test_states)
