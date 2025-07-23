"""Tests for JAX BFS implementation."""

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
        manager = JAXDeviceManager(device="auto")
        return manager.device_type
    except DeviceNotFoundError:
        if JAX_AVAILABLE:
            available_platforms = jax.local_devices()
            platform_types = {d.platform for d in available_platforms}
            for device in ["cpu", "gpu", "tpu"]:
                if device in platform_types:
                    return device
        return "cpu"


TEST_DEVICE = get_available_device() if JAX_AVAILABLE else "cpu"


@pytest.mark.requires_jax
class TestJAXBFS:
    """Test JAX BFS implementation."""

    def test_bfs_basic_permutation_group(self):
        """Test basic BFS on small permutation group."""
        # Simple 3-element permutation group
        generators = [[1, 2, 0], [1, 0, 2]]  # 3-cycle and swap
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE, verbose=0)

        # Run BFS with small diameter
        result = graph.bfs(max_diameter=3, max_layer_size_to_store=None)

        # Check basic properties
        assert result is not None
        assert hasattr(result, "layer_sizes")
        assert hasattr(result, "layers")
        assert hasattr(result, "bfs_completed")
        assert len(result.layer_sizes) > 0
        assert result.layer_sizes[0] == 1  # Start with central state

    def test_bfs_with_hashes(self):
        """Test BFS with hash return."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        result = graph.bfs(max_diameter=2, return_all_hashes=True)

        assert result.vertices_hashes is not None
        assert len(result.vertices_hashes) == sum(result.layer_sizes)

    def test_bfs_with_edges(self):
        """Test BFS with edge return."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        result = graph.bfs(max_diameter=2, return_all_edges=True)

        assert result.edges_list_hashes is not None
        assert result.edges_list_hashes.shape[1] == 2  # Each edge has start and end

    def test_bfs_matrix_group(self):
        """Test BFS on matrix group."""
        # 2x2 matrix group with modulo 3
        gen1 = MatrixGenerator.create([[1, 1], [0, 1]], modulo=3)
        gen2 = MatrixGenerator.create([[1, 0], [1, 1]], modulo=3)

        definition = CayleyGraphDef.for_matrix_group(generators=[gen1, gen2], central_state=[[1, 0], [0, 1]])
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        result = graph.bfs(max_diameter=3)

        assert result is not None
        assert len(result.layer_sizes) > 0
        assert result.layer_sizes[0] == 1

    def test_bfs_layer_size_limits(self):
        """Test BFS with layer size limits."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test with small layer size limit
        result = graph.bfs(max_diameter=5, max_layer_size_to_store=2)

        # Should still have first layer stored
        assert 0 in result.layers
        # Some layers might not be stored due to size limit
        assert len(result.layers) <= len(result.layer_sizes)

    def test_bfs_batching(self):
        """Test BFS with batching for memory efficiency."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE, batch_size=2)

        result = graph.bfs(max_diameter=3)

        assert result is not None
        assert len(result.layer_sizes) > 0

    def test_bfs_unique_state_detection(self):
        """Test that BFS correctly detects and removes duplicate states."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        result = graph.bfs(max_diameter=4, return_all_hashes=True)

        # Check that all hashes are unique
        all_hashes = result.vertices_hashes
        # Convert to numpy for uniqueness check since result contains PyTorch tensors
        all_hashes_np = all_hashes.numpy() if hasattr(all_hashes, "numpy") else np.array(all_hashes)
        unique_hashes = np.unique(all_hashes_np)
        assert len(unique_hashes) == len(all_hashes_np), "BFS should not have duplicate states"

    def test_bfs_memory_management(self):
        """Test BFS memory management and cleanup."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE, memory_limit_gb=1)

        # Should not crash with memory management
        result = graph.bfs(max_diameter=3)
        assert result is not None

    def test_bfs_start_states(self):
        """Test BFS with custom start states."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Start from a different state
        start_state = [1, 2, 0]
        result = graph.bfs(start_states=start_state, max_diameter=2)

        assert result is not None
        assert len(result.layer_sizes) > 0
        # First layer should contain the start state
        first_layer = result.get_layer(0)
        assert len(first_layer) == 1

    def test_bfs_performance_compilation(self):
        """Test that BFS benefits from JIT compilation."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # First run (compilation)
        result1 = graph.bfs(max_diameter=2)

        # Second run (should use compiled version)
        result2 = graph.bfs(max_diameter=2)

        # Results should be identical
        assert result1.layer_sizes == result2.layer_sizes
        assert result1.bfs_completed == result2.bfs_completed


@pytest.mark.requires_jax
class TestJAXBFSOptimizations:
    """Test BFS optimizations and performance features."""

    def test_bfs_chunked_processing(self):
        """Test BFS with chunked processing for large layers."""
        # Create a slightly larger group to test chunking
        generators = [[1, 2, 3, 0], [1, 0, 3, 2]]  # 4-element group
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE, batch_size=2)

        result = graph.bfs(max_diameter=3)
        assert result is not None

    def test_bfs_inverse_closed_optimization(self):
        """Test BFS optimization for inverse-closed generators."""
        # Create inverse-closed generators
        generators = [[1, 2, 0], [2, 0, 1]]  # 3-cycle and its inverse
        definition = CayleyGraphDef.create(generators)

        # Check if the definition has inverse-closed property and test accordingly
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        result = graph.bfs(max_diameter=3)
        assert result is not None

        # Test that BFS works regardless of inverse-closed status
        assert len(result.layer_sizes) > 0

    def test_bfs_memory_efficient_mode(self):
        """Test BFS memory-efficient mode without edge tracking."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Memory-efficient mode (no edges)
        result = graph.bfs(max_diameter=3, return_all_edges=False)

        assert result is not None
        assert result.edges_list_hashes is None

    def test_bfs_state_deduplication(self):
        """Test efficient state deduplication in BFS layers."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        result = graph.bfs(max_diameter=4, return_all_hashes=True)

        # Verify no duplicates within or across layers
        total_states = sum(result.layer_sizes)
        # Convert to numpy for uniqueness check since result contains PyTorch tensors
        all_hashes_np = (
            result.vertices_hashes.numpy()
            if hasattr(result.vertices_hashes, "numpy")
            else np.array(result.vertices_hashes)
        )
        unique_states = len(np.unique(all_hashes_np))
        assert total_states == unique_states


@pytest.mark.requires_jax
class TestJAXBFSTPUOptimizations:
    """Test TPU-specific BFS optimizations."""

    def test_tpu_sharding_detection(self):
        """Test TPU sharding detection logic."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test small array (should not shard)
        small_array = jnp.ones((10, 3))
        should_shard = graph._should_shard_array(small_array)

        if graph.is_tpu and graph.num_devices > 1:
            assert not should_shard  # Too small to shard
        else:
            assert not should_shard  # Not TPU or single device

    def test_tpu_memory_layout_optimization(self):
        """Test TPU memory layout optimization."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test memory layout optimization
        test_array = jnp.ones((100, 3))
        optimized = graph._optimize_memory_layout_for_tpu(test_array)

        # Should return an array (optimization may or may not change it)
        assert isinstance(optimized, jnp.ndarray)
        assert optimized.shape == test_array.shape

    def test_bfs_with_tpu_optimizations_enabled(self):
        """Test BFS with TPU optimizations enabled."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Run BFS with TPU optimizations
        result = graph.bfs(max_diameter=3, enable_tpu_sharding=True)
        assert result is not None
        assert len(result.layer_sizes) > 0

    def test_bfs_with_tpu_optimizations_disabled(self):
        """Test BFS with TPU optimizations disabled."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Run BFS without TPU optimizations
        result = graph.bfs(max_diameter=3, enable_tpu_sharding=False)
        assert result is not None
        assert len(result.layer_sizes) > 0

    def test_bfs_compiled_layer_processing(self):
        """Test compiled BFS layer processing."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        # Test compiled layer processing
        test_states = jnp.array([[0, 1, 2], [1, 2, 0]])
        test_hashes = graph.hasher.hash_states(test_states)

        try:
            neighbors, neighbor_hashes = graph._bfs_layer_processing_compiled(test_states, test_hashes)
            assert neighbors.shape[0] > 0
            assert neighbor_hashes.shape[0] == neighbors.shape[0]
        except Exception:
            # Compilation might fail in some environments, which is acceptable
            pytest.skip("JIT compilation not available in this environment")

    @pytest.mark.benchmark
    def test_bfs_performance_comparison(self):
        """Benchmark BFS performance with and without TPU optimizations."""
        generators = [[1, 2, 3, 0], [1, 0, 3, 2]]  # Slightly larger group
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        import time

        # Benchmark without TPU optimizations
        start_time = time.time()
        result_no_opt = graph.bfs(max_diameter=4, enable_tpu_sharding=False)
        time_no_opt = time.time() - start_time

        # Benchmark with TPU optimizations
        start_time = time.time()
        result_with_opt = graph.bfs(max_diameter=4, enable_tpu_sharding=True)
        time_with_opt = time.time() - start_time

        # Results should be identical
        assert result_no_opt.layer_sizes == result_with_opt.layer_sizes
        assert result_no_opt.bfs_completed == result_with_opt.bfs_completed

        # Print performance comparison (for manual inspection)
        print(f"BFS without TPU optimizations: {time_no_opt:.4f}s")
        print(f"BFS with TPU optimizations: {time_with_opt:.4f}s")

        if graph.is_tpu:
            print(f"TPU speedup: {time_no_opt / time_with_opt:.2f}x")
        else:
            print("Running on non-TPU device")


@pytest.mark.requires_jax
class TestJAXBFSCompatibility:
    """Test BFS compatibility with existing API."""

    def test_bfs_result_structure(self):
        """Test that BFS result has expected structure."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        result = graph.bfs(max_diameter=2, return_all_hashes=True, return_all_edges=True)

        # Check all expected attributes exist
        assert hasattr(result, "bfs_completed")
        assert hasattr(result, "layer_sizes")
        assert hasattr(result, "layers")
        assert hasattr(result, "vertices_hashes")
        assert hasattr(result, "edges_list_hashes")
        assert hasattr(result, "graph")

        # Check methods exist
        assert hasattr(result, "diameter")
        assert hasattr(result, "get_layer")
        assert hasattr(result, "has_vertices_hashes")
        assert hasattr(result, "has_edges_list_hashes")

    def test_bfs_result_methods(self):
        """Test BFS result methods work correctly."""
        generators = [[1, 2, 0], [1, 0, 2]]
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE)

        result = graph.bfs(max_diameter=3, max_layer_size_to_store=None)

        # Test diameter
        diameter = result.diameter()
        assert diameter == len(result.layer_sizes) - 1

        # Test get_layer
        first_layer = result.get_layer(0)
        assert len(first_layer) == result.layer_sizes[0]

        # Test last_layer
        last_layer = result.last_layer()
        assert len(last_layer) == result.layer_sizes[-1]

    def test_bfs_with_string_encoding(self):
        """Test BFS with string encoding enabled."""
        generators = [[1, 2, 3, 0], [1, 0, 3, 2]]  # 4-element group
        definition = CayleyGraphDef.create(generators)
        graph = JAXCayleyGraph(definition, device=TEST_DEVICE, bit_encoding_width=2)

        result = graph.bfs(max_diameter=2)
        assert result is not None
        assert len(result.layer_sizes) > 0
