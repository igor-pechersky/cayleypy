"""Integration tests for JAX device management with real JAX backend."""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from cayleypy.jax_device_manager import JAXDeviceManager, DeviceFallbackHandler


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXDeviceManagerIntegration:
    """Integration tests with real JAX backend."""

    def test_device_manager_initialization(self):
        """Test that device manager initializes with real JAX."""
        manager = JAXDeviceManager(device="auto")

        # Should have at least CPU available
        assert manager.get_device_count() >= 1
        assert manager.primary_device is not None

        # Should be able to identify device type
        assert manager.is_cpu() or manager.is_gpu() or manager.is_tpu()

    def test_put_array_on_device(self):
        """Test placing arrays on device with real JAX."""
        manager = JAXDeviceManager(device="auto")

        # Test with numpy array
        np_array = np.array([1, 2, 3, 4, 5])
        jax_array = manager.put_on_device(np_array)

        assert isinstance(jax_array, jnp.ndarray)
        assert jnp.array_equal(jax_array, jnp.array([1, 2, 3, 4, 5]))

    def test_put_list_on_device(self):
        """Test placing Python lists on device."""
        manager = JAXDeviceManager(device="auto")

        # Test with Python list
        python_list = [1, 2, 3, 4, 5]
        jax_array = manager.put_on_device(python_list)

        assert isinstance(jax_array, jnp.ndarray)
        assert jnp.array_equal(jax_array, jnp.array([1, 2, 3, 4, 5]))

    def test_memory_info_retrieval(self):
        """Test memory info retrieval with real devices."""
        manager = JAXDeviceManager(device="auto")

        memory_info = manager.get_memory_info()

        # Should have info for at least one device
        assert len(memory_info) >= 1

        # Check structure of memory info
        for device_str, info in memory_info.items():
            assert "device_id" in info
            assert "platform" in info
            assert "total_memory_gb" in info or "error" in info

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        manager = JAXDeviceManager(device="auto")

        # This should not raise an exception
        manager.clear_cache()

    def test_device_type_detection(self):
        """Test device type detection methods."""
        manager = JAXDeviceManager(device="auto")

        # Exactly one should be true
        device_types = [manager.is_cpu(), manager.is_gpu(), manager.is_tpu()]
        assert sum(device_types) == 1

    def test_string_representations(self):
        """Test string representation methods."""
        manager = JAXDeviceManager(device="auto")

        str_repr = str(manager)
        assert "JAXDeviceManager" in str_repr

        repr_str = repr(manager)
        assert "JAXDeviceManager" in repr_str
        assert "primary_device" in repr_str


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestDeviceFallbackHandlerIntegration:
    """Integration tests for device fallback with real JAX."""

    def test_get_best_device_manager(self):
        """Test getting best available device manager."""
        handler = DeviceFallbackHandler()

        manager = handler.get_best_device_manager()

        assert isinstance(manager, JAXDeviceManager)
        assert manager.get_device_count() >= 1

    def test_execute_with_fallback_success(self):
        """Test successful execution with fallback handler."""
        handler = DeviceFallbackHandler()

        def test_function(device_manager, test_value):
            # Simple function that uses the device manager
            array = device_manager.put_on_device([1, 2, 3])
            return jnp.sum(array) + test_value

        result = handler.execute_with_fallback(test_function, 10)

        # Should return 6 (sum of [1,2,3]) + 10 = 16
        assert result == 16


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXArrayOperations:
    """Test JAX array operations through device manager."""

    def test_basic_array_operations(self):
        """Test basic JAX array operations."""
        manager = JAXDeviceManager(device="auto")

        # Create arrays on device
        a = manager.put_on_device([1, 2, 3, 4])
        b = manager.put_on_device([5, 6, 7, 8])

        # Perform operations
        c = a + b
        d = jnp.dot(a, b)

        # Verify results
        expected_c = jnp.array([6, 8, 10, 12])
        assert jnp.array_equal(c, expected_c)
        assert d == 70  # 1*5 + 2*6 + 3*7 + 4*8

    def test_matrix_operations(self):
        """Test matrix operations on device."""
        manager = JAXDeviceManager(device="auto")

        # Create matrices
        matrix_a = manager.put_on_device([[1, 2], [3, 4]])
        matrix_b = manager.put_on_device([[5, 6], [7, 8]])

        # Matrix multiplication
        result = jnp.matmul(matrix_a, matrix_b)

        # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        expected = jnp.array([[19, 22], [43, 50]])
        assert jnp.array_equal(result, expected)

    def test_large_array_handling(self):
        """Test handling of larger arrays."""
        manager = JAXDeviceManager(device="auto")

        # Create a moderately large array (not huge to avoid memory issues in tests)
        large_array = np.random.rand(1000, 100)
        jax_array = manager.put_on_device(large_array)

        # Perform operation
        result = jnp.sum(jax_array, axis=1)

        # Verify shape and that it's a JAX array
        assert result.shape == (1000,)
        assert isinstance(result, jnp.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
