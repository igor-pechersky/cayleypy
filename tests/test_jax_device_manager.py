"""Unit tests for JAX device management system."""

import os
import pytest
import warnings
from unittest.mock import Mock, patch, MagicMock

# Test imports with proper error handling
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from cayleypy.jax_device_manager import (
    JAXDeviceManager,
    DeviceFallbackHandler,
    DeviceNotFoundError,
    OutOfMemoryError,
    JAX_AVAILABLE,
)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXDeviceManager:
    """Test cases for JAXDeviceManager."""

    def test_import_availability(self):
        """Test that JAX imports are handled correctly."""
        assert JAX_AVAILABLE is True
        assert jax is not None
        assert jnp is not None

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    def test_auto_device_selection_tpu_preferred(self, mock_local_devices):
        """Test that TPU is preferred in auto selection."""
        # Mock devices with TPU, GPU, and CPU available
        mock_tpu = Mock()
        mock_tpu.platform = "tpu"
        mock_gpu = Mock()
        mock_gpu.platform = "gpu"
        mock_cpu = Mock()
        mock_cpu.platform = "cpu"

        mock_local_devices.return_value = [mock_tpu, mock_gpu, mock_cpu]

        with patch("cayleypy.jax_device_manager.jax_devices", return_value=[mock_tpu]):
            manager = JAXDeviceManager(device="auto")
            assert manager.device_type == "tpu"
            assert manager.is_tpu()

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    def test_auto_device_selection_gpu_fallback(self, mock_local_devices):
        """Test that GPU is selected when TPU unavailable."""
        mock_gpu = Mock()
        mock_gpu.platform = "gpu"
        mock_cpu = Mock()
        mock_cpu.platform = "cpu"

        mock_local_devices.return_value = [mock_gpu, mock_cpu]

        with patch("cayleypy.jax_device_manager.jax_devices", return_value=[mock_gpu]):
            manager = JAXDeviceManager(device="auto")
            assert manager.device_type == "gpu"
            assert manager.is_gpu()

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    def test_auto_device_selection_cpu_fallback(self, mock_local_devices):
        """Test that CPU is selected when only CPU available."""
        mock_cpu = Mock()
        mock_cpu.platform = "cpu"

        mock_local_devices.return_value = [mock_cpu]

        with patch("cayleypy.jax_device_manager.jax_devices", return_value=[mock_cpu]):
            manager = JAXDeviceManager(device="auto")
            assert manager.device_type == "cpu"
            assert manager.is_cpu()

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    def test_specific_device_selection(self, mock_local_devices):
        """Test explicit device selection."""
        mock_gpu = Mock()
        mock_gpu.platform = "gpu"

        mock_local_devices.return_value = [mock_gpu]

        with patch("cayleypy.jax_device_manager.jax_devices", return_value=[mock_gpu]):
            manager = JAXDeviceManager(device="gpu")
            assert manager.device_type == "gpu"

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    def test_device_not_found_error(self, mock_local_devices):
        """Test error when requested device not available."""
        mock_cpu = Mock()
        mock_cpu.platform = "cpu"

        mock_local_devices.return_value = [mock_cpu]

        with pytest.raises(DeviceNotFoundError):
            JAXDeviceManager(device="tpu")

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    def test_no_devices_error(self, mock_local_devices):
        """Test error when no devices available."""
        mock_local_devices.return_value = []

        with pytest.raises(DeviceNotFoundError):
            JAXDeviceManager(device="auto")

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    @patch("cayleypy.jax_device_manager.jax.device_put")
    def test_put_on_device_small_array(self, mock_device_put, mock_jax_devices, mock_local_devices):
        """Test placing small array on device."""
        mock_device = Mock()
        mock_device.platform = "cpu"

        mock_local_devices.return_value = [mock_device]
        mock_jax_devices.return_value = [mock_device]

        manager = JAXDeviceManager(device="cpu")

        # Mock small array
        test_array = jnp.array([1, 2, 3, 4])
        mock_device_put.return_value = test_array

        result = manager.put_on_device(test_array)

        mock_device_put.assert_called_once_with(test_array, mock_device)
        assert result is test_array

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    def test_should_shard_array_large(self, mock_jax_devices, mock_local_devices):
        """Test that large arrays are identified for sharding."""
        mock_device1 = Mock()
        mock_device1.platform = "tpu"
        mock_device2 = Mock()
        mock_device2.platform = "tpu"

        mock_local_devices.return_value = [mock_device1, mock_device2]
        mock_jax_devices.return_value = [mock_device1, mock_device2]

        manager = JAXDeviceManager(device="tpu")

        # Create mock large array
        large_array = Mock()
        large_array.nbytes = 2e9  # 2GB
        large_array.size = 250000000

        assert manager._should_shard_array(large_array) is True

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    def test_should_not_shard_small_array(self, mock_jax_devices, mock_local_devices):
        """Test that small arrays are not sharded."""
        mock_device = Mock()
        mock_device.platform = "cpu"

        mock_local_devices.return_value = [mock_device]
        mock_jax_devices.return_value = [mock_device]

        manager = JAXDeviceManager(device="cpu")

        # Create mock small array
        small_array = Mock()
        small_array.nbytes = 1000  # 1KB
        small_array.size = 125

        assert manager._should_shard_array(small_array) is False

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    def test_get_memory_info_basic(self, mock_jax_devices, mock_local_devices):
        """Test basic memory info retrieval."""
        mock_device = Mock()
        mock_device.platform = "cpu"
        mock_device.device_kind = "cpu"
        # Ensure the mock device doesn't have memory_stats to trigger fallback path
        del mock_device.memory_stats

        mock_local_devices.return_value = [mock_device]
        mock_jax_devices.return_value = [mock_device]

        manager = JAXDeviceManager(device="cpu")
        memory_info = manager.get_memory_info()

        assert len(memory_info) == 1
        device_info = list(memory_info.values())[0]
        assert device_info["platform"] == "cpu"
        assert "total_memory_gb" in device_info
        assert device_info["total_memory_gb"] == 32.0  # CPU fallback value

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    def test_get_memory_info_with_stats(self, mock_jax_devices, mock_local_devices):
        """Test memory info with device stats."""
        mock_device = Mock()
        mock_device.platform = "gpu"
        mock_device.device_kind = "gpu"
        mock_device.memory_stats.return_value = {
            "bytes_limit": 8e9,  # 8GB
            "bytes_in_use": 2e9,  # 2GB
        }

        mock_local_devices.return_value = [mock_device]
        mock_jax_devices.return_value = [mock_device]

        manager = JAXDeviceManager(device="gpu")
        memory_info = manager.get_memory_info()

        device_info = list(memory_info.values())[0]
        assert device_info["total_memory_gb"] == 8.0
        assert device_info["used_memory_gb"] == 2.0
        assert device_info["free_memory_gb"] == 6.0

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    @patch("cayleypy.jax_device_manager.jax.clear_caches")
    def test_clear_cache(self, mock_clear_caches, mock_jax_devices, mock_local_devices):
        """Test cache clearing functionality."""
        mock_device = Mock()
        mock_device.platform = "cpu"

        mock_local_devices.return_value = [mock_device]
        mock_jax_devices.return_value = [mock_device]

        manager = JAXDeviceManager(device="cpu")
        manager.clear_cache()

        mock_clear_caches.assert_called_once()

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    def test_device_count(self, mock_jax_devices, mock_local_devices):
        """Test device count functionality."""
        mock_device1 = Mock()
        mock_device1.platform = "tpu"
        mock_device2 = Mock()
        mock_device2.platform = "tpu"

        mock_local_devices.return_value = [mock_device1, mock_device2]
        mock_jax_devices.return_value = [mock_device1, mock_device2]

        manager = JAXDeviceManager(device="tpu")
        assert manager.get_device_count() == 2

    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    def test_string_representations(self, mock_jax_devices, mock_local_devices):
        """Test string representation methods."""
        mock_device = Mock()
        mock_device.platform = "gpu"

        mock_local_devices.return_value = [mock_device]
        mock_jax_devices.return_value = [mock_device]

        manager = JAXDeviceManager(device="gpu")

        str_repr = str(manager)
        assert "JAXDeviceManager" in str_repr
        assert "gpu" in str_repr

        repr_str = repr(manager)
        assert "JAXDeviceManager" in repr_str
        assert "gpu" in repr_str


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestDeviceFallbackHandler:
    """Test cases for DeviceFallbackHandler."""

    def test_init_default_hierarchy(self):
        """Test default device hierarchy initialization."""
        handler = DeviceFallbackHandler()
        assert handler.device_hierarchy == ["tpu", "gpu", "cpu"]

    def test_init_custom_hierarchy(self):
        """Test custom device hierarchy initialization."""
        custom_hierarchy = ["gpu", "cpu"]
        handler = DeviceFallbackHandler(custom_hierarchy)
        assert handler.device_hierarchy == custom_hierarchy

    @patch("cayleypy.jax_device_manager.JAXDeviceManager")
    def test_execute_with_fallback_success_first_device(self, mock_device_manager_class):
        """Test successful execution on first device."""
        # Mock successful device manager creation
        mock_manager = Mock()
        mock_device_manager_class.return_value = mock_manager

        handler = DeviceFallbackHandler(["gpu", "cpu"])

        # Mock function that succeeds
        mock_func = Mock(return_value="success")

        result = handler.execute_with_fallback(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with(mock_manager, "arg1", kwarg1="value1")

    @patch("cayleypy.jax_device_manager.JAXDeviceManager")
    def test_execute_with_fallback_success_second_device(self, mock_device_manager_class):
        """Test successful execution on second device after first fails."""
        # Mock first device manager fails, second succeeds
        mock_manager1 = Mock()
        mock_manager2 = Mock()

        def side_effect(device_type):
            if device_type == "tpu":
                raise DeviceNotFoundError("TPU not found")
            return mock_manager2

        mock_device_manager_class.side_effect = side_effect

        handler = DeviceFallbackHandler(["tpu", "gpu"])

        # Mock function that succeeds on second call
        mock_func = Mock(return_value="success")

        result = handler.execute_with_fallback(mock_func, "arg1")

        assert result == "success"
        mock_func.assert_called_once_with(mock_manager2, "arg1")

    @patch("cayleypy.jax_device_manager.JAXDeviceManager")
    def test_execute_with_fallback_all_devices_fail(self, mock_device_manager_class):
        """Test error when all devices fail."""
        mock_device_manager_class.side_effect = DeviceNotFoundError("No devices")

        handler = DeviceFallbackHandler(["tpu", "gpu", "cpu"])

        mock_func = Mock()

        with pytest.raises(RuntimeError, match="All devices failed"):
            handler.execute_with_fallback(mock_func)

    @patch("cayleypy.jax_device_manager.JAXDeviceManager")
    def test_get_best_device_manager_success(self, mock_device_manager_class):
        """Test getting best available device manager."""
        mock_manager = Mock()
        mock_device_manager_class.return_value = mock_manager

        handler = DeviceFallbackHandler(["gpu", "cpu"])

        result = handler.get_best_device_manager()

        assert result is mock_manager
        mock_device_manager_class.assert_called_once_with("gpu")

    @patch("cayleypy.jax_device_manager.JAXDeviceManager")
    def test_get_best_device_manager_fallback(self, mock_device_manager_class):
        """Test fallback in getting best device manager."""
        mock_manager = Mock()

        def side_effect(device_type):
            if device_type == "tpu":
                raise DeviceNotFoundError("TPU not found")
            return mock_manager

        mock_device_manager_class.side_effect = side_effect

        handler = DeviceFallbackHandler(["tpu", "gpu"])

        result = handler.get_best_device_manager()

        assert result is mock_manager

    @patch("cayleypy.jax_device_manager.JAXDeviceManager")
    def test_get_best_device_manager_no_devices(self, mock_device_manager_class):
        """Test error when no devices available."""
        mock_device_manager_class.side_effect = DeviceNotFoundError("No devices")

        handler = DeviceFallbackHandler(["tpu", "gpu", "cpu"])

        with pytest.raises(RuntimeError, match="No compatible devices found"):
            handler.get_best_device_manager()


class TestJAXNotAvailable:
    """Test behavior when JAX is not available."""

    @patch("cayleypy.jax_device_manager.JAX_AVAILABLE", False)
    def test_device_manager_import_error(self):
        """Test that ImportError is raised when JAX not available."""
        with pytest.raises(ImportError, match="JAX is not available"):
            JAXDeviceManager()


class TestMemoryManagement:
    """Test memory management functionality."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    def test_memory_preallocation_disabled(self, mock_jax_devices, mock_local_devices):
        """Test memory preallocation can be disabled."""
        mock_device = Mock()
        mock_device.platform = "cpu"

        mock_local_devices.return_value = [mock_device]
        mock_jax_devices.return_value = [mock_device]

        with patch.dict(os.environ, {}, clear=True):
            manager = JAXDeviceManager(device="cpu", enable_memory_preallocation=False)
            assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    @patch("cayleypy.jax_device_manager.jax.local_devices")
    @patch("cayleypy.jax_device_manager.jax_devices")
    def test_tpu_environment_configuration(self, mock_jax_devices, mock_local_devices):
        """Test TPU environment configuration."""
        mock_device = Mock()
        mock_device.platform = "tpu"

        mock_local_devices.return_value = [mock_device]
        mock_jax_devices.return_value = [mock_device]

        with patch.dict(os.environ, {"TPU_NAME": "test-tpu"}, clear=True):
            manager = JAXDeviceManager(device="tpu")
            assert os.environ.get("JAX_PLATFORM_NAME") == "tpu"


if __name__ == "__main__":
    pytest.main([__file__])
