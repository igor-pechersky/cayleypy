"""Tests for TPU Backend with native int64 support."""

import pytest
import numpy as np

try:
    import jax.numpy as jnp
    from .tpu_backend import TPUBackend, TPUConfig, get_tpu_backend, is_tpu_available

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestTPUBackend:
    """Test TPU backend functionality."""

    def test_tpu_backend_initialization(self):
        """Test TPU backend can be initialized."""
        backend = TPUBackend()
        assert backend.is_available
        assert len(backend.devices) > 0
        assert backend.capabilities.value["supports_int64"]
        assert backend.capabilities.value["supports_float64"]

    def test_tpu_config(self):
        """Test TPU configuration."""
        config = TPUConfig(enable_x64=True, memory_fraction=0.8)
        backend = TPUBackend(config)
        assert backend.config.value.enable_x64
        assert backend.config.value.memory_fraction == 0.8
        del backend  # Silence unused variable warning

    def test_native_int64_support(self):
        """Test native int64 operations on TPU."""
        backend = TPUBackend()

        # Test int64 array creation
        arr = jnp.array([1, 2, 3, 4], dtype=jnp.int64)
        assert arr.dtype == jnp.int64

        # Test int64 arithmetic
        result = arr + jnp.int64(10)
        assert result.dtype == jnp.int64
        np.testing.assert_array_equal(result, [11, 12, 13, 14])
        del backend  # Silence unused variable warning

    def test_native_float64_support(self):
        """Test native float64 operations on TPU."""
        backend = TPUBackend()
        del backend  # Silence unused variable warning

        # Test float64 array creation
        arr = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        assert arr.dtype == jnp.float64

        # Test float64 arithmetic
        result = arr * 2.5
        assert result.dtype == jnp.float64
        np.testing.assert_array_almost_equal(result, [2.5, 5.0, 7.5, 10.0])

    def test_int64_precision_verification(self):
        """Test that int64 operations maintain full precision."""
        backend = TPUBackend()

        # Test with values that would overflow int32
        large_val = jnp.int64(2**40)
        result = large_val + jnp.int64(1)
        expected = 2**40 + 1

        assert int(result) == expected
        assert backend.verify_int64_precision()

    def test_dtype_support_checking(self):
        """Test dtype support checking."""
        backend = TPUBackend()

        assert backend.supports_dtype(jnp.int32)
        assert backend.supports_dtype(jnp.int64)
        assert backend.supports_dtype(jnp.float32)
        assert backend.supports_dtype(jnp.float64)
        assert backend.supports_dtype(jnp.bfloat16)

    def test_device_info(self):
        """Test device information retrieval."""
        backend = TPUBackend()
        info = backend.get_device_info()

        assert info["available"]
        assert info["device_type"] == "tpu"
        assert info["device_count"] > 0
        assert info["capabilities"]["supports_int64"]
        assert info["capabilities"]["hbm_per_chip_gb"] == 32

    def test_memory_usage(self):
        """Test memory usage tracking."""
        backend = TPUBackend()
        usage = backend.get_memory_usage()

        assert "total_memory_gb" in usage
        assert "used_memory_gb" in usage
        assert "available_memory_gb" in usage
        assert usage["total_memory_gb"] > 0

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        backend = TPUBackend()
        metrics = backend.get_performance_metrics()

        assert "device_count" in metrics
        assert "operations_count" in metrics
        assert "int64_operations_count" in metrics
        assert "device_info" in metrics
        assert "memory_usage" in metrics

    def test_global_backend_instance(self):
        """Test global backend instance management."""
        backend1 = get_tpu_backend()
        backend2 = get_tpu_backend()

        # Should return the same instance
        assert backend1 is backend2
        assert backend1.is_available

    def test_is_tpu_available(self):
        """Test TPU availability check."""
        assert is_tpu_available()


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_large_int64_operations():
    """Test operations with large int64 values that exceed int32 range."""
    backend = TPUBackend()
    del backend  # Silence unused variable warning

    # Create arrays with values larger than int32 max (2^31 - 1)
    large_vals = jnp.array([2**40, 2**50, 2**60], dtype=jnp.int64)

    # Test arithmetic operations
    result = large_vals + jnp.int64(1)
    expected = np.array([2**40 + 1, 2**50 + 1, 2**60 + 1], dtype=np.int64)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == jnp.int64


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def test_mixed_precision_operations():
    """Test mixed precision operations between int64 and float64."""
    backend = TPUBackend()
    del backend  # Silence unused variable warning

    int_arr = jnp.array([1, 2, 3], dtype=jnp.int64)
    float_arr = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64)

    # Test mixed operations
    result = int_arr.astype(jnp.float64) + float_arr
    expected = np.array([2.5, 4.5, 6.5], dtype=np.float64)

    np.testing.assert_array_almost_equal(result, expected)
    assert result.dtype == jnp.float64
