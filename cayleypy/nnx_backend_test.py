"""Tests for NNX Backend Infrastructure."""

import io
import sys
import warnings
from unittest.mock import patch

import pytest

from .nnx_backend import (
    NNXConfig,
    NNXBackend,
    create_nnx_backend,
    get_global_backend,
    set_global_backend,
    is_nnx_available,
    get_recommended_jax_installation,
    print_installation_guide,
    JAX_AVAILABLE,
)


class TestNNXConfig:
    """Test NNX configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NNXConfig()

        assert config.preferred_device == "auto"
        assert config.enable_x64 is True
        assert config.memory_fraction == 0.8
        assert config.enable_jit is True
        assert config.chunk_size == 10000
        assert config.max_memory_gb == 16.0
        assert isinstance(config.xla_flags, dict)
        assert isinstance(config.mesh_shape, tuple)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = NNXConfig(preferred_device="gpu", enable_x64=False, memory_fraction=0.5, chunk_size=5000)

        assert config.preferred_device == "gpu"
        assert config.enable_x64 is False
        assert config.memory_fraction == 0.5
        assert config.chunk_size == 5000


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestNNXBackend:
    """Test NNX backend functionality when JAX is available."""

    def test_backend_creation(self):
        """Test basic backend creation."""
        config = NNXConfig(preferred_device="cpu")
        backend = NNXBackend(config)

        assert backend.config == config
        assert backend.device_type == "cpu"
        assert backend.is_available()
        assert len(backend.devices) > 0

    def test_device_detection_cpu(self):
        """Test CPU device detection."""
        config = NNXConfig(preferred_device="cpu")
        backend = NNXBackend(config)

        assert backend.device_type == "cpu"
        assert backend.is_available()

    def test_device_detection_auto(self):
        """Test automatic device detection."""
        config = NNXConfig(preferred_device="auto")
        backend = NNXBackend(config)

        # Should detect some device (at least CPU)
        assert backend.device_type in ["cpu", "gpu", "tpu"]
        assert backend.is_available()

    def test_get_device_info(self):
        """Test device information retrieval."""
        backend = NNXBackend(NNXConfig(preferred_device="cpu"))
        info = backend.get_device_info()

        assert info["available"] is True
        assert info["device_type"] == "cpu"
        assert info["device_count"] > 0
        assert isinstance(info["devices"], list)
        assert isinstance(info["mesh_shape"], (tuple, dict))  # Can be tuple or OrderedDict

    def test_performance_metrics(self):
        """Test performance metrics collection."""
        backend = NNXBackend(NNXConfig(preferred_device="cpu"))
        metrics = backend.get_performance_metrics()

        assert "device_count" in metrics
        assert "memory_allocated" in metrics
        assert "device_info" in metrics
        assert "config" in metrics

        # Check that device info is included
        assert metrics["device_info"]["available"] is True

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        backend = NNXBackend(NNXConfig(preferred_device="cpu"))
        memory_stats = backend.get_memory_usage()

        # Should return dict (may be empty for CPU)
        assert isinstance(memory_stats, dict)

    def test_compilation_cache_management(self):
        """Test compilation cache management."""
        backend = NNXBackend(NNXConfig(preferred_device="cpu"))

        # Should not raise exception
        backend.clear_compilation_cache()

        # Metrics should be updated
        assert backend.metrics.value["compilation_cache_size"] == 0


class TestNNXBackendWithoutJAX:
    """Test NNX backend behavior when JAX is not available."""

    @patch("cayleypy.nnx_backend.JAX_AVAILABLE", False)
    def test_backend_creation_without_jax(self):
        """Test that backend creation fails gracefully without JAX."""
        with pytest.raises(ImportError, match="JAX and Flax are required"):
            NNXBackend()

    @patch("cayleypy.nnx_backend.JAX_AVAILABLE", False)
    def test_create_backend_without_jax(self):
        """Test factory function behavior without JAX."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend = create_nnx_backend()

            assert backend is None
            assert len(w) == 1
            assert "JAX and Flax are not available" in str(w[0].message)
            assert "cayleypy[jax]" in str(w[0].message)


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_nnx_backend_with_params(self):
        """Test backend creation with custom parameters."""
        backend = create_nnx_backend(preferred_device="cpu", enable_jit=False, memory_fraction=0.5)

        if JAX_AVAILABLE:
            assert backend is not None
            assert backend.config.preferred_device == "cpu"
            assert backend.config.enable_jit is False
            assert backend.config.memory_fraction == 0.5
        else:
            assert backend is None

    def test_global_backend_management(self):
        """Test global backend instance management."""
        # Clear global backend
        set_global_backend(None)

        # Get global backend (should create new one)
        backend1 = get_global_backend()
        backend2 = get_global_backend()

        if JAX_AVAILABLE:
            assert backend1 is not None
            assert backend1 is backend2  # Should be same instance
        else:
            assert backend1 is None
            assert backend2 is None

        # Set custom backend
        if JAX_AVAILABLE:
            custom_backend = create_nnx_backend(preferred_device="cpu")
            set_global_backend(custom_backend)

            backend3 = get_global_backend()
            assert backend3 is custom_backend

    def test_is_nnx_available(self):
        """Test NNX availability check."""
        available = is_nnx_available()

        if JAX_AVAILABLE:
            # Should be True if JAX is available and devices exist
            assert isinstance(available, bool)
        else:
            assert available is False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestNNXBackendIntegration:
    """Integration tests for NNX backend."""

    def test_sharded_array_creation(self):
        """Test sharded array creation."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Import JAX numpy locally since it's only used in this test
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

        backend = NNXBackend(NNXConfig(preferred_device="cpu"))

        # Create test array
        array = jnp.array([1, 2, 3, 4, 5])

        # Create sharded array
        sharded_array = backend.create_sharded_array(array)

        # Should not raise exception and preserve data
        assert jnp.array_equal(array, sharded_array)

    def test_backend_with_different_configs(self):
        """Test backend with various configuration combinations."""
        configs = [
            NNXConfig(preferred_device="cpu", enable_jit=True),
            NNXConfig(preferred_device="cpu", enable_jit=False),
            NNXConfig(preferred_device="auto", memory_fraction=0.5),
        ]

        for config in configs:
            backend = NNXBackend(config)
            assert backend.is_available()
            assert backend.device_type in ["cpu", "gpu", "tpu"]

            # Test basic functionality
            info = backend.get_device_info()
            assert info["available"] is True

            metrics = backend.get_performance_metrics()
            assert "device_count" in metrics


class TestInstallationGuide:
    """Test installation guide functionality."""

    def test_get_recommended_installation(self):
        """Test recommended installation detection."""
        recommendation = get_recommended_jax_installation()

        # Should return a string with pip install command
        assert isinstance(recommendation, str)
        assert "pip install" in recommendation
        assert "cayleypy[jax" in recommendation

    def test_print_installation_guide(self):
        """Test installation guide printing."""

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            print_installation_guide()
            output = captured_output.getvalue()

            # Check that guide contains expected content
            assert "Installation Guide" in output
            assert "cayleypy[jax]" in output
            assert "cayleypy[jax-gpu]" in output
            assert "cayleypy[jax-tpu]" in output
            assert "Recommended for your system" in output

        finally:
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    pytest.main([__file__])
