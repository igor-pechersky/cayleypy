"""TPU Backend Infrastructure for JAX/TPU acceleration in CayleyPy.

This module provides the core TPU backend system for integrating Flax NNX with CayleyPy,
enabling TPU v6e (Trillium) acceleration with native int64 support for large-scale graph operations.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    nnx = None  # type: ignore


@dataclass
class TPUConfig:
    """TPU v6e configuration with native int64 support."""

    enable_x64: bool = True  # Enable native int64 support
    memory_fraction: float = 0.9  # Use 90% of TPU v6e's 32GB HBM
    compilation_cache: bool = True

    def apply(self):
        """Apply TPU configuration."""
        if self.enable_x64:
            jax.config.update("jax_enable_x64", True)
        if self.compilation_cache:
            jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


class TPUBackend(nnx.Module):
    """NNX-based TPU backend for v6e (Trillium) with native int64 support."""

    def __init__(self, config: Optional[TPUConfig] = None, rngs: Optional[nnx.Rngs] = None):
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX and Flax are required for TPU backend. "
                "Install with: pip install 'cayleypy[jax-tpu]'"
            )

        self.config = nnx.Variable(config or TPUConfig())
        self.config.value.apply()

        # Initialize TPU devices
        self.devices = jax.devices("tpu")
        self.is_available = len(self.devices) > 0

        if not self.is_available:
            raise RuntimeError("No TPU devices found. This backend requires TPU v6e.")

        # TPU v6e specific capabilities
        self.capabilities = nnx.Variable(
            {
                "supports_int32": True,
                "supports_int64": True,  # Native int64 support on TPU v6e!
                "supports_float32": True,
                "supports_float64": True,  # Native float64 support on TPU v6e!
                "supports_bfloat16": True,
                "hbm_per_chip_gb": 32,  # TPU v6e spec
                "systolic_array_size": (256, 256),  # TPU v6e estimated spec
                "max_batch_size": 1024 * 1024,  # Estimated based on 32GB memory
            }
        )

        # Test native int64 support
        try:
            test_array = jnp.array([1, 2, 3], dtype=jnp.int64)
            test_result = test_array + 1  # Simple operation to verify support
            assert test_result.dtype == jnp.int64

            print(f"TPU v6e Backend initialized with {len(self.devices)} devices")
            print(f"HBM per chip: {self.capabilities.value['hbm_per_chip_gb']}GB")
            print("Native int64 support: VERIFIED ✓")
            print(f"Native float64 support: {self.capabilities.value['supports_float64']}")

        except Exception as e:
            raise RuntimeError(f"Failed to verify int64 support on TPU: {e}") from e

        # Initialize RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(42)
        self.rngs = rngs

        # Performance metrics
        self.metrics = nnx.Variable(
            {
                "device_count": len(self.devices),
                "memory_allocated_gb": 0.0,
                "compilation_cache_size": 0,
                "operations_count": 0,
                "int64_operations_count": 0,
            }
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("TPU v6e Backend initialized successfully")

    def get_device(self):
        """Get primary TPU device."""
        return self.devices[0] if self.is_available else None

    def supports_dtype(self, dtype) -> bool:
        """Check if TPU supports the given dtype."""
        dtype_support = {
            jnp.int32: self.capabilities.value["supports_int32"],
            jnp.int64: self.capabilities.value["supports_int64"],
            jnp.float32: self.capabilities.value["supports_float32"],
            jnp.float64: self.capabilities.value["supports_float64"],
            jnp.bfloat16: self.capabilities.value["supports_bfloat16"],
        }
        return bool(dtype_support.get(dtype, False))

    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed TPU device information."""
        if not self.is_available:
            return {"available": False, "reason": "No TPU devices found"}

        return {
            "available": True,
            "device_type": "tpu",
            "device_count": len(self.devices),
            "devices": [str(device) for device in self.devices],
            "capabilities": dict(self.capabilities.value),
            "config": {
                "enable_x64": self.config.value.enable_x64,
                "memory_fraction": self.config.value.memory_fraction,
                "compilation_cache": self.config.value.compilation_cache,
            },
        }

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        # TPU memory usage is harder to track than GPU
        # Return estimated usage based on operations
        hbm_per_chip: Any = self.capabilities.value["hbm_per_chip_gb"]
        total_memory_gb = len(self.devices) * float(hbm_per_chip)
        used_memory_gb = self.metrics.value["memory_allocated_gb"]

        return {
            "total_memory_gb": total_memory_gb,
            "used_memory_gb": used_memory_gb,
            "available_memory_gb": total_memory_gb - used_memory_gb,
            "memory_fraction": used_memory_gb / total_memory_gb if total_memory_gb > 0 else 0.0,
        }

    def clear_compilation_cache(self):
        """Clear JAX compilation cache."""
        try:
            jax.clear_caches()
            self.metrics.value["compilation_cache_size"] = 0
            self.logger.info("JAX compilation cache cleared")
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.warning("Could not clear compilation cache: %s", e)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            **dict(self.metrics.value),
            "device_info": self.get_device_info(),
            "memory_usage": self.get_memory_usage(),
        }

    def verify_int64_precision(self) -> bool:
        """Verify that int64 operations maintain precision."""
        try:
            # Test large int64 values that would overflow int32
            large_val = jnp.int64(2**40)  # Much larger than int32 max
            result = large_val + jnp.int64(1)
            expected = 2**40 + 1

            # Verify the computation was done with full precision
            return int(result) == expected
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("int64 precision verification failed: %s", e)
            return False


# Global TPU backend instance
_tpu_backend: Optional[TPUBackend] = None


def get_tpu_backend() -> TPUBackend:
    """Get global TPU backend instance."""
    global _tpu_backend  # pylint: disable=global-statement
    if _tpu_backend is None:
        try:
            _tpu_backend = TPUBackend()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.getLogger(__name__).error("Failed to initialize TPU backend: %s", e)
            raise RuntimeError(f"TPU backend initialization failed: {e}") from e
    return _tpu_backend


def create_tpu_backend(config: Optional[TPUConfig] = None) -> TPUBackend:
    """Factory function to create TPU backend with error handling."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX and Flax are not available. TPU acceleration requires: "
            "pip install 'cayleypy[jax-tpu]'"
        )

    try:
        backend = TPUBackend(config)

        if not backend.is_available:
            raise RuntimeError("No TPU devices available")

        # Verify int64 support
        if not backend.verify_int64_precision():
            raise RuntimeError("TPU int64 precision verification failed")

        return backend

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).error("Failed to create TPU backend: %s", e)
        raise


def is_tpu_available() -> bool:
    """Check if TPU acceleration is available."""
    if not JAX_AVAILABLE:
        return False

    try:
        backend = get_tpu_backend()
        return backend.is_available
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def print_tpu_info():
    """Print TPU device information."""
    print("CayleyPy TPU v6e Backend Information")
    print("=" * 40)

    if not JAX_AVAILABLE:
        print("JAX not available. Install with: pip install 'cayleypy[jax-tpu]'")
        return

    try:
        backend = get_tpu_backend()
        info = backend.get_device_info()

        print(f"TPU Available: {info['available']}")
        if info["available"]:
            print(f"Device Count: {info['device_count']}")
            print(f"Devices: {info['devices']}")
            print(f"Native int64 Support: {info['capabilities']['supports_int64']}")
            print(f"Native float64 Support: {info['capabilities']['supports_float64']}")
            print(f"HBM per Chip: {info['capabilities']['hbm_per_chip_gb']}GB")
            print(f"Total HBM: {info['device_count'] * info['capabilities']['hbm_per_chip_gb']}GB")

            # Test int64 precision
            precision_ok = backend.verify_int64_precision()
            print(f"int64 Precision Test: {'PASSED ✓' if precision_ok else 'FAILED ✗'}")
        else:
            print(f"Reason: {info.get('reason', 'Unknown')}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error getting TPU info: {e}")


if __name__ == "__main__":
    # If run as script, print TPU information
    print_tpu_info()
