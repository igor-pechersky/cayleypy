"""NNX Backend Infrastructure for JAX/GPU/TPU acceleration in CayleyPy.

This module provides the core backend system for integrating Flax NNX with CayleyPy,
enabling GPU and TPU acceleration for large-scale graph operations.
"""

import logging
import os
import subprocess
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    from flax import nnx

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None  # type: ignore
    jnp = None  # type: ignore
    Mesh = None  # type: ignore
    NamedSharding = None  # type: ignore
    PartitionSpec = None  # type: ignore
    nnx = None  # type: ignore


@dataclass
class NNXConfig:
    """Configuration for NNX backend with comprehensive settings."""

    # Device configuration
    preferred_device: str = "auto"  # "auto", "cpu", "gpu", "tpu"
    enable_x64: bool = True  # Enable 64-bit precision

    # Memory management
    memory_fraction: float = 0.8  # Fraction of GPU memory to use
    preallocate_memory: bool = False  # Whether to preallocate GPU memory

    # Compilation settings
    enable_jit: bool = True  # Enable JIT compilation
    jit_cache_size: int = 1000  # JIT compilation cache size

    # Optimization flags
    xla_flags: Optional[Dict[str, Any]] = None  # XLA optimization flags

    # Distributed computing
    enable_sharding: bool = True  # Enable tensor sharding
    mesh_shape: Optional[Tuple[int, ...]] = None  # Shape of device mesh

    # Performance tuning
    chunk_size: int = 10000  # Default chunk size for batch processing
    max_memory_gb: float = 16.0  # Maximum memory usage in GB

    def __post_init__(self):
        """Initialize default values that depend on runtime conditions."""
        if self.xla_flags is None:
            self.xla_flags = {
                "xla_cpu_multi_thread_eigen": False,
                "xla_force_host_platform_device_count": 1,
            }

        if self.mesh_shape is None:
            # Default mesh shape based on available devices
            if JAX_AVAILABLE:
                try:
                    num_devices = len(jax.devices())
                    self.mesh_shape = (num_devices,) if num_devices > 1 else (1,)
                except Exception:  # pylint: disable=broad-exception-caught
                    self.mesh_shape = (1,)
            else:
                self.mesh_shape = (1,)


class NNXBackend:
    """NNX-based backend for hardware acceleration and state management."""

    def __init__(self, config: Optional[NNXConfig] = None, rngs: Optional[Any] = None):
        """Initialize NNX backend with configuration.

        Args:
            config: Backend configuration. If None, uses default config.
            rngs: Random number generators for NNX. If None, creates default.
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX and Flax are required for NNX backend. "
                "Install with:\n"
                "  CPU: pip install 'cayleypy[jax]'\n"
                "  GPU: pip install 'cayleypy[jax-gpu]'\n"
                "  TPU: pip install 'cayleypy[jax-tpu]'"
            )

        self.config = config or NNXConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize JAX configuration
        self._configure_jax()

        # Detect and configure device
        self.device_type = self._detect_device(self.config.preferred_device)
        self.devices = jax.devices(self.device_type)

        # Create device mesh for distributed computation
        self.mesh = self._create_device_mesh()

        # Setup sharding strategy
        self.sharding = self._setup_sharding() if self.config.enable_sharding else None

        # Store optimization configuration as NNX Variable
        self.optimization_config = nnx.Variable(self._setup_optimization_flags())

        # Performance metrics tracking
        metrics_dict = {
            "device_count": len(self.devices),
            "memory_allocated": 0.0,
            "compilation_cache_size": 0,
            "operations_count": 0,
        }
        self.metrics = nnx.Variable(metrics_dict)

        # Initialize RNGs if not provided
        if rngs is None:
            rngs = nnx.Rngs(42)
        self.rngs = rngs

        self.logger.info("NNX Backend initialized with %d %s device(s)", len(self.devices), self.device_type)

    def _configure_jax(self):
        """Configure JAX with optimization flags."""
        # Set XLA flags
        for flag, value in self.config.xla_flags.items():
            os.environ[f"XLA_{flag.upper()}"] = str(value)

        # Configure JAX
        if self.config.enable_x64:
            jax.config.update("jax_enable_x64", True)

        # Memory configuration
        if self.config.preallocate_memory:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.config.memory_fraction)
        else:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    def _detect_device(self, preferred: str) -> str:
        """Detect and configure optimal device using NNX patterns."""
        if preferred == "auto":
            # Auto-detect best available device
            try:
                if jax.devices("tpu"):
                    return "tpu"
            except Exception:  # pylint: disable=broad-exception-caught
                pass

            try:
                if jax.devices("gpu"):
                    return "gpu"
            except Exception:  # pylint: disable=broad-exception-caught
                pass

            return "cpu"

        elif preferred in ["cpu", "gpu", "tpu"]:
            try:
                devices = jax.devices(preferred)
                if devices:
                    return preferred
                else:
                    self.logger.warning("Requested device '%s' not available, falling back to CPU", preferred)
                    return "cpu"
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.warning("Error accessing %s devices: %s, falling back to CPU", preferred, e)
                return "cpu"

        else:
            raise ValueError(f"Invalid device type: {preferred}. Must be 'auto', 'cpu', 'gpu', or 'tpu'")

    def _create_device_mesh(self) -> Mesh:
        """Create device mesh for distributed computation."""
        if len(self.devices) == 1:
            # Single device case
            return Mesh(self.devices, axis_names=["devices"])

        # Multi-device case - reshape according to config
        mesh_shape = self.config.mesh_shape
        if mesh_shape is not None and len(self.devices) != mesh_shape[0]:
            # Adjust mesh shape to match available devices
            mesh_shape = (len(self.devices),)
            self.logger.info("Adjusted mesh shape to %s to match %d devices", mesh_shape, len(self.devices))
        elif mesh_shape is None:
            mesh_shape = (len(self.devices),)

        return Mesh(self.devices[: mesh_shape[0]], axis_names=["devices"])

    def _setup_sharding(self) -> NamedSharding:
        """Setup sharding strategy for tensors."""
        if len(self.devices) == 1:
            # Single device - no sharding needed
            return NamedSharding(self.mesh, PartitionSpec())
        else:
            # Multi-device - shard across devices
            return NamedSharding(self.mesh, PartitionSpec("devices"))

    def _setup_optimization_flags(self) -> Dict[str, Any]:
        """Configure XLA and JAX optimization flags."""
        return {
            "jit_enabled": self.config.enable_jit,
            "cache_size": self.config.jit_cache_size,
            "chunk_size": self.config.chunk_size,
            "max_memory_bytes": int(self.config.max_memory_gb * (2**30)),
            "device_type": self.device_type,
            "num_devices": len(self.devices),
        }

    def is_available(self) -> bool:
        """Check if NNX acceleration is available."""
        return JAX_AVAILABLE and len(self.devices) > 0

    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        if not self.is_available():
            return {"available": False, "reason": "JAX not available"}

        device_info = {
            "available": True,
            "device_type": self.device_type,
            "device_count": len(self.devices),
            "devices": [str(device) for device in self.devices],
            "mesh_shape": self.mesh.shape,
            "sharding_enabled": self.config.enable_sharding,
            "memory_fraction": self.config.memory_fraction,
        }

        # Add device-specific information
        if self.device_type == "gpu":
            try:
                # Get GPU memory info if available
                for i, device in enumerate(self.devices):
                    memory_info = device.memory_stats() if hasattr(device, "memory_stats") else {}
                    device_info[f"gpu_{i}_memory"] = memory_info
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.debug("Could not get GPU memory info: %s", e)

        return device_info

    def create_sharded_array(self, array: jnp.ndarray, sharding: Optional[NamedSharding] = None) -> jnp.ndarray:
        """Create a sharded array using the backend's sharding strategy."""
        if not self.config.enable_sharding or sharding is None:
            sharding = self.sharding

        if sharding is None:
            return array

        return jax.device_put(array, sharding)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        memory_stats = {}

        try:
            if self.device_type == "gpu":
                for i, device in enumerate(self.devices):
                    if hasattr(device, "memory_stats"):
                        stats = device.memory_stats()
                        memory_stats[f"gpu_{i}"] = {
                            "bytes_in_use": stats.get("bytes_in_use", 0),
                            "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
                            "bytes_limit": stats.get("bytes_limit", 0),
                        }

            # Update metrics
            total_memory = sum(
                stats.get("bytes_in_use", 0) for stats in memory_stats.values() if isinstance(stats, dict)
            )
            if JAX_AVAILABLE and hasattr(self.metrics, "value"):
                self.metrics.value["memory_allocated"] = total_memory / (1024**3)  # Convert to GB

        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.debug("Could not get memory usage: %s", e)

        return memory_stats

    def clear_compilation_cache(self):
        """Clear JAX compilation cache."""
        try:
            jax.clear_caches()
            if JAX_AVAILABLE and hasattr(self.metrics, "value"):
                self.metrics.value["compilation_cache_size"] = 0
            else:
                self.metrics["compilation_cache_size"] = 0
            self.logger.info("JAX compilation cache cleared")
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.warning("Could not clear compilation cache: %s", e)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        if JAX_AVAILABLE and hasattr(self.metrics, "value"):
            metrics: Dict[str, Any] = dict(self.metrics.value)
        else:
            metrics = dict(self.metrics)

        metrics.update(
            {
                "device_info": self.get_device_info(),
                "memory_usage": self.get_memory_usage(),
                "config": {
                    "device_type": self.device_type,
                    "enable_jit": self.config.enable_jit,
                    "enable_sharding": self.config.enable_sharding,
                    "chunk_size": self.config.chunk_size,
                    "memory_fraction": self.config.memory_fraction,
                },
            }
        )
        return metrics


def create_nnx_backend(
    preferred_device: str = "auto", enable_jit: bool = True, memory_fraction: float = 0.8, **kwargs
) -> Optional[NNXBackend]:
    """Factory function to create NNX backend with error handling.

    Args:
        preferred_device: Preferred device type ("auto", "cpu", "gpu", "tpu")
        enable_jit: Whether to enable JIT compilation
        memory_fraction: Fraction of GPU memory to use
        **kwargs: Additional configuration options

    Returns:
        NNXBackend instance if successful, None if JAX is not available
    """
    if not JAX_AVAILABLE:
        warnings.warn(
            "JAX and Flax are not available. NNX acceleration will be disabled. "
            "Install with: pip install 'cayleypy[jax]' (CPU), 'cayleypy[jax-gpu]' (GPU), or 'cayleypy[jax-tpu]' (TPU)",
            UserWarning,
        )
        return None

    try:
        config = NNXConfig(
            preferred_device=preferred_device, enable_jit=enable_jit, memory_fraction=memory_fraction, **kwargs
        )

        backend = NNXBackend(config)

        if not backend.is_available():
            warnings.warn("NNX backend created but no devices are available", UserWarning)

        return backend

    except Exception as e:  # pylint: disable=broad-exception-caught
        warnings.warn(f"Failed to create NNX backend: {e}", UserWarning)
        return None


# Global backend instance for convenience
_global_backend: Optional[NNXBackend] = None


def get_global_backend() -> Optional[NNXBackend]:
    """Get or create the global NNX backend instance."""
    global _global_backend  # pylint: disable=global-statement
    if _global_backend is None:
        _global_backend = create_nnx_backend()
    return _global_backend


def set_global_backend(backend: Optional[NNXBackend]):
    """Set the global NNX backend instance."""
    global _global_backend  # pylint: disable=global-statement
    _global_backend = backend


def is_nnx_available() -> bool:
    """Check if NNX acceleration is available."""
    backend = get_global_backend()
    return backend is not None and backend.is_available()


def configure_nnx_from_environment() -> NNXConfig:
    """Create NNX configuration from environment variables.

    Supported environment variables:
    - CAYLEYPY_NNX_DEVICE: Preferred device ("auto", "cpu", "gpu", "tpu")
    - CAYLEYPY_NNX_ENABLE_JIT: Enable JIT compilation ("true"/"false")
    - CAYLEYPY_NNX_MEMORY_FRACTION: GPU memory fraction (0.0-1.0)
    - CAYLEYPY_NNX_CHUNK_SIZE: Default chunk size for batch processing
    - CAYLEYPY_NNX_MAX_MEMORY_GB: Maximum memory usage in GB
    - CAYLEYPY_NNX_ENABLE_X64: Enable 64-bit precision ("true"/"false")

    Returns:
        NNXConfig configured from environment variables
    """
    config = NNXConfig()

    # Device configuration
    if "CAYLEYPY_NNX_DEVICE" in os.environ:
        config.preferred_device = os.environ["CAYLEYPY_NNX_DEVICE"]

    # JIT configuration
    if "CAYLEYPY_NNX_ENABLE_JIT" in os.environ:
        config.enable_jit = os.environ["CAYLEYPY_NNX_ENABLE_JIT"].lower() == "true"

    # Memory configuration
    if "CAYLEYPY_NNX_MEMORY_FRACTION" in os.environ:
        config.memory_fraction = float(os.environ["CAYLEYPY_NNX_MEMORY_FRACTION"])

    if "CAYLEYPY_NNX_MAX_MEMORY_GB" in os.environ:
        config.max_memory_gb = float(os.environ["CAYLEYPY_NNX_MAX_MEMORY_GB"])

    # Processing configuration
    if "CAYLEYPY_NNX_CHUNK_SIZE" in os.environ:
        config.chunk_size = int(os.environ["CAYLEYPY_NNX_CHUNK_SIZE"])

    # Precision configuration
    if "CAYLEYPY_NNX_ENABLE_X64" in os.environ:
        config.enable_x64 = os.environ["CAYLEYPY_NNX_ENABLE_X64"].lower() == "true"

    return config


def auto_configure_nnx_backend() -> Optional[NNXBackend]:
    """Automatically configure NNX backend from environment and defaults.

    This function tries to create the best possible NNX backend configuration
    by checking environment variables and system capabilities.

    Returns:
        Configured NNXBackend or None if not available
    """
    if not JAX_AVAILABLE:
        return None

    try:
        # Start with environment configuration
        config = configure_nnx_from_environment()

        # Auto-tune configuration based on system capabilities
        if config.preferred_device == "auto":
            # Detect best device and adjust settings accordingly
            try:
                if jax.devices("tpu"):
                    config.preferred_device = "tpu"
                    config.chunk_size = min(config.chunk_size, 5000)  # TPUs prefer smaller chunks
                    config.memory_fraction = 0.9  # TPUs can use more memory
                elif jax.devices("gpu"):
                    config.preferred_device = "gpu"
                    # Keep default settings for GPU
                else:
                    config.preferred_device = "cpu"
                    config.enable_jit = True  # CPU benefits more from JIT
                    config.chunk_size = max(config.chunk_size, 20000)  # CPU can handle larger chunks
            except Exception:  # pylint: disable=broad-exception-caught
                config.preferred_device = "cpu"

        # Create backend with auto-tuned configuration
        backend = NNXBackend(config)

        # Log configuration for debugging
        logger = logging.getLogger(__name__)
        logger.info(
            "Auto-configured NNX backend: %s device, JIT=%s, chunk_size=%d",
            config.preferred_device,
            "enabled" if config.enable_jit else "disabled",
            config.chunk_size,
        )

        return backend

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger = logging.getLogger(__name__)
        logger.warning("Failed to auto-configure NNX backend: %s", e)
        return None


def get_recommended_jax_installation() -> str:
    """Get recommended JAX installation command based on available hardware.

    Returns:
        String with recommended pip install command
    """
    if not JAX_AVAILABLE:
        # Try to detect hardware without JAX
        try:

            # Check for NVIDIA GPU
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5, check=False)
                if result.returncode == 0:
                    return "pip install 'cayleypy[jax-gpu]'  # GPU detected"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Check for TPU (Google Cloud environment)
            if os.path.exists("/dev/accel0") or "TPU_NAME" in os.environ:
                return "pip install 'cayleypy[jax-tpu]'  # TPU detected"

            # Default to CPU
            return "pip install 'cayleypy[jax]'  # CPU (default)"

        except Exception:  # pylint: disable=broad-exception-caught
            return "pip install 'cayleypy[jax]'  # CPU (default)"

    else:
        # JAX is available, check what devices we have
        try:
            if jax.devices("tpu"):
                return "pip install 'cayleypy[jax-tpu]'  # TPU available"
            elif jax.devices("gpu"):
                return "pip install 'cayleypy[jax-gpu]'  # GPU available"
            else:
                return "pip install 'cayleypy[jax]'  # CPU available"
        except Exception:  # pylint: disable=broad-exception-caught
            return "pip install 'cayleypy[jax]'  # CPU (fallback)"


def print_installation_guide():
    """Print installation guide for JAX acceleration."""
    print("CayleyPy JAX Acceleration Installation Guide")
    print("=" * 50)
    print()
    print("Choose the appropriate installation based on your hardware:")
    print()
    print("CPU only:")
    print("  pip install 'cayleypy[jax]'")
    print()
    print("NVIDIA GPU (CUDA 12.x):")
    print("  pip install 'cayleypy[jax-gpu]'")
    print()
    print("Google Cloud TPU:")
    print("  pip install 'cayleypy[jax-tpu]'")
    print()
    print("Recommended for your system:")
    print(f"  {get_recommended_jax_installation()}")
    print()
    print("Note: You can also install multiple variants if needed.")
    print("The backend will automatically detect and use the best available device.")


if __name__ == "__main__":
    # If run as script, print installation guide
    print_installation_guide()
