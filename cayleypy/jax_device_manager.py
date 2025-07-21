"""JAX device management system with automatic TPU/GPU/CPU detection."""

import logging
import os
import warnings
from typing import Dict, List, Optional, Union, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax.numpy as jnp
    JaxArray = jnp.ndarray
else:
    JaxArray = Any

try:
    import jax
    import jax.numpy as jnp
    from jax import devices as jax_devices
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


logger = logging.getLogger(__name__)


class DeviceNotFoundError(Exception):
    """Raised when a requested device is not available."""
    pass


class OutOfMemoryError(Exception):
    """Raised when device runs out of memory."""
    pass


class JAXDeviceManager:
    """Manages JAX device selection and memory allocation with automatic fallback.
    
    Provides automatic device detection with preference order: TPU > GPU > CPU.
    Handles device placement, memory management, and graceful fallback between devices.
    """

    def __init__(self, device: str = "auto", enable_memory_preallocation: bool = True):
        """Initialize device manager.
        
        Args:
            device: Device preference ("auto", "tpu", "gpu", "cpu", or specific device)
            enable_memory_preallocation: Whether to enable JAX memory preallocation
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not available. Install with: pip install jax[tpu] or pip install jax[cuda]"
            )
        
        self.enable_memory_preallocation = enable_memory_preallocation
        self._configure_jax_memory()
        
        self.device_type = self._select_device(device)
        self.devices = self._get_devices(self.device_type)
        self.primary_device = self.devices[0]
        
        logger.info(f"JAX device manager initialized with {self.device_type} backend")
        logger.info(f"Available devices: {[str(d) for d in self.devices]}")
        logger.info(f"Primary device: {self.primary_device}")

    def _configure_jax_memory(self) -> None:
        """Configure JAX memory allocation settings."""
        if not self.enable_memory_preallocation:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        
        # Enable memory growth for GPU
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
        
        # Configure TPU settings
        if "TPU_NAME" in os.environ:
            os.environ.setdefault("JAX_PLATFORM_NAME", "tpu")

    def _select_device(self, device: str) -> str:
        """Select optimal device based on availability and preference.
        
        Args:
            device: Device preference string
            
        Returns:
            Selected device type string
            
        Raises:
            DeviceNotFoundError: If requested device is not available
        """
        if device == "auto":
            return self._auto_select_device()
        
        available_platforms = jax.local_devices()
        platform_types = {d.platform for d in available_platforms}
        
        if device in platform_types:
            return device
        
        # Handle specific device requests
        if device.startswith(("tpu:", "gpu:", "cpu:")):
            platform = device.split(":")[0]
            if platform in platform_types:
                return platform
        
        raise DeviceNotFoundError(f"Requested device '{device}' not available. Available: {platform_types}")

    def _auto_select_device(self) -> str:
        """Automatically select best available device with TPU > GPU > CPU preference."""
        available_platforms = jax.local_devices()
        platform_types = {d.platform for d in available_platforms}
        
        # Preference order: TPU > GPU > CPU
        for preferred in ["tpu", "gpu", "cpu"]:
            if preferred in platform_types:
                logger.info(f"Auto-selected {preferred} device")
                return preferred
        
        raise DeviceNotFoundError("No compatible devices found")

    def _get_devices(self, device_type: str) -> List:
        """Get list of devices for specified type.
        
        Args:
            device_type: Type of device to get
            
        Returns:
            List of JAX devices
        """
        all_devices = jax_devices()
        devices = [d for d in all_devices if d.platform == device_type]
        
        if not devices:
            raise DeviceNotFoundError(f"No {device_type} devices found")
        
        return devices

    def put_on_device(self, array: Union[JaxArray, list, tuple], device: Optional = None) -> JaxArray:
        """Place array on specified device with automatic sharding for large arrays.
        
        Args:
            array: Array to place on device
            device: Target device (uses primary if None)
            
        Returns:
            Array placed on device
        """
        target_device = device or self.primary_device
        
        # Convert to JAX array if needed
        if not isinstance(array, jnp.ndarray):
            array = jnp.array(array)
        
        # Check if array is too large for single device and needs sharding
        if self._should_shard_array(array):
            return self._shard_array(array)
        
        return jax.device_put(array, target_device)

    def _should_shard_array(self, array: JaxArray) -> bool:
        """Determine if array should be sharded across multiple devices.
        
        Args:
            array: Array to check
            
        Returns:
            True if array should be sharded
        """
        # Shard if array is larger than 1GB and we have multiple devices
        array_size_bytes = array.nbytes if hasattr(array, 'nbytes') else array.size * 8
        return array_size_bytes > 1e9 and len(self.devices) > 1

    def _shard_array(self, array: JaxArray) -> JaxArray:
        """Shard large array across available devices.
        
        Args:
            array: Array to shard
            
        Returns:
            Sharded array
        """
        num_devices = len(self.devices)
        
        # Shard along first axis
        if array.shape[0] >= num_devices:
            return jax.device_put_sharded(
                [array[i::num_devices] for i in range(num_devices)], 
                self.devices
            )
        else:
            # If first dimension is smaller than device count, replicate
            return jax.device_put_replicated(array, self.devices)

    def get_memory_info(self) -> Dict[str, Dict[str, float]]:
        """Get memory information for all devices.
        
        Returns:
            Dictionary with memory info per device
        """
        memory_info = {}
        
        for i, device in enumerate(self.devices):
            try:
                # JAX doesn't have direct memory query, so we estimate
                device_info = {
                    "device_id": i,
                    "platform": device.platform,
                    "device_kind": getattr(device, 'device_kind', 'unknown'),
                }
                
                # Try to get memory info if available
                if hasattr(device, 'memory_stats') and callable(getattr(device, 'memory_stats', None)):
                    try:
                        stats = device.memory_stats()
                        device_info.update({
                            "total_memory_gb": stats.get('bytes_limit', 0) / 1e9,
                            "used_memory_gb": stats.get('bytes_in_use', 0) / 1e9,
                            "free_memory_gb": (stats.get('bytes_limit', 0) - stats.get('bytes_in_use', 0)) / 1e9,
                        })
                    except Exception:
                        # Fall back to estimates if memory_stats fails
                        device_info.update(self._get_fallback_memory_estimates(device.platform))
                else:
                    # Fallback estimates based on device type
                    device_info.update(self._get_fallback_memory_estimates(device.platform))
                
                memory_info[str(device)] = device_info
                
            except Exception as e:
                logger.warning(f"Could not get memory info for device {device}: {e}")
                memory_info[str(device)] = {
                    "device_id": i,
                    "platform": device.platform,
                    "error": str(e)
                }
        
        return memory_info

    def _get_fallback_memory_estimates(self, platform: str) -> Dict[str, float]:
        """Get fallback memory estimates based on device platform.
        
        Args:
            platform: Device platform string
            
        Returns:
            Dictionary with memory estimates
        """
        if platform == 'tpu':
            return {
                "total_memory_gb": 16.0,  # Typical TPU v3 memory
                "used_memory_gb": 0.0,
                "free_memory_gb": 16.0,
            }
        elif platform == 'gpu':
            return {
                "total_memory_gb": 8.0,  # Conservative GPU estimate
                "used_memory_gb": 0.0,
                "free_memory_gb": 8.0,
            }
        else:  # CPU
            return {
                "total_memory_gb": 32.0,  # Conservative CPU estimate
                "used_memory_gb": 0.0,
                "free_memory_gb": 32.0,
            }

    def clear_cache(self) -> None:
        """Clear JAX compilation cache and device memory."""
        try:
            # Clear compilation cache
            jax.clear_caches()
            logger.info("JAX compilation cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear JAX cache: {e}")

    def get_device_count(self) -> int:
        """Get number of available devices of current type."""
        return len(self.devices)

    def is_tpu(self) -> bool:
        """Check if current primary device is TPU."""
        return self.primary_device.platform == 'tpu'

    def is_gpu(self) -> bool:
        """Check if current primary device is GPU."""
        return self.primary_device.platform == 'gpu'

    def is_cpu(self) -> bool:
        """Check if current primary device is CPU."""
        return self.primary_device.platform == 'cpu'

    def __str__(self) -> str:
        """String representation of device manager."""
        return f"JAXDeviceManager(device_type={self.device_type}, devices={len(self.devices)})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"JAXDeviceManager(device_type='{self.device_type}', "
                f"primary_device={self.primary_device}, "
                f"num_devices={len(self.devices)})")


class DeviceFallbackHandler:
    """Handles graceful fallback between devices when operations fail."""

    def __init__(self, preferred_devices: Optional[List[str]] = None):
        """Initialize fallback handler.
        
        Args:
            preferred_devices: List of devices in preference order
        """
        self.device_hierarchy = preferred_devices or ["tpu", "gpu", "cpu"]
        self.device_managers = {}

    def execute_with_fallback(self, func, *args, **kwargs):
        """Execute function with automatic device fallback.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RuntimeError: If all devices fail
        """
        last_error = None
        
        for device_type in self.device_hierarchy:
            try:
                # Get or create device manager for this device type
                if device_type not in self.device_managers:
                    try:
                        self.device_managers[device_type] = JAXDeviceManager(device_type)
                    except DeviceNotFoundError:
                        continue
                
                device_manager = self.device_managers[device_type]
                
                # Execute function with this device manager
                return func(device_manager, *args, **kwargs)
                
            except (OutOfMemoryError, DeviceNotFoundError, RuntimeError) as e:
                last_error = e
                logger.warning(f"Device {device_type} failed: {e}. Trying next device...")
                continue
        
        raise RuntimeError(f"All devices failed. Last error: {last_error}")

    def get_best_device_manager(self) -> JAXDeviceManager:
        """Get the best available device manager.
        
        Returns:
            Best available JAXDeviceManager
        """
        for device_type in self.device_hierarchy:
            try:
                return JAXDeviceManager(device_type)
            except DeviceNotFoundError:
                continue
        
        raise RuntimeError("No compatible devices found")