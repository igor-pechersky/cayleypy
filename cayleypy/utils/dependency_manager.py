"""
Intelligent dependency detection and management system for CayleyPy.

This module provides runtime backend detection, hardware detection for CPU, GPU (CUDA),
and TPU environments, and installation recommendation system based on detected hardware.
"""

import os
import platform
import subprocess
import sys
import warnings
from typing import Dict, List, Optional, Tuple, Union


class DependencyManager:
    """
    Manages dependencies and provides intelligent backend selection for CayleyPy.

    Detects available hardware (CPU, GPU, TPU) and provides installation recommendations
    for optimal JAX configurations based on the detected environment.
    """

    def __init__(self):
        """Initialize the dependency manager with hardware and backend detection."""
        self.hardware_info = self._detect_hardware()
        self.available_backends = self._detect_available_backends()
        self.recommended_backend = self._select_optimal_backend()
        self.installation_commands = self._generate_installation_commands()

    def _detect_hardware(self) -> Dict[str, Union[bool, str, int]]:
        """
        Detect available hardware capabilities.

        Returns:
            Dictionary containing hardware information including CPU, GPU, and TPU availability.
        """
        hardware = {
            "cpu": True,  # CPU always available
            "cpu_count": os.cpu_count() or 1,
            "platform": platform.system(),
            "architecture": platform.machine(),
            "gpu_available": False,
            "gpu_type": None,
            "gpu_count": 0,
            "cuda_available": False,
            "cuda_version": None,
            "tpu_available": False,
            "tpu_type": None,
        }

        # Detect CUDA/GPU
        hardware.update(self._detect_cuda())

        # Detect TPU
        hardware.update(self._detect_tpu())

        return hardware

    def _detect_cuda(self) -> Dict[str, Union[bool, str, int]]:
        """
        Detect CUDA availability and version.

        Returns:
            Dictionary with CUDA detection results.
        """
        cuda_info = {
            "gpu_available": False,
            "gpu_type": None,
            "gpu_count": 0,
            "cuda_available": False,
            "cuda_version": None,
        }

        try:
            # Try nvidia-smi command
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,count", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines and lines[0]:
                    cuda_info["gpu_available"] = True
                    cuda_info["gpu_count"] = len(lines)
                    cuda_info["gpu_type"] = lines[0].split(",")[0].strip()

                    # Try to get CUDA version
                    cuda_version_result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    if cuda_version_result.returncode == 0:
                        cuda_info["cuda_available"] = True
                        cuda_info["cuda_version"] = cuda_version_result.stdout.strip().split("\n")[0]

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # nvidia-smi not available or failed
            pass

        # Fallback: check for CUDA environment variables
        if not cuda_info["cuda_available"]:
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            if cuda_home and os.path.exists(cuda_home):
                cuda_info["cuda_available"] = True
                cuda_info["gpu_available"] = True

        return cuda_info

    def _detect_tpu(self) -> Dict[str, Union[bool, str]]:
        """
        Detect TPU availability.

        Returns:
            Dictionary with TPU detection results.
        """
        tpu_info = {
            "tpu_available": False,
            "tpu_type": None,
        }

        # Check for TPU environment variables (Google Cloud TPU)
        tpu_name = os.environ.get("TPU_NAME")
        if tpu_name:
            tpu_info["tpu_available"] = True
            tpu_info["tpu_type"] = tpu_name
            return tpu_info

        # Check for Colab TPU
        try:
            import requests

            response = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type",
                headers={"Metadata-Flavor": "Google"},
                timeout=1,
            )
            if response.status_code == 200 and "tpu" in response.text.lower():
                tpu_info["tpu_available"] = True
                tpu_info["tpu_type"] = response.text.strip()
        except:
            pass

        # Check for TPU via JAX (if available)
        try:
            import jax

            tpu_devices = jax.devices("tpu")
            if tpu_devices:
                tpu_info["tpu_available"] = True
                tpu_info["tpu_type"] = f"JAX-detected-{len(tpu_devices)}-cores"
        except:
            pass

        return tpu_info

    def _detect_available_backends(self) -> Dict[str, bool]:
        """
        Detect which backends are currently available.

        Returns:
            Dictionary mapping backend names to availability status.
        """
        backends = {
            "pytorch": self._check_pytorch(),
            "jax-cpu": self._check_jax_cpu(),
            "jax-gpu": self._check_jax_gpu(),
            "jax-tpu": self._check_jax_tpu(),
        }

        # Add convenience aliases
        backends["jax"] = any([backends["jax-cpu"], backends["jax-gpu"], backends["jax-tpu"]])
        backends["gpu"] = backends["jax-gpu"] or self._check_pytorch_gpu()

        return backends

    def _check_pytorch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch

            return True
        except ImportError:
            return False

    def _check_pytorch_gpu(self) -> bool:
        """Check if PyTorch with GPU support is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _check_jax_cpu(self) -> bool:
        """Check if JAX with CPU support is available."""
        try:
            import jax
            import jax.numpy as jnp

            # Test basic functionality
            _ = jnp.array([1, 2, 3])
            return True
        except ImportError:
            return False

    def _check_jax_gpu(self) -> bool:
        """Check if JAX with GPU support is available."""
        try:
            import jax

            gpu_devices = jax.devices("gpu")
            return len(gpu_devices) > 0
        except (ImportError, RuntimeError):
            return False

    def _check_jax_tpu(self) -> bool:
        """Check if JAX with TPU support is available."""
        try:
            import jax

            tpu_devices = jax.devices("tpu")
            return len(tpu_devices) > 0
        except (ImportError, RuntimeError):
            return False

    def _select_optimal_backend(self) -> str:
        """
        Select the optimal backend based on available hardware and software.

        Priority: TPU > GPU > CPU

        Returns:
            Name of the recommended backend.
        """
        if self.available_backends.get("jax-tpu", False):
            return "jax-tpu"
        elif self.available_backends.get("jax-gpu", False):
            return "jax-gpu"
        elif self.available_backends.get("jax-cpu", False):
            return "jax-cpu"
        elif self.available_backends.get("pytorch", False):
            return "pytorch"
        else:
            return "none"

    def _generate_installation_commands(self) -> Dict[str, str]:
        """
        Generate installation commands for different hardware configurations.

        Returns:
            Dictionary mapping hardware types to pip install commands.
        """
        base_package = "cayleypy"

        commands = {
            "cpu": f"pip install '{base_package}[jax-cpu]'",
            "gpu": f"pip install '{base_package}[jax-cuda]'",
            "tpu": f"pip install '{base_package}[jax-tpu]'",
            "auto": f"pip install '{base_package}[jax]'",
            "all": f"pip install '{base_package}[jax-cpu,jax-cuda,jax-tpu]'",
        }

        # Add CUDA version specific commands if detected
        if self.hardware_info.get("cuda_version"):
            cuda_version = self.hardware_info["cuda_version"]
            if "12." in cuda_version:
                commands["gpu-cuda12"] = f"pip install '{base_package}[jax-cuda12]'"
            elif "11." in cuda_version:
                commands["gpu-cuda11"] = f"pip install '{base_package}[jax-cuda11]'"

        return commands

    def get_hardware_summary(self) -> str:
        """
        Get a human-readable summary of detected hardware.

        Returns:
            Formatted string describing the hardware configuration.
        """
        lines = [
            "Hardware Detection Summary:",
            f"  Platform: {self.hardware_info['platform']} ({self.hardware_info['architecture']})",
            f"  CPU Cores: {self.hardware_info['cpu_count']}",
        ]

        if self.hardware_info["gpu_available"]:
            lines.append(f"  GPU: {self.hardware_info['gpu_type']} (Count: {self.hardware_info['gpu_count']})")
            if self.hardware_info["cuda_available"]:
                lines.append(f"  CUDA: {self.hardware_info['cuda_version']}")
        else:
            lines.append("  GPU: Not detected")

        if self.hardware_info["tpu_available"]:
            lines.append(f"  TPU: {self.hardware_info['tpu_type']}")
        else:
            lines.append("  TPU: Not detected")

        return "\n".join(lines)

    def get_backend_summary(self) -> str:
        """
        Get a human-readable summary of available backends.

        Returns:
            Formatted string describing available backends.
        """
        lines = ["Available Backends:"]

        for backend, available in self.available_backends.items():
            status = "✓" if available else "✗"
            lines.append(f"  {status} {backend}")

        lines.append(f"\nRecommended: {self.recommended_backend}")

        return "\n".join(lines)

    def get_installation_recommendation(self, hardware_type: Optional[str] = None) -> str:
        """
        Get installation recommendation based on detected or specified hardware.

        Args:
            hardware_type: Optional hardware type override ('cpu', 'gpu', 'tpu', 'auto').
                          If None, uses auto-detection.

        Returns:
            Installation command string.
        """
        if hardware_type is None:
            # Auto-detect based on hardware
            if self.hardware_info["tpu_available"]:
                hardware_type = "tpu"
            elif self.hardware_info["gpu_available"]:
                hardware_type = "gpu"
            else:
                hardware_type = "cpu"

        return self.installation_commands.get(hardware_type, self.installation_commands["cpu"])

    def validate_environment(self) -> Tuple[bool, List[str]]:
        """
        Validate the current environment for CayleyPy usage.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues = []

        # Check if any backend is available
        if not any(self.available_backends.values()):
            issues.append("No compatible backend found. Install JAX or PyTorch.")
            issues.append(f"Recommended: {self.get_installation_recommendation()}")

        # Check for hardware/software mismatches
        if self.hardware_info["tpu_available"] and not self.available_backends["jax-tpu"]:
            issues.append("TPU detected but JAX TPU support not available.")
            issues.append(f"Install with: {self.installation_commands['tpu']}")

        if self.hardware_info["gpu_available"] and not (
            self.available_backends["jax-gpu"] or self._check_pytorch_gpu()
        ):
            issues.append("GPU detected but no GPU-accelerated backend available.")
            issues.append(f"Install with: {self.installation_commands['gpu']}")

        # Check Python version compatibility
        python_version = sys.version_info
        if python_version < (3, 9):
            issues.append(
                f"Python {python_version.major}.{python_version.minor} detected. "
                "CayleyPy requires Python 3.9 or higher."
            )

        return len(issues) == 0, issues

    def get_optimal_device_config(self) -> Dict[str, Union[str, int, bool]]:
        """
        Get optimal device configuration for the current environment.

        Returns:
            Dictionary with device configuration recommendations.
        """
        config = {
            "backend": self.recommended_backend,
            "device": "auto",
            "enable_jit": True,
            "memory_fraction": 0.8,
        }

        if self.hardware_info["tpu_available"]:
            config.update(
                {
                    "device": "tpu",
                    "enable_sharding": True,
                    "batch_size_multiplier": 8,  # TPUs work well with larger batches
                }
            )
        elif self.hardware_info["gpu_available"]:
            config.update(
                {
                    "device": "gpu",
                    "enable_sharding": self.hardware_info["gpu_count"] > 1,
                    "batch_size_multiplier": 4,
                }
            )
        else:
            config.update(
                {
                    "device": "cpu",
                    "enable_sharding": False,
                    "batch_size_multiplier": 1,
                    "enable_jit": False,  # JIT may be slower on CPU for small problems
                }
            )

        return config

    def print_environment_report(self) -> None:
        """Print a comprehensive environment report."""
        print("=" * 60)
        print("CayleyPy Environment Report")
        print("=" * 60)
        print(self.get_hardware_summary())
        print()
        print(self.get_backend_summary())
        print()

        is_valid, issues = self.validate_environment()
        if is_valid:
            print("✓ Environment validation: PASSED")
            print(f"Recommended installation: {self.get_installation_recommendation()}")
        else:
            print("✗ Environment validation: FAILED")
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")

        print()
        print("Optimal Configuration:")
        config = self.get_optimal_device_config()
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 60)


def get_dependency_manager() -> DependencyManager:
    """
    Get a singleton instance of the DependencyManager.

    Returns:
        DependencyManager instance.
    """
    if not hasattr(get_dependency_manager, "_instance"):
        get_dependency_manager._instance = DependencyManager()
    return get_dependency_manager._instance


def check_environment() -> None:
    """Convenience function to check and print environment status."""
    manager = get_dependency_manager()
    manager.print_environment_report()


if __name__ == "__main__":
    # CLI interface for environment checking
    check_environment()
