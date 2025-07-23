"""
Unit tests for the DependencyManager class.

Tests dependency detection, hardware detection, and installation recommendations
across different environments.
"""

import os
import platform
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import pytest

from cayleypy.utils.dependency_manager import DependencyManager, get_dependency_manager, check_environment


class TestDependencyManager(unittest.TestCase):
    """Test cases for DependencyManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear singleton instance for clean tests
        if hasattr(get_dependency_manager, "_instance"):
            delattr(get_dependency_manager, "_instance")

    def test_initialization(self):
        """Test DependencyManager initialization."""
        manager = DependencyManager()

        # Check that all required attributes are initialized
        self.assertIsInstance(manager.hardware_info, dict)
        self.assertIsInstance(manager.available_backends, dict)
        self.assertIsInstance(manager.recommended_backend, str)
        self.assertIsInstance(manager.installation_commands, dict)

        # Check required hardware info keys
        required_keys = ["cpu", "cpu_count", "platform", "architecture", "gpu_available", "tpu_available"]
        for key in required_keys:
            self.assertIn(key, manager.hardware_info)

    def test_hardware_detection_basic(self):
        """Test basic hardware detection."""
        manager = DependencyManager()

        # CPU should always be available
        self.assertTrue(manager.hardware_info["cpu"])
        self.assertGreater(manager.hardware_info["cpu_count"], 0)
        self.assertIn(manager.hardware_info["platform"], ["Linux", "Darwin", "Windows"])
        self.assertIsInstance(manager.hardware_info["architecture"], str)

    @patch("subprocess.run")
    def test_cuda_detection_success(self, mock_run):
        """Test successful CUDA detection."""
        # Mock nvidia-smi success
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 3080, 1\nNVIDIA GeForce RTX 3090, 1"
        mock_run.return_value = mock_result

        manager = DependencyManager()

        self.assertTrue(manager.hardware_info["gpu_available"])
        self.assertEqual(manager.hardware_info["gpu_count"], 2)
        self.assertEqual(manager.hardware_info["gpu_type"], "NVIDIA GeForce RTX 3080")

    @patch("subprocess.run")
    def test_cuda_detection_failure(self, mock_run):
        """Test CUDA detection when nvidia-smi fails."""
        # Mock nvidia-smi failure
        mock_run.side_effect = FileNotFoundError()

        manager = DependencyManager()

        self.assertFalse(manager.hardware_info["gpu_available"])
        self.assertEqual(manager.hardware_info["gpu_count"], 0)
        self.assertIsNone(manager.hardware_info["gpu_type"])

    @patch.dict(os.environ, {"CUDA_HOME": "/usr/local/cuda"})
    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_cuda_detection_fallback(self, mock_run, mock_exists):
        """Test CUDA detection fallback to environment variables."""
        # Mock nvidia-smi failure but CUDA_HOME exists
        mock_run.side_effect = FileNotFoundError()
        mock_exists.return_value = True

        manager = DependencyManager()

        self.assertTrue(manager.hardware_info["cuda_available"])
        self.assertTrue(manager.hardware_info["gpu_available"])

    @patch.dict(os.environ, {"TPU_NAME": "test-tpu-v3-8"})
    def test_tpu_detection_env_var(self):
        """Test TPU detection via environment variable."""
        manager = DependencyManager()

        self.assertTrue(manager.hardware_info["tpu_available"])
        self.assertEqual(manager.hardware_info["tpu_type"], "test-tpu-v3-8")

    @patch("requests.get")
    def test_tpu_detection_colab(self, mock_get):
        """Test TPU detection in Colab environment."""
        # Mock Colab metadata response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "tpu-v2"
        mock_get.return_value = mock_response

        manager = DependencyManager()

        self.assertTrue(manager.hardware_info["tpu_available"])
        self.assertEqual(manager.hardware_info["tpu_type"], "tpu-v2")

    @patch("requests.get")
    def test_tpu_detection_no_colab(self, mock_get):
        """Test TPU detection when not in Colab."""
        # Mock failed metadata request
        mock_get.side_effect = Exception("Connection failed")

        manager = DependencyManager()

        # Should not crash and should handle gracefully
        self.assertIsInstance(manager.hardware_info["tpu_available"], bool)

    def test_backend_detection_no_imports(self):
        """Test backend detection when no packages are available."""
        with (
            patch("cayleypy.utils.dependency_manager.DependencyManager._check_pytorch", return_value=False),
            patch("cayleypy.utils.dependency_manager.DependencyManager._check_jax_cpu", return_value=False),
            patch("cayleypy.utils.dependency_manager.DependencyManager._check_jax_gpu", return_value=False),
            patch("cayleypy.utils.dependency_manager.DependencyManager._check_jax_tpu", return_value=False),
        ):
            manager = DependencyManager()

            self.assertFalse(manager.available_backends["pytorch"])
            self.assertFalse(manager.available_backends["jax-cpu"])
            self.assertFalse(manager.available_backends["jax-gpu"])
            self.assertFalse(manager.available_backends["jax-tpu"])

    @patch("cayleypy.utils.dependency_manager.DependencyManager._check_pytorch")
    @patch("cayleypy.utils.dependency_manager.DependencyManager._check_jax_cpu")
    def test_backend_detection_pytorch_only(self, mock_jax, mock_pytorch):
        """Test backend detection with only PyTorch available."""
        mock_pytorch.return_value = True
        mock_jax.return_value = False

        manager = DependencyManager()

        self.assertTrue(manager.available_backends["pytorch"])
        self.assertFalse(manager.available_backends["jax-cpu"])

    @patch("cayleypy.utils.dependency_manager.DependencyManager._check_jax_tpu")
    @patch("cayleypy.utils.dependency_manager.DependencyManager._check_jax_gpu")
    @patch("cayleypy.utils.dependency_manager.DependencyManager._check_jax_cpu")
    def test_optimal_backend_selection(self, mock_cpu, mock_gpu, mock_tpu):
        """Test optimal backend selection priority."""
        # Test TPU priority
        mock_tpu.return_value = True
        mock_gpu.return_value = True
        mock_cpu.return_value = True

        manager = DependencyManager()
        self.assertEqual(manager.recommended_backend, "jax-tpu")

        # Test GPU priority when no TPU
        mock_tpu.return_value = False
        manager = DependencyManager()
        self.assertEqual(manager.recommended_backend, "jax-gpu")

        # Test CPU fallback
        mock_gpu.return_value = False
        manager = DependencyManager()
        self.assertEqual(manager.recommended_backend, "jax-cpu")

    def test_installation_commands_generation(self):
        """Test generation of installation commands."""
        manager = DependencyManager()

        # Check that all expected commands are generated
        expected_commands = ["cpu", "gpu", "tpu", "auto", "all"]
        for cmd in expected_commands:
            self.assertIn(cmd, manager.installation_commands)
            self.assertIn("cayleypy", manager.installation_commands[cmd])
            self.assertIn("pip install", manager.installation_commands[cmd])

    def test_hardware_summary(self):
        """Test hardware summary generation."""
        manager = DependencyManager()
        summary = manager.get_hardware_summary()

        self.assertIsInstance(summary, str)
        self.assertIn("Hardware Detection Summary", summary)
        self.assertIn("Platform:", summary)
        self.assertIn("CPU Cores:", summary)

    def test_backend_summary(self):
        """Test backend summary generation."""
        manager = DependencyManager()
        summary = manager.get_backend_summary()

        self.assertIsInstance(summary, str)
        self.assertIn("Available Backends:", summary)
        self.assertIn("Recommended:", summary)

    def test_installation_recommendation_auto(self):
        """Test automatic installation recommendation."""
        manager = DependencyManager()
        recommendation = manager.get_installation_recommendation()

        self.assertIsInstance(recommendation, str)
        self.assertIn("pip install", recommendation)
        self.assertIn("cayleypy", recommendation)

    def test_installation_recommendation_specific(self):
        """Test specific hardware installation recommendations."""
        manager = DependencyManager()

        # Test CPU recommendation
        cpu_rec = manager.get_installation_recommendation("cpu")
        self.assertIn("jax-cpu", cpu_rec)
        self.assertIn("cayleypy", cpu_rec)

        # Test GPU recommendation (contains 'cuda' not 'gpu')
        gpu_rec = manager.get_installation_recommendation("gpu")
        self.assertIn("cuda", gpu_rec.lower())
        self.assertIn("cayleypy", gpu_rec)

        # Test TPU recommendation
        tpu_rec = manager.get_installation_recommendation("tpu")
        self.assertIn("tpu", tpu_rec.lower())
        self.assertIn("cayleypy", tpu_rec)

    def test_environment_validation_no_backends(self):
        """Test environment validation when no backends are available."""
        with patch.object(DependencyManager, "_detect_available_backends") as mock_backends:
            mock_backends.return_value = {
                key: False for key in ["pytorch", "jax-cpu", "jax-gpu", "jax-tpu", "jax", "gpu"]
            }

            manager = DependencyManager()
            is_valid, issues = manager.validate_environment()

            self.assertFalse(is_valid)
            self.assertGreater(len(issues), 0)
            self.assertTrue(any("No compatible backend" in issue for issue in issues))

    def test_environment_validation_success(self):
        """Test environment validation when backends are available."""
        with patch.object(DependencyManager, "_detect_available_backends") as mock_backends:
            mock_backends.return_value = {
                "pytorch": False,
                "jax-cpu": True,
                "jax-gpu": False,
                "jax-tpu": False,
                "jax": True,
                "gpu": False,
            }

            manager = DependencyManager()
            is_valid, issues = manager.validate_environment()

            self.assertTrue(is_valid)
            self.assertEqual(len(issues), 0)

    def test_python_version_validation(self):
        """Test Python version validation."""
        # Create a proper version_info-like object that supports comparison
        mock_version = type(
            "version_info",
            (),
            {
                "major": 3,
                "minor": 8,
                "__lt__": lambda self, other: (self.major, self.minor) < other,
                "__getitem__": lambda self, i: [self.major, self.minor][i],
            },
        )()

        with patch("sys.version_info", mock_version):
            manager = DependencyManager()
            is_valid, issues = manager.validate_environment()

            # Should fail due to Python version
            python_issue = any("Python" in issue and "3.9" in issue for issue in issues)
            self.assertTrue(python_issue)

    def test_optimal_device_config_tpu(self):
        """Test optimal device configuration for TPU."""
        with patch.object(DependencyManager, "_detect_hardware") as mock_hardware:
            mock_hardware.return_value = {
                "tpu_available": True,
                "gpu_available": False,
                "cpu": True,
                "cpu_count": 4,
                "platform": "Linux",
                "architecture": "x86_64",
            }

            manager = DependencyManager()
            config = manager.get_optimal_device_config()

            self.assertEqual(config["device"], "tpu")
            self.assertTrue(config["enable_sharding"])
            self.assertEqual(config["batch_size_multiplier"], 8)

    def test_optimal_device_config_gpu(self):
        """Test optimal device configuration for GPU."""
        with patch.object(DependencyManager, "_detect_hardware") as mock_hardware:
            mock_hardware.return_value = {
                "tpu_available": False,
                "gpu_available": True,
                "gpu_count": 2,
                "cpu": True,
                "cpu_count": 8,
                "platform": "Linux",
                "architecture": "x86_64",
            }

            manager = DependencyManager()
            config = manager.get_optimal_device_config()

            self.assertEqual(config["device"], "gpu")
            self.assertTrue(config["enable_sharding"])  # Multiple GPUs
            self.assertEqual(config["batch_size_multiplier"], 4)

    def test_optimal_device_config_cpu(self):
        """Test optimal device configuration for CPU only."""
        with patch.object(DependencyManager, "_detect_hardware") as mock_hardware:
            mock_hardware.return_value = {
                "tpu_available": False,
                "gpu_available": False,
                "cpu": True,
                "cpu_count": 4,
                "platform": "Linux",
                "architecture": "x86_64",
            }

            manager = DependencyManager()
            config = manager.get_optimal_device_config()

            self.assertEqual(config["device"], "cpu")
            self.assertFalse(config["enable_sharding"])
            self.assertEqual(config["batch_size_multiplier"], 1)
            self.assertFalse(config["enable_jit"])  # JIT disabled for CPU

    def test_singleton_pattern(self):
        """Test that get_dependency_manager returns the same instance."""
        manager1 = get_dependency_manager()
        manager2 = get_dependency_manager()

        self.assertIs(manager1, manager2)

    @patch("builtins.print")
    def test_print_environment_report(self, mock_print):
        """Test environment report printing."""
        manager = DependencyManager()
        manager.print_environment_report()

        # Check that print was called
        self.assertTrue(mock_print.called)

        # Check that report contains expected sections
        printed_text = "".join(str(call[0][0]) if call[0] else "" for call in mock_print.call_args_list)
        self.assertIn("Environment Report", printed_text)
        self.assertIn("Hardware Detection", printed_text)
        self.assertIn("Available Backends", printed_text)

    @patch("builtins.print")
    def test_check_environment_function(self, mock_print):
        """Test the check_environment convenience function."""
        check_environment()

        # Should call print (via print_environment_report)
        self.assertTrue(mock_print.called)


class TestDependencyManagerIntegration(unittest.TestCase):
    """Integration tests for DependencyManager with real environment."""

    def test_real_environment_detection(self):
        """Test dependency manager with real environment (no mocking)."""
        manager = DependencyManager()

        # Basic sanity checks that should always pass
        self.assertTrue(manager.hardware_info["cpu"])
        self.assertGreater(manager.hardware_info["cpu_count"], 0)
        self.assertIsInstance(manager.hardware_info["platform"], str)
        self.assertIsInstance(manager.available_backends, dict)
        self.assertIsInstance(manager.recommended_backend, str)

    def test_installation_commands_valid(self):
        """Test that generated installation commands are valid."""
        manager = DependencyManager()

        for cmd_type, command in manager.installation_commands.items():
            # All commands should be pip install commands
            self.assertIn("pip install", command)
            self.assertIn("cayleypy", command)

            # Commands should be properly quoted
            if "[" in command:
                self.assertIn("'", command)  # Should have quotes around package[extras]

    def test_environment_validation_real(self):
        """Test environment validation with real environment."""
        manager = DependencyManager()
        is_valid, issues = manager.validate_environment()

        # Should not crash
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(issues, list)

        # If invalid, should have meaningful issues
        if not is_valid:
            self.assertGreater(len(issues), 0)
            for issue in issues:
                self.assertIsInstance(issue, str)
                self.assertGreater(len(issue), 0)


if __name__ == "__main__":
    unittest.main()
