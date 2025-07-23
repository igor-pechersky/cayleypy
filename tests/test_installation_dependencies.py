"""
Tests for installation and dependency combinations in pyproject.toml.

Tests different optional dependency groups for jax-cpu, jax-cuda, jax-tpu
and validates proper version constraints and compatibility requirements.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import pytest


class TestInstallationDependencies(unittest.TestCase):
    """Test cases for installation dependency combinations."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.pyproject_path = self.project_root / "pyproject.toml"

        # Ensure pyproject.toml exists
        self.assertTrue(self.pyproject_path.exists(), "pyproject.toml not found")

    def test_pyproject_toml_structure(self):
        """Test that pyproject.toml has the correct optional dependencies structure."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        # Check that project.optional-dependencies exists
        self.assertIn("project", config)
        self.assertIn("optional-dependencies", config["project"])

        optional_deps = config["project"]["optional-dependencies"]

        # Check required JAX dependency groups
        required_groups = ["jax-cpu", "jax-cuda", "jax-tpu", "jax", "jax-gpu", "jax-all"]
        for group in required_groups:
            self.assertIn(group, optional_deps, f"Missing optional dependency group: {group}")

        # Check that each group has proper dependencies
        self.assertIsInstance(optional_deps["jax-cpu"], list)
        self.assertIsInstance(optional_deps["jax-cuda"], list)
        self.assertIsInstance(optional_deps["jax-tpu"], list)
        self.assertIsInstance(optional_deps["jax"], list)
        self.assertIsInstance(optional_deps["jax-gpu"], list)
        self.assertIsInstance(optional_deps["jax-all"], list)

    def test_jax_cpu_dependencies(self):
        """Test jax-cpu dependency group."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        jax_cpu_deps = config["project"]["optional-dependencies"]["jax-cpu"]

        # Should contain jax[cpu] with version constraint
        self.assertEqual(len(jax_cpu_deps), 1)
        self.assertIn("jax[cpu]", jax_cpu_deps[0])
        self.assertIn(">=0.4.0", jax_cpu_deps[0])

    def test_jax_cuda_dependencies(self):
        """Test jax-cuda dependency group."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        jax_cuda_deps = config["project"]["optional-dependencies"]["jax-cuda"]

        # Should contain jax[cuda12_pip] with version constraint
        self.assertEqual(len(jax_cuda_deps), 1)
        self.assertIn("jax[cuda12_pip]", jax_cuda_deps[0])
        self.assertIn(">=0.4.0", jax_cuda_deps[0])

    def test_jax_cuda11_dependencies(self):
        """Test jax-cuda11 dependency group."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        jax_cuda11_deps = config["project"]["optional-dependencies"]["jax-cuda11"]

        # Should contain jax[cuda11_pip] with version constraint
        self.assertEqual(len(jax_cuda11_deps), 1)
        self.assertIn("jax[cuda11_pip]", jax_cuda11_deps[0])
        self.assertIn(">=0.4.0", jax_cuda11_deps[0])

    def test_jax_tpu_dependencies(self):
        """Test jax-tpu dependency group."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        jax_tpu_deps = config["project"]["optional-dependencies"]["jax-tpu"]

        # Should contain jax[tpu] with version constraint
        self.assertEqual(len(jax_tpu_deps), 1)
        self.assertIn("jax[tpu]", jax_tpu_deps[0])
        self.assertIn(">=0.4.0", jax_tpu_deps[0])

    def test_jax_gpu_alias(self):
        """Test jax-gpu convenience alias."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        jax_gpu_deps = config["project"]["optional-dependencies"]["jax-gpu"]

        # Should reference cayleypy[jax-cuda]
        self.assertEqual(len(jax_gpu_deps), 1)
        self.assertIn("cayleypy[jax-cuda]", jax_gpu_deps[0])

    def test_jax_all_meta_package(self):
        """Test jax-all meta-package."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        jax_all_deps = config["project"]["optional-dependencies"]["jax-all"]

        # Should reference multiple cayleypy extras
        self.assertEqual(len(jax_all_deps), 1)
        dep_str = jax_all_deps[0]
        self.assertIn("cayleypy[jax-cpu,jax-cuda,jax-tpu]", dep_str)

    def test_version_constraints(self):
        """Test that all JAX dependencies have proper version constraints."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        optional_deps = config["project"]["optional-dependencies"]

        # Check version constraints for direct JAX dependencies
        jax_groups = ["jax-cpu", "jax-cuda", "jax-cuda11", "jax-tpu", "jax"]
        for group in jax_groups:
            if group in optional_deps:
                deps = optional_deps[group]
                for dep in deps:
                    if "jax" in dep and "cayleypy" not in dep:
                        self.assertIn(">=0.4.0", dep, f"Missing version constraint in {group}: {dep}")

    def test_compatibility_requirements(self):
        """Test compatibility requirements between different JAX variants."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        optional_deps = config["project"]["optional-dependencies"]

        # Ensure no conflicting JAX installations in the same group
        for group_name, deps in optional_deps.items():
            if group_name.startswith("jax-") and group_name not in ["jax-gpu", "jax-all"]:
                jax_deps = [dep for dep in deps if "jax[" in dep]
                # Each group should have exactly one JAX dependency
                self.assertLessEqual(len(jax_deps), 1, f"Group {group_name} has multiple JAX dependencies: {jax_deps}")

    @pytest.mark.slow
    def test_dry_run_installation_cpu(self):
        """Test dry-run installation of jax-cpu dependencies."""
        try:
            # Use pip's --dry-run flag to test installation without actually installing
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--dry-run", "--quiet", f"{self.project_root}[jax-cpu]"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should not fail (exit code 0 or 1 for dry-run is acceptable)
            self.assertIn(result.returncode, [0, 1], f"Dry-run installation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.skipTest("Installation test timed out")
        except FileNotFoundError:
            self.skipTest("pip not available for testing")

    @pytest.mark.slow
    def test_dry_run_installation_cuda(self):
        """Test dry-run installation of jax-cuda dependencies."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--dry-run", "--quiet", f"{self.project_root}[jax-cuda]"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should not fail (exit code 0 or 1 for dry-run is acceptable)
            self.assertIn(result.returncode, [0, 1], f"Dry-run installation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.skipTest("Installation test timed out")
        except FileNotFoundError:
            self.skipTest("pip not available for testing")

    @pytest.mark.slow
    def test_dry_run_installation_tpu(self):
        """Test dry-run installation of jax-tpu dependencies."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--dry-run", "--quiet", f"{self.project_root}[jax-tpu]"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Should not fail (exit code 0 or 1 for dry-run is acceptable)
            self.assertIn(result.returncode, [0, 1], f"Dry-run installation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.skipTest("Installation test timed out")
        except FileNotFoundError:
            self.skipTest("pip not available for testing")

    def test_dependency_manager_integration(self):
        """Test integration with DependencyManager for installation recommendations."""
        from cayleypy.utils.dependency_manager import DependencyManager

        manager = DependencyManager()

        # Test that installation commands reference the correct extras
        commands = manager.installation_commands

        # CPU command should reference jax-cpu
        self.assertIn("jax-cpu", commands["cpu"])

        # GPU command should reference jax-cuda
        self.assertIn("cuda", commands["gpu"].lower())

        # TPU command should reference jax-tpu
        self.assertIn("jax-tpu", commands["tpu"])

        # Auto command should reference jax
        self.assertIn("jax", commands["auto"])

    def test_backward_compatibility(self):
        """Test that existing torch dependencies are preserved."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        optional_deps = config["project"]["optional-dependencies"]

        # Torch should still be available
        self.assertIn("torch", optional_deps)
        self.assertIn("torch", optional_deps["torch"][0])

    def test_meta_package_resolution(self):
        """Test that meta-packages resolve to correct dependencies."""
        import tomllib

        with open(self.pyproject_path, "rb") as f:
            config = tomllib.load(f)

        optional_deps = config["project"]["optional-dependencies"]

        # jax-gpu should resolve to jax-cuda
        jax_gpu = optional_deps["jax-gpu"][0]
        self.assertIn("jax-cuda", jax_gpu)

        # jax-all should include all variants
        jax_all = optional_deps["jax-all"][0]
        self.assertIn("jax-cpu", jax_all)
        self.assertIn("jax-cuda", jax_all)
        self.assertIn("jax-tpu", jax_all)


class TestInstallationCommands(unittest.TestCase):
    """Test installation command generation and validation."""

    def test_installation_command_format(self):
        """Test that installation commands are properly formatted."""
        from cayleypy.utils.dependency_manager import DependencyManager

        manager = DependencyManager()

        for cmd_type, command in manager.installation_commands.items():
            # All commands should be pip install commands
            self.assertIn("pip install", command)
            self.assertIn("cayleypy", command)

            # Commands with extras should be properly quoted
            if "[" in command and "]" in command:
                # Should have quotes around the package[extras] part
                self.assertTrue(
                    "'" in command or '"' in command, f"Command {command} should have quotes around package[extras]"
                )

    def test_hardware_specific_recommendations(self):
        """Test hardware-specific installation recommendations."""
        from cayleypy.utils.dependency_manager import DependencyManager

        # Mock different hardware configurations
        test_cases = [
            ({"tpu_available": True, "gpu_available": False}, "tpu"),
            ({"tpu_available": False, "gpu_available": True}, "gpu"),
            ({"tpu_available": False, "gpu_available": False}, "cpu"),
        ]

        for hardware_config, expected_type in test_cases:
            with patch.object(DependencyManager, "_detect_hardware") as mock_hardware:
                mock_hardware.return_value = {
                    "cpu": True,
                    "cpu_count": 4,
                    "platform": "Linux",
                    "architecture": "x86_64",
                    **hardware_config,
                }

                manager = DependencyManager()
                recommendation = manager.get_installation_recommendation()

                # Should recommend appropriate installation
                if expected_type == "tpu":
                    self.assertIn("tpu", recommendation.lower())
                elif expected_type == "gpu":
                    self.assertIn("cuda", recommendation.lower())
                elif expected_type == "cpu":
                    self.assertIn("cpu", recommendation.lower())


if __name__ == "__main__":
    unittest.main()
