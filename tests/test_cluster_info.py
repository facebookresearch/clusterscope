# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from clusterscope.cluster_info import (
    AWSClusterInfo,
    DarwinInfo,
    LinuxInfo,
    LocalNodeInfo,
    ResourceShape,
    run_cli,
    SlurmClusterInfo,
    UnifiedInfo,
)


class TestUnifiedInfo(unittest.TestCase):

    def test_get_cluster_name(self):
        unified_info = UnifiedInfo()
        unified_info.is_slurm_cluster = False
        self.assertIn(unified_info.get_cluster_name(), ["local-node", "github"])

    def test_get_cluster_name_with_partition(self):
        unified_info = UnifiedInfo(partition="gpu_partition")
        unified_info.is_slurm_cluster = False
        self.assertIn(unified_info.get_cluster_name(), ["local-node", "github"])
        self.assertEqual(unified_info.partition, "gpu_partition")

    def test_get_gpu_generation_and_count(self):
        unified_info = UnifiedInfo()
        unified_info.is_slurm_cluster = False
        unified_info.has_nvidia_gpus = False
        self.assertEqual(unified_info.get_gpu_generation_and_count(), {})

    def test_partition_passed_to_slurm_cluster_info(self):
        unified_info = UnifiedInfo(partition="test_partition")
        self.assertEqual(unified_info.slurm_cluster_info.partition, "test_partition")


class TestLinuxInfo(unittest.TestCase):
    def setUp(self):
        self.linux_info = LinuxInfo()

    @patch("clusterscope.cluster_info.run_cli", return_value="1234")
    def test_get_cpu_count(self, mock_run):
        self.assertEqual(self.linux_info.get_cpu_count(), 1234)

    @patch(
        "clusterscope.cluster_info.run_cli",
        return_value="               total        used\nMem:     12345    123\n",
    )
    def test_get_mem_per_node_MB(self, mock_run):
        self.assertEqual(self.linux_info.get_mem_MB(), 12345)


class TestDarwinInfo(unittest.TestCase):
    def setUp(self):
        self.darwin_info = DarwinInfo()

    @patch("clusterscope.cluster_info.run_cli", return_value="10")
    def test_get_cpu_count(self, mock_run):
        self.assertEqual(self.darwin_info.get_cpu_count(), 10)

    @patch(
        "clusterscope.cluster_info.run_cli",
        return_value="34359738368",
    )
    def test_get_mem_per_node_MB(self, mock_run):
        self.assertEqual(self.darwin_info.get_mem_MB(), 32768)


class TestSlurmClusterInfo(unittest.TestCase):
    def setUp(self):
        self.cluster_info = SlurmClusterInfo()
        self.cluster_info_with_partition = SlurmClusterInfo(partition="test_partition")

    @patch("subprocess.run")
    def test_get_cluster_name(self, mock_run):
        # Mock successful cluster name retrieval
        mock_run.return_value = MagicMock(
            stdout="ClusterName=test_cluster\nOther=value", returncode=0
        )
        self.assertEqual(self.cluster_info.get_cluster_name(), "test_cluster")

    @patch("subprocess.run")
    def test_get_cpu_per_node(self, mock_run):
        # Mock successful cluster name retrieval
        mock_run.return_value = MagicMock(stdout="128", returncode=0)
        self.assertEqual(self.cluster_info.get_cpus_per_node(), 128)

    @patch("subprocess.run")
    @patch("clusterscope.cache.load", return_value={})  # Mock empty cache
    @patch("clusterscope.cache.save")  # Mock cache save function
    def test_get_cpu_per_node_with_partition(self, mock_save, mock_load, mock_run):
        # Mock successful CPU per node retrieval with partition
        mock_run.return_value = MagicMock(stdout="128", returncode=0)
        result = self.cluster_info_with_partition.get_cpus_per_node()
        self.assertEqual(result, 128)
        # Verify that partition argument was passed to subprocess.run
        mock_run.assert_called_with(
            ["sinfo", "-o", "%100c", "--noheader", "-p", "test_partition"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_get_mem_per_node_MB(self, mock_run):
        # Mock successful cluster name retrieval
        mock_run.return_value = MagicMock(stdout="123456+", returncode=0)
        self.assertEqual(self.cluster_info.get_mem_per_node_MB(), 123456)

    @patch("subprocess.run")
    @patch("clusterscope.cache.load", return_value={})  # Mock empty cache
    @patch("clusterscope.cache.save")  # Mock cache save function
    def test_get_mem_per_node_MB_with_partition(self, mock_save, mock_load, mock_run):
        # Mock successful memory per node retrieval with partition
        mock_run.return_value = MagicMock(stdout="123456+", returncode=0)
        result = self.cluster_info_with_partition.get_mem_per_node_MB()
        self.assertEqual(result, 123456)
        # Verify that partition argument was passed to subprocess.run
        mock_run.assert_called_with(
            [
                "sinfo",
                "-o",
                "%100m",
                "--noconvert",
                "--noheader",
                "-p",
                "test_partition",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_get_max_job_lifetime(self, mock_run):
        # Mock successful max job lifetime retrieval
        mock_run.return_value = MagicMock(
            stdout="MaxJobTime=1-00:00:00\nOther=value", returncode=0
        )
        self.assertEqual(self.cluster_info.get_max_job_lifetime(), "1-00:00:00")

    @patch("subprocess.run")
    def test_get_max_job_lifetime_error(self, mock_run):
        # Mock failed command
        mock_run.side_effect = subprocess.SubprocessError()
        with self.assertRaises(RuntimeError):
            self.cluster_info.get_max_job_lifetime()
        mock_run.side_effect = FileNotFoundError()
        with self.assertRaises(RuntimeError):
            self.cluster_info.get_max_job_lifetime()

    @patch("subprocess.run")
    def test_get_max_job_lifetime_not_found(self, mock_run):
        # Mock successful command but MaxJobTime not in output
        mock_run.return_value = MagicMock(
            stdout="SomeOtherSetting=value\nAnotherSetting=value", returncode=0
        )
        with self.assertRaises(RuntimeError):
            self.cluster_info.get_max_job_lifetime()

    @patch("subprocess.run")
    def test_get_gpu_generations(self, mock_run):
        # Mock successful GPU generations retrieval using 'sinfo -o %G'
        mock_run.return_value = MagicMock(
            stdout="GRES\ngres:gpu:a100:4\ngres:gpu:v100:2\ngres:gpu:p100:8\nother:resource:1",
            returncode=0,
        )

        # Create an instance of the class
        cluster_info = SlurmClusterInfo()

        # Call the method and check the result
        result = cluster_info.get_gpu_generations()
        expected = {"A100", "V100", "P100"}
        self.assertEqual(result, expected)

    @patch("subprocess.run")
    def test_get_gpu_generations_no_gpus(self, mock_run):
        # Mock output with no GPU information
        mock_run.return_value = MagicMock(
            stdout="GRES\nother:resource:1\n", returncode=0
        )

        # Create an instance of the class
        cluster_info = SlurmClusterInfo()

        # Call the method and check the result
        result = cluster_info.get_gpu_generations()
        self.assertEqual(result, set())  # Should return an empty set

    @patch("subprocess.run")
    def test_get_gpu_generations_with_partition(self, mock_run):
        # Mock successful GPU generations retrieval with partition
        mock_run.return_value = MagicMock(
            stdout="GRES\ngres:gpu:a100:4\ngres:gpu:v100:2\n",
            returncode=0,
        )

        result = self.cluster_info_with_partition.get_gpu_generations()
        expected = {"A100", "V100"}
        self.assertEqual(result, expected)

        # Verify that partition argument was passed to subprocess.run
        mock_run.assert_called_with(
            ["sinfo", "-o", "%G", "-p", "test_partition"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_get_gpu_generation_and_count_with_partition(self, mock_run):
        # Mock successful GPU generation and count retrieval with partition
        mock_run.return_value = MagicMock(
            stdout="gpu:a100:4(S:0-1)\ngpu:v100:2(S:0)\n",
            returncode=0,
        )

        result = self.cluster_info_with_partition.get_gpu_generation_and_count()
        expected = {"a100": 4, "v100": 2}
        self.assertEqual(result, expected)

        # Verify that partition argument was passed to subprocess.run
        mock_run.assert_called_with(
            ["sinfo", "-o", "%G", "-p", "test_partition"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_get_gpu_generations_error(self, mock_run):
        # Create an instance of the class
        cluster_info = SlurmClusterInfo()

        # Mock failed command
        mock_run.side_effect = subprocess.SubprocessError()
        # Check that RuntimeError is raised
        with self.assertRaises(RuntimeError):
            cluster_info.get_gpu_generations()
        mock_run.side_effect = FileNotFoundError()
        # Check that RuntimeError is raised
        with self.assertRaises(RuntimeError):
            cluster_info.get_gpu_generations()

    @patch("clusterscope.cluster_info.SlurmClusterInfo.get_gpu_generation_and_count")
    def test_has_gpu_type_true(self, mock_get_gpu_generation_and_count):
        # Set up the mock to return a dictionary with the GPU type we're looking for
        mock_get_gpu_generation_and_count.return_value = {"A100": 4, "V100": 2}

        # Create an instance of the class containing the has_gpu_type method
        gpu_manager = SlurmClusterInfo()

        result = gpu_manager.has_gpu_type("A100")
        self.assertTrue(result)

        result = gpu_manager.has_gpu_type("H100")
        self.assertFalse(result)

        result = gpu_manager.has_gpu_type("V100")
        self.assertTrue(result)


class TestRunCli(unittest.TestCase):
    """Test cases for the run_cli function."""

    def test_run_cli_empty_command(self):
        """Test that run_cli raises RuntimeError for empty command list."""
        with self.assertRaises(RuntimeError) as context:
            run_cli([])
        self.assertIn("Command list cannot be empty", str(context.exception))

    @patch("shutil.which")
    def test_run_cli_command_not_available(self, mock_which):
        """Test that run_cli raises RuntimeError when command is not available."""
        mock_which.return_value = None

        with self.assertRaises(RuntimeError) as context:
            run_cli(["nonexistent_command"])
        self.assertIn(
            "Command 'nonexistent_command' is not available", str(context.exception)
        )

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_run_cli_successful_execution(self, mock_check_output, mock_which):
        """Test successful command execution."""
        mock_which.return_value = "/usr/bin/echo"
        mock_check_output.return_value = "Hello World\n"

        result = run_cli(["echo", "Hello World"])
        self.assertEqual(result, "Hello World\n")
        mock_check_output.assert_called_once_with(
            ["echo", "Hello World"], text=True, timeout=60, stderr=None
        )

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_run_cli_with_custom_parameters(self, mock_check_output, mock_which):
        """Test run_cli with custom text, timeout, and stderr parameters."""
        mock_which.return_value = "/usr/bin/echo"
        mock_check_output.return_value = b"Binary output"

        result = run_cli(
            ["echo", "test"], text=False, timeout=30, stderr=subprocess.STDOUT
        )
        self.assertEqual(result, b"Binary output")
        mock_check_output.assert_called_once_with(
            ["echo", "test"], text=False, timeout=30, stderr=subprocess.STDOUT
        )

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_run_cli_called_process_error(self, mock_check_output, mock_which):
        """Test that run_cli handles CalledProcessError properly."""
        mock_which.return_value = "/usr/bin/false"
        mock_check_output.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["false"], output="Command failed"
        )

        with self.assertRaises(RuntimeError) as context:
            run_cli(["false"])
        self.assertIn(
            "Command 'false' failed with return code 1", str(context.exception)
        )

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_run_cli_timeout_expired(self, mock_check_output, mock_which):
        """Test that run_cli handles TimeoutExpired properly."""
        mock_which.return_value = "/usr/bin/sleep"
        mock_check_output.side_effect = subprocess.TimeoutExpired(
            cmd=["sleep", "10"], timeout=1
        )

        with self.assertRaises(RuntimeError) as context:
            run_cli(["sleep", "10"], timeout=1)
        self.assertIn(
            "Command 'sleep 10' timed out after 1 seconds", str(context.exception)
        )

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_run_cli_subprocess_error(self, mock_check_output, mock_which):
        """Test that run_cli handles SubprocessError properly."""
        mock_which.return_value = "/usr/bin/echo"
        mock_check_output.side_effect = subprocess.SubprocessError(
            "Generic subprocess error"
        )

        with self.assertRaises(RuntimeError) as context:
            run_cli(["echo", "test"])
        self.assertIn("Failed to execute command 'echo test'", str(context.exception))

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_run_cli_file_not_found_error(self, mock_check_output, mock_which):
        """Test that run_cli handles FileNotFoundError properly."""
        mock_which.return_value = "/usr/bin/echo"
        mock_check_output.side_effect = FileNotFoundError("File not found")

        with self.assertRaises(RuntimeError) as context:
            run_cli(["echo", "test"])
        self.assertIn("Failed to execute command 'echo test'", str(context.exception))

    @patch("shutil.which")
    @patch("subprocess.check_output")
    def test_run_cli_real_command_integration(self, mock_check_output, mock_which):
        """Integration test with a real command that should exist on most systems."""
        # Test with 'echo' command which should be available on most systems
        mock_which.return_value = "/bin/echo"
        mock_check_output.return_value = "integration test\n"

        result = run_cli(["echo", "integration test"])
        self.assertEqual(result, "integration test\n")


class TestLocalNodeInfo(unittest.TestCase):
    """Test cases for LocalNodeInfo class, including AMD GPU support."""

    def setUp(self):
        self.local_node_info = LocalNodeInfo()

    @patch("subprocess.run")
    def test_has_nvidia_gpus_true(self, mock_run):
        """Test has_nvidia_gpus returns True when nvidia-smi is available."""
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(self.local_node_info.has_nvidia_gpus())

    @patch("subprocess.run")
    def test_has_nvidia_gpus_false_file_not_found(self, mock_run):
        """Test has_nvidia_gpus returns False when nvidia-smi is not found."""
        mock_run.side_effect = FileNotFoundError()
        self.assertFalse(self.local_node_info.has_nvidia_gpus())

    @patch("subprocess.run")
    def test_has_nvidia_gpus_false_called_process_error(self, mock_run):
        """Test has_nvidia_gpus returns False when nvidia-smi fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["nvidia-smi"])
        self.assertFalse(self.local_node_info.has_nvidia_gpus())

    @patch("subprocess.run")
    def test_has_amd_gpus_true(self, mock_run):
        """Test has_amd_gpus returns True when rocm-smi is available."""
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(self.local_node_info.has_amd_gpus())

    @patch("subprocess.run")
    def test_has_amd_gpus_false_file_not_found(self, mock_run):
        """Test has_amd_gpus returns False when rocm-smi is not found."""
        mock_run.side_effect = FileNotFoundError()
        self.assertFalse(self.local_node_info.has_amd_gpus())

    @patch("subprocess.run")
    def test_has_amd_gpus_false_called_process_error(self, mock_run):
        """Test has_amd_gpus returns False when rocm-smi fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, ["rocm-smi"])
        self.assertFalse(self.local_node_info.has_amd_gpus())

    @patch.object(LocalNodeInfo, "has_nvidia_gpus")
    @patch.object(LocalNodeInfo, "has_amd_gpus")
    def test_get_gpu_vendor_nvidia(self, mock_has_amd, mock_has_nvidia):
        """Test get_gpu_vendor returns 'nvidia' when NVIDIA GPUs are present."""
        mock_has_nvidia.return_value = True
        mock_has_amd.return_value = False
        self.assertEqual(self.local_node_info.get_gpu_vendor(), "nvidia")

    @patch.object(LocalNodeInfo, "has_nvidia_gpus")
    @patch.object(LocalNodeInfo, "has_amd_gpus")
    def test_get_gpu_vendor_amd(self, mock_has_amd, mock_has_nvidia):
        """Test get_gpu_vendor returns 'amd' when AMD GPUs are present."""
        mock_has_nvidia.return_value = False
        mock_has_amd.return_value = True
        self.assertEqual(self.local_node_info.get_gpu_vendor(), "amd")

    @patch.object(LocalNodeInfo, "has_nvidia_gpus")
    @patch.object(LocalNodeInfo, "has_amd_gpus")
    def test_get_gpu_vendor_none(self, mock_has_amd, mock_has_nvidia):
        """Test get_gpu_vendor returns 'none' when no GPUs are present."""
        mock_has_nvidia.return_value = False
        mock_has_amd.return_value = False
        self.assertEqual(self.local_node_info.get_gpu_vendor(), "none")

    @patch.object(LocalNodeInfo, "has_nvidia_gpus")
    @patch.object(LocalNodeInfo, "has_amd_gpus")
    def test_get_gpu_vendor_nvidia_priority(self, mock_has_amd, mock_has_nvidia):
        """Test get_gpu_vendor returns 'nvidia' when both GPU types are present (NVIDIA has priority)."""
        mock_has_nvidia.return_value = True
        mock_has_amd.return_value = True
        self.assertEqual(self.local_node_info.get_gpu_vendor(), "nvidia")

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_nvidia_gpu_info_success(self, mock_run_cli):
        """Test successful NVIDIA GPU information retrieval."""
        mock_run_cli.return_value = (
            "NVIDIA A100-SXM4-40GB\nNVIDIA A100-SXM4-40GB\nTesla V100-SXM2-16GB"
        )

        result = self.local_node_info.get_nvidia_gpu_info()
        expected = {"A100": 2, "V100": 1}
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_nvidia_gpu_info_empty_lines(self, mock_run_cli):
        """Test NVIDIA GPU info parsing with empty lines."""
        mock_run_cli.return_value = "NVIDIA A100-SXM4-40GB\n\n\nTesla V100-SXM2-16GB\n"

        result = self.local_node_info.get_nvidia_gpu_info()
        expected = {"A100": 1, "V100": 1}
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_mi300x(self, mock_run_cli):
        """Test AMD GPU information retrieval for MI300X."""
        mock_run_cli.return_value = """GPU[0]: AMD Instinct MI300X
GPU[1]: AMD Instinct MI300X"""

        result = self.local_node_info.get_amd_gpu_info()
        expected = {"MI300X": 2}
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_mi300a(self, mock_run_cli):
        """Test AMD GPU information retrieval for MI300A."""
        mock_run_cli.return_value = "GPU[0]: AMD Instinct MI300A"

        result = self.local_node_info.get_amd_gpu_info()
        expected = {"MI300A": 1}
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_various_models(self, mock_run_cli):
        """Test AMD GPU info parsing with various GPU models."""
        mock_run_cli.return_value = """GPU[0]: AMD Instinct MI250X
GPU[1]: AMD Instinct MI210
GPU[2]: AMD Instinct MI100
GPU[3]: AMD Radeon RX 7900 XTX"""

        result = self.local_node_info.get_amd_gpu_info()
        expected = {"MI250X": 1, "MI210": 1, "MI100": 1, "RX7900XTX": 1}
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_generic_fallback(self, mock_run_cli):
        """Test AMD GPU info parsing with generic fallback for unknown models."""
        mock_run_cli.return_value = "GPU[0]: AMD Radeon RX 6800 XT"

        result = self.local_node_info.get_amd_gpu_info()
        # Should fall back to extracting "6800" as the model
        expected = {"6800": 1}
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_no_gpu_lines(self, mock_run_cli):
        """Test AMD GPU info parsing with no GPU lines."""
        mock_run_cli.return_value = "Some other output\nNo GPU information here"

        result = self.local_node_info.get_amd_gpu_info()
        self.assertEqual(result, {})

    @patch.object(LocalNodeInfo, "has_nvidia_gpus")
    @patch.object(LocalNodeInfo, "has_amd_gpus")
    @patch.object(LocalNodeInfo, "get_nvidia_gpu_info")
    @patch.object(LocalNodeInfo, "get_amd_gpu_info")
    def test_get_gpu_generation_and_count_nvidia_only(
        self, mock_amd_info, mock_nvidia_info, mock_has_amd, mock_has_nvidia
    ):
        """Test get_gpu_generation_and_count with NVIDIA GPUs only."""
        mock_has_nvidia.return_value = True
        mock_has_amd.return_value = False
        mock_nvidia_info.return_value = {"A100": 2, "V100": 1}

        result = self.local_node_info.get_gpu_generation_and_count()
        expected = {"A100": 2, "V100": 1}
        self.assertEqual(result, expected)

    @patch.object(LocalNodeInfo, "has_nvidia_gpus")
    @patch.object(LocalNodeInfo, "has_amd_gpus")
    @patch.object(LocalNodeInfo, "get_nvidia_gpu_info")
    @patch.object(LocalNodeInfo, "get_amd_gpu_info")
    def test_get_gpu_generation_and_count_amd_only(
        self, mock_amd_info, mock_nvidia_info, mock_has_amd, mock_has_nvidia
    ):
        """Test get_gpu_generation_and_count with AMD GPUs only."""
        mock_has_nvidia.return_value = False
        mock_has_amd.return_value = True
        mock_amd_info.return_value = {"MI300X": 4, "MI250X": 2}

        result = self.local_node_info.get_gpu_generation_and_count()
        expected = {"MI300X": 4, "MI250X": 2}
        self.assertEqual(result, expected)

    @patch.object(LocalNodeInfo, "has_nvidia_gpus")
    @patch.object(LocalNodeInfo, "has_amd_gpus")
    @patch.object(LocalNodeInfo, "get_nvidia_gpu_info")
    @patch.object(LocalNodeInfo, "get_amd_gpu_info")
    def test_get_gpu_generation_and_count_both_vendors(
        self, mock_amd_info, mock_nvidia_info, mock_has_amd, mock_has_nvidia
    ):
        """Test get_gpu_generation_and_count with both NVIDIA and AMD GPUs."""
        mock_has_nvidia.return_value = True
        mock_has_amd.return_value = True
        mock_nvidia_info.return_value = {"A100": 2}
        mock_amd_info.return_value = {"MI300X": 4}

        result = self.local_node_info.get_gpu_generation_and_count()
        expected = {"A100": 2, "MI300X": 4}
        self.assertEqual(result, expected)

    @patch.object(LocalNodeInfo, "has_nvidia_gpus")
    @patch.object(LocalNodeInfo, "has_amd_gpus")
    @patch("logging.warning")
    def test_get_gpu_generation_and_count_no_gpus(
        self, mock_logging, mock_has_amd, mock_has_nvidia
    ):
        """Test get_gpu_generation_and_count with no GPUs available."""
        mock_has_nvidia.return_value = False
        mock_has_amd.return_value = False

        result = self.local_node_info.get_gpu_generation_and_count()
        self.assertEqual(result, {})
        mock_logging.assert_called_with(
            "No GPUs found or unable to retrieve GPU information"
        )

    @patch.object(LocalNodeInfo, "get_gpu_generation_and_count")
    def test_has_gpu_type_true(self, mock_get_gpu_info):
        """Test has_gpu_type returns True for available GPU types."""
        mock_get_gpu_info.return_value = {"A100": 2, "MI300X": 4}

        self.assertTrue(self.local_node_info.has_gpu_type("A100"))
        self.assertTrue(self.local_node_info.has_gpu_type("MI300X"))
        self.assertTrue(self.local_node_info.has_gpu_type("a100"))  # Case insensitive
        self.assertTrue(self.local_node_info.has_gpu_type("mi300x"))  # Case insensitive

    @patch.object(LocalNodeInfo, "get_gpu_generation_and_count")
    def test_has_gpu_type_false(self, mock_get_gpu_info):
        """Test has_gpu_type returns False for unavailable GPU types."""
        mock_get_gpu_info.return_value = {"A100": 2, "MI300X": 4}

        self.assertFalse(self.local_node_info.has_gpu_type("V100"))
        self.assertFalse(self.local_node_info.has_gpu_type("MI250X"))
        self.assertFalse(self.local_node_info.has_gpu_type("H100"))

    @patch.object(LocalNodeInfo, "get_gpu_generation_and_count")
    def test_has_gpu_type_runtime_error(self, mock_get_gpu_info):
        """Test has_gpu_type returns False when get_gpu_generation_and_count raises RuntimeError."""
        mock_get_gpu_info.side_effect = RuntimeError("No GPUs found")

        self.assertFalse(self.local_node_info.has_gpu_type("A100"))


class TestUnifiedInfoAMDSupport(unittest.TestCase):
    """Test cases for UnifiedInfo class AMD GPU support."""

    def setUp(self):
        self.unified_info = UnifiedInfo()

    @patch.object(LocalNodeInfo, "get_gpu_vendor")
    def test_get_gpu_vendor(self, mock_get_gpu_vendor):
        """Test UnifiedInfo.get_gpu_vendor delegates to LocalNodeInfo."""
        mock_get_gpu_vendor.return_value = "amd"

        result = self.unified_info.get_gpu_vendor()
        self.assertEqual(result, "amd")
        mock_get_gpu_vendor.assert_called_once()

    @patch.object(LocalNodeInfo, "has_gpu_type")
    def test_has_gpu_type_local_node(self, mock_has_gpu_type):
        """Test UnifiedInfo.has_gpu_type with local node (non-Slurm)."""
        self.unified_info.is_slurm_cluster = False
        mock_has_gpu_type.return_value = True

        result = self.unified_info.has_gpu_type("MI300X")
        self.assertTrue(result)
        mock_has_gpu_type.assert_called_once_with("MI300X")

    @patch.object(SlurmClusterInfo, "has_gpu_type")
    def test_has_gpu_type_slurm_cluster(self, mock_has_gpu_type):
        """Test UnifiedInfo.has_gpu_type with Slurm cluster."""
        self.unified_info.is_slurm_cluster = True
        mock_has_gpu_type.return_value = True

        result = self.unified_info.has_gpu_type("MI300X")
        self.assertTrue(result)
        mock_has_gpu_type.assert_called_once_with("MI300X")

    def test_get_gpu_generation_and_count_with_amd_gpus(self):
        """Test get_gpu_generation_and_count includes AMD GPUs."""
        self.unified_info.is_slurm_cluster = False
        self.unified_info.has_nvidia_gpus = False
        self.unified_info.has_amd_gpus = True

        with patch.object(
            LocalNodeInfo, "get_gpu_generation_and_count"
        ) as mock_get_gpu_info:
            mock_get_gpu_info.return_value = {"MI300X": 4}

            result = self.unified_info.get_gpu_generation_and_count()
            self.assertEqual(result, {"MI300X": 4})


class TestAWSClusterInfo(unittest.TestCase):
    def setUp(self):
        self.aws_cluster_info = AWSClusterInfo()

    @patch("subprocess.run")
    def test_is_aws_cluster(self, mock_run):
        # Mock AWS environment
        mock_run.return_value = MagicMock(stdout="amazon_ec2", returncode=0)
        self.assertTrue(self.aws_cluster_info.is_aws_cluster())

        # Mock non-AWS environment
        mock_run.return_value = MagicMock(stdout="other_system", returncode=0)
        self.assertFalse(self.aws_cluster_info.is_aws_cluster())

    def test_get_aws_nccl_settings(self):
        # Test with AWS cluster
        with patch.object(AWSClusterInfo, "is_aws_cluster", return_value=True):
            settings = self.aws_cluster_info.get_aws_nccl_settings()
            self.assertIn("FI_PROVIDER", settings)
            self.assertEqual(settings["FI_PROVIDER"], "efa")

        # Test with non-AWS cluster
        with patch.object(AWSClusterInfo, "is_aws_cluster", return_value=False):
            settings = self.aws_cluster_info.get_aws_nccl_settings()
            self.assertEqual(settings, {})


class TestResourceRequirementMethods(unittest.TestCase):
    """Test cases for get_task_resource_requirements and get_array_job_requirements methods."""

    def setUp(self):
        self.unified_info = UnifiedInfo()

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_single_gpu_8gpu_node(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test get_task_resource_requirements with 1 GPU on an 8-GPU node."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = 192
        mock_mem.return_value = 1843200  # 1.8TB in MB

        result = self.unified_info.get_task_resource_requirements(num_gpus=1)

        self.assertEqual(result.cpu_cores, 24)  # 192/8 = 24
        self.assertEqual(result.memory, "225G")  # 1843200/8/1024 = 225GB
        self.assertEqual(result.tasks_per_node, 1)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_four_gpus_8gpu_node(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test get_task_resource_requirements with 4 GPUs on an 8-GPU node."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = 192
        mock_mem.return_value = 1843200

        result = self.unified_info.get_task_resource_requirements(num_gpus=4)

        self.assertEqual(result.cpu_cores, 96)  # 192/8*4 = 96
        self.assertEqual(result.memory, "900G")  # 1843200/8*4/1024 = 900GB
        self.assertEqual(result.tasks_per_node, 1)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_full_node_8gpu(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test get_task_resource_requirements with all 8 GPUs (full node)."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = 192
        mock_mem.return_value = 1843200

        result = self.unified_info.get_task_resource_requirements(num_gpus=8)

        self.assertEqual(result.cpu_cores, 192)  # All CPUs
        self.assertEqual(
            result.memory, "2T"
        )  # All memory: 1843200/1024/1024 = 1.8TB rounds to 2T
        self.assertEqual(result.tasks_per_node, 1)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_4gpu_node_configuration(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test get_task_resource_requirements on a 4-GPU node configuration."""
        mock_total_gpus.return_value = 4
        mock_cpus.return_value = 64
        mock_mem.return_value = 524288  # 512GB in MB

        # Test 1 GPU on 4-GPU node
        result = self.unified_info.get_task_resource_requirements(num_gpus=1)
        self.assertEqual(result.cpu_cores, 16)  # 64/4 = 16
        self.assertEqual(result.memory, "128G")  # 524288/4/1024 = 128GB

        # Test full 4-GPU node
        result = self.unified_info.get_task_resource_requirements(num_gpus=4)
        self.assertEqual(result.cpu_cores, 64)  # All CPUs
        self.assertEqual(result.memory, "512G")  # All memory

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_getResRequirements_with_multiple_tasks_per_node(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test getResRequirements with multiple tasks per node."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = 192
        mock_mem.return_value = 1843200

        result = self.unified_info.get_task_resource_requirements(
            num_gpus=4, num_tasks_per_node=2
        )

        self.assertEqual(result.cpu_cores, 48)  # (192/8*4)/2 = 48 per task
        self.assertEqual(result.memory, "900G")  # Memory is per node, not per task
        self.assertEqual(result.tasks_per_node, 2)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_memory_terabyte_format(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test get_task_resource_requirements returns TB format for very large memory."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = 192
        mock_mem.return_value = 8388608  # 8TB in MB

        result = self.unified_info.get_task_resource_requirements(num_gpus=8)

        self.assertEqual(result.memory, "8T")  # 8388608/1024/1024 = 8TB

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_cpu_rounding_up(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test get_task_resource_requirements rounds up CPU cores when fractional."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = 191  # Odd number that doesn't divide evenly
        mock_mem.return_value = 1843200

        result = self.unified_info.get_task_resource_requirements(num_gpus=1)

        # 191/8 = 23.875, should round up to 24
        self.assertEqual(result.cpu_cores, 24)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    def test_getResRequirements_invalid_num_gpus(self, mock_total_gpus):
        """Test getResRequirements raises ValueError for invalid num_gpus."""
        mock_total_gpus.return_value = 8

        # Test zero GPUs
        with self.assertRaises(ValueError) as context:
            self.unified_info.get_task_resource_requirements(num_gpus=0)
        self.assertIn("num_gpus must be between 1 and 8", str(context.exception))

        # Test more than max GPUs
        with self.assertRaises(ValueError) as context:
            self.unified_info.get_task_resource_requirements(num_gpus=9)
        self.assertIn("num_gpus must be between 1 and 8", str(context.exception))

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    def test_getResRequirements_invalid_num_tasks_per_node(self, mock_total_gpus):
        """Test getResRequirements raises ValueError for invalid num_tasks_per_node."""
        mock_total_gpus.return_value = 8

        with self.assertRaises(ValueError) as context:
            self.unified_info.get_task_resource_requirements(
                num_gpus=1, num_tasks_per_node=0
            )
        self.assertIn("num_tasks_per_node must be at least 1", str(context.exception))

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_getArrayJobRequirements_single_gpu(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test getArrayJobRequirements with 1 GPU per task."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = 192
        mock_mem.return_value = 1843200

        result = self.unified_info.get_array_job_requirements(num_gpus_per_task=1)

        self.assertEqual(result.cpu_cores, 24)  # 192/8 = 24
        self.assertEqual(result.memory, "225G")  # 1843200/8/1024 = 225GB
        self.assertEqual(result.tasks_per_node, 1)  # Always 1 for array jobs

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_getArrayJobRequirements_multiple_gpus(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test getArrayJobRequirements with multiple GPUs per task."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = 192
        mock_mem.return_value = 1843200

        result = self.unified_info.get_array_job_requirements(num_gpus_per_task=4)

        self.assertEqual(result.cpu_cores, 96)  # 192/8*4 = 96
        self.assertEqual(result.memory, "900G")  # 1843200/8*4/1024 = 900GB
        self.assertEqual(result.tasks_per_node, 1)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_getArrayJobRequirements_full_node(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test getArrayJobRequirements with all GPUs (full node per task)."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = 192
        mock_mem.return_value = 1843200

        result = self.unified_info.get_array_job_requirements(num_gpus_per_task=8)

        self.assertEqual(result.cpu_cores, 192)  # All CPUs
        self.assertEqual(
            result.memory, "2T"
        )  # All memory: 1843200/1024/1024 = 1.8TB rounds to 2T
        self.assertEqual(result.tasks_per_node, 1)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_getArrayJobRequirements_4gpu_node(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test getArrayJobRequirements on a 4-GPU node configuration."""
        mock_total_gpus.return_value = 4
        mock_cpus.return_value = 64
        mock_mem.return_value = 524288

        result = self.unified_info.get_array_job_requirements(num_gpus_per_task=2)

        self.assertEqual(result.cpu_cores, 32)  # 64/4*2 = 32
        self.assertEqual(result.memory, "256G")  # 524288/4*2/1024 = 256GB
        self.assertEqual(result.tasks_per_node, 1)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    def test_getArrayJobRequirements_invalid_num_gpus_per_task(self, mock_total_gpus):
        """Test getArrayJobRequirements raises ValueError for invalid num_gpus_per_task."""
        mock_total_gpus.return_value = 8

        # Test zero GPUs
        with self.assertRaises(ValueError) as context:
            self.unified_info.get_array_job_requirements(num_gpus_per_task=0)
        self.assertIn(
            "num_gpus_per_task must be between 1 and 8", str(context.exception)
        )

        # Test more than max GPUs
        with self.assertRaises(ValueError) as context:
            self.unified_info.get_array_job_requirements(num_gpus_per_task=9)
        self.assertIn(
            "num_gpus_per_task must be between 1 and 8", str(context.exception)
        )

    @patch.object(UnifiedInfo, "get_gpu_generation_and_count")
    def test_get_total_gpus_per_node_with_gpus(self, mock_gpu_info):
        """Test get_total_gpus_per_node with actual GPU detection."""
        mock_gpu_info.return_value = {"A100": 4, "V100": 4}

        result = self.unified_info.get_total_gpus_per_node()
        self.assertEqual(result, 8)  # 4 + 4 = 8

    @patch.object(UnifiedInfo, "get_gpu_generation_and_count")
    def test_get_total_gpus_per_node_no_gpus_detected(self, mock_gpu_info):
        """Test get_total_gpus_per_node defaults to 8 when no GPUs detected."""
        mock_gpu_info.return_value = {}

        result = self.unified_info.get_total_gpus_per_node()
        self.assertEqual(result, 8)  # Default fallback

    @patch.object(UnifiedInfo, "get_gpu_generation_and_count")
    def test_get_total_gpus_per_node_single_gpu_type(self, mock_gpu_info):
        """Test get_total_gpus_per_node with single GPU type."""
        mock_gpu_info.return_value = {"A100": 8}

        result = self.unified_info.get_total_gpus_per_node()
        self.assertEqual(result, 8)

    @patch.object(UnifiedInfo, "get_gpu_generation_and_count")
    def test_get_total_gpus_per_node_mixed_gpu_types(self, mock_gpu_info):
        """Test get_total_gpus_per_node with mixed GPU types."""
        mock_gpu_info.return_value = {"A100": 2, "V100": 4, "P100": 2}

        result = self.unified_info.get_total_gpus_per_node()
        self.assertEqual(result, 8)  # 2 + 4 + 2 = 8

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_dynamic_gpu_boundary_validation(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test that validation boundaries adapt to detected GPU count."""
        # Test with 4-GPU node
        mock_total_gpus.return_value = 4
        mock_cpus.return_value = 64
        mock_mem.return_value = 524288

        # Valid requests for 4-GPU node
        result = self.unified_info.get_task_resource_requirements(num_gpus=1)
        self.assertIsInstance(result, ResourceShape)

        result = self.unified_info.get_task_resource_requirements(num_gpus=4)
        self.assertIsInstance(result, ResourceShape)

        # Invalid request (more than available)
        with self.assertRaises(ValueError) as context:
            self.unified_info.get_task_resource_requirements(num_gpus=5)
        self.assertIn("num_gpus must be between 1 and 4", str(context.exception))

        # Test with 16-GPU node
        mock_total_gpus.return_value = 16

        # Should now accept up to 16 GPUs
        result = self.unified_info.get_task_resource_requirements(num_gpus=16)
        self.assertIsInstance(result, ResourceShape)

        # But not 17
        with self.assertRaises(ValueError) as context:
            self.unified_info.get_task_resource_requirements(num_gpus=17)
        self.assertIn("num_gpus must be between 1 and 16", str(context.exception))

    def test_resource_shape_namedtuple(self):
        """Test ResourceShape NamedTuple structure."""
        resource = ResourceShape(cpu_cores=24, memory="225G", tasks_per_node=1)

        self.assertEqual(resource.cpu_cores, 24)
        self.assertEqual(resource.memory, "225G")
        self.assertEqual(resource.tasks_per_node, 1)

        # Test that it's immutable (characteristic of NamedTuple)
        with self.assertRaises(AttributeError):
            resource.cpu_cores = 48


if __name__ == "__main__":
    unittest.main()
