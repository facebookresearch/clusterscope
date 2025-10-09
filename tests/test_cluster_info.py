# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from clusterscope.cluster_info import (
    AWSClusterInfo,
    CPUInfo,
    DarwinInfo,
    GPUInfo,
    LinuxInfo,
    LocalNodeInfo,
    MemInfo,
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
        self.assertEqual(unified_info.get_gpu_generation_and_count(), [])

    def test_partition_passed_to_slurm_cluster_info(self):
        unified_info = UnifiedInfo(partition="test_partition")
        self.assertEqual(unified_info.slurm_cluster_info.partition, "test_partition")


class TestLinuxInfo(unittest.TestCase):
    def setUp(self):
        self.linux_info = LinuxInfo()

    @patch("clusterscope.cluster_info.run_cli", return_value="1234")
    def test_get_cpu_count(self, mock_run):
        self.assertEqual(self.linux_info.get_cpu_count(), CPUInfo(cpu_count=1234))

    @patch(
        "clusterscope.cluster_info.run_cli",
        return_value="               total        used\nMem:     12345    123\n",
    )
    def test_get_mem_per_node_MB(self, mock_run):
        self.assertEqual(
            self.linux_info.get_mem_MB(), MemInfo(mem_total_MB=12345, mem_total_GB=12)
        )


class TestDarwinInfo(unittest.TestCase):
    def setUp(self):
        self.darwin_info = DarwinInfo()

    @patch("clusterscope.cluster_info.run_cli", return_value="10")
    def test_get_cpu_count(self, mock_run):
        self.assertEqual(self.darwin_info.get_cpu_count(), CPUInfo(cpu_count=10))

    @patch(
        "clusterscope.cluster_info.run_cli",
        return_value="34359738368",
    )
    def test_get_mem_per_node_MB(self, mock_run):
        self.assertEqual(
            self.darwin_info.get_mem_MB(),
            MemInfo(mem_total_MB=32768, mem_total_GB=32),
        )


class TestSlurmClusterInfo(unittest.TestCase):
    def setUp(self):
        self.cluster_info = SlurmClusterInfo()
        self.cluster_info_with_partition = SlurmClusterInfo(partition="test_partition")

    @patch("subprocess.run")
    @patch("clusterscope.cache.load")
    def test_get_cluster_name(self, mock_cache, mock_run):
        # Mock successful cluster name retrieval
        mock_run.return_value = MagicMock(
            stdout="ClusterName=test_cluster\nOther=value", returncode=0
        )
        mock_cache.return_value = {"SLURM_CLUSTER_NAME": "test_cluster"}
        self.assertEqual(self.cluster_info.get_cluster_name(), "test_cluster")

    @patch("subprocess.run")
    def test_get_cpu_per_node(self, mock_run):
        # Mock successful cluster name retrieval
        mock_run.return_value = MagicMock(stdout="128, test_partition", returncode=0)
        self.assertEqual(
            self.cluster_info.get_cpus_per_node(),
            [CPUInfo(cpu_count=128, partition="test_partition")],
        )

    @patch("subprocess.run")
    @patch("clusterscope.cache.load", return_value={})  # Mock empty cache
    @patch("clusterscope.cache.save")  # Mock cache save function
    def test_get_cpu_per_node_with_partition(self, mock_save, mock_load, mock_run):
        # Mock successful CPU per node retrieval with partition
        mock_run.return_value = MagicMock(stdout="128, test_partition", returncode=0)
        result = self.cluster_info_with_partition.get_cpus_per_node()
        self.assertEqual(result, [CPUInfo(cpu_count=128, partition="test_partition")])
        # Verify that partition argument was passed to subprocess.run
        mock_run.assert_called_with(
            ["sinfo", "-o", "%100c,%100P", "--noheader", "-p", "test_partition"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    @patch("subprocess.run")
    def test_get_mem_per_node_MB(self, mock_run):
        # Mock successful cluster name retrieval
        mock_run.return_value = MagicMock(
            stdout="123456+, test_partition", returncode=0
        )
        self.assertEqual(
            self.cluster_info.get_mem_per_node_MB()[0].mem_total_MB, 123456
        )

    @patch("subprocess.run")
    @patch("clusterscope.cache.load", return_value={})  # Mock empty cache
    @patch("clusterscope.cache.save")  # Mock cache save function
    def test_get_mem_per_node_MB_with_partition(self, mock_save, mock_load, mock_run):
        # Mock successful memory per node retrieval with partition
        mock_run.return_value = MagicMock(
            stdout="123456+, test_partition", returncode=0
        )
        result = self.cluster_info_with_partition.get_mem_per_node_MB()
        self.assertEqual(result[0].mem_total_MB, 123456)
        # Verify that partition argument was passed to subprocess.run
        mock_run.assert_called_with(
            [
                "sinfo",
                "-o",
                "%100m,%100P",
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
    def test_get_gpu_generation_and_count_duplicated_partitions(self, mock_run):
        # Mock successful GPU generation and count retrieval with partition
        mock_run.return_value = MagicMock(
            stdout="gpu:a100:4(S:0-1), test_partition\ngpu:a100:4, test_partition\ngpu:v100:2(S:0), test_partition\n",
            returncode=0,
        )

        result = self.cluster_info_with_partition.get_gpu_generation_and_count()
        expected = [
            GPUInfo(
                gpu_count=4,
                gpu_gen="a100",
                vendor="nvidia",
                partition="test_partition",
            ),
            GPUInfo(
                gpu_count=2,
                gpu_gen="v100",
                vendor="nvidia",
                partition="test_partition",
            ),
        ]
        self.assertEqual(result, expected)

    @patch("subprocess.run")
    def test_get_gpu_generation_and_count_with_partition(self, mock_run):
        # Mock successful GPU generation and count retrieval with partition
        mock_run.return_value = MagicMock(
            stdout="gpu:a100:4(S:0-1), test_partition\ngpu:v100:2(S:0), test_partition\n",
            returncode=0,
        )

        result = self.cluster_info_with_partition.get_gpu_generation_and_count()
        expected = [
            GPUInfo(
                gpu_count=4,
                gpu_gen="a100",
                vendor="nvidia",
                partition="test_partition",
            ),
            GPUInfo(
                gpu_count=2,
                gpu_gen="v100",
                vendor="nvidia",
                partition="test_partition",
            ),
        ]
        self.assertEqual(result, expected)

        # Verify that partition argument was passed to subprocess.run
        mock_run.assert_called_with(
            ["sinfo", "-o", "%G,%P", "-p", "test_partition"],
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
        mock_get_gpu_generation_and_count.return_value = [
            GPUInfo(
                gpu_count=4,
                gpu_gen="A100",
                vendor="nvidia",
            ),
            GPUInfo(
                gpu_count=2,
                gpu_gen="V100",
                vendor="nvidia",
            ),
        ]

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

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_nvidia_gpu_info_success(self, mock_run_cli):
        """Test successful NVIDIA GPU information retrieval."""
        mock_run_cli.return_value = "NVIDIA A100-SXM4-40GB, 2\nNVIDIA A100-SXM4-40GB, 2\nTesla V100-SXM2-16GB, 1"

        result = self.local_node_info.get_nvidia_gpu_info()
        expected = [
            GPUInfo(gpu_gen="A100", gpu_count=2, vendor="nvidia"),
            GPUInfo(gpu_gen="V100", gpu_count=1, vendor="nvidia"),
        ]
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_nvidia_gpu_info_empty_lines(self, mock_run_cli):
        """Test NVIDIA GPU info parsing with empty lines."""
        mock_run_cli.return_value = (
            "NVIDIA A100-SXM4-40GB, 1\n\n\nTesla V100-SXM2-16GB, 1\n"
        )

        result = self.local_node_info.get_nvidia_gpu_info()
        expected = [
            GPUInfo(gpu_gen="A100", gpu_count=1, vendor="nvidia"),
            GPUInfo(gpu_gen="V100", gpu_count=1, vendor="nvidia"),
        ]
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_mi300x(self, mock_run_cli):
        """Test AMD GPU information retrieval for MI300X."""
        mock_run_cli.return_value = """GPU[0]: AMD Instinct MI300X
GPU[1]: AMD Instinct MI300X"""

        result = self.local_node_info.get_amd_gpu_info()
        expected = [GPUInfo(gpu_gen="MI300X", gpu_count=2, vendor="amd")]
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_mi300a(self, mock_run_cli):
        """Test AMD GPU information retrieval for MI300A."""
        mock_run_cli.return_value = "GPU[0]: AMD Instinct MI300A"

        result = self.local_node_info.get_amd_gpu_info()
        expected = [GPUInfo(gpu_gen="MI300A", gpu_count=1, vendor="amd")]
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_various_models(self, mock_run_cli):
        """Test AMD GPU info parsing with various GPU models."""
        mock_run_cli.return_value = """GPU[0]: AMD Instinct MI250X
GPU[1]: AMD Instinct MI210
GPU[2]: AMD Instinct MI100
GPU[3]: AMD Radeon RX 7900 XTX"""

        result = self.local_node_info.get_amd_gpu_info()

        expected = [
            GPUInfo(
                gpu_count=1,
                gpu_gen="MI250X",
                vendor="amd",
            ),
            GPUInfo(
                gpu_count=1,
                gpu_gen="MI210",
                vendor="amd",
            ),
            GPUInfo(
                gpu_count=1,
                gpu_gen="MI100",
                vendor="amd",
            ),
            GPUInfo(
                gpu_count=1,
                gpu_gen="RX7900XTX",
                vendor="amd",
            ),
        ]
        for gpu in expected:
            self.assertIn(gpu, result)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_generic_fallback(self, mock_run_cli):
        """Test AMD GPU info parsing with generic fallback for unknown models."""
        mock_run_cli.return_value = "GPU[0]: AMD Radeon RX 6800 XT"

        result = self.local_node_info.get_amd_gpu_info()
        # Should fall back to extracting "6800" as the model
        expected = [GPUInfo(gpu_gen="6800", gpu_count=1, vendor="amd", partition=None)]
        self.assertEqual(result, expected)

    @patch("clusterscope.cluster_info.run_cli")
    def test_get_amd_gpu_info_no_gpu_lines(self, mock_run_cli):
        """Test AMD GPU info parsing with no GPU lines."""
        mock_run_cli.return_value = "Some other output\nNo GPU information here"

        result = self.local_node_info.get_amd_gpu_info()
        self.assertEqual(result, [])

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
        nvidia_return = [
            GPUInfo(
                gpu_count=2,
                gpu_gen="A100",
                vendor="nvidia",
            )
        ]
        mock_nvidia_info.return_value = nvidia_return
        amd_return = [
            GPUInfo(
                gpu_count=4,
                gpu_gen="MI300X",
                vendor="amd",
            )
        ]
        mock_amd_info.return_value = amd_return

        result = self.local_node_info.get_gpu_generation_and_count()
        expected = nvidia_return + amd_return
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
        self.assertEqual(result, [])
        mock_logging.assert_called_with(
            "No GPUs found or unable to retrieve GPU information"
        )

    @patch.object(LocalNodeInfo, "get_gpu_generation_and_count")
    def test_has_gpu_type_true(self, mock_get_gpu_info):
        """Test has_gpu_type returns True for available GPU types."""
        mock_get_gpu_info.return_value = [
            GPUInfo(
                gpu_count=2,
                gpu_gen="A100",
                vendor="nvidia",
            ),
            GPUInfo(
                gpu_count=4,
                gpu_gen="MI300X",
                vendor="amd",
            ),
        ]

        self.assertTrue(self.local_node_info.has_gpu_type("A100"))
        self.assertTrue(self.local_node_info.has_gpu_type("MI300X"))
        self.assertTrue(self.local_node_info.has_gpu_type("a100"))  # Case insensitive
        self.assertTrue(self.local_node_info.has_gpu_type("mi300x"))  # Case insensitive

    @patch.object(LocalNodeInfo, "get_gpu_generation_and_count")
    def test_has_gpu_type_false(self, mock_get_gpu_info):
        """Test has_gpu_type returns False for unavailable GPU types."""
        mock_get_gpu_info.return_value = [
            GPUInfo(
                gpu_count=2,
                gpu_gen="A100",
                vendor="nvidia",
            ),
            GPUInfo(
                gpu_count=4,
                gpu_gen="MI300X",
                vendor="amd",
            ),
        ]

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
        mock_cpus.return_value = [CPUInfo(cpu_count=192)]
        mock_mem.return_value = [MemInfo(mem_total_MB=1843200, mem_total_GB=1800)]

        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition", gpus_per_task=1
        )

        self.assertEqual(result.cpus_per_task, 24)  # 192/8 = 24
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
        mock_cpus.return_value = [CPUInfo(cpu_count=192)]
        mock_mem.return_value = [MemInfo(mem_total_MB=1843200, mem_total_GB=1800)]

        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition", gpus_per_task=4
        )

        self.assertEqual(result.cpus_per_task, 96)  # 192/8*4 = 96
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
        mock_cpus.return_value = [CPUInfo(cpu_count=192)]
        mock_mem.return_value = [MemInfo(mem_total_MB=1843200, mem_total_GB=1800)]

        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition", gpus_per_task=8
        )

        self.assertEqual(result.cpus_per_task, 192)  # All CPUs
        self.assertEqual(result.memory, "1800G")  # All memory: 1843200/1024 = 1800GB
        self.assertEqual(result.tasks_per_node, 1)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_4gpu_node_configuration(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test get_task_resource_requirements on a 4-GPU node configuration."""
        mock_total_gpus.return_value = 4
        mock_cpus.return_value = [CPUInfo(cpu_count=64)]
        mock_mem.return_value = [MemInfo(mem_total_MB=524288, mem_total_GB=512)]

        # Test 1 GPU on 4-GPU node
        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition", gpus_per_task=1
        )
        self.assertEqual(result.cpus_per_task, 16)  # 64/4 = 16
        self.assertEqual(result.memory, "128G")  # 524288/4/1024 = 128GB

        # Test full 4-GPU node
        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition", gpus_per_task=4
        )
        self.assertEqual(result.cpus_per_task, 64)  # All CPUs
        self.assertEqual(result.memory, "512G")  # All memory

        """Test get_task_resource_requirements with all 8 GPUs (full node)."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = [CPUInfo(cpu_count=192)]
        mock_mem.return_value = [MemInfo(mem_total_MB=1843200, mem_total_GB=1800)]

        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition", gpus_per_task=8
        )

        self.assertEqual(result.cpus_per_task, 192)  # All CPUs
        self.assertEqual(result.memory, "1800G")  # All memory: 1843200/1024 = 1800GB
        self.assertEqual(result.tasks_per_node, 1)

    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_full_cpu_node_configuration(
        self, mock_mem, mock_cpus
    ):
        """Test get_task_resource_requirements on a 192-CPU node configuration."""
        mock_cpus.return_value = [CPUInfo(cpu_count=192)]
        mock_mem.return_value = [MemInfo(mem_total_MB=524288, mem_total_GB=512)]

        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition",
            gpus_per_task=0,
            cpus_per_task=192,
            tasks_per_node=1,
        )
        self.assertEqual(result.cpus_per_task, 192)
        self.assertEqual(result.memory, "512G")

    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_96cpu_node_configuration(
        self, mock_mem, mock_cpus
    ):
        """Test get_task_resource_requirements on a 192-CPU node configuration."""
        mock_cpus.return_value = [CPUInfo(cpu_count=192)]
        mock_mem.return_value = [MemInfo(mem_total_MB=524288, mem_total_GB=512)]

        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition",
            gpus_per_task=0,
            cpus_per_task=96,
            tasks_per_node=1,
        )
        self.assertEqual(result.cpus_per_task, 96)
        self.assertEqual(result.memory, "256G")

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_getResRequirements_with_multiple_tasks_per_node(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test getResRequirements with multiple tasks per node."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = [CPUInfo(cpu_count=192)]
        mock_mem.return_value = [MemInfo(mem_total_MB=1843200, mem_total_GB=1800)]

        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition", gpus_per_task=4, tasks_per_node=2
        )

        self.assertEqual(result.cpus_per_task, 48)  # (192/8*4)/2 = 48 per task
        self.assertEqual(result.memory, "1800G")
        self.assertEqual(result.tasks_per_node, 2)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_memory_terabyte_format(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test get_task_resource_requirements returns TB format for very large memory."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = [CPUInfo(cpu_count=192)]
        mock_mem.return_value = [MemInfo(mem_total_MB=8388608, mem_total_GB=8192)]

        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition", gpus_per_task=8
        )

        self.assertEqual(result.memory, "8192G")  # 8388608/1024 = 8192GB

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    @patch.object(UnifiedInfo, "get_cpus_per_node")
    @patch.object(UnifiedInfo, "get_mem_per_node_MB")
    def test_get_task_resource_requirements_cpu_rounding_up(
        self, mock_mem, mock_cpus, mock_total_gpus
    ):
        """Test get_task_resource_requirements rounds up CPU cores when fractional."""
        mock_total_gpus.return_value = 8
        mock_cpus.return_value = [CPUInfo(cpu_count=191)]
        mock_mem.return_value = [MemInfo(mem_total_MB=1843200, mem_total_GB=1800)]

        result = self.unified_info.get_task_resource_requirements(
            partition="test_partition", gpus_per_task=1
        )

        # 191/8 = 23.875, should round down to 23
        self.assertEqual(result.cpus_per_task, 23)

    @patch.object(UnifiedInfo, "get_total_gpus_per_node")
    def test_getResRequirements_invalid_tasks_per_node(self, mock_total_gpus):
        """Test getResRequirements raises ValueError for invalid tasks_per_node."""
        mock_total_gpus.return_value = 8

        with self.assertRaises(ValueError) as context:
            self.unified_info.get_task_resource_requirements(
                partition="test_partition", gpus_per_task=1, tasks_per_node=0
            )
        self.assertIn("tasks_per_node must be at least 1", str(context.exception))

    @patch.object(UnifiedInfo, "get_gpu_generation_and_count")
    def test_get_total_gpus_per_node_with_gpus(self, mock_gpu_info):
        """Test get_total_gpus_per_node with actual GPU detection."""
        mock_gpu_info.return_value = [
            GPUInfo(
                gpu_count=4,
                gpu_gen="a100",
                vendor="nvidia",
            ),
            GPUInfo(
                gpu_count=4,
                gpu_gen="v100",
                vendor="nvidia",
            ),
        ]

        result = self.unified_info.get_total_gpus_per_node()
        self.assertEqual(result, 8)  # 4 + 4 = 8

    @patch.object(UnifiedInfo, "get_gpu_generation_and_count")
    def test_get_total_gpus_per_node_no_gpus_detected(self, mock_gpu_info):
        """Test get_total_gpus_per_node defaults to 8 when no GPUs detected."""
        mock_gpu_info.return_value = []

        result = self.unified_info.get_total_gpus_per_node()
        self.assertEqual(result, 8)  # Default fallback

    @patch.object(UnifiedInfo, "get_gpu_generation_and_count")
    def test_get_total_gpus_per_node_single_gpu_type(self, mock_gpu_info):
        """Test get_total_gpus_per_node with single GPU type."""
        mock_gpu_info.return_value = [
            GPUInfo(
                gpu_count=8,
                gpu_gen="a100",
                vendor="nvidia",
            ),
        ]

        result = self.unified_info.get_total_gpus_per_node()
        self.assertEqual(result, 8)

    @patch.object(UnifiedInfo, "get_gpu_generation_and_count")
    def test_get_total_gpus_per_node_mixed_gpu_types(self, mock_gpu_info):
        """Test get_total_gpus_per_node with mixed GPU types."""
        mock_gpu_info.return_value = [
            GPUInfo(
                gpu_count=2,
                gpu_gen="a100",
                vendor="nvidia",
            ),
            GPUInfo(
                gpu_count=4,
                gpu_gen="v100",
                vendor="nvidia",
            ),
            GPUInfo(
                gpu_count=2,
                gpu_gen="p100",
                vendor="nvidia",
            ),
        ]

        result = self.unified_info.get_total_gpus_per_node()
        self.assertEqual(result, 8)  # 2 + 4 + 2 = 8

    def test_resource_shape_namedtuple(self):
        """Test ResourceShape NamedTuple structure."""
        resource = ResourceShape(
            slurm_partition="test",
            gpus_per_task=4,
            cpus_per_task=24,
            memory="225G",
            tasks_per_node=1,
            nodes=1,
        )

        # Test that it's immutable (characteristic of NamedTuple)
        with self.assertRaises(AttributeError):
            resource.cpus_per_task = 48


if __name__ == "__main__":
    unittest.main()
