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
    run_cli,
    SlurmClusterInfo,
    UnifiedInfo,
)


class TestUnifiedInfo(unittest.TestCase):

    def test_get_cluster_name(self):
        unified_info = UnifiedInfo()
        unified_info.is_slurm_cluster = False
        self.assertEqual(unified_info.get_cluster_name(), "local-node")

    def test_get_gpu_generation_and_count(self):
        unified_info = UnifiedInfo()
        unified_info.is_slurm_cluster = False
        unified_info.has_nvidia_gpus = False
        self.assertEqual(unified_info.get_gpu_generation_and_count(), {})


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
    def test_get_mem_per_node_MB(self, mock_run):
        # Mock successful cluster name retrieval
        mock_run.return_value = MagicMock(stdout="123456+", returncode=0)
        self.assertEqual(self.cluster_info.get_mem_per_node_MB(), 123456)

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


if __name__ == "__main__":
    unittest.main()
