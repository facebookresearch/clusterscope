# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from clusterscope.cli import cli
from clusterscope.cluster_info import ResourceShape, UnifiedInfo


class TestCLIArgumentCombinations(unittest.TestCase):
    """Test various CLI argument combinations to validate for unexpected behaviors."""

    def setUp(self):
        self.runner = CliRunner()

    def test_version_command(self):
        """Test version command works independently."""
        with patch("clusterscope.cli.get_version", return_value="1.0.0"):
            result = self.runner.invoke(cli, ["version"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("clusterscope version 1.0.0", result.output)

    def test_info_command_without_partition(self):
        """Test info command without partition argument."""
        with (
            patch.object(UnifiedInfo, "get_cluster_name", return_value="test-cluster"),
            patch.object(UnifiedInfo, "get_slurm_version", return_value="20.11.0"),
        ):
            result = self.runner.invoke(cli, ["info"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Cluster Name: test-cluster", result.output)
            self.assertIn("Slurm Version: 20.11.0", result.output)

    def test_info_command_with_partition(self):
        """Test info command with partition argument."""
        with (
            patch.object(UnifiedInfo, "get_cluster_name", return_value="test-cluster"),
            patch.object(UnifiedInfo, "get_slurm_version", return_value="20.11.0"),
        ):
            result = self.runner.invoke(cli, ["info", "--partition", "gpu"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Cluster Name: test-cluster", result.output)

    def test_cpus_command_without_partition(self):
        """Test cpus command without partition argument."""
        with patch.object(UnifiedInfo, "get_cpus_per_node", return_value=128):
            result = self.runner.invoke(cli, ["cpus"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("CPU counts per node:", result.output)
            self.assertIn("128", result.output)

    def test_cpus_command_with_partition(self):
        """Test cpus command with partition argument."""
        with patch.object(UnifiedInfo, "get_cpus_per_node", return_value=128):
            result = self.runner.invoke(cli, ["cpus", "--partition", "compute"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("CPU counts per node:", result.output)

    def test_mem_command_default_options(self):
        """Test memory command with default options (GB unit, no flags)."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_mem_per_node_MB.return_value = 262144  # 256GB in MB
        mock_unified_info.get_cpu_memory_usage_percentage.return_value = 95.0

        with (
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
            patch(
                "clusterscope.cli.get_cpu_memory_usage_percentage", return_value=95.0
            ),
        ):
            result = self.runner.invoke(cli, ["mem"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Total CPU memory per node: 256.00 GB", result.output)
            self.assertIn("Available for applications (95.0%)", result.output)

    def test_mem_command_mb_unit(self):
        """Test memory command with MB unit."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_mem_per_node_MB.return_value = 262144

        with (
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
            patch(
                "clusterscope.cli.get_cpu_memory_usage_percentage", return_value=95.0
            ),
        ):
            result = self.runner.invoke(cli, ["mem", "--unit", "MB"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Total CPU memory per node: 262144 MB", result.output)

    def test_mem_command_with_partition(self):
        """Test memory command with partition argument."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_mem_per_node_MB.return_value = 131072  # 128GB in MB

        with (
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
            patch(
                "clusterscope.cli.get_cpu_memory_usage_percentage", return_value=95.0
            ),
        ):
            result = self.runner.invoke(
                cli, ["mem", "--partition", "highmem", "--unit", "GB"]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Total CPU memory per node: 128.00 GB", result.output)

    def test_mem_command_detailed_flag(self):
        """Test memory command with detailed flag."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_cpu_memory_info.return_value = {
            "total_cpu_mb": 262144,
            "total_cpu_gb": 256.0,
            "available_cpu_mb": 248832,
            "available_cpu_gb": 243.0,
            "percentage": 95.0,
        }

        with (
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
            patch(
                "clusterscope.cli.get_cpu_memory_usage_percentage", return_value=95.0
            ),
        ):
            result = self.runner.invoke(cli, ["mem", "--detailed"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("CPU Memory (RAM) Information:", result.output)
            self.assertIn("Total CPU Memory: 262144 MB (256.0 GB)", result.output)
            self.assertIn("Available for Apps (95.0%)", result.output)

    def test_mem_command_available_only_flag_mb(self):
        """Test memory command with available-only flag in MB."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_MB.return_value = 248832

        with (
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
            patch(
                "clusterscope.cli.get_cpu_memory_usage_percentage", return_value=95.0
            ),
        ):
            result = self.runner.invoke(
                cli, ["mem", "--available-only", "--unit", "MB"]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn(
                "Available CPU memory for applications: 248832 MB", result.output
            )

    def test_mem_command_available_only_flag_gb(self):
        """Test memory command with available-only flag in GB."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_GB.return_value = 243.0

        with (
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
            patch(
                "clusterscope.cli.get_cpu_memory_usage_percentage", return_value=95.0
            ),
        ):
            result = self.runner.invoke(
                cli, ["mem", "--available-only", "--unit", "GB"]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn(
                "Available CPU memory for applications: 243.00 GB", result.output
            )

    def test_mem_command_conflicting_flags(self):
        """Test memory command with both detailed and available-only flags."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_cpu_memory_info.return_value = {
            "total_cpu_mb": 262144,
            "total_cpu_gb": 256.0,
            "available_cpu_mb": 248832,
            "available_cpu_gb": 243.0,
            "percentage": 95.0,
        }

        with (
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
            patch(
                "clusterscope.cli.get_cpu_memory_usage_percentage", return_value=95.0
            ),
        ):
            result = self.runner.invoke(cli, ["mem", "--detailed", "--available-only"])
            self.assertEqual(result.exit_code, 0)
            # Detailed should take precedence based on the code order
            self.assertIn("CPU Memory (RAM) Information:", result.output)

    def test_gpus_command_default(self):
        """Test gpus command with default behavior."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_gpu_vendor.return_value = "nvidia"
        mock_unified_info.get_gpu_generation_and_count.return_value = {
            "A100": 4,
            "V100": 2,
        }

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            result = self.runner.invoke(cli, ["gpus"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("GPU vendor: nvidia", result.output)
            self.assertIn("GPU information:", result.output)
            self.assertIn("A100: 4", result.output)
            self.assertIn("V100: 2", result.output)

    def test_gpus_command_generations_flag(self):
        """Test gpus command with generations flag."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_gpu_generation_and_count.return_value = {
            "A100": 4,
            "V100": 2,
        }

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            result = self.runner.invoke(cli, ["gpus", "--generations"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("GPU generations available:", result.output)
            self.assertIn("- A100", result.output)
            self.assertIn("- V100", result.output)

    def test_gpus_command_counts_flag(self):
        """Test gpus command with counts flag."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_gpu_generation_and_count.return_value = {
            "A100": 4,
            "V100": 2,
        }

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            result = self.runner.invoke(cli, ["gpus", "--counts"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("GPU counts by type:", result.output)
            self.assertIn("A100: 4", result.output)
            self.assertIn("V100: 2", result.output)

    def test_gpus_command_vendor_flag(self):
        """Test gpus command with vendor flag."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_gpu_vendor.return_value = "amd"

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            result = self.runner.invoke(cli, ["gpus", "--vendor"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Primary GPU vendor: amd", result.output)

    def test_gpus_command_no_gpus_found(self):
        """Test gpus command when no GPUs are found."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_gpu_vendor.return_value = "none"
        mock_unified_info.get_gpu_generation_and_count.return_value = {}

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            result = self.runner.invoke(cli, ["gpus"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("GPU vendor: none", result.output)
            self.assertIn("No GPUs found", result.output)

    def test_gpus_command_multiple_flags(self):
        """Test gpus command with multiple flags (generations and vendor)."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_gpu_vendor.return_value = "nvidia"
        mock_unified_info.get_gpu_generation_and_count.return_value = {"A100": 4}

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            result = self.runner.invoke(cli, ["gpus", "--vendor", "--generations"])
            self.assertEqual(result.exit_code, 0)
            # Should show vendor info based on the order in the conditional logic
            self.assertIn("Primary GPU vendor: nvidia", result.output)

    def test_check_gpu_command_available(self):
        """Test check-gpu command for available GPU type."""
        mock_unified_info = MagicMock()
        mock_unified_info.has_gpu_type.return_value = True

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            result = self.runner.invoke(cli, ["check-gpu", "A100"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("GPU type A100 is available in the cluster.", result.output)

    def test_check_gpu_command_not_available(self):
        """Test check-gpu command for unavailable GPU type."""
        mock_unified_info = MagicMock()
        mock_unified_info.has_gpu_type.return_value = False

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            result = self.runner.invoke(cli, ["check-gpu", "H100"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn(
                "GPU type H100 is NOT available in the cluster.", result.output
            )

    def test_check_gpu_command_with_partition(self):
        """Test check-gpu command with partition argument."""
        mock_unified_info = MagicMock()
        mock_unified_info.has_gpu_type.return_value = True

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            result = self.runner.invoke(
                cli, ["check-gpu", "A100", "--partition", "gpu"]
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("GPU type A100 is available in the cluster.", result.output)

    def test_aws_command_is_aws_cluster(self):
        """Test aws command when running on AWS cluster."""
        mock_aws_info = MagicMock()
        mock_aws_info.is_aws_cluster.return_value = True
        mock_aws_info.get_aws_nccl_settings.return_value = {
            "FI_PROVIDER": "efa",
            "NCCL_PROTO": "simple",
        }

        with patch("clusterscope.cli.AWSClusterInfo", return_value=mock_aws_info):
            result = self.runner.invoke(cli, ["aws"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("This is an AWS cluster.", result.output)
            self.assertIn("Recommended NCCL settings:", result.output)
            self.assertIn("FI_PROVIDER", result.output)

    def test_aws_command_not_aws_cluster(self):
        """Test aws command when not running on AWS cluster."""
        mock_aws_info = MagicMock()
        mock_aws_info.is_aws_cluster.return_value = False

        with patch("clusterscope.cli.AWSClusterInfo", return_value=mock_aws_info):
            result = self.runner.invoke(cli, ["aws"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("This is NOT an AWS cluster.", result.output)

    def test_job_gen_task_slurm_json_format(self):
        """Test job-gen task slurm command with JSON format."""
        mock_resource_shape = ResourceShape(
            cpu_cores=24, memory="256G", tasks_per_node=1
        )
        mock_unified_info = MagicMock()
        mock_unified_info.get_task_resource_requirements.return_value = (
            mock_resource_shape
        )

        mock_partition_info = [MagicMock(name="gpu")]

        with (
            patch(
                "clusterscope.cli.get_partition_info", return_value=mock_partition_info
            ),
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
        ):

            # Mock the to_json method
            mock_resource_shape.to_json = MagicMock(
                return_value='{"cpu_cores": 24, "memory": "256G", "tasks_per_node": 1}'
            )

            result = self.runner.invoke(
                cli,
                [
                    "job-gen",
                    "task",
                    "slurm",
                    "--num-gpus",
                    "2",
                    "--partition",
                    "gpu",
                    "--format",
                    "json",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("cpu_cores", result.output)

    def test_job_gen_task_slurm_sbatch_format(self):
        """Test job-gen task slurm command with sbatch format."""
        mock_resource_shape = ResourceShape(
            cpu_cores=24, memory="256G", tasks_per_node=1
        )
        mock_unified_info = MagicMock()
        mock_unified_info.get_task_resource_requirements.return_value = (
            mock_resource_shape
        )

        mock_partition_info = [MagicMock(name="gpu")]

        with (
            patch(
                "clusterscope.cli.get_partition_info", return_value=mock_partition_info
            ),
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
        ):

            # Mock the to_sbatch method
            mock_resource_shape.to_sbatch = MagicMock(
                return_value="#SBATCH --cpus-per-task=24\n#SBATCH --mem=256G"
            )

            result = self.runner.invoke(
                cli,
                [
                    "job-gen",
                    "task",
                    "slurm",
                    "--num-gpus",
                    "4",
                    "--partition",
                    "gpu",
                    "--format",
                    "sbatch",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("#SBATCH", result.output)

    def test_job_gen_task_slurm_invalid_partition(self):
        """Test job-gen task slurm command with invalid partition."""
        mock_partition_info = [MagicMock(name="gpu"), MagicMock(name="cpu")]

        with patch(
            "clusterscope.cli.get_partition_info", return_value=mock_partition_info
        ):
            result = self.runner.invoke(
                cli,
                [
                    "job-gen",
                    "task",
                    "slurm",
                    "--num-gpus",
                    "2",
                    "--partition",
                    "invalid",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Partition invalid not found", str(result.output))

    def test_job_gen_task_slurm_with_tasks_per_node(self):
        """Test job-gen task slurm command with custom num-tasks-per-node."""
        mock_resource_shape = ResourceShape(
            cpu_cores=12, memory="256G", tasks_per_node=2
        )
        mock_unified_info = MagicMock()
        mock_unified_info.get_task_resource_requirements.return_value = (
            mock_resource_shape
        )

        mock_partition_info = [MagicMock(name="gpu")]

        with (
            patch(
                "clusterscope.cli.get_partition_info", return_value=mock_partition_info
            ),
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
        ):

            mock_resource_shape.to_json = MagicMock(
                return_value='{"cpu_cores": 12, "memory": "256G", "tasks_per_node": 2}'
            )

            result = self.runner.invoke(
                cli,
                [
                    "job-gen",
                    "task",
                    "slurm",
                    "--num-gpus",
                    "2",
                    "--partition",
                    "gpu",
                    "--num-tasks-per-node",
                    "2",
                    "--format",
                    "json",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            # Verify the method was called with correct parameters
            mock_unified_info.get_task_resource_requirements.assert_called_with(
                num_gpus=2, num_tasks_per_node=2
            )

    def test_job_gen_array_json_format(self):
        """Test job-gen array command with JSON format."""
        mock_resource_shape = ResourceShape(
            cpu_cores=24, memory="256G", tasks_per_node=1
        )
        mock_unified_info = MagicMock()
        mock_unified_info.get_array_job_requirements.return_value = mock_resource_shape

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            mock_resource_shape.to_json = MagicMock(
                return_value='{"cpu_cores": 24, "memory": "256G", "tasks_per_node": 1}'
            )

            result = self.runner.invoke(
                cli,
                ["job-gen", "array", "--num-gpus-per-task", "2", "--format", "json"],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("cpu_cores", result.output)

    def test_job_gen_array_srun_format(self):
        """Test job-gen array command with srun format."""
        mock_resource_shape = ResourceShape(
            cpu_cores=24, memory="256G", tasks_per_node=1
        )
        mock_unified_info = MagicMock()
        mock_unified_info.get_array_job_requirements.return_value = mock_resource_shape

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            mock_resource_shape.to_srun = MagicMock(
                return_value="srun --cpus-per-task=24 --mem=256G"
            )

            result = self.runner.invoke(
                cli,
                ["job-gen", "array", "--num-gpus-per-task", "1", "--format", "srun"],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("srun", result.output)

    def test_job_gen_array_with_partition(self):
        """Test job-gen array command with partition argument."""
        mock_resource_shape = ResourceShape(
            cpu_cores=24, memory="256G", tasks_per_node=1
        )
        mock_unified_info = MagicMock()
        mock_unified_info.get_array_job_requirements.return_value = mock_resource_shape

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            mock_resource_shape.to_submitit = MagicMock(return_value="submitit config")

            result = self.runner.invoke(
                cli,
                [
                    "job-gen",
                    "array",
                    "--num-gpus-per-task",
                    "2",
                    "--partition",
                    "highmem",
                    "--format",
                    "submitit",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("submitit", result.output)

    def test_invalid_command(self):
        """Test invalid command returns appropriate error."""
        result = self.runner.invoke(cli, ["invalid-command"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No such command", result.output)

    def test_missing_required_arguments(self):
        """Test commands with missing required arguments."""
        # check-gpu without GPU_TYPE argument
        result = self.runner.invoke(cli, ["check-gpu"])
        self.assertNotEqual(result.exit_code, 0)

        # job-gen task slurm without required arguments
        result = self.runner.invoke(cli, ["job-gen", "task", "slurm"])
        self.assertNotEqual(result.exit_code, 0)

        # job-gen array without required arguments
        result = self.runner.invoke(cli, ["job-gen", "array"])
        self.assertNotEqual(result.exit_code, 0)

    def test_invalid_option_values(self):
        """Test commands with invalid option values."""
        mock_partition_info = [MagicMock(name="gpu")]

        with patch(
            "clusterscope.cli.get_partition_info", return_value=mock_partition_info
        ):
            # Invalid format for job-gen task slurm
            result = self.runner.invoke(
                cli,
                [
                    "job-gen",
                    "task",
                    "slurm",
                    "--num-gpus",
                    "2",
                    "--partition",
                    "gpu",
                    "--format",
                    "invalid-format",
                ],
            )
            self.assertNotEqual(result.exit_code, 0)

            # Invalid unit for mem command
            result = self.runner.invoke(cli, ["mem", "--unit", "TB"])
            self.assertNotEqual(result.exit_code, 0)

    def test_help_messages(self):
        """Test help messages for various commands."""
        # Main help
        result = self.runner.invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Command-line tool to query Slurm cluster information", result.output
        )

        # Command-specific help
        result = self.runner.invoke(cli, ["mem", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Show CPU memory (RAM) information per node", result.output)

        # Subcommand help
        result = self.runner.invoke(cli, ["job-gen", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "Generate job requirements for different job types", result.output
        )

    def test_edge_case_argument_combinations(self):
        """Test edge case argument combinations that might cause issues."""
        mock_unified_info = MagicMock()

        # Memory command with all possible flags and options
        mock_unified_info.get_cpu_memory_info.return_value = {
            "total_cpu_mb": 262144,
            "total_cpu_gb": 256.0,
            "available_cpu_mb": 248832,
            "available_cpu_gb": 243.0,
            "percentage": 95.0,
        }

        with (
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
            patch(
                "clusterscope.cli.get_cpu_memory_usage_percentage", return_value=95.0
            ),
        ):

            # This should work but detailed flag takes precedence
            result = self.runner.invoke(
                cli,
                [
                    "mem",
                    "--partition",
                    "test",
                    "--unit",
                    "MB",
                    "--detailed",
                    "--available-only",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertIn("CPU Memory (RAM) Information:", result.output)

    def test_extreme_values(self):
        """Test commands with extreme or boundary values."""
        mock_partition_info = [MagicMock(name="gpu")]
        mock_resource_shape = ResourceShape(cpu_cores=1, memory="1G", tasks_per_node=1)
        mock_unified_info = MagicMock()
        mock_unified_info.get_task_resource_requirements.return_value = (
            mock_resource_shape
        )

        with (
            patch(
                "clusterscope.cli.get_partition_info", return_value=mock_partition_info
            ),
            patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info),
        ):

            mock_resource_shape.to_json = MagicMock(
                return_value='{"cpu_cores": 1, "memory": "1G", "tasks_per_node": 1}'
            )

            # Minimum values
            result = self.runner.invoke(
                cli,
                [
                    "job-gen",
                    "task",
                    "slurm",
                    "--num-gpus",
                    "1",
                    "--partition",
                    "gpu",
                    "--num-tasks-per-node",
                    "1",
                ],
            )
            self.assertEqual(result.exit_code, 0)

    def test_concurrent_flag_combinations(self):
        """Test multiple boolean flags used together."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_gpu_vendor.return_value = "nvidia"
        mock_unified_info.get_gpu_generation_and_count.return_value = {"A100": 4}

        with patch("clusterscope.cli.UnifiedInfo", return_value=mock_unified_info):
            # All GPU flags together - should prioritize based on if-elif order
            result = self.runner.invoke(
                cli, ["gpus", "--vendor", "--counts", "--generations"]
            )
            self.assertEqual(result.exit_code, 0)
            # Vendor should be shown since it's first in the if-elif chain
            self.assertIn("Primary GPU vendor: nvidia", result.output)


if __name__ == "__main__":
    unittest.main()
