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

    @patch("subprocess.check_output", return_value="1234")
    def test_get_cpu_count(self, mock_run):
        self.assertEqual(self.linux_info.get_cpu_count(), 1234)

    @patch(
        "subprocess.check_output",
        return_value="               total        used\nMem:     12345    123\n",
    )
    def test_get_mem_per_node(self, mock_run):
        self.assertEqual(self.linux_info.get_mem(), 12345)


class TestDarwinInfo(unittest.TestCase):
    def setUp(self):
        self.darwin_info = DarwinInfo()

    @patch("subprocess.check_output", return_value="10")
    def test_get_cpu_count(self, mock_run):
        self.assertEqual(self.darwin_info.get_cpu_count(), 10)

    @patch(
        "subprocess.check_output",
        return_value="34359738368",
    )
    def test_get_mem_per_node(self, mock_run):
        self.assertEqual(self.darwin_info.get_mem(), 32768)


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
    def test_get_mem_per_node(self, mock_run):
        # Mock successful cluster name retrieval
        mock_run.return_value = MagicMock(stdout="123456+", returncode=0)
        self.assertEqual(self.cluster_info.get_mem_per_node(), 123456)

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
