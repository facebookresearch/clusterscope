import subprocess
import unittest
from unittest.mock import MagicMock, patch

from cluster_info import ClusterInfo


class TestClusterInfo(unittest.TestCase):
    def setUp(self):
        self.cluster_info = ClusterInfo()

    @patch("subprocess.run")
    def test_get_cluster_name(self, mock_run):
        # Mock successful cluster name retrieval
        mock_run.return_value = MagicMock(
            stdout="ClusterName=test_cluster\nOther=value", returncode=0
        )
        self.assertEqual(self.cluster_info.get_cluster_name(), "test_cluster")

        # Mock failed command
        mock_run.side_effect = subprocess.SubprocessError()
        with self.assertRaises(RuntimeError):
            self.cluster_info.get_cluster_name()

    @patch("subprocess.run")
    def test_has_gpu_type(self, mock_run):
        # Mock node with A100 GPUs
        mock_run.return_value = MagicMock(
            stdout="NodeName=node1 Gres=gpu:a100:4", returncode=0
        )
        self.assertTrue(self.cluster_info.has_gpu_type("A100"))
        self.assertFalse(self.cluster_info.has_gpu_type("V100"))

    @patch("subprocess.run")
    def test_is_aws_cluster(self, mock_run):
        # Mock AWS environment
        mock_run.return_value = MagicMock(stdout="amazon_ec2", returncode=0)
        self.assertTrue(self.cluster_info.is_aws_cluster())

        # Mock non-AWS environment
        mock_run.return_value = MagicMock(stdout="other_system", returncode=0)
        self.assertFalse(self.cluster_info.is_aws_cluster())

    def test_get_aws_nccl_settings(self):
        # Test with AWS cluster
        with patch.object(ClusterInfo, "is_aws_cluster", return_value=True):
            settings = self.cluster_info.get_aws_nccl_settings()
            self.assertIn("FI_PROVIDER", settings)
            self.assertEqual(settings["FI_PROVIDER"], "efa")

        # Test with non-AWS cluster
        with patch.object(ClusterInfo, "is_aws_cluster", return_value=False):
            settings = self.cluster_info.get_aws_nccl_settings()
            self.assertEqual(settings, {})


if __name__ == "__main__":
    unittest.main()
