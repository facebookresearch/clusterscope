# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import clusterscope.lib as cs_lib


class TestCPUMemoryConfiguration(unittest.TestCase):
    """Test cases for CPU memory configuration constants and functions."""

    def setUp(self):
        """Reset CPU memory percentage to default before each test."""
        cs_lib.CPU_MEMORY_USAGE_PERCENTAGE = 95.0

    def test_default_cpu_memory_usage_percentage(self):
        """Test that the default CPU memory usage percentage is 95%."""
        self.assertEqual(cs_lib.CPU_MEMORY_USAGE_PERCENTAGE, 95.0)
        self.assertEqual(cs_lib.get_cpu_memory_usage_percentage(), 95.0)

    def test_set_cpu_memory_usage_percentage_valid_values(self):
        """Test setting valid CPU memory usage percentages."""
        test_percentages = [1.0, 50.0, 75.0, 80.0, 85.0, 90.0, 95.0, 99.0, 100.0]

        for percentage in test_percentages:
            cs_lib.set_cpu_memory_usage_percentage(percentage)
            self.assertEqual(cs_lib.get_cpu_memory_usage_percentage(), percentage)
            self.assertEqual(cs_lib.CPU_MEMORY_USAGE_PERCENTAGE, percentage)

    def test_set_cpu_memory_usage_percentage_invalid_values(self):
        """Test that invalid percentages raise ValueError."""
        invalid_percentages = [0.0, -5.0, -10.0, 150.0, 200.0, 0.5]

        for invalid_pct in invalid_percentages:
            with self.assertRaises(ValueError) as context:
                cs_lib.set_cpu_memory_usage_percentage(invalid_pct)
            self.assertIn(
                "Percentage must be between 1.0 and 100.0", str(context.exception)
            )

    def test_set_cpu_memory_usage_percentage_edge_cases(self):
        """Test edge cases for setting CPU memory usage percentage."""
        # Test minimum valid value
        cs_lib.set_cpu_memory_usage_percentage(1.0)
        self.assertEqual(cs_lib.get_cpu_memory_usage_percentage(), 1.0)

        # Test maximum valid value
        cs_lib.set_cpu_memory_usage_percentage(100.0)
        self.assertEqual(cs_lib.get_cpu_memory_usage_percentage(), 100.0)

    def test_configuration_persistence(self):
        """Test that configuration changes persist across function calls."""
        # Set to 80%
        cs_lib.set_cpu_memory_usage_percentage(80.0)

        # Verify it persists
        self.assertEqual(cs_lib.get_cpu_memory_usage_percentage(), 80.0)

        # Change to 90%
        cs_lib.set_cpu_memory_usage_percentage(90.0)

        # Verify change
        self.assertEqual(cs_lib.get_cpu_memory_usage_percentage(), 90.0)


class TestCPUMemoryLibraryFunctions(unittest.TestCase):
    """Test cases for CPU memory library functions."""

    def setUp(self):
        """Reset CPU memory percentage to default and clear unified info cache."""
        cs_lib.CPU_MEMORY_USAGE_PERCENTAGE = 95.0
        cs_lib._unified_info = None
        cs_lib._current_partition = None

    @patch("clusterscope.lib.get_unified_info")
    def test_available_cpu_memory_MB_default_percentage(self, mock_get_unified_info):
        """Test available_cpu_memory_MB with default percentage."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_MB.return_value = 95000
        mock_get_unified_info.return_value = mock_unified_info

        result = cs_lib.available_cpu_memory_MB()

        self.assertEqual(result, 95000)
        mock_unified_info.get_available_cpu_memory_MB.assert_called_once_with(95.0)
        mock_get_unified_info.assert_called_once_with(None)

    @patch("clusterscope.lib.get_unified_info")
    def test_available_cpu_memory_MB_custom_percentage(self, mock_get_unified_info):
        """Test available_cpu_memory_MB with custom percentage."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_MB.return_value = 80000
        mock_get_unified_info.return_value = mock_unified_info

        # Set custom percentage
        cs_lib.set_cpu_memory_usage_percentage(80.0)
        result = cs_lib.available_cpu_memory_MB()

        self.assertEqual(result, 80000)
        mock_unified_info.get_available_cpu_memory_MB.assert_called_once_with(80.0)

    @patch("clusterscope.lib.get_unified_info")
    def test_available_cpu_memory_MB_with_partition(self, mock_get_unified_info):
        """Test available_cpu_memory_MB with partition parameter."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_MB.return_value = 95000
        mock_get_unified_info.return_value = mock_unified_info

        result = cs_lib.available_cpu_memory_MB(partition="gpu_partition")

        self.assertEqual(result, 95000)
        mock_unified_info.get_available_cpu_memory_MB.assert_called_once_with(95.0)
        mock_get_unified_info.assert_called_once_with("gpu_partition")

    @patch("clusterscope.lib.get_unified_info")
    def test_available_cpu_memory_GB_default_percentage(self, mock_get_unified_info):
        """Test available_cpu_memory_GB with default percentage."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_GB.return_value = 92.77
        mock_get_unified_info.return_value = mock_unified_info

        result = cs_lib.available_cpu_memory_GB()

        self.assertEqual(result, 92.77)
        mock_unified_info.get_available_cpu_memory_GB.assert_called_once_with(95.0)
        mock_get_unified_info.assert_called_once_with(None)

    @patch("clusterscope.lib.get_unified_info")
    def test_available_cpu_memory_GB_custom_percentage(self, mock_get_unified_info):
        """Test available_cpu_memory_GB with custom percentage."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_GB.return_value = 78.12
        mock_get_unified_info.return_value = mock_unified_info

        # Set custom percentage
        cs_lib.set_cpu_memory_usage_percentage(85.0)
        result = cs_lib.available_cpu_memory_GB()

        self.assertEqual(result, 78.12)
        mock_unified_info.get_available_cpu_memory_GB.assert_called_once_with(85.0)

    @patch("clusterscope.lib.get_unified_info")
    def test_available_cpu_memory_GB_with_partition(self, mock_get_unified_info):
        """Test available_cpu_memory_GB with partition parameter."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_GB.return_value = 92.77
        mock_get_unified_info.return_value = mock_unified_info

        result = cs_lib.available_cpu_memory_GB(partition="high_mem")

        self.assertEqual(result, 92.77)
        mock_unified_info.get_available_cpu_memory_GB.assert_called_once_with(95.0)
        mock_get_unified_info.assert_called_once_with("high_mem")

    @patch("clusterscope.lib.get_unified_info")
    def test_cpu_memory_info_default_percentage(self, mock_get_unified_info):
        """Test cpu_memory_info with default percentage."""
        mock_unified_info = MagicMock()
        expected_info = {
            "total_cpu_mb": 100000,
            "total_cpu_gb": 97.66,
            "available_cpu_mb": 95000,
            "available_cpu_gb": 92.77,
            "percentage": 95.0,
        }
        mock_unified_info.get_cpu_memory_info.return_value = expected_info
        mock_get_unified_info.return_value = mock_unified_info

        result = cs_lib.cpu_memory_info()

        self.assertEqual(result, expected_info)
        mock_unified_info.get_cpu_memory_info.assert_called_once_with(95.0)
        mock_get_unified_info.assert_called_once_with(None)

    @patch("clusterscope.lib.get_unified_info")
    def test_cpu_memory_info_custom_percentage(self, mock_get_unified_info):
        """Test cpu_memory_info with custom percentage."""
        mock_unified_info = MagicMock()
        expected_info = {
            "total_cpu_mb": 100000,
            "total_cpu_gb": 97.66,
            "available_cpu_mb": 90000,
            "available_cpu_gb": 87.89,
            "percentage": 90.0,
        }
        mock_unified_info.get_cpu_memory_info.return_value = expected_info
        mock_get_unified_info.return_value = mock_unified_info

        # Set custom percentage
        cs_lib.set_cpu_memory_usage_percentage(90.0)
        result = cs_lib.cpu_memory_info()

        self.assertEqual(result, expected_info)
        mock_unified_info.get_cpu_memory_info.assert_called_once_with(90.0)

    @patch("clusterscope.lib.get_unified_info")
    def test_cpu_memory_info_with_partition(self, mock_get_unified_info):
        """Test cpu_memory_info with partition parameter."""
        mock_unified_info = MagicMock()
        expected_info = {
            "total_cpu_mb": 200000,
            "total_cpu_gb": 195.31,
            "available_cpu_mb": 190000,
            "available_cpu_gb": 185.55,
            "percentage": 95.0,
        }
        mock_unified_info.get_cpu_memory_info.return_value = expected_info
        mock_get_unified_info.return_value = mock_unified_info

        result = cs_lib.cpu_memory_info(partition="cpu_partition")

        self.assertEqual(result, expected_info)
        mock_unified_info.get_cpu_memory_info.assert_called_once_with(95.0)
        mock_get_unified_info.assert_called_once_with("cpu_partition")


class TestUnifiedInfoCaching(unittest.TestCase):
    """Test cases for UnifiedInfo instance caching with partition support."""

    def setUp(self):
        """Clear unified info cache before each test."""
        cs_lib._unified_info = None
        cs_lib._current_partition = None

    @patch("clusterscope.cluster_info.UnifiedInfo")
    def test_get_unified_info_creates_instance(self, mock_unified_info_class):
        """Test that get_unified_info creates a new UnifiedInfo instance."""
        mock_instance = MagicMock()
        mock_unified_info_class.return_value = mock_instance

        result = cs_lib.get_unified_info()

        self.assertEqual(result, mock_instance)
        mock_unified_info_class.assert_called_once_with(partition=None)

    @patch("clusterscope.cluster_info.UnifiedInfo")
    def test_get_unified_info_caches_instance(self, mock_unified_info_class):
        """Test that get_unified_info caches the UnifiedInfo instance."""
        mock_instance = MagicMock()
        mock_unified_info_class.return_value = mock_instance

        # First call should create instance
        result1 = cs_lib.get_unified_info()
        # Second call should return cached instance
        result2 = cs_lib.get_unified_info()

        self.assertEqual(result1, mock_instance)
        self.assertEqual(result2, mock_instance)
        # Should only create instance once
        mock_unified_info_class.assert_called_once_with(partition=None)

    @patch("clusterscope.cluster_info.UnifiedInfo")
    def test_get_unified_info_with_partition(self, mock_unified_info_class):
        """Test get_unified_info with partition parameter."""
        mock_instance = MagicMock()
        mock_unified_info_class.return_value = mock_instance

        result = cs_lib.get_unified_info(partition="gpu_partition")

        self.assertEqual(result, mock_instance)
        mock_unified_info_class.assert_called_once_with(partition="gpu_partition")

    @patch("clusterscope.cluster_info.UnifiedInfo")
    def test_get_unified_info_partition_change_creates_new_instance(
        self, mock_unified_info_class
    ):
        """Test that changing partition creates a new UnifiedInfo instance."""
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_unified_info_class.side_effect = [mock_instance1, mock_instance2]

        # First call with no partition
        result1 = cs_lib.get_unified_info()
        # Second call with partition should create new instance
        result2 = cs_lib.get_unified_info(partition="gpu_partition")

        self.assertEqual(result1, mock_instance1)
        self.assertEqual(result2, mock_instance2)
        # Should create two instances due to partition change
        self.assertEqual(mock_unified_info_class.call_count, 2)

    @patch("clusterscope.cluster_info.UnifiedInfo")
    def test_get_unified_info_same_partition_uses_cache(self, mock_unified_info_class):
        """Test that using same partition uses cached instance."""
        mock_instance = MagicMock()
        mock_unified_info_class.return_value = mock_instance

        # Multiple calls with same partition
        result1 = cs_lib.get_unified_info(partition="gpu_partition")
        result2 = cs_lib.get_unified_info(partition="gpu_partition")
        result3 = cs_lib.get_unified_info(partition="gpu_partition")

        self.assertEqual(result1, mock_instance)
        self.assertEqual(result2, mock_instance)
        self.assertEqual(result3, mock_instance)
        # Should only create instance once
        mock_unified_info_class.assert_called_once_with(partition="gpu_partition")


class TestCPUMemoryIntegration(unittest.TestCase):
    """Integration tests for CPU memory functionality."""

    def setUp(self):
        """Reset configuration before each test."""
        cs_lib.CPU_MEMORY_USAGE_PERCENTAGE = 95.0
        cs_lib._unified_info = None
        cs_lib._current_partition = None

    @patch("clusterscope.lib.get_unified_info")
    def test_cpu_memory_workflow_conservative_app(self, mock_get_unified_info):
        """Test complete workflow for a conservative application."""
        mock_unified_info = MagicMock()
        mock_get_unified_info.return_value = mock_unified_info

        # Configure mock to return consistent values
        mock_unified_info.get_available_cpu_memory_MB.return_value = 80000
        mock_unified_info.get_available_cpu_memory_GB.return_value = 78.12
        mock_unified_info.get_cpu_memory_info.return_value = {
            "total_cpu_mb": 100000,
            "total_cpu_gb": 97.66,
            "available_cpu_mb": 80000,
            "available_cpu_gb": 78.12,
            "percentage": 80.0,
        }

        # Conservative application sets 80% usage
        cs_lib.set_cpu_memory_usage_percentage(80.0)

        # Verify configuration
        self.assertEqual(cs_lib.get_cpu_memory_usage_percentage(), 80.0)

        # Get available memory
        available_mb = cs_lib.available_cpu_memory_MB()
        available_gb = cs_lib.available_cpu_memory_GB()
        memory_info = cs_lib.cpu_memory_info()

        # Verify results
        self.assertEqual(available_mb, 80000)
        self.assertEqual(available_gb, 78.12)
        self.assertEqual(memory_info["percentage"], 80.0)
        self.assertEqual(memory_info["available_cpu_mb"], 80000)

        # Verify all calls use the configured percentage
        mock_unified_info.get_available_cpu_memory_MB.assert_called_with(80.0)
        mock_unified_info.get_available_cpu_memory_GB.assert_called_with(80.0)
        mock_unified_info.get_cpu_memory_info.assert_called_with(80.0)

    @patch("clusterscope.lib.get_unified_info")
    def test_cpu_memory_workflow_with_partitions(self, mock_get_unified_info):
        """Test workflow with different partitions."""
        mock_unified_info = MagicMock()
        mock_get_unified_info.return_value = mock_unified_info

        # Configure different returns for different partitions
        mock_unified_info.get_available_cpu_memory_GB.side_effect = [95.0, 190.0, 380.0]

        # Set percentage
        cs_lib.set_cpu_memory_usage_percentage(90.0)

        # Query different partitions
        regular_memory = cs_lib.available_cpu_memory_GB()
        gpu_memory = cs_lib.available_cpu_memory_GB(partition="gpu_partition")
        high_mem_memory = cs_lib.available_cpu_memory_GB(partition="high_mem")

        # Verify results
        self.assertEqual(regular_memory, 95.0)
        self.assertEqual(gpu_memory, 190.0)
        self.assertEqual(high_mem_memory, 380.0)

        # Verify partition parameters were passed correctly
        calls = mock_get_unified_info.call_args_list
        self.assertEqual(calls[0][1]["partition"], None)
        self.assertEqual(calls[1][1]["partition"], "gpu_partition")
        self.assertEqual(calls[2][1]["partition"], "high_mem")

    def test_cpu_memory_configuration_isolation(self):
        """Test that configuration changes don't affect other imports."""
        # This test verifies that the configuration is module-level
        # and changes persist across function calls

        # Set initial percentage
        cs_lib.set_cpu_memory_usage_percentage(75.0)
        self.assertEqual(cs_lib.get_cpu_memory_usage_percentage(), 75.0)

        # Simulate different parts of application
        def app_component_1():
            return cs_lib.get_cpu_memory_usage_percentage()

        def app_component_2():
            cs_lib.set_cpu_memory_usage_percentage(85.0)
            return cs_lib.get_cpu_memory_usage_percentage()

        def app_component_3():
            return cs_lib.get_cpu_memory_usage_percentage()

        # Test that configuration is shared
        self.assertEqual(app_component_1(), 75.0)
        self.assertEqual(app_component_2(), 85.0)
        self.assertEqual(
            app_component_3(), 85.0
        )  # Should reflect the change from component 2


class TestCPUMemoryErrorHandling(unittest.TestCase):
    """Test error handling in CPU memory functions."""

    def setUp(self):
        """Reset configuration before each test."""
        cs_lib.CPU_MEMORY_USAGE_PERCENTAGE = 95.0

    @patch("clusterscope.lib.get_unified_info")
    def test_available_cpu_memory_MB_error_propagation(self, mock_get_unified_info):
        """Test that errors from UnifiedInfo are properly propagated."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_MB.side_effect = RuntimeError(
            "Memory query failed"
        )
        mock_get_unified_info.return_value = mock_unified_info

        with self.assertRaises(RuntimeError) as context:
            cs_lib.available_cpu_memory_MB()

        self.assertIn("Memory query failed", str(context.exception))

    @patch("clusterscope.lib.get_unified_info")
    def test_available_cpu_memory_GB_error_propagation(self, mock_get_unified_info):
        """Test that errors from UnifiedInfo are properly propagated."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_available_cpu_memory_GB.side_effect = ValueError(
            "Invalid memory configuration"
        )
        mock_get_unified_info.return_value = mock_unified_info

        with self.assertRaises(ValueError) as context:
            cs_lib.available_cpu_memory_GB()

        self.assertIn("Invalid memory configuration", str(context.exception))

    @patch("clusterscope.lib.get_unified_info")
    def test_cpu_memory_info_error_propagation(self, mock_get_unified_info):
        """Test that errors from UnifiedInfo are properly propagated."""
        mock_unified_info = MagicMock()
        mock_unified_info.get_cpu_memory_info.side_effect = RuntimeError(
            "Info query failed"
        )
        mock_get_unified_info.return_value = mock_unified_info

        with self.assertRaises(RuntimeError) as context:
            cs_lib.cpu_memory_info()

        self.assertIn("Info query failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
