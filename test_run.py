#!/usr/bin/env python3
"""Simple test runner to verify unit tests work with run_cli() changes."""

import sys
import unittest
from unittest.mock import patch

# Add the clusterscope package to the path
sys.path.insert(0, '/storage/home/skalyan/clusterscope')

from clusterscope.cluster_info import LinuxInfo, DarwinInfo

def test_linux_cpu_count():
    """Test LinuxInfo.get_cpu_count() with mocked run_cli."""
    linux_info = LinuxInfo()

    with patch("clusterscope.cluster_info.run_cli", return_value="8") as mock_run_cli:
        result = linux_info.get_cpu_count()
        print(f"✓ LinuxInfo.get_cpu_count() returned: {result}")
        assert result == 8
        mock_run_cli.assert_called_once_with(["nproc", "--all"], text=True, timeout=60)

def test_linux_memory():
    """Test LinuxInfo.get_mem_MB() with mocked run_cli."""
    linux_info = LinuxInfo()

    mock_output = "               total        used\nMem:     16384    8192\n"
    with patch("clusterscope.cluster_info.run_cli", return_value=mock_output) as mock_run_cli:
        result = linux_info.get_mem_MB()
        print(f"✓ LinuxInfo.get_mem_MB() returned: {result}")
        assert result == 16384
        mock_run_cli.assert_called_once_with(["free", "-m"], text=True, timeout=60)

def test_darwin_cpu_count():
    """Test DarwinInfo.get_cpu_count() with mocked run_cli."""
    darwin_info = DarwinInfo()

    with patch("clusterscope.cluster_info.run_cli", return_value="12") as mock_run_cli:
        result = darwin_info.get_cpu_count()
        print(f"✓ DarwinInfo.get_cpu_count() returned: {result}")
        assert result == 12
        mock_run_cli.assert_called_once_with(["sysctl", "-n", "hw.ncpu"], text=True, timeout=60)

def test_darwin_memory():
    """Test DarwinInfo.get_mem_MB() with mocked run_cli."""
    darwin_info = DarwinInfo()

    # 34359738368 bytes = 32768 MB
    with patch("clusterscope.cluster_info.run_cli", return_value="34359738368") as mock_run_cli:
        result = darwin_info.get_mem_MB()
        print(f"✓ DarwinInfo.get_mem_MB() returned: {result}")
        assert result == 32768
        mock_run_cli.assert_called_once_with(["sysctl", "-n", "hw.memsize"], text=True, timeout=60)

def test_run_cli_command_not_found():
    """Test run_cli() with a command that doesn't exist."""
    from clusterscope.cluster_info import run_cli

    try:
        run_cli(["nonexistent_command"])
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"✓ run_cli() correctly raised RuntimeError: {e}")
        assert "not available on this system" in str(e)

def main():
    """Run all tests."""
    print("Testing run_cli() functionality...")
    print("=" * 50)

    try:
        test_linux_cpu_count()
        test_linux_memory()
        test_darwin_cpu_count()
        test_darwin_memory()
        test_run_cli_command_not_found()

        print("=" * 50)
        print("✓ All tests passed! The run_cli() replacement is working correctly.")
        return 0

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
