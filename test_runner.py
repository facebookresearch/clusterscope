#!/usr/bin/env python3
"""Simple test runner to verify unit tests pass."""

import sys
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr

# Add the clusterscope package to the path
sys.path.insert(0, '/storage/home/skalyan/clusterscope')

def run_unit_tests():
    """Run the unit tests and capture results."""
    try:
        # Import the test module
        from tests.test_cluster_info import (
            TestLinuxInfo,
            TestDarwinInfo,
            TestUnifiedInfo,
            TestSlurmClusterInfo,
            TestAWSClusterInfo
        )

        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        # Add specific tests that use run_cli()
        suite.addTest(TestLinuxInfo('test_get_cpu_count'))
        suite.addTest(TestLinuxInfo('test_get_mem_per_node_MB'))
        suite.addTest(TestDarwinInfo('test_get_cpu_count'))
        suite.addTest(TestDarwinInfo('test_get_mem_per_node_MB'))

        # Run tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)

        # Report results
        if result.wasSuccessful():
            print(f"\n✓ All {result.testsRun} unit tests passed!")
            print("✓ The run_cli() replacement is working correctly with unit tests.")
            return True
        else:
            print(f"\n✗ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
            for test, traceback in result.failures:
                print(f"FAILED: {test}")
                print(traceback)
            for test, traceback in result.errors:
                print(f"ERROR: {test}")
                print(traceback)
            return False

    except Exception as e:
        print(f"✗ Failed to run unit tests: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("Running unit tests to verify run_cli() changes...")
    print("=" * 60)

    success = run_unit_tests()

    print("=" * 60)
    if success:
        print("✓ Unit test verification completed successfully!")
        return 0
    else:
        print("✗ Unit test verification failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
