#!/usr/bin/env python3
"""Verify that the run_cli() changes work correctly."""

import sys
import os

# Add the clusterscope package to the path
sys.path.insert(0, '/storage/home/skalyan/clusterscope')

def test_imports():
    """Test that all imports work correctly."""
    try:
        from clusterscope.cluster_info import run_cli, LinuxInfo, DarwinInfo, UnifiedInfo
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_run_cli_function():
    """Test that run_cli function exists and has correct signature."""
    try:
        from clusterscope.cluster_info import run_cli
        import inspect

        # Check function signature
        sig = inspect.signature(run_cli)
        params = list(sig.parameters.keys())
        expected_params = ['cmd', 'text', 'timeout', 'stderr']

        if params == expected_params:
            print("✓ run_cli() function has correct signature")
            return True
        else:
            print(f"✗ run_cli() signature mismatch. Expected: {expected_params}, Got: {params}")
            return False
    except Exception as e:
        print(f"✗ run_cli() function test failed: {e}")
        return False

def test_run_cli_with_invalid_command():
    """Test that run_cli raises RuntimeError for invalid commands."""
    try:
        from clusterscope.cluster_info import run_cli

        try:
            run_cli(["this_command_does_not_exist_12345"])
            print("✗ run_cli() should have raised RuntimeError for invalid command")
            return False
        except RuntimeError as e:
            if "not available on this system" in str(e):
                print("✓ run_cli() correctly raises RuntimeError for invalid commands")
                return True
            else:
                print(f"✗ run_cli() raised RuntimeError but with unexpected message: {e}")
                return False
    except Exception as e:
        print(f"✗ run_cli() invalid command test failed: {e}")
        return False

def test_run_cli_with_valid_command():
    """Test that run_cli works with a valid command."""
    try:
        from clusterscope.cluster_info import run_cli

        # Test with 'echo' command which should be available on most systems
        result = run_cli(["echo", "test"])
        if result.strip() == "test":
            print("✓ run_cli() works correctly with valid commands")
            return True
        else:
            print(f"✗ run_cli() returned unexpected result: {result}")
            return False
    except Exception as e:
        print(f"✗ run_cli() valid command test failed: {e}")
        return False

def test_class_instantiation():
    """Test that classes can be instantiated without errors."""
    try:
        from clusterscope.cluster_info import LinuxInfo, DarwinInfo, UnifiedInfo

        # Test instantiation
        linux_info = LinuxInfo()
        darwin_info = DarwinInfo()
        unified_info = UnifiedInfo()

        print("✓ All classes can be instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Class instantiation failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("Verifying run_cli() implementation...")
    print("=" * 50)

    tests = [
        test_imports,
        test_run_cli_function,
        test_run_cli_with_invalid_command,
        test_run_cli_with_valid_command,
        test_class_instantiation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All verification tests passed! The run_cli() implementation is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
