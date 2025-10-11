#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import shutil
import subprocess


def run_cli(
    cmd: list[str],
    text: bool = True,
    timeout: int = 60,
    stderr: int | None = None,
) -> str:
    """
    Run a CLI command after verifying it's available.

    Args:
        cmd: List of command and arguments
        text: Whether to return text output (default: True)
        timeout: Command timeout in seconds (default: 60)
        stderr: How to handle stderr (default: None)

    Returns:
        str: Command output

    Raises:
        RuntimeError: If command is not available or execution fails
    """
    if not cmd:
        raise RuntimeError("Command list cannot be empty")

    command_name = cmd[0]

    # Check if command is available
    if shutil.which(command_name) is None:
        raise RuntimeError(f"Command '{command_name}' is not available on this system")

    try:
        result = subprocess.check_output(cmd, text=text, timeout=timeout, stderr=stderr)
        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command '{' '.join(cmd)}' failed with return code {e.returncode}: {e.output}, {e}"
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Command '{' '.join(cmd)}' timed out after {timeout} seconds"
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to execute command '{' '.join(cmd)}': {str(e)}")
