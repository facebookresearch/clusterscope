#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
def parse_memory_to_gb(memory) -> int:
    """Parse memory string and convert to GB.

    Returns:
        int: Memory in GB
    """
    mem_value = memory.rstrip("GT")
    if memory.endswith("T"):
        return int(mem_value) * 1024
    elif memory.endswith("G"):
        return int(mem_value)
    else:
        raise RuntimeError(f"Invalid memory format: {memory}")
