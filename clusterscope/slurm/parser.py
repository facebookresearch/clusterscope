#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import re


def parse_gpu_gres(gres_str: str) -> int:
    """Parse GPU count from GRES string (e.g., 'gpu:4' or 'gpu:a100:2')."""
    if not gres_str or gres_str == "(null)":
        return 0

    # Match patterns like gpu:4, gpu:a100:2, gpu:tesla:4, etc.
    match = re.search(r"gpu(?::\w+)?:(\d+)", gres_str)
    if match:
        return int(match.group(1))

    # If just 'gpu' with no count, assume 0
    return 0
