#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import re


def parse_gres(gres_str: str) -> int:
    """Parse GPU count from GRES string.

    Handles formats like:
    - 'gpu:4'
    - 'gpu:a100:2'
    - 'gpu:volta:8(S:0-1)'
    - 'gpu:pascal:2'
    - '(null)'
    """
    if not gres_str or gres_str == "(null)":
        return 0

    match = re.search(r"gpu(?::\w+)?:(\d+)", gres_str)
    if match:
        return int(match.group(1))

    return 0
