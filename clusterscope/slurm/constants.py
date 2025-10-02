#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

NODE_DOWN_STATES = frozenset(
    [
        "drained",
        "down",
        "maint",
        "powered_down",
        "powering_down",
        "powering_up",
        "fail",
        "future",
        "inval",
        "perfctrs",
    ]
)
