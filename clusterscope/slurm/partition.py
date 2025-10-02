#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from clusterscope.shell import run_cli


@dataclass
class PartitionInfo:
    """Store partition information from scontrol."""

    name: str


def get_partition_info() -> list[PartitionInfo]:
    """
    Query Slurm for partition information using scontrol.
    Returns a list of PartitionInfo objects.
    """
    result = run_cli(["scontrol", "show", "partition", "-o"])

    partitions = []
    for line in result.strip().split("\n"):
        if not line:
            continue

        partition_data = {}
        for item in line.split():
            if "=" in item:
                key, value = item.split("=", 1)
                partition_data[key] = value

        # Extract partition name
        name = partition_data.get("PartitionName", "Unknown")

        partition = PartitionInfo(
            name=name,
        )
        partitions.append(partition)

    return partitions
