#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from clusterscope.shell import run_cli
from clusterscope.slurm.parser import parse_gres


@dataclass
class PartitionInfo:
    """Store partition information from scontrol."""

    name: str
    max_gpus_per_node: int
    max_cpus_per_node: int


def get_partition_resources(partition: str) -> dict:
    result = run_cli(
        [
            "sinfo",
            "-o",
            "%G,%c",
            f"--partition={partition}",
            "--noheader",
        ],
    )

    stdout = result.strip().split("\n")[0]
    gres, cpus = stdout.split(",")

    return {
        "max_gpus": parse_gres(gres),
        "max_cpus": int(cpus),
    }


def get_partition_info() -> list[PartitionInfo]:
    """
    Query Slurm for partition information using scontrol.
    Returns a list of PartitionInfo objects.
    """
    result = run_cli(["scontrol", "show", "partition", "-o"])

    max_gpus = 0

    partitions = []
    for line in result.strip().split("\n"):
        if not line:
            continue

        partition_data = {}
        for item in line.split():
            if "=" in item:
                key, value = item.split("=", 1)
                partition_data[key] = value

        partition_name = partition_data.get("PartitionName", "Unknown")

        nodes = partition_data.get("Nodes", "")
        if nodes and nodes != "(null)":
            partition_info = get_partition_resources(partition=partition_name)
        else:
            partition_info = {
                "max_gpus": 0,
                "max_cpus": 0,
            }

        partition = PartitionInfo(
            name=partition_name,
            max_cpus_per_node=partition_info["max_cpus"],
            max_gpus_per_node=partition_info["max_gpus"],
        )
        partitions.append(partition)

    return partitions
