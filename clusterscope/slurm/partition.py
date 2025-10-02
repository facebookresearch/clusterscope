#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
from dataclasses import dataclass

from clusterscope.shell import run_cli
from clusterscope.slurm.parser import extract_gpus_from_gres


@dataclass
class PartitionInfo:
    """Store partition information from scontrol."""

    name: str
    max_gpus_per_node: int


def get_node_resources(node_spec: str) -> dict:
    """
    Query node resources for given node specification.
    Returns dict with max resources across nodes in the spec.
    """
    # not checking the return code here because `scontrol show node` can return non-zero exit code even
    # though stdout has what we need.
    result = subprocess.run(
        ["scontrol", "show", "node", node_spec, "-o"], capture_output=True, text=True
    )

    max_gpus = 0

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        node_data = {}
        for item in line.split():
            if "=" in item:
                key, value = item.split("=", 1)
                node_data[key] = value

        gres = node_data.get("Gres", "")
        gpu_count = extract_gpus_from_gres(gres)
        max_gpus = max(max_gpus, gpu_count)

    return {
        "max_gpus": max_gpus,
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

        name = partition_data.get("PartitionName", "Unknown")

        nodes = partition_data.get("Nodes", "")
        if nodes and nodes != "(null)":
            node_info = get_node_resources(nodes)
        else:
            node_info = {
                "max_gpus": 0,
            }

        partition = PartitionInfo(
            name=name,
            max_gpus_per_node=node_info["max_gpus"],
        )
        partitions.append(partition)

    return partitions
