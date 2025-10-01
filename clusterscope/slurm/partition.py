#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
from dataclasses import dataclass

from clusterscope.shell import run_cli
from clusterscope.slurm.constants import NODE_DOWN_STATES
from clusterscope.slurm.parser import parse_gpu_gres, parse_gpu_memory


@dataclass
class PartitionInfo:
    """Store partition information from scontrol."""

    name: str
    max_cpus_per_node: int
    max_mem_per_node: int  # MB
    max_gpus_per_node: int
    gpu_mem_per_gpu: int  # MB (if available)
    available_nodes: int
    state: str


def get_node_resources(node_spec: str) -> dict:
    """
    Query node resources for given node specification.
    Returns dict with max resources across nodes in the spec.
    """
    result = subprocess.run(
        ["scontrol", "show", "node", node_spec, "-o"], capture_output=True, text=True
    )
    # result = run_cli(['scontrol', 'show', 'node', node_spec, '-o'])

    max_cpus = 0
    max_mem = 0
    max_gpus = 0
    gpu_mem = 0
    available = 0

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue

        node_data = {}
        for item in line.split():
            if "=" in item:
                key, value = item.split("=", 1)
                node_data[key] = value

        state = node_data.get("State", "")
        if state.lower() not in NODE_DOWN_STATES:
            available += 1

        cpus = int(node_data.get("CPUTot", 0))
        max_cpus = max(max_cpus, cpus)

        mem_str = node_data.get("RealMemory", "0")
        mem = int(mem_str) if mem_str.isdigit() else 0
        max_mem = max(max_mem, mem)

        gres = node_data.get("Gres", "")
        gpu_count = parse_gpu_gres(gres)
        max_gpus = max(max_gpus, gpu_count)

        gres_used = node_data.get("GresUsed", "")
        gpu_mem_parsed = parse_gpu_memory(gres + " " + gres_used)
        if gpu_mem_parsed > 0:
            gpu_mem = max(gpu_mem, gpu_mem_parsed)

    return {
        "max_cpus": max_cpus,
        "max_mem": max_mem,
        "max_gpus": max_gpus,
        "gpu_mem": gpu_mem,
        "available": available,
    }


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

        # Get node information for this partition
        nodes = partition_data.get("Nodes", "")
        if nodes and nodes != "(null)":
            node_info = get_node_resources(nodes)
        else:
            node_info = {
                "max_cpus": 0,
                "max_mem": 0,
                "max_gpus": 0,
                "gpu_mem": 0,
                "available": 0,
            }

        partition = PartitionInfo(
            name=name,
            max_cpus_per_node=node_info["max_cpus"],
            max_mem_per_node=node_info["max_mem"],
            max_gpus_per_node=node_info["max_gpus"],
            gpu_mem_per_gpu=node_info["gpu_mem"],
            available_nodes=node_info["available"],
            state=partition_data.get("State", "UNKNOWN"),
        )
        partitions.append(partition)

    return partitions
