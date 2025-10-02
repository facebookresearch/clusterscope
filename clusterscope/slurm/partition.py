#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import subprocess
from dataclasses import dataclass

from clusterscope.cluster_info import UnifiedInfo
from clusterscope.parser import parse_memory_to_gb

from clusterscope.shell import run_cli
from clusterscope.slurm.constants import NODE_DOWN_STATES
from clusterscope.slurm.parser import parse_gpu_gres


@dataclass
class PartitionInfo:
    """Store partition information from scontrol."""

    name: str
    max_cpus_per_node: int
    max_mem_per_node: int  # MB
    max_gpus_per_node: int
    available_nodes: int
    state: str


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

    max_cpus = 0
    max_mem = 0
    max_gpus = 0
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

    return {
        "max_cpus": max_cpus,
        "max_mem": max_mem,
        "max_gpus": max_gpus,
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
                "available": 0,
            }

        partition = PartitionInfo(
            name=name,
            max_cpus_per_node=node_info["max_cpus"],
            max_mem_per_node=node_info["max_mem"],
            max_gpus_per_node=node_info["max_gpus"],
            available_nodes=node_info["available"],
            state=partition_data.get("State", "UNKNOWN"),
        )
        partitions.append(partition)

    return partitions


def find_matching_partitions(
    partitions: list[PartitionInfo],
    cpus_per_node: int,
    gpus_per_node: int,
    cpu_ram: int,
) -> list[PartitionInfo]:
    """
    Find partitions that satisfy the given requirements.

    Args:
        partitions: List of available partitions
        cpus_per_node: Required number of CPUs
        gpus_per_node: Required number of GPUs
        cpu_ram: Required CPU RAM in MB (optional)

    Returns:
        List of matching PartitionInfo objects
    """
    matching = []

    for partition in partitions:
        if partition.state.upper() != "UP":
            continue

        if cpus_per_node > partition.max_cpus_per_node:
            continue

        if gpus_per_node > partition.max_gpus_per_node:
            continue

        if cpu_ram > partition.max_mem_per_node:
            continue

        # Get CPU cores and CPU RAM allowed based on number of GPUs per node
        resource = UnifiedInfo(partition=partition.name).get_task_resource_requirements(
            num_gpus=gpus_per_node
        )
        if gpus_per_node > 0:
            if cpus_per_node > resource.cpu_cores:
                continue
            if cpu_ram > parse_memory_to_gb(resource.memory):
                continue

        matching.append(partition)

    return matching
