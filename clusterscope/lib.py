# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Literal, Optional, Tuple

from clusterscope.cluster_info import LocalNodeInfo, UnifiedInfo
from clusterscope.job_info import JobInfo

# Partition-aware unified info instance
_unified_info: Optional[UnifiedInfo] = None
_current_partition: Optional[str] = None

local_info = LocalNodeInfo()

# init only if clusterscope is queried for job info
_job: Optional[JobInfo] = None


def get_unified_info(partition: Optional[str] = None) -> UnifiedInfo:
    """Get the unified info instance, creating a new one if partition changes."""
    global _unified_info, _current_partition

    if _unified_info is None or _current_partition != partition:
        _unified_info = UnifiedInfo(partition=partition)
        _current_partition = partition

    return _unified_info


def get_job() -> JobInfo:
    global _job
    if _job is None:
        _job = JobInfo()
    return _job


def cluster(partition: Optional[str] = None) -> str:
    """Get the cluster name. Returns `local-node` if not on a cluster.

    Args:
        partition (str, optional): Slurm partition name to filter queries.
    """
    return get_unified_info(partition).get_cluster_name()


def slurm_version(partition: Optional[str] = None) -> Tuple[int, ...]:
    """Get the slurm version. Returns `0` if not a Slurm cluster.

    Args:
        partition (str, optional): Slurm partition name to filter queries.
    """
    slurm_version = get_unified_info(partition).get_slurm_version()
    version = tuple(int(v) for v in slurm_version.split("."))
    return version


def cpus(partition: Optional[str] = None) -> int:
    """Get the number of CPUs for each node in the cluster. Returns the number of local cpus if not on a cluster.

    Args:
        partition (str, optional): Slurm partition name to filter queries.
    """
    return get_unified_info(partition).get_cpus_per_node()


def mem(
    to_unit: Literal["MB", "GB"] = "GB",
    partition: Optional[str] = None,
) -> int:
    """Get the amount of memory for each node in the cluster. Returns the local memory if not on a cluster.

    Args:
        to_unit: Unit to return memory in ("MB" or "GB").
        partition (str, optional): Slurm partition name to filter queries.
    """
    mem = get_unified_info(partition).get_mem_per_node_MB()
    if to_unit == "MB":
        pass
    elif to_unit == "GB":
        mem //= 1000
    else:
        raise ValueError(
            f"{to_unit} is not a supported unit. Currently supported units: MB, GB"
        )
    return mem


def local_node_gpu_generation_and_count() -> Dict[str, int]:
    """Get the GPU generation and count for the local node."""
    return local_info.get_gpu_generation_and_count()
