# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

from clusterscope.cluster_info import (
    CPUInfo,
    GPUInfo,
    LocalNodeInfo,
    MemInfo,
    UnifiedInfo,
)
from clusterscope.job_info import JobInfo
from clusterscope.validate import (
    job_gen_task_slurm_validator,
    validate_partition_exists,
)

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


def cpus(partition: Optional[str] = None) -> list[CPUInfo] | CPUInfo:
    """Get the number of CPUs for each node in the cluster. Returns the number of local cpus if not on a cluster.

    Args:
        partition (str, optional): Slurm partition name to filter queries.
    """
    if partition is not None:
        validate_partition_exists(partition=partition)
    cpu_info = get_unified_info(partition).get_cpus_per_node()
    cpu_info_list = cpu_info if isinstance(cpu_info, list) else [cpu_info]
    for cpu in cpu_info_list:
        if partition is not None and partition == cpu.partition:
            return cpu
    return cpu_info_list


def mem(
    partition: Optional[str] = None,
) -> list[MemInfo] | MemInfo:
    """Get the amount of memory for each node in the cluster. Returns the local memory if not on a cluster.

    Args:
        to_unit: Unit to return memory in ("MB" or "GB").
        partition (str, optional): Slurm partition name to filter queries.
    """
    if partition is not None:
        validate_partition_exists(partition=partition)
    mem_info = get_unified_info(partition).get_mem_per_node_MB()
    mem_info_list = mem_info if isinstance(mem_info, list) else [mem_info]
    for mem in mem_info_list:
        if partition is not None and partition == mem.partition:
            return mem
    return mem_info_list


def get_tmp_dir():
    tmp = get_unified_info().get_tmp_dir()
    return tmp


def local_node_gpu_generation_and_count() -> list[GPUInfo]:
    """Get the GPU generation and count for the local node."""
    return local_info.get_gpu_generation_and_count()


def job_gen_task_slurm(
    partition: str,
    gpus_per_task: Optional[int] = None,
    cpus_per_task: Optional[int] = None,
    tasks_per_node: int = 1,
) -> dict:
    """Get the number of CPUs/RAM for each task in the job."""
    job_gen_task_slurm_validator(
        partition=partition,
        gpus_per_task=gpus_per_task,
        cpus_per_task=cpus_per_task,
        tasks_per_node=tasks_per_node,
    )

    unified_info = UnifiedInfo(partition=partition)
    job_requirements = unified_info.get_task_resource_requirements(
        partition=partition,
        cpus_per_task=cpus_per_task,
        gpus_per_task=gpus_per_task,
        tasks_per_node=tasks_per_node,
    )
    return job_requirements.to_dict()
