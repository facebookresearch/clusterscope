# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Literal, Tuple

from clusterscope.cluster_info import UnifiedInfo
from clusterscope.job_info import JobInfo

unified_info = UnifiedInfo()
job_info = JobInfo()


def cluster() -> str:
    """Get the cluster name. Returns `local-node` if not on a cluster."""
    return unified_info.get_cluster_name()


def slurm_version() -> Tuple[int, ...]:
    """Get the slurm version. Returns `0` if not a Slurm cluster."""
    slurm_version = unified_info.get_slurm_version()
    version = tuple(int(v) for v in slurm_version.split("."))
    return version


def cpus() -> int:
    """Get the number of CPUs for each node in the cluster. Returns the number of local cpus if not on a cluster."""
    return unified_info.get_cpus_per_node()


def mem(
    to_unit: Literal["MB", "GB"] = "GB",
) -> int:
    """Get the amount of memory for each node in the cluster. Returns the local memory if not on a cluster."""
    mem = unified_info.get_mem_per_node_MB()
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
    """Get the GPU generation and count for the slurm cluster. Returns local gpus if not on a cluster."""
    return unified_info.get_gpu_generation_and_count()


is_torch_run = job_info.is_torch_run
is_slurm_job = job_info.is_slurm_job
job_id = job_info.job_id
job_name = job_info.job_name
global_rank = job_info.global_rank
local_rank = job_info.local_rank
world_size = job_info.world_size
is_rank_zero = job_info.is_rank_zero
