# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
__version__ = "0.0.0"

from clusterscope.lib import (
    cluster,
    cpus,
    global_rank,
    is_rank_zero,
    is_slurm_job,
    is_torch_run,
    job_id,
    job_name,
    local_node_gpu_generation_and_count,
    local_rank,
    mem,
    slurm_version,
    world_size,
)

__all__ = [
    "cluster",
    "slurm_version",
    "cpus",
    "mem",
    "local_node_gpu_generation_and_count",
    "is_torch_run",
    "is_slurm_job",
    "job_id",
    "job_name",
    "global_rank",
    "local_rank",
    "world_size",
    "is_rank_zero",
]
