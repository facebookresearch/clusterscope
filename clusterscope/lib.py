# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

from clusterscope.cluster_info import UnifiedInfo

unified_info = UnifiedInfo()


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


def mem() -> int:
    """Get the amount of memory for each node in the cluster. Returns the local memory if not on a cluster."""
    return unified_info.get_mem_per_node()
