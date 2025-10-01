# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Literal, Optional, Tuple, Union

from clusterscope.cluster_info import LocalNodeInfo, UnifiedInfo
from clusterscope.job_info import JobInfo

# Configurable CPU memory usage percentage
# This determines what percentage of total CPU memory is made available to applications
# Can be modified by calling set_cpu_memory_usage_percentage()
CPU_MEMORY_USAGE_PERCENTAGE = 95.0

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


def set_cpu_memory_usage_percentage(percentage: float) -> None:
    """Set the global CPU memory usage percentage for applications.

    This configures what percentage of total CPU memory is made available to applications
    when calling available_cpu_memory_MB(), available_cpu_memory_GB(), etc.

    Args:
        percentage (float): Percentage of total CPU memory to make available (1.0-100.0)

    Raises:
        ValueError: If percentage is not between 1 and 100

    Example:
        # Configure to use 90% of CPU memory for applications
        cs.set_cpu_memory_usage_percentage(90.0)

        # Now all calls use 90% by default
        available_mem = cs.available_cpu_memory_GB()  # Uses 90%
    """
    global CPU_MEMORY_USAGE_PERCENTAGE

    if not (1.0 <= percentage <= 100.0):
        raise ValueError("Percentage must be between 1.0 and 100.0")

    CPU_MEMORY_USAGE_PERCENTAGE = percentage


def get_cpu_memory_usage_percentage() -> float:
    """Get the current CPU memory usage percentage setting.

    Returns:
        float: Current percentage of total CPU memory made available to applications
    """
    return CPU_MEMORY_USAGE_PERCENTAGE


def available_cpu_memory_MB(partition: Optional[str] = None) -> int:
    """Get the available CPU memory (RAM) for applications in MB.

    Uses the configured CPU memory usage percentage (see set_cpu_memory_usage_percentage).
    By default, returns 95% of total CPU memory.

    This is useful for applications that need to know how much CPU memory (RAM) they can safely use
    without exhausting system resources.

    Note: This refers to system RAM, not GPU memory. For GPU memory information,
    use get_gpu_generation_and_count() or related GPU methods.

    Args:
        partition (str, optional): Slurm partition name to filter queries.

    Returns:
        int: Available CPU memory in MB for applications

    Example:
        # Use default 95% of CPU memory
        available_mb = cs.available_cpu_memory_MB()

        # Configure for 90% usage
        cs.set_cpu_memory_usage_percentage(90.0)
        available_mb = cs.available_cpu_memory_MB()  # Now uses 90%
    """
    return get_unified_info(partition).get_available_cpu_memory_MB(
        CPU_MEMORY_USAGE_PERCENTAGE
    )


def available_cpu_memory_GB(partition: Optional[str] = None) -> float:
    """Get the available CPU memory (RAM) for applications in GB.

    Uses the configured CPU memory usage percentage (see set_cpu_memory_usage_percentage).
    By default, returns 95% of total CPU memory.

    Note: This refers to system RAM, not GPU memory. For GPU memory information,
    use get_gpu_generation_and_count() or related GPU methods.

    Args:
        partition (str, optional): Slurm partition name to filter queries.

    Returns:
        float: Available CPU memory in GB for applications

    Example:
        # Configure for 85% usage, then get available memory
        cs.set_cpu_memory_usage_percentage(85.0)
        available_gb = cs.available_cpu_memory_GB()
    """
    return get_unified_info(partition).get_available_cpu_memory_GB(
        CPU_MEMORY_USAGE_PERCENTAGE
    )


def cpu_memory_info(partition: Optional[str] = None) -> Dict[str, Union[int, float]]:
    """Get comprehensive CPU memory (RAM) information including total and available memory.

    Uses the configured CPU memory usage percentage (see set_cpu_memory_usage_percentage).

    Note: This refers to system RAM, not GPU memory. For GPU memory information,
    use get_gpu_generation_and_count() or related GPU methods.

    Args:
        partition (str, optional): Slurm partition name to filter queries.

    Returns:
        Dict[str, Union[int, float]]: Dictionary containing:
            - total_cpu_mb: Total CPU memory in MB
            - total_cpu_gb: Total CPU memory in GB
            - available_cpu_mb: Available CPU memory in MB (at configured percentage)
            - available_cpu_gb: Available CPU memory in GB (at configured percentage)
            - percentage: The percentage used for calculation

    Example:
        # Set custom percentage and get info
        cs.set_cpu_memory_usage_percentage(80.0)
        info = cs.cpu_memory_info()
        print(f"Available: {info['available_cpu_gb']} GB at {info['percentage']}%")
    """
    return get_unified_info(partition).get_cpu_memory_info(CPU_MEMORY_USAGE_PERCENTAGE)


def local_node_gpu_generation_and_count() -> Dict[str, int]:
    """Get the GPU generation and count for the local node."""
    return local_info.get_gpu_generation_and_count()
