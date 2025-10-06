import logging
import sys
from typing import Optional

from clusterscope.slurm.partition import get_partition_info


def job_gen_task_slurm_validator(
    partition: str,
    gpus_per_task: Optional[int],
    cpus_per_task: Optional[int],
    tasks_per_node: int,
    exit_on_error: bool = False,
) -> None:
    """Validate the job requirements for a task of a Slurm job based on GPU or CPU per task requirements.
    This validation is used for CLI and API calls.

    Returns: None

    Raises or Exits depending on exit_on_error(bool) flag
    """
    if gpus_per_task is None and cpus_per_task is None:
        if exit_on_error:
            logging.error("Either gpus_per_task or cpus_per_task must be specified.")
            sys.exit(1)
        raise ValueError("Either gpus_per_task or cpus_per_task must be specified.")
    if cpus_per_task and cpus_per_task < 0:
        if exit_on_error:
            logging.error("cpus_per_task has to be >= 0.")
            sys.exit(1)
        raise ValueError("cpus_per_task has to be >= 0.")
    if gpus_per_task and gpus_per_task < 0:
        if exit_on_error:
            logging.error("gpus_per_task has to be >= 0.")
            sys.exit(1)
        raise ValueError("gpus_per_task has to be >= 0.")
    if gpus_per_task == 0 and cpus_per_task == 0:
        if exit_on_error:
            logging.error("One of gpus_per_task or cpus_per_task has to be non-zero.")
            sys.exit(1)
        raise ValueError("One of gpus_per_task or cpus_per_task has to be non-zero.")
    if gpus_per_task and cpus_per_task:
        if exit_on_error:
            logging.error(
                "Only one of gpus_per_task or cpus_per_task can be specified. For GPU requests, use gpus_per_task and cpus_per_task will be generated automatically. For CPU requests, use cpus_per_task only."
            )
            sys.exit(1)
        raise ValueError(
            "Only one of gpus_per_task or cpus_per_task can be specified. For GPU requests, use gpus_per_task and cpus_per_task will be generated automatically. For CPU requests, use cpus_per_task only."
        )

    partitions = get_partition_info()
    req_partition = next((p for p in partitions if p.name == partition), None)

    if req_partition is None:
        if exit_on_error:
            logging.error(
                f"Partition {partition} not found. Available partitions: {[p.name for p in partitions]}"
            )
            sys.exit(1)
        raise ValueError(
            f"Partition {partition} not found. Available partitions: {[p.name for p in partitions]}"
        )

    # reject if requires more GPUs than the max GPUs per node for the partition
    if (
        gpus_per_task
        and gpus_per_task * tasks_per_node > req_partition.max_gpus_per_node
    ):
        if exit_on_error:
            logging.error(
                f"Requested {gpus_per_task=} GPUs with {tasks_per_node=} exceeds the maximum {req_partition.max_gpus_per_node} GPUs per node available in partition '{partition}'"
            )
            sys.exit(1)
        raise ValueError(
            f"Requested {gpus_per_task=} GPUs with {tasks_per_node=} exceeds the maximum {req_partition.max_gpus_per_node} GPUs per node available in partition '{partition}'"
        )

    # reject if requires more CPUs than the max CPUs at the partition
    if (
        cpus_per_task
        and cpus_per_task * tasks_per_node > req_partition.max_cpus_per_node
    ):
        if exit_on_error:
            logging.error(
                f"Requested {cpus_per_task=} CPUs with {tasks_per_node=} exceeds the maximum {req_partition.max_cpus_per_node} CPUs per node available in partition '{partition}'"
            )
            sys.exit(1)
        raise ValueError(
            f"Requested {cpus_per_task=} CPUs with {tasks_per_node=} exceeds the maximum {req_partition.max_cpus_per_node} CPUs per node available in partition '{partition}'"
        )
