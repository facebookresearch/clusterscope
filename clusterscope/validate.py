#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import sys
from typing import Optional

from clusterscope.slurm.partition import get_partition_info, PartitionInfo


def validate_partition_exists(
    partition: str, exit_on_error: bool = False
) -> PartitionInfo:
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
    return req_partition


def job_gen_task_slurm_validator(
    partition: str,
    tasks_per_node: int = 1,
    gpus_per_task: Optional[int] = None,
    cpus_per_task: Optional[int] = None,
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
    if gpus_per_task is not None and cpus_per_task is not None:
        if exit_on_error:
            logging.error(
                "Only one of gpus_per_task or cpus_per_task can be specified. For GPU requests, use gpus_per_task and cpus_per_task will be generated automatically. For CPU requests, use cpus_per_task only."
            )
            sys.exit(1)
        raise ValueError(
            "Only one of gpus_per_task or cpus_per_task can be specified. For GPU requests, use gpus_per_task and cpus_per_task will be generated automatically. For CPU requests, use cpus_per_task only."
        )
    if cpus_per_task is not None and cpus_per_task <= 0:
        if exit_on_error:
            logging.error("cpus_per_task has to be > 0.")
            sys.exit(1)
        raise ValueError("cpus_per_task has to be > 0.")
    if gpus_per_task is not None and gpus_per_task <= 0:
        if exit_on_error:
            logging.error("gpus_per_task has to be > 0.")
            sys.exit(1)
        raise ValueError("gpus_per_task has to be > 0.")
    if tasks_per_node <= 0:
        if exit_on_error:
            logging.error("tasks_per_node has to be > 0.")
            sys.exit(1)
        raise ValueError("tasks_per_node has to be > 0.")

    req_partition = validate_partition_exists(
        partition=partition,
        exit_on_error=exit_on_error,
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
