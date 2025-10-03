import logging
import sys

from clusterscope.slurm.partition import get_partition_info


def job_gen_task_slurm_validator(
    partition: str,
    gpus_per_task: int,
    cpus_per_task: int,
    tasks_per_node: int,
) -> None:
    if gpus_per_task is None and cpus_per_task is None:
        logging.error("Either gpus_per_task or cpus_per_task must be specified.")
        sys.exit(1)
    if cpus_per_task < 0:
        logging.error("cpus_per_task has to be >= 0.")
        sys.exit(1)
    if gpus_per_task < 0:
        logging.error("gpus_per_task has to be >= 0.")
        sys.exit(1)
    if gpus_per_task == 0 and cpus_per_task == 0:
        logging.error("One of gpus_per_task or cpus_per_task has to be non-zero.")
        sys.exit(1)
    if gpus_per_task and cpus_per_task:
        logging.error(
            "Only one of gpus_per_task or cpus_per_task can be specified. For GPU requests, use gpus_per_task and cpus_per_task will be generated automatically. For CPU requests, use cpus_per_task only."
        )
        sys.exit(1)

    partitions = get_partition_info()
    req_partition = next((p for p in partitions if p.name == partition), None)

    if req_partition is None:
        logging.error(
            f"Partition {partition} not found. Available partitions: {[p.name for p in partitions]}"
        )
        sys.exit(1)

    # reject if requires more GPUs than the max GPUs per node for the partition
    if gpus_per_task * tasks_per_node > req_partition.max_gpus_per_node:
        logging.error(
            f"Requested {gpus_per_task=} GPUs with {tasks_per_node=} exceeds the maximum {req_partition.max_gpus_per_node} GPUs per node available in partition '{partition}'"
        )
        sys.exit(1)

    # reject if requires more CPUs than the max CPUs at the partition
    if cpus_per_task * tasks_per_node > req_partition.max_cpus_per_node:
        logging.error(
            f"Requested {cpus_per_task=} CPUs with {tasks_per_node=} exceeds the maximum {req_partition.max_cpus_per_node} CPUs per node available in partition '{partition}'"
        )
        sys.exit(1)
