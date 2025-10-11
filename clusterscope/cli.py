#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from typing import Any, Dict, Optional

import click
from click_option_group import optgroup, RequiredMutuallyExclusiveOptionGroup

from clusterscope.cluster_info import AWSClusterInfo, UnifiedInfo
from clusterscope.validate import (
    job_gen_task_slurm_validator,
    validate_partition_exists,
)


def format_dict(data: Dict[str, Any]) -> str:
    """Format a dictionary for display."""
    return json.dumps(data, indent=2)


@click.group()
def cli():
    """Command-line tool to query Slurm cluster information."""
    pass


@cli.command()
def version():
    """Show the version of clusterscope."""
    try:
        from importlib.metadata import version as get_version

        pkg_version = get_version("clusterscope")
    except Exception:
        # Fallback to the version in __init__.py if setuptools-scm isn't available
        import clusterscope

        pkg_version = clusterscope.__version__
    click.echo(f"clusterscope version {pkg_version}")


@cli.command()
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Slurm partition name to filter queries (optional)",
)
def info(partition: str):
    """Show basic cluster information."""
    if partition is not None:
        validate_partition_exists(partition=partition, exit_on_error=True)
    unified_info = UnifiedInfo(partition=partition)
    cluster_name = unified_info.get_cluster_name()
    slurm_version = unified_info.get_slurm_version()
    click.echo(f"Cluster Name: {cluster_name}")
    click.echo(f"Slurm Version: {slurm_version}")


@cli.command()
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Slurm partition name to filter queries (optional)",
)
def cpus(partition: str):
    """Show CPU counts per node."""
    if partition is not None:
        validate_partition_exists(partition=partition, exit_on_error=True)
    unified_info = UnifiedInfo(partition=partition)
    cpu_info = unified_info.get_cpus_per_node()
    cpu_info_list = cpu_info if isinstance(cpu_info, list) else [cpu_info]
    click.echo("CPU Count, Slurm Partition:")
    for cpu in cpu_info_list:
        if partition is not None and partition != cpu.partition:
            continue
        click.echo(f"{cpu.cpu_count}, {cpu.partition}")


@cli.command()
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Slurm partition name to filter queries (optional)",
)
def mem(partition: str):
    """Show memory information per node."""
    if partition is not None:
        validate_partition_exists(partition=partition, exit_on_error=True)
    unified_info = UnifiedInfo(partition=partition)
    mem_info = unified_info.get_mem_per_node_MB()
    mem_info_list = mem_info if isinstance(mem_info, list) else [mem_info]
    click.echo("Mem total MB, Mem total GB, Slurm Partition:")
    for mem in mem_info_list:
        if partition is not None and partition != mem.partition:
            continue
        click.echo(f"{mem.mem_total_MB}, {mem.mem_total_GB}, {mem.partition}")


@cli.command()
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Slurm partition name to filter queries (optional)",
)
@click.option("--generations", is_flag=True, help="Show only GPU generations")
@click.option("--counts", is_flag=True, help="Show only GPU counts by type")
@click.option("--vendor", is_flag=True, help="Show GPU vendor information")
def gpus(partition: str, generations: bool, counts: bool, vendor: bool):
    """Show GPU information."""
    if partition is not None:
        validate_partition_exists(partition=partition, exit_on_error=True)
    unified_info = UnifiedInfo(partition=partition)

    if vendor:
        gpus = unified_info.get_gpu_generation_and_count()
        all_vendors = set()
        if gpus:
            click.echo("GPU Vendors:")
            for gpu in gpus:
                if partition is not None and partition != gpu.partition:
                    continue
                if gpu.vendor in all_vendors:
                    continue
                all_vendors.add(gpu.vendor)
                click.echo(f"{gpu.vendor}")
    elif counts:
        gpus = unified_info.get_gpu_generation_and_count()
        if gpus:
            click.echo("GPU Gen, GPU Count, Slurm Partition:")
            for gpu in gpus:
                if partition is not None and partition != gpu.partition:
                    continue
                click.echo(f"{gpu.gpu_gen}, {gpu.gpu_count}, {gpu.partition}")
        else:
            click.echo("No GPUs found")
    elif generations:
        gpus = unified_info.get_gpu_generation_and_count()
        if gpus:
            click.echo("GPU Gen, Slurm Partition:")
            for gpu in gpus:
                if partition is not None and partition != gpu.partition:
                    continue
                click.echo(f"{gpu.gpu_gen}, {gpu.partition}")
        else:
            click.echo("No GPUs found")
    else:
        gpus = unified_info.get_gpu_generation_and_count()
        if gpus:
            click.echo("GPU Gen, GPU Count, GPU Vendor, Slurm Partition:")
            for gpu in gpus:
                if partition is not None and partition != gpu.partition:
                    continue
                click.echo(
                    f"{gpu.gpu_gen}, {gpu.gpu_count}, {gpu.vendor}, {gpu.partition}"
                )
        else:
            click.echo("No GPUs found")


@cli.command(name="check-gpu")
@click.argument("gpu_type")
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Slurm partition name to filter queries (optional)",
)
def check_gpu(gpu_type: str, partition: str):
    """Check if a specific GPU type exists.

    GPU_TYPE: GPU type to check for (e.g., A100, MI300X)
    """
    unified_info = UnifiedInfo(partition=partition)
    has_gpu = unified_info.has_gpu_type(gpu_type)
    if has_gpu:
        click.echo(f"GPU type {gpu_type} is available in the cluster.")
    else:
        click.echo(f"GPU type {gpu_type} is NOT available in the cluster.")


@cli.command()
def aws():
    """Check if running on AWS and show NCCL settings."""
    aws_cluster_info = AWSClusterInfo()
    is_aws = aws_cluster_info.is_aws_cluster()
    if is_aws:
        click.echo("This is an AWS cluster.")
        nccl_settings = aws_cluster_info.get_aws_nccl_settings()
        click.echo("\nRecommended NCCL settings:")
        click.echo(format_dict(nccl_settings))
    else:
        click.echo("This is NOT an AWS cluster.")


@cli.group(name="job-gen")
def job_gen():
    """Generate job requirements for different job types."""
    pass


@job_gen.group(name="task")
def task():
    """Generate job requirements for a task of a job."""
    pass


@task.command()  # type: ignore[arg-type]
@click.option("--partition", type=str, required=True, help="Partition to query")
@click.option(
    "--tasks-per-node",
    type=int,
    default=1,
    help="Number of tasks per node to request",
)
@click.option(
    "--nodes",
    type=int,
    default=1,
    help="Number nodes to request",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "sbatch", "slurm_directives", "slurm_cli", "submitit"]),
    default="json",
    help="Format to output the job requirements in",
)
@optgroup.group(
    "GPU or CPU Job Request",
    cls=RequiredMutuallyExclusiveOptionGroup,
    help="Only one of --gpus-per-task or --cpus-per-task can be specified. For GPU requests, use --gpus-per-task and cpus-per-task will be generated automatically. For CPU requests, use --cpus-per-task.",
)
@optgroup.option(
    "--gpus-per-task",
    default=None,
    type=click.IntRange(min=1),
    help="Number of GPUs per task to request",
)
@optgroup.option(  # type: ignore[arg-type]
    "--cpus-per-task",
    default=None,
    type=click.IntRange(min=1),
    help="Number of CPUs per task to request",
)
def slurm(
    tasks_per_node: int,
    nodes: int,
    output_format: str,
    partition: str,
    gpus_per_task: Optional[int],
    cpus_per_task: Optional[int],
):
    """Generate job requirements for a task of a Slurm job based on GPU or CPU per task requirements."""
    job_gen_task_slurm_validator(
        partition=partition,
        gpus_per_task=gpus_per_task,
        cpus_per_task=cpus_per_task,
        tasks_per_node=tasks_per_node,
        exit_on_error=True,
    )

    unified_info = UnifiedInfo(partition=partition)
    job_requirements = unified_info.get_task_resource_requirements(
        partition=partition,
        cpus_per_task=cpus_per_task,
        gpus_per_task=gpus_per_task,
        tasks_per_node=tasks_per_node,
        nodes=nodes,
    )

    # Route to the correct format method based on CLI option
    format_methods = {
        "json": job_requirements.to_json,
        "sbatch": job_requirements.to_sbatch,
        "slurm_directives": job_requirements.to_sbatch,
        "slurm_cli": job_requirements.to_srun,
        "submitit": job_requirements.to_submitit,
    }
    click.echo(format_methods[output_format]())


def main():
    """Main entry point for the Slurm information CLI."""
    cli()


if __name__ == "__main__":
    main()
