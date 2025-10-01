#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from typing import Any, Dict

import click

from clusterscope.cluster_info import AWSClusterInfo, UnifiedInfo
from clusterscope.lib import get_cpu_memory_usage_percentage

# Default CPU memory usage percentage for CLI operations
DEFAULT_CPU_MEMORY_USAGE_PERCENTAGE = 95.0


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
    unified_info = UnifiedInfo(partition=partition)
    cpus_per_node = unified_info.get_cpus_per_node()
    click.echo("CPU counts per node:")
    click.echo(cpus_per_node)


@cli.command()
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Slurm partition name to filter queries (optional)",
)
@click.option(
    "--unit",
    type=click.Choice(["MB", "GB"]),
    default="GB",
    help="Unit to display CPU memory in (default: GB)",
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed CPU memory information including total and available",
)
@click.option(
    "--available-only",
    is_flag=True,
    help="Show only available CPU memory for applications",
)
def mem(partition: str, unit: str, detailed: bool, available_only: bool):
    """Show CPU memory (RAM) information per node.

    Note: This shows system RAM, not GPU memory. Use 'gpus' command for GPU information.
    The percentage of memory made available to applications is configured in the library
    (default: 95%). Use clusterscope.set_cpu_memory_usage_percentage() to change it.
    """
    unified_info = UnifiedInfo(partition=partition)

    if detailed:
        # Show comprehensive CPU memory information using current configured percentage
        current_percentage = get_cpu_memory_usage_percentage()
        info = unified_info.get_cpu_memory_info(current_percentage)
        click.echo("CPU Memory (RAM) Information:")
        click.echo(
            f"  Total CPU Memory: {info['total_cpu_mb']} MB ({info['total_cpu_gb']} GB)"
        )
        click.echo(
            f"  Available for Apps ({info['percentage']}%): {info['available_cpu_mb']} MB ({info['available_cpu_gb']} GB)"
        )
        click.echo(
            f"  Reserved for System: {info['total_cpu_mb'] - info['available_cpu_mb']} MB ({info['total_cpu_gb'] - info['available_cpu_gb']:.2f} GB)"
        )
        click.echo(
            f"  Note: Use clusterscope.set_cpu_memory_usage_percentage() to change the {info['percentage']}% setting"
        )
    elif available_only:
        # Show available CPU memory using current configured percentage
        current_percentage = get_cpu_memory_usage_percentage()
        if unit == "MB":
            available_mem_mb = unified_info.get_available_cpu_memory_MB(
                current_percentage
            )
            click.echo(f"Available CPU memory for applications: {available_mem_mb} MB")
        else:  # GB
            available_mem_gb = unified_info.get_available_cpu_memory_GB(
                current_percentage
            )
            click.echo(
                f"Available CPU memory for applications: {available_mem_gb:.2f} GB"
            )
        click.echo(
            f"Note: Currently using {current_percentage}% of total memory. Use clusterscope.set_cpu_memory_usage_percentage() to change."
        )
    else:
        # Show total CPU memory (default behavior)
        total_mem = unified_info.get_mem_per_node_MB()
        if unit == "MB":
            click.echo(f"Total CPU memory per node: {total_mem} MB")
        else:  # GB
            click.echo(f"Total CPU memory per node: {total_mem / 1024:.2f} GB")
        # Show available memory using current configured percentage
        current_percentage = get_cpu_memory_usage_percentage()
        divisor = 1024.0 if unit == "GB" else 1.0
        available_mem = (total_mem * (current_percentage / 100.0)) / divisor

        click.echo(
            f"Available for applications ({current_percentage}%): {available_mem:.2f} {unit}"
        )


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
    unified_info = UnifiedInfo(partition=partition)

    if vendor:
        vendor_info = unified_info.get_gpu_vendor()
        click.echo(f"Primary GPU vendor: {vendor_info}")
    elif counts:
        gpu_counts = unified_info.get_gpu_generation_and_count()
        if gpu_counts:
            click.echo("GPU counts by type:")
            for gpu_type, count in sorted(gpu_counts.items()):
                click.echo(f"  {gpu_type}: {count}")
        else:
            click.echo("No GPUs found")
    elif generations:
        gpu_counts = unified_info.get_gpu_generation_and_count()
        if gpu_counts:
            click.echo("GPU generations available:")
            for gen in sorted(gpu_counts.keys()):
                click.echo(f"- {gen}")
        else:
            click.echo("No GPUs found")
    else:
        # Default: show both vendor and detailed info
        vendor_info = unified_info.get_gpu_vendor()
        gpu_counts = unified_info.get_gpu_generation_and_count()

        click.echo(f"GPU vendor: {vendor_info}")
        if gpu_counts:
            click.echo("GPU information:")
            for gpu_type, count in sorted(gpu_counts.items()):
                click.echo(f"  {gpu_type}: {count}")
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


@job_gen.command()
@click.option("--num-gpus", type=int, required=True, help="Number of GPUs to request")
@click.option(
    "--num-tasks-per-node",
    type=int,
    default=1,
    help="Number of tasks per node to request",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "sbatch", "srun", "submitit"]),
    default="json",
    help="Format to output the job requirements in",
)
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Slurm partition name to filter queries (optional)",
)
def task(num_gpus: int, num_tasks_per_node: int, output_format: str, partition: str):
    """Generate job requirements for a task job."""
    unified_info = UnifiedInfo(partition=partition)
    job_requirements = unified_info.get_task_resource_requirements(
        num_gpus=num_gpus,
        num_tasks_per_node=num_tasks_per_node,
    )

    # Route to the correct format method based on CLI option
    format_methods = {
        "json": job_requirements.to_json,
        "sbatch": job_requirements.to_sbatch,
        "srun": job_requirements.to_srun,
        "submitit": job_requirements.to_submitit,
    }
    click.echo(format_methods[output_format]())


@job_gen.command()
@click.option(
    "--num-gpus-per-task", type=int, required=True, help="Number of GPUs per task"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "sbatch", "srun", "submitit"]),
    default="json",
    help="Format to output the job requirements in",
)
@click.option(
    "--partition",
    type=str,
    default=None,
    help="Slurm partition name to filter queries (optional)",
)
def array(num_gpus_per_task: int, output_format: str, partition: str):
    """Generate job requirements for an array job."""
    unified_info = UnifiedInfo(partition=partition)
    job_requirements = unified_info.get_array_job_requirements(
        num_gpus_per_task=num_gpus_per_task,
    )

    # Route to the correct format method based on CLI option
    format_methods = {
        "json": job_requirements.to_json,
        "sbatch": job_requirements.to_sbatch,
        "srun": job_requirements.to_srun,
        "submitit": job_requirements.to_submitit,
    }
    click.echo(format_methods[output_format]())


def main():
    """Main entry point for the Slurm information CLI."""
    cli()


if __name__ == "__main__":
    main()
