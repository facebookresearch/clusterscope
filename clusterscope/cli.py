#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import sys
from typing import Any, Dict

import click

from clusterscope.cluster_info import AWSClusterInfo, UnifiedInfo


def format_dict(data: Dict[str, Any]) -> str:
    """Format a dictionary for display."""
    return json.dumps(data, indent=2)


@click.group()
def cli():
    """Command-line tool to query Slurm cluster information."""
    pass


@cli.command()
def info():
    """Show basic cluster information."""
    try:
        unified_info = UnifiedInfo()
        cluster_name = unified_info.get_cluster_name()
        slurm_version = unified_info.get_slurm_version()
        click.echo(f"Cluster Name: {cluster_name}")
        click.echo(f"Slurm Version: {slurm_version}")
    except RuntimeError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def cpus():
    """Show CPU counts per node."""
    try:
        unified_info = UnifiedInfo()
        cpus_per_node = unified_info.get_cpus_per_node()
        click.echo("CPU counts per node:")
        click.echo(cpus_per_node)
    except RuntimeError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def mem():
    """Show memory information per node."""
    try:
        unified_info = UnifiedInfo()
        mem_per_node = unified_info.get_mem_per_node_MB()
        click.echo("Mem per node MB:")
        click.echo(mem_per_node)
    except RuntimeError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--generations", is_flag=True, help="Show only GPU generations")
@click.option("--counts", is_flag=True, help="Show only GPU counts by type")
@click.option("--vendor", is_flag=True, help="Show GPU vendor information")
def gpus(generations, counts, vendor):
    """Show GPU information."""
    try:
        unified_info = UnifiedInfo()

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
    except RuntimeError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command(name="check-gpu")
@click.argument("gpu_type")
def check_gpu(gpu_type):
    """Check if a specific GPU type exists.

    GPU_TYPE: GPU type to check for (e.g., A100, MI300X)
    """
    try:
        unified_info = UnifiedInfo()
        has_gpu = unified_info.has_gpu_type(gpu_type)
        if has_gpu:
            click.echo(f"GPU type {gpu_type} is available in the cluster.")
        else:
            click.echo(f"GPU type {gpu_type} is NOT available in the cluster.")
    except RuntimeError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def aws():
    """Check if running on AWS and show NCCL settings."""
    try:
        aws_cluster_info = AWSClusterInfo()
        is_aws = aws_cluster_info.is_aws_cluster()
        if is_aws:
            click.echo("This is an AWS cluster.")
            nccl_settings = aws_cluster_info.get_aws_nccl_settings()
            click.echo("\nRecommended NCCL settings:")
            click.echo(format_dict(nccl_settings))
        else:
            click.echo("This is NOT an AWS cluster.")
    except RuntimeError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


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
def task(num_gpus: int, num_tasks_per_node: int, output_format: str):
    """Generate job requirements for a task job."""
    try:
        unified_info = UnifiedInfo()
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
    except (RuntimeError, ValueError) as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


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
def array(num_gpus_per_task: int, output_format: str):
    """Generate job requirements for an array job."""
    try:
        unified_info = UnifiedInfo()
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
    except (RuntimeError, ValueError) as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the Slurm information CLI."""
    try:
        cli()
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
