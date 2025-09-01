#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import sys
from typing import Any, Dict

from clusterscope.cluster_info import AWSClusterInfo, UnifiedInfo


def format_dict(data: Dict[str, Any]) -> str:
    """Format a dictionary for display."""
    return json.dumps(data, indent=2)


def main():
    """Main entry point for the Slurm information CLI."""
    parser = argparse.ArgumentParser(
        description="Command-line tool to query Slurm cluster information"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show basic cluster information")

    # CPUs command
    cpus_parser = subparsers.add_parser("cpus", help="Show CPU counts per node")

    # GPUs command
    gpus_parser = subparsers.add_parser("gpus", help="Show GPU information")
    gpus_parser.add_argument(
        "--generations", action="store_true", help="Show only GPU generations"
    )
    gpus_parser.add_argument(
        "--counts", action="store_true", help="Show only GPU counts by type"
    )
    gpus_parser.add_argument(
        "--vendor", action="store_true", help="Show GPU vendor information"
    )

    # Check GPU command
    check_gpu_parser = subparsers.add_parser(
        "check-gpu", help="Check if a specific GPU type exists"
    )
    check_gpu_parser.add_argument(
        "gpu_type", help="GPU type to check for (e.g., A100, MI300X)"
    )

    # Memory command
    mem_parser = subparsers.add_parser("mem", help="Show memory information per node")

    # AWS command
    aws_parser = subparsers.add_parser(
        "aws", help="Check if running on AWS and show NCCL settings"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        unified_info = UnifiedInfo()
        aws_cluster_info = AWSClusterInfo()

        if args.command == "info":
            cluster_name = unified_info.get_cluster_name()
            slurm_version = unified_info.get_slurm_version()
            print(f"Cluster Name: {cluster_name}")
            print(f"Slurm Version: {slurm_version}")

        elif args.command == "cpus":
            cpus_per_node = unified_info.get_cpus_per_node()
            print("CPU counts per node:")
            print(cpus_per_node)

        elif args.command == "mem":
            mem_per_node = unified_info.get_mem_per_node_MB()
            print("Mem per node MB:")
            print(mem_per_node)

        elif args.command == "gpus":
            if args.vendor:
                vendor = unified_info.get_gpu_vendor()
                print(f"Primary GPU vendor: {vendor}")
            elif args.counts:
                gpu_counts = unified_info.get_gpu_generation_and_count()
                if gpu_counts:
                    print("GPU counts by type:")
                    for gpu_type, count in sorted(gpu_counts.items()):
                        print(f"  {gpu_type}: {count}")
                else:
                    print("No GPUs found")
            elif args.generations:
                gpu_counts = unified_info.get_gpu_generation_and_count()
                if gpu_counts:
                    print("GPU generations available:")
                    for gen in sorted(gpu_counts.keys()):
                        print(f"- {gen}")
                else:
                    print("No GPUs found")
            else:
                # Default: show both vendor and detailed info
                vendor = unified_info.get_gpu_vendor()
                gpu_counts = unified_info.get_gpu_generation_and_count()

                print(f"GPU vendor: {vendor}")
                if gpu_counts:
                    print("GPU information:")
                    for gpu_type, count in sorted(gpu_counts.items()):
                        print(f"  {gpu_type}: {count}")
                else:
                    print("No GPUs found")

        elif args.command == "check-gpu":
            gpu_type = args.gpu_type
            has_gpu = unified_info.has_gpu_type(gpu_type)
            if has_gpu:
                print(f"GPU type {gpu_type} is available in the cluster.")
            else:
                print(f"GPU type {gpu_type} is NOT available in the cluster.")

        elif args.command == "aws":
            is_aws = aws_cluster_info.is_aws_cluster()
            if is_aws:
                print("This is an AWS cluster.")
                nccl_settings = aws_cluster_info.get_aws_nccl_settings()
                print("\nRecommended NCCL settings:")
                print(format_dict(nccl_settings))
            else:
                print("This is NOT an AWS cluster.")

        return 0

    except RuntimeError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
