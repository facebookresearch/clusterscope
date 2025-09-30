# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import math
import os
import platform
import shutil
import subprocess
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, NamedTuple, Optional, Set, Union

from clusterscope.cache import fs_cache


class ResourceShape(NamedTuple):
    """Represents resource requirements for a job in Slurm SBATCH format."""

    cpu_cores: int
    memory: str
    tasks_per_node: int

    def _parse_memory_to_gb(self) -> int:
        """Parse memory string and convert to GB.

        Returns:
            int: Memory in GB
        """
        mem_value = self.memory.rstrip("GT")
        if self.memory.endswith("T"):
            return int(mem_value) * 1024
        elif self.memory.endswith("G"):
            return int(mem_value)
        else:
            raise RuntimeError(f"Invalid memory format: {self.memory}")

    def to_json(self) -> str:
        """Convert ResourceShape to JSON format.

        Returns:
            str: JSON representation of the resource requirements
        """
        mem_gb = self._parse_memory_to_gb()

        data = {
            "cpu_cores": self.cpu_cores,
            "memory": self.memory,
            "tasks_per_node": self.tasks_per_node,
            "mem_gb": mem_gb,
        }
        return json.dumps(data, indent=2)

    def to_sbatch(self) -> str:
        """Convert ResourceShape to SBATCH script format.

        Returns:
            str: SBATCH script with resource directives
        """
        lines = [
            "#!/bin/bash",
            f"#SBATCH --cpus-per-task={self.cpu_cores}",
            f"#SBATCH --mem={self.memory}",
            f"#SBATCH --ntasks-per-node={self.tasks_per_node}",
        ]
        return "\n".join(lines)

    def to_srun(self) -> str:
        """Convert ResourceShape to srun command format.

        Returns:
            str: srun command with resource specifications
        """
        cmd_parts = [
            "srun",
            f"--cpus-per-task={self.cpu_cores}",
            f"--mem={self.memory}",
            f"--ntasks-per-node={self.tasks_per_node}",
        ]
        return " ".join(cmd_parts)

    def to_submitit(self) -> str:
        """Convert ResourceShape to submitit parameters format.

        Returns:
            str: JSON representation of submitit parameters
        """
        mem_gb = self._parse_memory_to_gb()

        params = {
            "cpus_per_task": self.cpu_cores,
            "mem_gb": mem_gb,
            "tasks_per_node": self.tasks_per_node,
        }
        return json.dumps(params, indent=2)


# Common NVIDIA GPU types
NVIDIA_GPU_TYPES = {
    "A100": "A100",
    "A40": "A40",
    "A30": "A30",
    "A10": "A10",
    "V100": "V100",
    "P100": "P100",
    "T4": "T4",
    "H100": "H100",
    "H200": "H200",
}

# Common AMD GPU types
AMD_GPU_TYPES = {
    "MI300X": "MI300X",
    "MI300A": "MI300A",
    "MI300": "MI300",
    "MI250X": "MI250X",
    "MI210": "MI210",
    "MI100": "MI100",
    "RX7900XTX": "RX 7900",
    "RX6900XT": "RX 6900",
}


def run_cli(
    cmd: List[str],
    text: bool = True,
    timeout: int = 60,
    stderr: Union[int, None] = None,
) -> str:
    """
    Run a CLI command after verifying it's available.

    Args:
        cmd: List of command and arguments
        text: Whether to return text output (default: True)
        timeout: Command timeout in seconds (default: 60)
        stderr: How to handle stderr (default: None)

    Returns:
        str: Command output

    Raises:
        RuntimeError: If command is not available or execution fails
    """
    if not cmd:
        raise RuntimeError("Command list cannot be empty")

    command_name = cmd[0]

    # Check if command is available
    if shutil.which(command_name) is None:
        raise RuntimeError(f"Command '{command_name}' is not available on this system")

    try:
        result = subprocess.check_output(cmd, text=text, timeout=timeout, stderr=stderr)
        return result
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command '{' '.join(cmd)}' failed with return code {e.returncode}: {e.output}"
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Command '{' '.join(cmd)}' timed out after {timeout} seconds"
        )
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to execute command '{' '.join(cmd)}': {str(e)}")


class UnifiedInfo:
    def __init__(self, partition: Optional[str] = None):
        """Initialize the UnifiedInfo instance.

        Args:
            partition (str, optional): Slurm partition name to filter queries.
                                     If None, queries all partitions.
        """
        self.partition = partition
        self.local_node_info = LocalNodeInfo()
        self.slurm_cluster_info = SlurmClusterInfo(partition=partition)
        self.is_slurm_cluster = self.slurm_cluster_info.verify_slurm_available()
        self.has_nvidia_gpus = self.local_node_info.has_nvidia_gpus()
        self.has_amd_gpus = self.local_node_info.has_amd_gpus()
        self.aws_cluster_info = AWSClusterInfo()

    def get_cluster_name(self) -> str:
        """Get the name of the Slurm cluster. Returns `local-node` if not a Slurm cluster.

        Returns:
            str: The name of the Slurm cluster.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_cluster_name()
        if "GITHUB_ACTIONS" in os.environ:
            return "github"
        if platform.system() == "Darwin":
            return "macos"
        return "local-node"

    def get_slurm_version(self) -> str:
        """Get the slurm version. Returns `0` if not a Slurm cluster.

        Returns:
            str: Slurm version as a string: "24.11.4"
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_slurm_version()
        return "0"

    def get_cpus_per_node(self) -> int:
        """Get the number of CPUs for each node in the cluster. Returns 0 if not a Slurm cluster.

        Returns:
            int: The number of CPUs per node, assuming all nodes have the same CPU count.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_cpus_per_node()
        return self.local_node_info.get_cpu_count()

    def get_mem_per_node_MB(self) -> int:
        """Return the lowest amount of mem configured across all nodes in the cluster. Returns 0 if not a Slurm cluster.

        Returns:
            int: The memory per node in the cluster.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_mem_per_node_MB()
        return self.local_node_info.get_mem_MB()

    def get_gpu_generation_and_count(self) -> Dict[str, int]:
        """Get the number of GPUs on the slurm cluster node.

        Returns:
            dict: A dictionary with GPU generation as keys and counts as values.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_gpu_generation_and_count()
        if self.has_nvidia_gpus or self.has_amd_gpus:
            return self.local_node_info.get_gpu_generation_and_count()
        return {}

    def has_gpu_type(self, gpu_type: str) -> bool:
        """Check if a specific GPU type is available.

        Args:
            gpu_type (str): The GPU type to check for (e.g., "A100", "MI300X", "V100")

        Returns:
            bool: True if the GPU type is available, False otherwise
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.has_gpu_type(gpu_type)
        else:
            return self.local_node_info.has_gpu_type(gpu_type)

    def get_gpu_vendor(self) -> str:
        """Get the primary GPU vendor available on the system.

        Returns:
            str: 'nvidia', 'amd', or 'none'
        """
        return self.local_node_info.get_gpu_vendor()

    def get_total_gpus_per_node(self) -> int:
        """Get the total number of GPUs available per node.

        Returns:
            int: Total number of GPUs per node. Returns 8 as default if no GPUs are detected.
        """
        gpu_counts = self.get_gpu_generation_and_count()
        if not gpu_counts:
            # Default to 8 if no GPUs detected (common configuration)
            return 8

        # Sum all GPU counts across different types
        total_gpus = sum(gpu_counts.values())
        return max(total_gpus, 1)  # Ensure at least 1 to avoid division by zero

    def get_task_resource_requirements(
        self, num_gpus: int, num_tasks_per_node: int = 1
    ) -> ResourceShape:
        """Calculate resource requirements for better GPU packing based on node's GPU configuration.

        Uses proportional allocation of node resources per GPU based on the node's total GPU count.
        For the maximum GPU count, returns all available node resources.
        Returns values in Slurm SBATCH format for direct use in job scripts.

        NOTE: For array jobs, use get_array_job_requirements() instead, which provides
        per-array-element resource allocation.

        Args:
            num_gpus (int): Total number of GPUs required per node (1 to max available)
            num_tasks_per_node (int): Number of tasks to run per node (default: 1)

        Returns:
            ResourceShape: Tuple containing CPU cores per task (int), memory per node (str),
                          and tasks per node (int) in Slurm format
                          e.g., ResourceShape(cpu_cores=24, memory="225G", tasks_per_node=1)

        Raises:
            ValueError: If num_gpus is not between 1 and max available GPUs, or if num_tasks_per_node < 1
        """
        # Get the total number of GPUs available per node
        max_gpus_per_node = self.get_total_gpus_per_node()

        if not (1 <= num_gpus <= max_gpus_per_node):
            raise ValueError(f"num_gpus must be between 1 and {max_gpus_per_node}")

        if num_tasks_per_node < 1:
            raise ValueError("num_tasks_per_node must be at least 1")

        # Get total resources per node
        total_cpu_cores = self.get_cpus_per_node()
        total_ram_mb = self.get_mem_per_node_MB()

        if num_gpus == max_gpus_per_node:
            # For max GPUs, use all available resources
            total_required_cpu_cores = total_cpu_cores
            total_required_ram_mb = total_ram_mb
        else:
            # Calculate per-GPU allocation based on actual GPU count per node
            cpu_cores_per_gpu = total_cpu_cores / max_gpus_per_node
            ram_mb_per_gpu = total_ram_mb / max_gpus_per_node

            # Calculate total requirements for the requested number of GPUs
            total_required_cpu_cores = math.ceil(cpu_cores_per_gpu * num_gpus)
            total_required_ram_mb = math.ceil(ram_mb_per_gpu * num_gpus)

        # Calculate per-task CPU allocation
        cpu_cores_per_task = total_required_cpu_cores / num_tasks_per_node

        # Convert to Slurm SBATCH format
        # CPU cores per task: Round up to ensure we don't under-allocate
        sbatch_cpu_cores = math.ceil(cpu_cores_per_task)

        # Memory per node: Convert MB to GB and format for Slurm
        # Note: Memory is allocated per node, not per task in most Slurm configurations
        required_ram_gb = total_required_ram_mb / 1024
        if required_ram_gb >= 1024:
            # Use TB format for very large memory
            sbatch_memory = f"{required_ram_gb / 1024:.0f}T"
        else:
            # Use GB format (most common)
            sbatch_memory = f"{required_ram_gb:.0f}G"

        return ResourceShape(
            cpu_cores=sbatch_cpu_cores,
            memory=sbatch_memory,
            tasks_per_node=num_tasks_per_node,
        )

    def get_array_job_requirements(self, num_gpus_per_task: int) -> ResourceShape:
        """Calculate resource requirements for array jobs with optimal GPU packing.

        For array jobs, each array element gets its own resource allocation.
        This method calculates per-array-element resources based on proportional
        allocation of node resources per GPU. For maximum GPUs, returns all
        available node resources.

        Args:
            num_gpus_per_task (int): Number of GPUs required per array task (1 to max available)

        Returns:
            ResourceShape: Tuple containing CPU cores per array element (int),
                          memory per array element (str), and tasks_per_node=1
                          e.g., ResourceShape(cpu_cores=24, memory="225G", tasks_per_node=1)

        Raises:
            ValueError: If num_gpus_per_task is not between 1 and max available GPUs
        """
        # Get the total number of GPUs available per node
        max_gpus_per_node = self.get_total_gpus_per_node()

        if not (1 <= num_gpus_per_task <= max_gpus_per_node):
            raise ValueError(
                f"num_gpus_per_task must be between 1 and {max_gpus_per_node}"
            )

        # Get total resources per node
        total_cpu_cores = self.get_cpus_per_node()
        total_ram_mb = self.get_mem_per_node_MB()

        if num_gpus_per_task == max_gpus_per_node:
            # For max GPUs, use all available resources
            required_cpu_cores = total_cpu_cores
            required_ram_mb = total_ram_mb
        else:
            # Calculate per-GPU allocation based on actual GPU count per node
            cpu_cores_per_gpu = total_cpu_cores / max_gpus_per_node
            ram_mb_per_gpu = total_ram_mb / max_gpus_per_node

            # Calculate requirements per array element
            required_cpu_cores = math.ceil(cpu_cores_per_gpu * num_gpus_per_task)
            required_ram_mb = math.ceil(ram_mb_per_gpu * num_gpus_per_task)

        # Convert to Slurm SBATCH format
        # CPU cores: Round up to ensure we don't under-allocate
        sbatch_cpu_cores = math.ceil(required_cpu_cores)

        # Memory: Convert MB to GB and format for Slurm
        required_ram_gb = required_ram_mb / 1024
        if required_ram_gb >= 1024:
            # Use TB format for very large memory
            sbatch_memory = f"{required_ram_gb / 1024:.0f}T"
        else:
            # Use GB format (most common)
            sbatch_memory = f"{required_ram_gb:.0f}G"

        # Array jobs always have 1 task per array element
        return ResourceShape(
            cpu_cores=sbatch_cpu_cores, memory=sbatch_memory, tasks_per_node=1
        )


class DarwinInfo:
    def get_cpu_count(self, timeout: int = 60) -> int:
        """Get the number of CPUs on the local node.

        Returns:
            int: The number of CPUs on the local node.

        Raises:
            RuntimeError: If unable to retrieve CPU information.
        """
        try:
            result = run_cli(["sysctl", "-n", "hw.ncpu"], text=True, timeout=timeout)
            return int(result.strip())
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get CPU information: {str(e)}")

    def get_mem_MB(self, timeout: int = 60) -> int:
        """Get the amount of memory on the local node.

        Returns:
            int: The amount of memory on the local node.

        Raises:
            RuntimeError: If unable to retrieve memory information.
        """
        try:
            result = run_cli(["sysctl", "-n", "hw.memsize"], text=True, timeout=timeout)
            return int(result.strip()) // 1024 // 1024
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get memory information: {str(e)}")


class LinuxInfo:
    def get_cpu_count(self, timeout: int = 60) -> int:
        """Get the number of CPUs on the local node.

        Returns:
            int: The number of CPUs on the local node.

        Raises:
            RuntimeError: If unable to retrieve CPU information.
        """
        try:
            result = run_cli(["nproc", "--all"], text=True, timeout=timeout)
            return int(result.strip())
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get CPU information: {str(e)}")

    def get_mem_MB(self, timeout: int = 60) -> int:
        """Get the amount of memory on the local node.

        Returns:
            int: The amount of memory on the local node.

        Raises:
            RuntimeError: If unable to retrieve memory information.
        """
        try:
            result = run_cli(["free", "-m"], text=True, timeout=timeout)
            for line in result.strip().split("\n"):
                if "Mem:" in line:
                    parts = line.split()
                    return int(parts[1])
            raise RuntimeError("Could not find memory information in free output")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get memory information: {str(e)}")


class LocalNodeInfo:
    """A class to provide information about the local node.

    This class offers methods to query various aspects of the local node,
    such as CPU and GPU information.
    """

    @lru_cache(maxsize=1)
    def has_nvidia_gpus(self) -> bool:
        """Verify that nvidia GPU is available on the system."""
        try:
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    @lru_cache(maxsize=1)
    def has_amd_gpus(self) -> bool:
        """Verify that AMD GPU is available on the system."""
        try:
            subprocess.run(
                ["rocm-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def get_gpu_vendor(self) -> str:
        """Determine the primary GPU vendor on the system.

        Returns:
            str: 'nvidia', 'amd', or 'none'
        """
        if self.has_nvidia_gpus():
            return "nvidia"
        elif self.has_amd_gpus():
            return "amd"
        else:
            return "none"

    @fs_cache(var_name="LOCAL_NODE_CPU_COUNT")
    def get_cpu_count(self, timeout: int = 60) -> int:
        """Get the number of CPUs on the local node.

        Returns:
            int: The number of CPUs on the local node.

        Raises:
            RuntimeError: If unable to retrieve CPU information.
        """
        system = platform.system()
        if system == "Linux":
            return LinuxInfo().get_cpu_count(timeout)
        if system == "Darwin":
            return DarwinInfo().get_cpu_count(timeout)
        raise RuntimeError(f"Unsupported system: {system}")

    @fs_cache(var_name="LOCAL_NODE_MEM_MB")
    def get_mem_MB(self, timeout: int = 60) -> int:
        """Get the amount of memory on the local node.

        Returns:
            int: The amount of memory on the local node.

        Raises:
            RuntimeError: If unable to retrieve memory information.
        """
        system = platform.system()
        if system == "Linux":
            mem = LinuxInfo().get_mem_MB(timeout)
        elif system == "Darwin":
            mem = DarwinInfo().get_mem_MB(timeout)
        else:
            raise RuntimeError(f"Unsupported system: {system}")
        assert 0 < mem <= 10**12, f"Likely invalid memory: {mem}"
        return mem

    def get_nvidia_gpu_info(self, timeout: int = 60) -> Dict[str, int]:
        """Get NVIDIA GPU information using nvidia-smi.

        Returns:
            Dict[str, int]: Dictionary with GPU generation as keys and counts as values.
        """
        # Check if NVIDIA GPUs are available
        if not self.has_nvidia_gpus():
            try:
                # Try to run nvidia-smi command
                result = run_cli(
                    ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                    text=True,
                    timeout=timeout,
                )
            except RuntimeError:
                raise RuntimeError("No NVIDIA GPUs found")
        try:
            result = run_cli(
                ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                text=True,
                timeout=timeout,
            )

            gpu_info: Dict[str, int] = defaultdict(int)
            for line in result.strip().split("\n"):
                if line.strip():
                    # Extract GPU generation from full name
                    # Examples: "NVIDIA A100-SXM4-40GB" -> "A100"
                    #          "Tesla V100-SXM2-16GB" -> "V100"
                    gpu_name_upper = line.strip().upper()

                    # Check for known NVIDIA GPU types
                    found_gpu = False
                    for gpu_key, gpu_pattern in NVIDIA_GPU_TYPES.items():
                        if gpu_pattern in gpu_name_upper:
                            gpu_info[gpu_key] += 1
                            found_gpu = True
                            break

                    # If no known GPU type was found
                    if not found_gpu:
                        # Generic fallback - try to extract model number
                        words = gpu_name_upper.split()
                        for word in words:
                            if any(char.isdigit() for char in word) and len(word) > 2:
                                gpu_info[word] += 1
                                break

            return gpu_info
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get NVIDIA GPU information: {str(e)}")

    def get_amd_gpu_info(self, timeout: int = 60) -> Dict[str, int]:
        """Get AMD GPU information using rocm-smi.

        Returns:
            Dict[str, int]: Dictionary with GPU generation as keys and counts as values.
        """
        # Check if AMD GPUs are available
        if not self.has_amd_gpus():
            try:
                # Try to run rocm-smi command
                result = run_cli(
                    ["rocm-smi", "--showproductname"],
                    text=True,
                    timeout=timeout,
                )
            except RuntimeError:
                raise RuntimeError("No AMD GPUs found")
        try:
            result = run_cli(
                ["rocm-smi", "--showproductname"], text=True, timeout=timeout
            )

            gpu_info: Dict[str, int] = defaultdict(int)
            for line in result.strip().split("\n"):
                if "GPU" in line and ":" in line:
                    # Parse lines like "GPU[0]: AMD Instinct MI300X"
                    parts = line.split(":")
                    if len(parts) >= 2:
                        gpu_name = parts[1].strip()
                        # Extract GPU generation from full name
                        # Examples: "AMD Instinct MI300X" -> "MI300X"
                        #          "AMD Instinct MI250X" -> "MI250X"
                        #          "AMD Radeon RX 7900 XTX" -> "RX7900XTX"
                        gpu_name_upper = gpu_name.upper()

                        # Check for known AMD GPU types
                        found_gpu = False
                        for gpu_key, gpu_pattern in AMD_GPU_TYPES.items():
                            if gpu_pattern in gpu_name_upper:
                                gpu_info[gpu_key] += 1
                                found_gpu = True
                                break

                        # If no known GPU type was found
                        if not found_gpu:
                            # Generic fallback - try to extract model number
                            words = gpu_name_upper.split()
                            for word in words:
                                if (
                                    any(char.isdigit() for char in word)
                                    and len(word) > 2
                                ):
                                    gpu_info[word] += 1
                                    break

            return gpu_info
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get AMD GPU information: {str(e)}")

    def get_gpu_generation_and_count(self, timeout: int = 60) -> Dict[str, int]:
        """Get GPU information for all available GPUs on the local node.

        Returns:
            Dict[str, int]: Dictionary with GPU generation as keys and counts as values.

        Raises:
            RuntimeError: If unable to retrieve GPU information.
        """
        gpu_info: Dict[str, int] = {}

        # Try NVIDIA GPUs
        if self.has_nvidia_gpus():
            try:
                nvidia_info = self.get_nvidia_gpu_info(timeout)
                gpu_info.update(nvidia_info)
            except RuntimeError as e:
                logging.warning(f"Failed to get NVIDIA GPU info: {e}")

        # Try AMD GPUs
        if self.has_amd_gpus():
            try:
                amd_info = self.get_amd_gpu_info(timeout)
                gpu_info.update(amd_info)
            except RuntimeError as e:
                logging.warning(f"Failed to get AMD GPU info: {e}")

        # Raise an error if no GPUs were found
        if not gpu_info:
            logging.warning("No GPUs found or unable to retrieve GPU information")

        return gpu_info

    def has_gpu_type(self, gpu_type: str) -> bool:
        """Check if a specific GPU type is available on the local node.

        Args:
            gpu_type (str): The GPU type to check for (e.g., "A100", "MI300X")

        Returns:
            bool: True if the GPU type is available, False otherwise
        """
        try:
            gpu_counts = self.get_gpu_generation_and_count()
            return gpu_type.upper() in [k.upper() for k in gpu_counts.keys()]
        except RuntimeError:
            return False


class SlurmClusterInfo:
    """A class to provide information about the Slurm cluster configuration.

    This class offers methods to query various aspects of a Slurm cluster,
    such as cluster name, available resources, and node configurations.
    """

    def __init__(self, partition: Optional[str] = None):
        """Initialize the Cluster instance.

        Args:
            partition (str, optional): Slurm partition name to filter queries.
                                     If None, queries all partitions.
        """
        self.partition = partition
        self.is_slurm_cluster = False
        if shutil.which("sinfo") is not None:
            self.is_slurm_cluster = self.verify_slurm_available()

    @lru_cache(maxsize=1)
    def verify_slurm_available(self) -> bool:
        """Verify that Slurm commands are available on the system."""
        try:
            subprocess.run(
                ["sinfo", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    @fs_cache(var_name="SLURM_VERSION")
    def get_slurm_version(self, timeout: int = 60) -> str:
        """Get the slurm version

        ```
        $ sinfo -V
        slurm 24.11.4
        ```

        Returns:
            str: Slurm version as a string: "24.11.4"

        Raises:
            RuntimeError: If unable to retrieve cluster information.
        """
        try:
            slurm_version = run_cli(["sinfo", "-V"], text=True, timeout=timeout)
            return str(slurm_version.strip().split(" ")[1])
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get slurm version: {str(e)}")

    @fs_cache(var_name="SLURM_CLUSTER_NAME")
    def get_cluster_name(self) -> str:
        """Get the name of the Slurm cluster.

        Returns:
            str: The name of the Slurm cluster.

        Raises:
            RuntimeError: If unable to retrieve cluster information.
        """
        try:
            result = subprocess.run(
                ["scontrol", "show", "config"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            for line in result.stdout.split("\n"):
                if "ClusterName" in line:
                    return line.split("=")[1].strip()

            raise RuntimeError("Could not find cluster name in scontrol output")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to get cluster name: {str(e)}")

    @fs_cache(var_name="SLURM_MEM_PER_NODE_MB")
    def get_mem_per_node_MB(self) -> int:
        """Get the lowest memory available per node in the cluster.

        Returns:
            int: The memory per node in the cluster.

        Raises:
            RuntimeError: If unable to retrieve node information.
        """
        try:
            cmd = ["sinfo", "-o", "%100m", "--noconvert", "--noheader"]
            if self.partition:
                cmd.extend(["-p", self.partition])

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            logging.debug("Parsing node information...")
            for line in result.stdout.splitlines():
                mem = int(line.strip("+ "))
                return mem
            raise RuntimeError(f"No mem information found in: {result.stdout}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logging.error(f"Failed to get Slurm memory information: {str(e)}")
            raise RuntimeError(f"Failed to get Slurm memory information: {str(e)}")

    @fs_cache(var_name="SLURM_CPUS_PER_NODE")
    def get_cpus_per_node(self) -> int:
        """Get the minimum number of CPUs for each node in the cluster.

        Returns:
            int: The number of CPUs per node, assuming all nodes have the same CPU count.

        Raises:
            RuntimeError: If unable to retrieve node information or if nodes have different CPU counts.
        """
        try:
            cmd = ["sinfo", "-o", "%100c", "--noheader"]
            if self.partition:
                cmd.extend(["-p", self.partition])

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            logging.debug("Parsing node information...")
            for line in result.stdout.splitlines():
                cpus = int(line.strip("+ "))
                return cpus
            raise RuntimeError(f"No CPU information found in: {result.stdout}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logging.error(f"Failed to get CPU information: {str(e)}")
            raise RuntimeError(f"Failed to get CPU information: {str(e)}")

    def get_gpu_generation_and_count(self) -> Dict[str, int]:
        """
        Detects the GPU generation and count per server using `sinfo`.

        Returns:
            dict: A dictionary with GPU generation as keys and counts as values.
        """
        try:
            # Run sinfo command
            cmd = ["sinfo", "-o", "%G"]
            if self.partition:
                cmd.extend(["-p", self.partition])

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            # Parse output
            gpu_info: Dict[str, int] = {}
            logging.debug("Parsing node information...")
            for line in result.stdout.splitlines():
                parts = line.split(":")
                if len(parts) >= 3:
                    gpu_gen = parts[1]
                    gpu_count = int(parts[2].split("(")[0])
                    gpu_info[gpu_gen] = gpu_info.get(gpu_gen, 0) + gpu_count

            return gpu_info
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logging.error(f"Failed to get GPU information: {str(e)}")
            raise RuntimeError(f"Failed to get GPU information: {str(e)}")

    def get_gpu_generations(self) -> Set[str]:
        """Get the set of GPU generations available in the cluster.

        Returns:
            Set[str]: A set of GPU generation names (e.g., {"A100", "V100", "P100"})

        Raises:
            RuntimeError: If unable to retrieve GPU information.
        """
        try:
            cmd = ["sinfo", "-o", "%G"]
            if self.partition:
                cmd.extend(["-p", self.partition])

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            gpu_generations = set()

            for line in result.stdout.split("\n"):
                if line.strip():
                    parts = line.split(":")
                    if len(parts) >= 2 and not parts[2].isdigit():
                        gpu_generations.add(parts[2].upper())

            if not gpu_generations:
                return set()  # Return empty set if no GPUs found

            return gpu_generations

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to get GPU information: {str(e)}")

    def has_gpu_type(self, gpu_type: str) -> bool:
        """Check if a specific GPU type is available in the cluster.

        Args:
            gpu_type (str): The GPU type to check for (e.g., "A100", "V100")

        Returns:
            bool: True if the GPU type is available, False otherwise
        """
        gpu_counts = self.get_gpu_generation_and_count()
        return gpu_type.upper() in [k.upper() for k in gpu_counts.keys()]

    def get_max_job_lifetime(self) -> str:
        """Get the maximum job lifetime specified in the Slurm configuration.

        Note: MaxJobTime is a global configuration setting, not partition-specific,
        so the partition parameter doesn't affect this method.

        Returns:
            str: The maximum job lifetime in the format "days-hours:minutes:seconds".

        Raises:
            RuntimeError: If unable to retrieve the maximum job lifetime information.
        """
        try:
            result = subprocess.run(
                ["scontrol", "show", "config"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            for line in result.stdout.split("\n"):
                if "MaxJobTime" in line:
                    return line.split("=")[1].strip()

            raise RuntimeError("Could not find MaxJobTime in scontrol output")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to get maximum job lifetime: {str(e)}")


class AWSClusterInfo:
    def is_aws_cluster(self) -> bool:
        """Check if the cluster is running on AWS.

        Returns:
            bool: True if running on AWS, False otherwise
        """
        try:
            # Check for AWS-specific system files
            result = subprocess.run(
                ["cat", "/sys/devices/virtual/dmi/id/sys_vendor"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return "amazon" in result.stdout.lower()
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_aws_nccl_settings(self) -> Dict[str, str]:
        """Get recommended NCCL environment settings for AWS clusters with EFA.

        Returns:
            Dict[str, str]: Dictionary of environment variables and their recommended values
                           for optimal NCCL performance on AWS with EFA.
        """
        if not self.is_aws_cluster():
            return {}

        return {
            "FI_PROVIDER": "efa",
            "FI_EFA_USE_DEVICE_RDMA": "1",
            "NCCL_DEBUG": "INFO",
            "NCCL_PROTO": "simple",
            "NCCL_IB_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "ens,eth,en",
        }
