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
from typing import Dict, NamedTuple, Optional, Set

from clusterscope.cache import fs_cache
from clusterscope.parser import parse_memory_to_gb
from clusterscope.shell import run_cli


class ResourceShape(NamedTuple):
    """Represents resource requirements for a job in Slurm SBATCH format."""

    cpus_per_task: int
    memory: str
    tasks_per_node: int
    slurm_partition: str
    nodes: int
    gpus_per_task: Optional[int] = None

    def to_dict(self) -> dict:
        data = {k: v for k, v in self._asdict().items() if v is not None}
        data["mem_gb"] = parse_memory_to_gb(data["memory"])
        if self.gpus_per_task == 0:
            data.pop("gpus_per_task")
        return data

    def to_json(self) -> str:
        """Convert ResourceShape to JSON format.

        Returns:
            str: JSON representation of the resource requirements
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sbatch(self) -> str:
        """Convert ResourceShape to SBATCH script format.

        Returns:
            str: SBATCH script with resource directives
        """
        lines = [
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --mem={self.memory}",
            f"#SBATCH --ntasks-per-node={self.tasks_per_node}",
            f"#SBATCH --partition={self.slurm_partition}",
            f"#SBATCH --nodes={self.nodes}",
        ]
        if self.gpus_per_task and self.gpus_per_task > 0:
            lines.append(f"#SBATCH --gpus-per-task={self.gpus_per_task}")
        return "\n".join(lines)

    def to_srun(self) -> str:
        """Convert ResourceShape to srun command format.

        Returns:
            str: srun command with resource specifications
        """
        cmd_parts = [
            f"--cpus-per-task={self.cpus_per_task}",
            f"--mem={self.memory}",
            f"--ntasks-per-node={self.tasks_per_node}",
            f"--partition={self.slurm_partition}",
            f"--nodes={self.nodes}",
        ]
        if self.gpus_per_task and self.gpus_per_task > 0:
            cmd_parts.append(f"--gpus-per-task={self.gpus_per_task}")
        return " ".join(cmd_parts)

    def to_submitit(self) -> str:
        """Convert ResourceShape to submitit parameters format.

        Returns:
            str: JSON representation of submitit parameters
        """
        mem_gb = parse_memory_to_gb(self.memory)

        params = {
            "cpus_per_task": self.cpus_per_task,
            "mem_gb": mem_gb,
        }
        attrs = [
            "slurm_partition",
            "tasks_per_node",
            "nodes",
        ]
        if self.gpus_per_task and self.gpus_per_task > 0:
            attrs.append("gpus_per_task")
        for attr_name in attrs:
            value = getattr(self, attr_name)
            if value is not None:
                params[attr_name] = value
        return json.dumps(params, indent=2)


class GPUInfo(NamedTuple):
    """Represents resource requirements for a job in Slurm SBATCH format."""

    gpu_gen: str
    gpu_count: int
    vendor: str
    partition: Optional[str] = None


class MemInfo(NamedTuple):
    """Represents memory information for a host."""

    mem_total_MB: int
    mem_total_GB: int
    partition: Optional[str] = None


class CPUInfo(NamedTuple):
    """Represents CPU information for a host."""

    cpu_count: int
    partition: Optional[str] = None


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
    "VOLTA": "V100",
    "AMPERE": "A100",
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
        if os.environ.get("MAST_HPC_JOB_NAME", None) is not None:
            return "mast"
        if os.environ.get("GITHUB_ACTIONS", None) is not None:
            return "github"
        if platform.system() == "Darwin":
            return "macos"
        return "local-node"

    def get_tmp_dir(self) -> str:
        """Get the TMP directory for the current cluster.

        Returns:
            str: The TMP directory for the current cluster.
        """
        if self.is_slurm_cluster and "SLURM_JOB_ID" in os.environ:
            return self.slurm_cluster_info.get_tmp_dir()
        return self.local_node_info.get_tmp_dir()

    def get_slurm_version(self) -> str:
        """Get the slurm version. Returns `0` if not a Slurm cluster.

        Returns:
            str: Slurm version as a string: "24.11.4"
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_slurm_version()
        return "0"

    def get_cpus_per_node(self) -> list[CPUInfo] | CPUInfo:
        """Get the number of CPUs for each node in the cluster. Returns 0 if not a Slurm cluster.

        Returns:
            int: The number of CPUs per node, assuming all nodes have the same CPU count.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_cpus_per_node()
        return self.local_node_info.get_cpu_count()

    def get_mem_per_node_MB(self) -> list[MemInfo] | MemInfo:
        """Return the lowest amount of mem configured across all nodes in the cluster. Returns 0 if not a Slurm cluster.

        Returns:
            int: The memory per node in the cluster.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_mem_per_node_MB()
        return self.local_node_info.get_mem_MB()

    def get_gpu_generation_and_count(self) -> list[GPUInfo]:
        """Get the number of GPUs on the slurm cluster node.

        Returns:
            dict: A dictionary with GPU generation as keys and counts as values.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_gpu_generation_and_count()
        if self.has_nvidia_gpus or self.has_amd_gpus:
            return self.local_node_info.get_gpu_generation_and_count()
        return []

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

    def get_total_gpus_per_node(self) -> int:
        """Get the total number of GPUs available per node.

        Returns:
            int: Total number of GPUs per node. Returns 8 as default if no GPUs are detected.
        """
        gpus = self.get_gpu_generation_and_count()
        if not gpus:
            # Default to 8 if no GPUs detected (common configuration)
            return 8

        # Sum all GPU counts across different types
        total_gpus = sum([g.gpu_count for g in gpus])
        return max(total_gpus, 1)  # Ensure at least 1 to avoid division by zero

    def get_task_resource_requirements(
        self,
        partition: str,
        cpus_per_task: Optional[int] = None,
        gpus_per_task: Optional[int] = None,
        tasks_per_node: int = 1,
        nodes: int = 1,
    ) -> ResourceShape:
        """Calculate resource requirements for better GPU packing based on node's GPU configuration.

        Uses proportional allocation of node resources per GPU based on the node's total GPU count.
        For the maximum GPU count, returns all available node resources.
        Returns values in Slurm SBATCH format for direct use in job scripts.

        NOTE: For array jobs, use get_array_job_requirements() instead, which provides
        per-array-element resource allocation.

        Args:
            (mutually exclusive, at least 1 required):
                cpus_per_task (int): Total number of CPUs required per task (1 to max available)
                gpus_per_task (int): Total number of GPUs required per task (1 to max available)
            tasks_per_node (int): Number of tasks to run per node (default: 1)

        Returns:
            ResourceShape: Tuple containing CPU cores per task (int), memory per node (str),
                          and tasks per node (int)
        """
        assert not (
            gpus_per_task is None and cpus_per_task is None
        ), "gpus_per_task, and cpus_per_task are mutually exclusive, at least 1 required"
        if tasks_per_node < 1:
            raise ValueError("tasks_per_node must be at least 1")

        self.partition = partition
        cpus_per_node = self.get_cpus_per_node()
        total_cpus_per_node = (
            cpus_per_node[0] if isinstance(cpus_per_node, list) else cpus_per_node
        )
        mem_per_node = self.get_mem_per_node_MB()
        total_ram_per_node = (
            mem_per_node[0] if isinstance(mem_per_node, list) else mem_per_node
        )

        # CPU Request
        if cpus_per_task is not None:
            ram_mb_per_cpu = (
                total_ram_per_node.mem_total_MB / total_cpus_per_node.cpu_count
            )
            total_required_ram_mb = math.floor(
                ram_mb_per_cpu * cpus_per_task * tasks_per_node
            )
        # GPU Request
        elif gpus_per_task is not None:
            total_gpus_per_node = self.get_total_gpus_per_node()

            cpu_cores_per_gpu = total_cpus_per_node.cpu_count / total_gpus_per_node
            total_required_cpu_cores_per_task = math.floor(
                cpu_cores_per_gpu * gpus_per_task
            )

            ram_mb_per_gpu = total_ram_per_node.mem_total_MB / total_gpus_per_node
            total_required_ram_mb = math.floor(
                ram_mb_per_gpu * gpus_per_task * tasks_per_node
            )

            cpu_cores_per_task = total_required_cpu_cores_per_task / tasks_per_node

            cpus_per_task = math.floor(cpu_cores_per_task)
        else:
            raise ValueError(
                "gpus_per_task, and cpus_per_task are mutually exclusive, at least 1 required."
            )

        # Memory per node: Convert MB to GB and format for Slurm
        # Note: Memory is allocated per node, not per task in most Slurm configurations
        required_ram_gb = total_required_ram_mb // 1024
        # Default to GB for higher precision
        memory = f"{required_ram_gb:.0f}G"

        return ResourceShape(
            slurm_partition=partition,
            cpus_per_task=cpus_per_task,
            gpus_per_task=gpus_per_task,
            memory=memory,
            tasks_per_node=tasks_per_node,
            nodes=nodes,
        )


class DarwinInfo:
    def get_cpu_count(self, timeout: int = 60) -> CPUInfo:
        """Get the number of CPUs on the local node.

        Returns:
            int: The number of CPUs on the local node.

        Raises:
            RuntimeError: If unable to retrieve CPU information.
        """
        try:
            result = run_cli(["sysctl", "-n", "hw.ncpu"], text=True, timeout=timeout)
            return CPUInfo(cpu_count=int(result.strip()))
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get CPU information: {str(e)}")

    def get_mem_MB(self, timeout: int = 60) -> MemInfo:
        """Get the amount of memory on the local node.

        Returns:
            int: The amount of memory on the local node.

        Raises:
            RuntimeError: If unable to retrieve memory information.
        """
        try:
            result = run_cli(["sysctl", "-n", "hw.memsize"], text=True, timeout=timeout)
            mem_total_MB = int(result.strip()) // 1024 // 1024
            return MemInfo(
                mem_total_MB=mem_total_MB,
                mem_total_GB=mem_total_MB // 1024,
            )
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get memory information: {str(e)}")


class LinuxInfo:
    def get_cpu_count(self, timeout: int = 60) -> CPUInfo:
        """Get the number of CPUs on the local node.

        Returns:
            int: The number of CPUs on the local node.

        Raises:
            RuntimeError: If unable to retrieve CPU information.
        """
        try:
            result = run_cli(["nproc", "--all"], text=True, timeout=timeout)
            return CPUInfo(cpu_count=int(result.strip()))
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get CPU information: {str(e)}")

    def get_mem_MB(self, timeout: int = 60) -> MemInfo:
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
                    mem_total_MB = int(parts[1])
                    return MemInfo(
                        mem_total_MB=mem_total_MB,
                        mem_total_GB=mem_total_MB // 1024,
                    )
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

    def get_cpu_count(self, timeout: int = 60) -> CPUInfo:
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

    def get_mem_MB(self, timeout: int = 60) -> MemInfo:
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
        assert 0 < mem.mem_total_MB <= 10**12, f"Likely invalid memory: {mem}"
        return mem

    def get_nvidia_gpu_info(self, timeout: int = 60) -> list[GPUInfo]:
        """Get NVIDIA GPU information using nvidia-smi."""
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
                ["nvidia-smi", "--query-gpu=gpu_name,count", "--format=csv,noheader"],
                text=True,
                timeout=timeout,
            )

            gpu_info: Dict[str, int] = defaultdict(int)
            lines = result.strip().split("\n")
            all_lines = set()
            for line in lines:
                if line in all_lines:
                    continue
                all_lines.add(line)
                if ", " not in line:
                    continue
                gpu_gen = ""
                gpu_name_upper, count = line.split(", ")
                gpu_name_upper = gpu_name_upper.strip().upper()
                count_parsed = int(count)

                # Check for known NVIDIA GPU types
                found_gpu = False
                for gpu_key, gpu_pattern in NVIDIA_GPU_TYPES.items():
                    if gpu_pattern in gpu_name_upper:
                        gpu_gen = gpu_key
                        found_gpu = True
                        break

                # If no known GPU type was found
                if not found_gpu:
                    # Generic fallback - try to extract model number
                    words = gpu_name_upper.split()
                    for word in words:
                        if any(char.isdigit() for char in word) and len(word) > 2:
                            gpu_gen = word
                            break

                # If no GPU generation was found, use the full name
                if not found_gpu and gpu_gen == "":
                    gpu_gen = gpu_name_upper

                gpu_info[gpu_gen] += count_parsed

            return [
                GPUInfo(gpu_gen=gpu_gen, gpu_count=count, vendor="nvidia")
                for gpu_gen, count in gpu_info.items()
            ]
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get NVIDIA GPU information: {str(e)}")

    def get_amd_gpu_info(self, timeout: int = 60) -> list[GPUInfo]:
        """Get AMD GPU information using rocm-smi."""
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
                                gpu_gen = gpu_key
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
                                    gpu_gen = word
                                    found_gpu = True
                                    break

                        if not found_gpu and gpu_gen is None:
                            gpu_gen = gpu_name_upper

                        gpu_info[gpu_gen] += 1
            return [
                GPUInfo(gpu_gen=gpu_gen, gpu_count=count, vendor="amd")
                for gpu_gen, count in gpu_info.items()
            ]
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get AMD GPU information: {str(e)}")

    def get_gpu_generation_and_count(self, timeout: int = 60) -> list[GPUInfo]:
        """Get GPU information for all available GPUs on the local node.

        Returns:
            Dict[str, int]: Dictionary with GPU generation as keys and counts as values.

        Raises:
            RuntimeError: If unable to retrieve GPU information.
        """
        gpu_info = []

        # Try NVIDIA GPUs
        if self.has_nvidia_gpus():
            try:
                nvidia_info = self.get_nvidia_gpu_info(timeout)
                gpu_info.extend(nvidia_info)
            except RuntimeError as e:
                logging.warning(f"Failed to get NVIDIA GPU info: {e}")

        # Try AMD GPUs
        if self.has_amd_gpus():
            try:
                amd_info = self.get_amd_gpu_info(timeout)
                gpu_info.extend(amd_info)
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
            gpus = self.get_gpu_generation_and_count()
            return gpu_type.upper() in [g.gpu_gen for g in gpus]
        except RuntimeError:
            return False

    def get_tmp_dir(self) -> str:
        return "/tmp"


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

    def get_tmp_dir(self) -> str:
        return f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}/"

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

    def get_mem_per_node_MB(self) -> list[MemInfo]:
        """Get the lowest memory available per node in the cluster.

        Returns:
            int: The memory per node in the cluster.

        Raises:
            RuntimeError: If unable to retrieve node information.
        """
        try:
            cmd = ["sinfo", "-o", "%100m,%100P", "--noconvert", "--noheader"]
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
            results = []
            for line in result.stdout.splitlines():
                mem, partition = line.split(",")
                mem_MB = int(mem.strip("+ "))
                results.append(
                    MemInfo(
                        mem_total_MB=mem_MB,
                        mem_total_GB=mem_MB // 1024,
                        partition=partition.strip("* "),
                    )
                )
            return results
            raise RuntimeError(f"No mem information found in: {result.stdout}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logging.error(f"Failed to get Slurm memory information: {str(e)}")
            raise RuntimeError(f"Failed to get Slurm memory information: {str(e)}")

    def get_cpus_per_node(self) -> list[CPUInfo]:
        """Get the minimum number of CPUs for each node in the cluster.

        Returns:
            int: The number of CPUs per node, assuming all nodes have the same CPU count.

        Raises:
            RuntimeError: If unable to retrieve node information or if nodes have different CPU counts.
        """
        try:
            cmd = ["sinfo", "-o", "%100c,%100P", "--noheader"]
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
            all_cpus = []
            for line in result.stdout.splitlines():
                cpus, partition = line.split(",")
                cpu_count = int(cpus.strip("+ "))
                all_cpus.append(
                    CPUInfo(cpu_count=cpu_count, partition=partition.strip("* "))
                )
            if all_cpus == []:
                raise RuntimeError(f"No CPU information found in: {result.stdout}")
            return all_cpus
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logging.error(f"Failed to get CPU information: {str(e)}")
            raise RuntimeError(f"Failed to get CPU information: {str(e)}")

    def get_gpu_generation_and_count(self) -> list[GPUInfo]:
        """
        Detects the GPU generation and count per server using `sinfo`.

        Returns:
            dict: A dictionary with GPU generation as keys and counts as values.
        """
        try:
            # Run sinfo command
            cmd = ["sinfo", "-o", "%G,%P"]
            if self.partition:
                cmd.extend(["-p", self.partition])

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            results = []
            all_lines = set()

            # Parse output
            logging.debug("Parsing node information...")
            for line in result.stdout.splitlines():
                gres, partition = line.split(",")
                gres_gpu_gen_and_count = gres.split("(")[0]
                uniq_gpus = gres_gpu_gen_and_count + partition
                if uniq_gpus in all_lines:
                    continue
                all_lines.add(uniq_gpus)
                partition = partition.strip("* ")
                gres_parts = gres.split(":")
                if len(gres_parts) >= 3:
                    gpu_gen = gres_parts[1]
                    gpu_count = int(gres_parts[2].split("(")[0])
                    vendor = "Vendor Not Found"
                    if gpu_gen.upper() in NVIDIA_GPU_TYPES:
                        vendor = "nvidia"
                    elif gpu_gen.upper() in AMD_GPU_TYPES:
                        vendor = "amd"

                    results.append(
                        GPUInfo(
                            partition=partition,
                            gpu_gen=gpu_gen,
                            gpu_count=gpu_count,
                            vendor=vendor,
                        )
                    )

            return results
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
        gpus = self.get_gpu_generation_and_count()
        return gpu_type.upper() in [g.gpu_gen.upper() for g in gpus]

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
            "NCCL_SOCKET_IFNAME": "ens,eth,en",
        }
