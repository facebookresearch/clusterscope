# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import platform
import shutil
import subprocess
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Set

from clusterscope.cache import fs_cache


class UnifiedInfo:
    def __init__(self):
        self.local_node_info = LocalNodeInfo()
        self.slurm_cluster_info = SlurmClusterInfo()
        self.is_slurm_cluster = self.slurm_cluster_info.verify_slurm_available()
        self.has_nvidia_gpus = self.local_node_info.has_nvidia_gpus()
        self.aws_cluster_info = AWSClusterInfo()

    def get_cluster_name(self) -> str:
        """Get the name of the Slurm cluster. Returns `local-node` if not a Slurm cluster.

        Returns:
            str: The name of the Slurm cluster.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_cluster_name()
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

    def get_mem_per_node(self) -> int:
        """Return the lowest amount of mem configured across all nodes in the cluster. Returns 0 if not a Slurm cluster.

        Returns:
            int: The memory per node in the cluster.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_mem_per_node()
        return self.local_node_info.get_mem()

    def get_gpu_generation_and_count(self) -> Dict[str, int]:
        """Get the number of GPUs on the slurm cluster node.

        Returns:
            dict: A dictionary with GPU generation as keys and counts as values.
        """
        if self.is_slurm_cluster:
            return self.slurm_cluster_info.get_gpu_generation_and_count()
        if self.has_nvidia_gpus:
            return self.local_node_info.get_gpu_generation_and_count()
        return {}


class DarwinInfo:
    def get_cpu_count(self, timeout: int = 60) -> int:
        """Get the number of CPUs on the local node.

        Returns:
            int: The number of CPUs on the local node.

        Raises:
            RuntimeError: If unable to retrieve CPU information.
        """
        try:
            result = subprocess.check_output(
                ["sysctl", "-n", "hw.ncpu"], text=True, timeout=timeout
            )

            return int(result.strip())
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to get CPU information: {str(e)}")

    def get_mem(self, timeout: int = 60) -> int:
        """Get the amount of memory on the local node.

        Returns:
            int: The amount of memory on the local node.

        Raises:
            RuntimeError: If unable to retrieve memory information.
        """
        try:
            result = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=timeout
            )

            return int(result.strip()) // 1024 // 1024
        except (subprocess.SubprocessError, FileNotFoundError) as e:
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
            result = subprocess.check_output(
                ["nproc", "--all"], text=True, timeout=timeout
            )

            return int(result.strip())
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to get CPU information: {str(e)}")

    @fs_cache(var_name="LOCAL_NODE_MEM")
    def get_mem(self, timeout: int = 60) -> int:
        """Get the amount of memory on the local node.

        Returns:
            int: The amount of memory on the local node.

        Raises:
            RuntimeError: If unable to retrieve memory information.
        """
        try:
            result = subprocess.check_output(["free", "-m"], text=True, timeout=timeout)

            for line in result.strip().split("\n"):
                if "Mem:" in line:
                    parts = line.split()
                    return int(parts[1])
            raise RuntimeError("Could not find memory information in free output")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
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
        except FileNotFoundError:
            return False

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

    @fs_cache(var_name="LOCAL_NODE_MEM")
    def get_mem(self, timeout: int = 60) -> int:
        """Get the amount of memory on the local node.

        Returns:
            int: The amount of memory on the local node.

        Raises:
            RuntimeError: If unable to retrieve memory information.
        """
        system = platform.system()
        if system == "Linux":
            return LinuxInfo().get_mem(timeout)
        if system == "Darwin":
            return DarwinInfo().get_mem(timeout)
        raise RuntimeError(f"Unsupported system: {system}")

    def get_gpu_generation_and_count(self, timeout: int = 60) -> Dict[str, int]:
        """Get the number of GPUs on the local node.

        Returns:
            int: The number of GPUs on the local node.

        Raises:
            RuntimeError: If unable to retrieve GPU information.
        """
        assert self.has_nvidia_gpus() is True
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                text=True,
                timeout=timeout,
            )

            gpu_info: Dict[str, int] = defaultdict(int)
            for line in result.strip().split("\n"):
                parts = line.split()
                gpu_gen = parts[1]
                gpu_info[gpu_gen] += 1
            return gpu_info

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to get GPU information: {str(e)}")


class SlurmClusterInfo:
    """A class to provide information about the Slurm cluster configuration.

    This class offers methods to query various aspects of a Slurm cluster,
    such as cluster name, available resources, and node configurations.
    """

    def __init__(self):
        """Initialize the Cluster instance."""
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
            slurm_version = subprocess.check_output(
                ["sinfo", "-V"], text=True, timeout=timeout
            )

            return str(slurm_version.strip().split(" ")[1])
        except (subprocess.SubprocessError, FileNotFoundError) as e:
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

    @fs_cache(var_name="SLURM_MEM_PER_NODE")
    def get_mem_per_node(self) -> int:
        """Get the lowest memory available per node in the cluster.

        Returns:
            int: The memory per node in the cluster.

        Raises:
            RuntimeError: If unable to retrieve node information.
        """
        try:
            result = subprocess.run(
                ["sinfo", "-o", "%100m", "--noconvert", "--noheader"],
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
            result = subprocess.run(
                ["sinfo", "-o", "%100c", "--noheader"],
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
            result = subprocess.run(
                ["sinfo", "-o", "%G"],
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
            logging.error(f"Failed to get CPU information: {str(e)}")
            raise RuntimeError(f"Failed to get CPU information: {str(e)}")

    def get_gpu_generations(self) -> Set[str]:
        """Get the set of GPU generations available in the cluster.

        Returns:
            Set[str]: A set of GPU generation names (e.g., {"A100", "V100", "P100"})

        Raises:
            RuntimeError: If unable to retrieve GPU information.
        """
        try:
            result = subprocess.run(
                ["sinfo", "-o", "%G"],
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
        return gpu_type.upper() in gpu_counts

    def get_max_job_lifetime(self) -> str:
        """Get the maximum job lifetime specified in the Slurm configuration.

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
