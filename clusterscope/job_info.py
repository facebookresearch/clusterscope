# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import socket
import subprocess
import time

from contextlib import closing
from functools import lru_cache
from typing import Any, Dict, MutableMapping, Optional, Union

import ray

MIN_MASTER_PORT, MAX_MASTER_PORT = (20_000, 60_000)


class RayCoordinator:
    LEADER_MAX_RETRIES = 30
    LEADER_RETRY_INTERVAL = 1.0

    def __init__(
        self,
        job_id: int,
        world_size: int,
    ):
        self.job_id = job_id
        self.worker_info: Dict[int, Dict[str, Any]] = {}
        self.leader_port = None
        self.ready_workers = 0
        self.world_size = world_size

    def register_worker(
        self, hostname: str, rank: int, free_port: Union[int, None]
    ) -> Dict[str, Any]:
        """Register a worker with its placement group ID and GPU ID"""
        self.ready_workers += 1
        info = {
            "hostname": hostname,
            "rank": rank,
            "ready_workers": self.ready_workers,
            "world_size": self.world_size,
            "leader_port": free_port,
        }
        self.worker_info[rank] = info
        return info

    def get_leader_info(self) -> Union[Dict[str, Any], None]:
        if self.ready_workers == self.world_size:
            return self.worker_info[0]
        else:
            return None


class JobInfo:
    """
    This class is used to get information about the current job.

    It prefers torch distributed env variables and it falls back to slurm env variables:

    Job ID: SLURM_JOB_ID
    Job Name: SLURM_JOB_NAME
    Global Rank: RANK, SLURM_PROCID
    Local Rank: LOCAL_RANK, SLURM_LOCALID
    World Size: WORLD_SIZE, SLURM_NTASKS
    Master Address: MASTER_ADDR, SLURM_JOB_NODELIST[0] (first hostname in the job)
    Master Port: MASTER_PORT, rand(MIN_MASTER_PORT, MAX_MASTER_PORT)

    To set all torch distributed env vars from slurm env vars, see `set_torch_distributed_env_from_slurm`
    """

    def __init__(self):
        self.is_torch_run = lambda: "LOCAL_RANK" in os.environ
        self.is_torchelastic_run = lambda: "TORCHELASTIC_RUN_ID" in os.environ
        self.is_slurm_job = lambda: "SLURM_JOB_ID" in os.environ
        self.is_ray_job = lambda: ray.is_initialized()
        self.job_id = self.get_job_id()
        self.job_name = self.get_job_name()
        self.global_rank = self.get_global_rank()
        self.local_rank = self.get_local_rank()
        self.world_size = self.get_world_size()
        self.is_rank_zero = self.get_is_rank_zero()

    @lru_cache(maxsize=1)
    def get_job_id(self) -> int:
        if self.is_slurm_job():
            job_id = os.environ.get("SLURM_JOB_ID")
            # is_slurm_job() checks if SLURM_JOB_ID variable exists in the env.
            # this assert should always pass, unless something undefines the variable.
            assert job_id is not None, "SLURM_JOB_ID is not set"
            try:
                parsed_job_id = int(job_id)
            except ValueError:
                raise RuntimeError(f"Slurm job ID cannot be parsed. {job_id=}")
            return parsed_job_id
        return 0

    @lru_cache(maxsize=1)
    def get_job_name(self) -> str:
        if self.is_slurm_job():
            return os.environ.get("SLURM_JOB_NAME", "")
        return "local"

    @lru_cache(maxsize=1)
    def get_global_rank(self) -> int:
        maybe_global_rank = os.environ.get("RANK")
        if maybe_global_rank is not None:
            try:
                global_rank = int(maybe_global_rank)
            except ValueError:
                raise RuntimeError(f"RANK cannot be parsed. {global_rank=}")
            return global_rank
        if self.is_slurm_job():
            return int(os.environ["SLURM_PROCID"])
        return 0

    @lru_cache(maxsize=1)
    def get_local_rank(self) -> int:
        maybe_local_rank = os.environ.get("LOCAL_RANK")
        if maybe_local_rank is not None:
            try:
                local_rank = int(maybe_local_rank)
            except ValueError:
                raise RuntimeError(f"LOCAL_RANK cannot be parsed. {local_rank=}")
            return local_rank
        if self.is_slurm_job():
            return int(os.environ["SLURM_LOCALID"])
        return 0

    @lru_cache(maxsize=1)
    def get_world_size(self) -> int:
        maybe_world_size = os.environ.get("WORLD_SIZE")
        if maybe_world_size is not None:
            try:
                world_size = int(maybe_world_size)
            except ValueError:
                raise RuntimeError(f"WORLD_SIZE cannot be parsed. {world_size=}")
            return world_size
        if self.is_ray_job():
            return len(ray.nodes())
        if self.is_slurm_job():
            return int(os.environ["SLURM_NTASKS"])
        return 1

    @lru_cache(maxsize=1)
    def get_is_rank_zero(self) -> bool:
        return self.get_global_rank() == 0

    @lru_cache(maxsize=1)
    def get_master_port(self) -> int:
        maybe_master_port = os.environ.get("MASTER_PORT")
        if maybe_master_port is not None:
            try:
                master_port = int(maybe_master_port)
            except ValueError:
                raise RuntimeError(f"master port cannot be parsed. {master_port=}")
            return master_port
        rng = random.Random(int(os.environ.get("SLURM_JOB_ID", -1)))
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    @lru_cache(maxsize=1)
    def get_master_addr(self) -> str:
        maybe_master_addr = os.environ.get("MASTER_ADDR")
        if maybe_master_addr is not None:
            return maybe_master_addr
        if self.is_slurm_job():
            result = subprocess.run(
                ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                if node_list := result.stdout.split("\n"):
                    return node_list[0]

            raise RuntimeError(
                f"`scontrol show hostnames` failed: {result.returncode=}, {result.stdout=}, {result.stderr=}"
            )
        return "127.0.0.1"

    def set_env_if_exists(
        self,
        target_key: str,
        source_key: str,
        source_dict: MutableMapping[str, str] = os.environ,
    ):
        if source_key in source_dict:
            os.environ[target_key] = str(source_dict[source_key])

    def set_torch_distributed_env(
        self, ray_coordinator_name: Optional[str] = None
    ) -> None:
        """
        Set torch distributed env variables from slurm env variables, and ray env variables.

        `ray_coordinator_name`: the name of the ray coordinator actor. If not provided, it skips setting vars from ray env.

        Preferece: Ray > Slurm.
        """
        self.set_torch_distributed_env_from_slurm()

        if ray_coordinator_name:
            self.set_torch_distributed_env_from_ray(
                ray_coordinator_name=ray_coordinator_name
            )

    def set_torch_distributed_env_from_slurm(self) -> None:
        if not self.is_slurm_job():
            return

        self.set_env_if_exists(
            target_key="WORLD_SIZE",
            source_key="SLURM_NTASKS",
        )
        self.set_env_if_exists(
            target_key="RANK",
            source_key="SLURM_PROCID",
        )
        self.set_env_if_exists(
            target_key="LOCAL_WORLD_SIZE",
            source_key="SLURM_NTASKS_PER_NODE",
        )
        if "LOCAL_WORLD_SIZE" not in os.environ:
            os.environ["LOCAL_WORLD_SIZE"] = "1"
        self.set_env_if_exists(
            target_key="LOCAL_RANK",
            source_key="SLURM_LOCALID",
        )
        self.set_env_if_exists(
            target_key="MASTER_ADDR",
            source_key=self.get_master_addr(),
        )
        self.set_env_if_exists(
            target_key="MASTER_PORT",
            source_key=str(self.get_master_port()),
        )
        self.set_env_if_exists(
            target_key="CUDA_VISIBLE_DEVICES",
            source_key="SLURM_LOCALID",
        )

    def set_torch_distributed_env_from_ray(self, ray_coordinator_name: str) -> None:
        if not self.is_ray_job():
            return

        hostname = socket.gethostname()

        coordinator_name = os.environ.get(ray_coordinator_name)
        if coordinator_name is None:
            raise RuntimeError(
                f"Ray coordinator name not found in environment variable {coordinator_name=}"
            )

        coordinator = ray.get_actor(*coordinator_name.split(":"))

        free_port = None
        rank = self.get_global_rank()
        if rank == 0:
            free_port = self._find_free_port()

        worker_info = ray.get(
            coordinator.register_worker.remote(
                hostname=hostname, rank=rank, free_port=free_port
            )
        )

        leader = None
        for attempts in range(RayCoordinator.LEADER_MAX_RETRIES):
            leader = ray.get(coordinator.get_leader_info.remote())
            if leader is not None:
                break
            time.sleep(RayCoordinator.LEADER_RETRY_INTERVAL * (1.1**attempts))

        if not leader:
            raise TimeoutError(f"Worker {rank} timed out waiting for leader")

        self.set_env_if_exists(
            target_key="WORLD_SIZE",
            source_key="world_size",
            source_dict=worker_info,
        )
        self.set_env_if_exists(
            target_key="MASTER_ADDR",
            source_key="hostname",
            source_dict=leader,
        )
        self.set_env_if_exists(
            target_key="MASTER_PORT",
            source_key="leader_port",
            source_dict=leader,
        )

    def _find_free_port(self) -> int:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
            return int(port)
