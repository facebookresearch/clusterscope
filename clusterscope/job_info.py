# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import subprocess

from functools import lru_cache

MIN_MASTER_PORT, MAX_MASTER_PORT = (20_000, 60_000)


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

    def set_torch_distributed_env_from_slurm(self) -> None:
        if self.is_slurm_job():
            os.environ["WORLD_SIZE"] = str(os.environ.get("SLURM_NTASKS"))
            os.environ["RANK"] = str(os.environ.get("SLURM_PROCID"))
            os.environ["LOCAL_WORLD_SIZE"] = os.environ.get(
                "SLURM_NTASKS_PER_NODE", "1"
            )
            os.environ["LOCAL_RANK"] = str(os.environ.get("SLURM_LOCALID"))
            os.environ["MASTER_ADDR"] = self.get_master_addr()
            os.environ["MASTER_PORT"] = str(self.get_master_port())
            os.environ["CUDA_VISIBLE_DEVICES"] = str(os.environ.get("SLURM_LOCALID"))
