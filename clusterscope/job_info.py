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
    def __init__(self):
        self.is_torch_run = lambda: os.environ.get("LOCAL_RANK") is not None
        self.is_slurm_job = (
            lambda: "SLURM_JOB_ID" in os.environ and not self.is_torch_run
        )
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
        if self.is_slurm_job():
            return int(os.environ["SLURM_PROCID"])
        if self.is_torch_run():
            return int(os.environ["RANK"])
        return 0

    @lru_cache(maxsize=1)
    def get_local_rank(self) -> int:
        if self.is_slurm_job():
            return int(os.environ["SLURM_LOCALID"])
        if self.is_torch_run():
            return int(os.environ["LOCAL_RANK"])
        return 0

    @lru_cache(maxsize=1)
    def get_world_size(self) -> int:
        if self.is_torch_run():
            return int(os.environ["WORLD_SIZE"])
        if self.is_slurm_job():
            return int(os.environ["SLURM_NTASKS"])
        return 1

    @lru_cache(maxsize=1)
    def get_is_rank_zero(self) -> bool:
        return self.get_global_rank() == 0

    @lru_cache(maxsize=1)
    def get_master_port(self) -> int:
        maybe_master_port = os.environ["MASTER_PORT"]
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
        if self.is_torch_run():
            return os.environ["MASTER_ADDR"]
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
