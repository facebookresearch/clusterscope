# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import subprocess

from functools import lru_cache

MIN_MASTER_PORT, MAX_MASTER_PORT = (20000, 60000)


class JobInfo:
    def __init__(self):
        self.is_torch_run = os.environ.get("LOCAL_RANK") is not None
        self.is_slurm_job = "SLURM_JOB_ID" in os.environ and not self.is_torch_run
        self.job_id = self.get_job_id()
        self.job_name = self.get_job_name()
        self.global_rank = self.get_global_rank()
        self.local_rank = self.get_local_rank()
        self.world_size = self.get_world_size()
        self.is_rank_zero = self.get_is_rank_zero()

    @lru_cache(maxsize=1)
    def get_job_id(self) -> int:
        if self.is_slurm_job:
            return int(os.environ.get("SLURM_JOB_ID", -1))
        return 0

    @lru_cache(maxsize=1)
    def get_job_name(self) -> str:
        if self.is_slurm_job:
            return os.environ.get("SLURM_JOB_NAME", "")
        return "local"

    @lru_cache(maxsize=1)
    def get_global_rank(self) -> int:
        if self.is_slurm_job:
            return int(os.environ["SLURM_PROCID"])
        if self.is_torch_run:
            return int(os.environ["RANK"])
        return 0

    @lru_cache(maxsize=1)
    def get_local_rank(self) -> int:
        if self.is_slurm_job:
            return int(os.environ["SLURM_LOCALID"])
        if self.is_torch_run:
            return int(os.environ["LOCAL_RANK"])
        return 0

    @lru_cache(maxsize=1)
    def get_world_size(self) -> int:
        if self.is_torch_run:
            return int(os.environ["WORLD_SIZE"])
        if self.is_slurm_job:
            return int(os.environ["SLURM_NTASKS"])
        return 1

    @lru_cache(maxsize=1)
    def get_is_rank_zero(self) -> bool:
        return self.get_global_rank() == 0

    @lru_cache(maxsize=1)
    def get_master_port(self) -> int:
        if self.is_torch_run:
            return int(os.environ["MASTER_PORT"])
        rng = random.Random(int(os.environ.get("SLURM_JOB_ID", -1)))
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    @lru_cache(maxsize=1)
    def get_master_addr(self) -> str:
        if self.is_torch_run:
            return os.environ["MASTER_ADDR"]
        if self.is_slurm_job:
            hostnames = subprocess.check_output(
                ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            )
            return hostnames.split()[0].decode("utf-8")
        return "127.0.0.1"
