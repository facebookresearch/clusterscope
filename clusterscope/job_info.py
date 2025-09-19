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
            return int(os.environ.get("SLURM_PROCID", "0"))
        if self.is_torch_run:
            return int(os.environ.get("RANK", "0"))
        return 0

    @lru_cache(maxsize=1)
    def get_local_rank(self) -> int:
        if self.is_slurm_job:
            return int(os.environ.get("SLURM_LOCALID", "0"))
        if self.is_torch_run:
            return int(os.environ.get("LOCAL_RANK", "0"))
        return 0

    @lru_cache(maxsize=1)
    def get_world_size(self) -> int:
        if self.is_torch_run:
            return int(os.environ.get("WORLD_SIZE", "1"))
        if self.is_slurm_job:
            return int(os.environ.get("SLURM_NTASKS", "1"))
        return 1

    @lru_cache(maxsize=1)
    def get_is_rank_zero(self) -> bool:
        return self.get_global_rank() == 0

    @lru_cache(maxsize=1)
    def get_master_port(self) -> int:
        if self.is_torch_run:
            return int(os.environ.get("MASTER_PORT", "29500"))
        rng = random.Random(int(os.environ.get("SLURM_JOB_ID", -1)))
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    @lru_cache(maxsize=1)
    def get_master_addr(self) -> str:
        if self.is_torch_run:
            return os.environ.get("MASTER_ADDR", "127.0.0.1")
        if self.is_slurm_job:
            try:
                slurm_job_nodelist = os.environ.get("SLURM_JOB_NODELIST")
                if slurm_job_nodelist:
                    hostnames = subprocess.check_output(
                        ["scontrol", "show", "hostnames", slurm_job_nodelist],
                    )
                    return hostnames.split()[0].decode("utf-8")
            except (subprocess.CalledProcessError, KeyError, IndexError):
                pass
        return "127.0.0.1"
