# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest
from unittest.mock import patch

from clusterscope.job_info import JobInfo


class TestJobInfo(unittest.TestCase):

    def test_set_torch_distributed_env_from_slurm_default(self):
        slurm_env = {
            "SLURM_PROCID": "2",
            "SLURM_NTASKS": "8",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS_PER_NODE": "4",
        }
        with patch.dict(os.environ, slurm_env, clear=True):
            job = JobInfo()
            job.set_torch_distributed_env_from_slurm()

            self.assertEqual(os.environ.get("WORLD_SIZE"), "8")
            self.assertEqual(os.environ.get("RANK"), "2")
            self.assertEqual(os.environ.get("LOCAL_WORLD_SIZE"), "4")
            self.assertEqual(os.environ.get("LOCAL_RANK"), "1")
            self.assertEqual(os.environ.get("MASTER_ADDR"), "127.0.0.1")
            self.assertIsNotNone(os.environ.get("MASTER_PORT"))
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_set_torch_distributed_env_from_slurm_with_cuda_visible_devices(self):
        slurm_env = {
            "SLURM_PROCID": "3",
            "SLURM_NTASKS": "4",
            "SLURM_LOCALID": "3",
            "SLURM_NTASKS_PER_NODE": "4",
        }
        with patch.dict(os.environ, slurm_env, clear=True):
            job = JobInfo()
            job.set_torch_distributed_env_from_slurm(set_cuda_visible_devices=True)

            self.assertEqual(os.environ.get("WORLD_SIZE"), "4")
            self.assertEqual(os.environ.get("RANK"), "3")
            self.assertEqual(os.environ.get("LOCAL_WORLD_SIZE"), "4")
            self.assertEqual(os.environ.get("LOCAL_RANK"), "3")
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "3")

    def test_set_torch_distributed_env_from_slurm_preserves_existing_cuda_visible_devices(
        self,
    ):
        slurm_env = {
            "SLURM_PROCID": "1",
            "SLURM_NTASKS": "2",
            "SLURM_LOCALID": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        }
        with patch.dict(os.environ, slurm_env, clear=True):
            job = JobInfo()
            job.set_torch_distributed_env_from_slurm(set_cuda_visible_devices=False)

            self.assertEqual(os.environ.get("WORLD_SIZE"), "2")
            self.assertEqual(os.environ.get("RANK"), "1")
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0,1,2,3")

    def test_set_torch_distributed_env_from_slurm_not_srun(self):
        with patch.dict(os.environ, {}, clear=True):
            job = JobInfo()
            job.set_torch_distributed_env_from_slurm()

            self.assertNotIn("WORLD_SIZE", os.environ)
            self.assertNotIn("RANK", os.environ)
            self.assertNotIn("LOCAL_RANK", os.environ)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_job_info_ranks_and_world_size_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            job = JobInfo()
            self.assertEqual(job.get_job_id(), 0)
            self.assertEqual(job.get_job_name(), "local")
            self.assertEqual(job.get_global_rank(), 0)
            self.assertEqual(job.get_local_rank(), 0)
            self.assertEqual(job.get_world_size(), 1)
            self.assertTrue(job.get_is_rank_zero())
            self.assertEqual(job.get_master_addr(), "127.0.0.1")

    def test_job_info_ranks_from_torch_env(self):
        torch_env = {
            "RANK": "3",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "MASTER_PORT": "29500",
            "MASTER_ADDR": "10.0.0.1",
        }
        with patch.dict(os.environ, torch_env, clear=True):
            job = JobInfo()
            self.assertEqual(job.get_global_rank(), 3)
            self.assertEqual(job.get_local_rank(), 1)
            self.assertEqual(job.get_world_size(), 8)
            self.assertFalse(job.get_is_rank_zero())
            self.assertEqual(job.get_master_port(), 29500)
            self.assertEqual(job.get_master_addr(), "10.0.0.1")

    def test_job_info_ranks_from_slurm_env(self):
        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NAME": "my_job",
            "SLURM_PROCID": "0",
            "SLURM_LOCALID": "0",
            "SLURM_NTASKS": "4",
        }
        with patch.dict(os.environ, slurm_env, clear=True):
            job = JobInfo()
            self.assertEqual(job.get_job_id(), 12345)
            self.assertEqual(job.get_job_name(), "my_job")
            self.assertEqual(job.get_global_rank(), 0)
            self.assertEqual(job.get_local_rank(), 0)
            self.assertEqual(job.get_world_size(), 4)
            self.assertTrue(job.get_is_rank_zero())
