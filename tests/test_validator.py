#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import shutil
import unittest

from unittest.mock import patch

from clusterscope.slurm.partition import PartitionInfo
from clusterscope.validate import job_gen_task_slurm_validator


def has_slurm():
    """Check if Slurm is installed."""
    return shutil.which("scontrol") is not None


class TestValidator(unittest.TestCase):

    def test_job_gen_task_slurm_validator_no_args(self):
        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(partition="test-partition")

    def test_job_gen_task_slurm_validator_cpu_and_gpu(self):
        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="test-partition",
                gpus_per_task=1,
                cpus_per_task=24,
            )

    def test_job_gen_task_slurm_validator_0_gpus(self):
        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="test-partition",
                gpus_per_task=0,
            )

    def test_job_gen_task_slurm_validator_negative_gpus(self):
        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="test-partition",
                gpus_per_task=-1,
            )

    def test_job_gen_task_slurm_validator_0_cpus(self):
        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="test-partition",
                cpus_per_task=0,
            )

    def test_job_gen_task_slurm_validator_negative_cpus(self):
        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="test-partition",
                cpus_per_task=-1,
            )

    def test_job_gen_task_slurm_validator_0_tasks_per_node(self):
        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="test-partition",
                tasks_per_node=0,
            )

    def test_job_gen_task_slurm_validator_negative_tasks_per_node(self):
        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="test-partition",
                tasks_per_node=-1,
            )

    @unittest.skipIf(not has_slurm(), "Slurm not available")
    def test_job_gen_task_slurm_validator_non_existent_partition(self):
        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="non-existent",
                cpus_per_task=1,
            )

    @patch("clusterscope.validate.get_partition_info")
    def test_job_gen_task_slurm_validator_valid_cpu(self, mock_run):
        mock_run.return_value = [
            PartitionInfo(
                name="test-partition",
                max_cpus_per_node=10,
                max_gpus_per_node=10,
            )
        ]

        job_gen_task_slurm_validator(
            partition="test-partition",
            cpus_per_task=5,
            tasks_per_node=2,
        )

    @patch("clusterscope.validate.get_partition_info")
    def test_job_gen_task_slurm_validator_valid_gpu(self, mock_run):
        mock_run.return_value = [
            PartitionInfo(
                name="test-partition",
                max_cpus_per_node=10,
                max_gpus_per_node=10,
            )
        ]

        job_gen_task_slurm_validator(
            partition="test-partition",
            gpus_per_task=5,
            tasks_per_node=2,
        )

    @patch("clusterscope.validate.get_partition_info")
    def test_job_gen_task_slurm_validator_invalid_cpu(self, mock_run):
        mock_run.return_value = [
            PartitionInfo(
                name="test-partition",
                max_cpus_per_node=10,
                max_gpus_per_node=10,
            )
        ]

        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="test-partition",
                cpus_per_task=11,
                tasks_per_node=1,
            )

    @patch("clusterscope.validate.get_partition_info")
    def test_job_gen_task_slurm_validator_invalid_gpu(self, mock_run):
        mock_run.return_value = [
            PartitionInfo(
                name="test-partition",
                max_cpus_per_node=10,
                max_gpus_per_node=10,
            )
        ]

        with self.assertRaises(ValueError):
            job_gen_task_slurm_validator(
                partition="test-partition",
                gpus_per_task=11,
                tasks_per_node=1,
            )
