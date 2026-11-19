#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import unittest

from unittest.mock import patch

from clusterscope.slurm.parser import parse_gres
from clusterscope.slurm.partition import get_partition_resources


class TestParseGres(unittest.TestCase):

    def test_parse_gres(self):
        cases = {
            "(null)": 0,
            "gpu:4": 4,
            "gpu:a100:2": 2,
            "gpu:volta:8(S:0-1)": 8,
            # Type names produced by AutoDetect=nvml
            "gpu:nvidia_a100-sxm4-80gb:8(S:0-1)": 8,
            "gpu:nvidia_a100_1g.10gb:7": 7,
            "gpu:h100:8(S:0-1),nvme:1": 8,
        }
        for gres, expected in cases.items():
            with self.subTest(gres=gres):
                self.assertEqual(parse_gres(gres), expected)


class TestGetPartitionResources(unittest.TestCase):

    @patch("clusterscope.slurm.partition.run_cli")
    def test_multiple_gres(self, mock_run_cli):
        mock_run_cli.return_value = (
            "gpu:nvidia_h100_80gb_hbm3:8(S:0-1),nvme:1,192\n(null),64\n"
        )
        self.assertEqual(
            get_partition_resources("test_partition"),
            {"max_gpus": 8, "max_cpus": 192},
        )


if __name__ == "__main__":
    unittest.main()
