# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import unittest

from clusterscope.cluster_info import ResourceShape
from clusterscope.parser import parse_memory_to_gb

TEST_CONFIGS = [
    # (partition, gpus_per_task, cpus_per_task, memory, tasks_per_node, expected_mem_gb)
    ("test_partition", 4, 24, "225G", 1, 225),
    ("test_partition", 4, 64, "1T", 2, 1024),
    ("test_partition", 4, 8, "32G", 1, 32),
    ("test_partition", 8, 128, "4T", 1, 4096),
    ("test_partition", 8, 1, "1G", 1, 1),
    ("test_partition", 8, 256, "16T", 4, 16384),
]


class TestResourceShape(unittest.TestCase):
    """Test cases for ResourceShape class and its to_X methods."""

    def test_resource_shape_creation(self):
        """Test ResourceShape creation and basic properties."""
        resource = ResourceShape(
            slurm_partition="test_partition",
            gpus_per_task=1,
            cpus_per_task=24,
            memory="225G",
            tasks_per_node=1,
            nodes=1,
        )

        # Test immutability (NamedTuple characteristic)
        with self.assertRaises(AttributeError):
            resource.cpus_per_task = 48

    def test_memory_parsing(self):
        """Test memory parsing with various formats."""
        test_cases = [
            # Valid formats (memory_str, expected_gb)
            ("1G", 1),
            ("10G", 10),
            ("225G", 225),
            ("512G", 512),
            ("1000G", 1000),
            ("1T", 1024),
            ("2T", 2048),
            ("4T", 4096),
            ("10T", 10240),
            ("16T", 16384),
        ]

        for memory_str, expected_gb in test_cases:
            with self.subTest(memory=memory_str, expected=expected_gb):
                resource = ResourceShape(
                    slurm_partition="test_partition",
                    gpus_per_task=1,
                    cpus_per_task=8,
                    memory=memory_str,
                    tasks_per_node=1,
                    nodes=2,
                )
                self.assertEqual(parse_memory_to_gb(resource.memory), expected_gb)

    def test_to_json(self):
        """Test to_json format method with various configurations."""

        for (
            partition,
            gpus_per_task,
            cpus_per_task,
            memory,
            tasks_per_node,
            expected_mem_gb,
        ) in TEST_CONFIGS:
            with self.subTest(
                config=f"{gpus_per_task}gpu_{cpus_per_task}cpu_{memory}_{tasks_per_node}tasks"
            ):
                resource = ResourceShape(
                    slurm_partition=partition,
                    gpus_per_task=gpus_per_task,
                    cpus_per_task=cpus_per_task,
                    memory=memory,
                    tasks_per_node=tasks_per_node,
                    nodes=1,
                )
                result = json.loads(resource.to_json())

                self.assertEqual(result["slurm_partition"], partition)
                self.assertEqual(result["gpus_per_task"], gpus_per_task)
                self.assertEqual(result["cpus_per_task"], cpus_per_task)
                self.assertEqual(result["memory"], memory)
                self.assertEqual(result["tasks_per_node"], tasks_per_node)
                self.assertEqual(result["mem_gb"], expected_mem_gb)

    def test_to_sbatch(self):
        """Test to_sbatch format method with various configurations."""

        for (
            partition,
            gpus_per_task,
            cpus_per_task,
            memory,
            tasks_per_node,
            _,
        ) in TEST_CONFIGS:
            with self.subTest(
                config=f"{gpus_per_task}gpu_{cpus_per_task}cpu_{memory}_{tasks_per_node}tasks"
            ):
                resource = ResourceShape(
                    slurm_partition=partition,
                    gpus_per_task=gpus_per_task,
                    cpus_per_task=cpus_per_task,
                    memory=memory,
                    tasks_per_node=tasks_per_node,
                    nodes=1,
                )
                result = resource.to_sbatch()
                lines = result.split("\n")

                sbatch_lines = [line for line in lines if line.startswith("#SBATCH")]
                self.assertIn(f"#SBATCH --cpus-per-task={cpus_per_task}", result)
                self.assertIn(f"#SBATCH --mem={memory}", result)
                self.assertIn(f"#SBATCH --ntasks-per-node={tasks_per_node}", result)
                self.assertIn(f"#SBATCH --partition={partition}", result)
                self.assertIn(f"#SBATCH --gpus-per-task={gpus_per_task}", result)

    def test_to_srun(self):
        """Test to_srun format method with various configurations."""

        for (
            partition,
            gpus_per_task,
            cpus_per_task,
            memory,
            tasks_per_node,
            _,
        ) in TEST_CONFIGS:
            with self.subTest(
                config=f"{gpus_per_task}gpu_{cpus_per_task}cpu_{memory}_{tasks_per_node}tasks"
            ):
                resource = ResourceShape(
                    slurm_partition=partition,
                    gpus_per_task=gpus_per_task,
                    cpus_per_task=cpus_per_task,
                    memory=memory,
                    tasks_per_node=tasks_per_node,
                    nodes=1,
                )

                result = resource.to_srun()

                parts = result.split()
                self.assertIn(f"--cpus-per-task={cpus_per_task}", result)
                self.assertIn(f"--mem={memory}", result)
                self.assertIn(f"--ntasks-per-node={tasks_per_node}", result)
                self.assertIn(f"--partition={partition}", result)
                self.assertIn(f"--gpus-per-task={gpus_per_task}", result)

    def test_to_submitit(self):
        """Test to_submitit format method with various configurations."""

        for (
            partition,
            gpus_per_task,
            cpus_per_task,
            memory,
            tasks_per_node,
            expected_mem_gb,
        ) in TEST_CONFIGS:
            with self.subTest(
                config=f"{gpus_per_task}gpu_{cpus_per_task}cpu_{memory}_{tasks_per_node}tasks"
            ):
                resource = ResourceShape(
                    slurm_partition=partition,
                    gpus_per_task=gpus_per_task,
                    cpus_per_task=cpus_per_task,
                    memory=memory,
                    tasks_per_node=tasks_per_node,
                    nodes=1,
                )
                result = json.loads(resource.to_submitit())

                self.assertEqual(result["slurm_partition"], partition)
                self.assertEqual(result["gpus_per_task"], gpus_per_task)
                self.assertEqual(result["cpus_per_task"], cpus_per_task)
                self.assertEqual(result["mem_gb"], expected_mem_gb)
                self.assertEqual(result["tasks_per_node"], tasks_per_node)


if __name__ == "__main__":
    unittest.main()
