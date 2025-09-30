# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import unittest

from clusterscope.cluster_info import ResourceShape


class TestResourceShape(unittest.TestCase):
    """Test cases for ResourceShape class and its to_X methods."""

    def test_resource_shape_creation(self):
        """Test ResourceShape creation and basic properties."""
        resource = ResourceShape(cpu_cores=24, memory="225G", tasks_per_node=1)

        self.assertEqual(resource.cpu_cores, 24)
        self.assertEqual(resource.memory, "225G")
        self.assertEqual(resource.tasks_per_node, 1)

        # Test immutability (NamedTuple characteristic)
        with self.assertRaises(AttributeError):
            resource.cpu_cores = 48

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
                    cpu_cores=8, memory=memory_str, tasks_per_node=1
                )
                self.assertEqual(resource._parse_memory_to_gb(), expected_gb)

    def test_format_methods(self):
        """Test all format methods with various configurations."""
        test_configs = [
            # (cpu_cores, memory, tasks_per_node, expected_mem_gb)
            (24, "225G", 1, 225),
            (64, "1T", 2, 1024),
            (8, "32G", 1, 32),
            (128, "4T", 1, 4096),
        ]

        for cpu_cores, memory, tasks_per_node, expected_mem_gb in test_configs:
            with self.subTest(config=f"{cpu_cores}cpu_{memory}_{tasks_per_node}tasks"):
                resource = ResourceShape(
                    cpu_cores=cpu_cores, memory=memory, tasks_per_node=tasks_per_node
                )

                # Test JSON format
                json_result = json.loads(resource.to_json())
                self.assertEqual(json_result["cpu_cores"], cpu_cores)
                self.assertEqual(json_result["memory"], memory)
                self.assertEqual(json_result["tasks_per_node"], tasks_per_node)
                self.assertEqual(json_result["mem_gb"], expected_mem_gb)

                # Test SBATCH format
                sbatch_result = resource.to_sbatch()
                self.assertIn("#!/bin/bash", sbatch_result)
                self.assertIn(f"--cpus-per-task={cpu_cores}", sbatch_result)
                self.assertIn(f"--mem={memory}", sbatch_result)
                self.assertIn(f"--ntasks-per-node={tasks_per_node}", sbatch_result)

                # Test SRUN format
                srun_result = resource.to_srun()
                expected_srun = f"srun --cpus-per-task={cpu_cores} --mem={memory} --ntasks-per-node={tasks_per_node}"
                self.assertEqual(srun_result, expected_srun)

                # Test Submitit format
                submitit_result = json.loads(resource.to_submitit())
                self.assertEqual(submitit_result["cpus_per_task"], cpu_cores)
                self.assertEqual(submitit_result["mem_gb"], expected_mem_gb)
                self.assertEqual(submitit_result["tasks_per_node"], tasks_per_node)

    def test_to_json(self):
        """Test to_json format method with various configurations."""
        test_configs = [
            # (cpu_cores, memory, tasks_per_node, expected_mem_gb)
            (24, "225G", 1, 225),
            (64, "1T", 2, 1024),
            (8, "32G", 1, 32),
            (128, "4T", 1, 4096),
            (1, "1G", 1, 1),  # Minimum values
            (256, "16T", 4, 16384),  # Large values
        ]

        for cpu_cores, memory, tasks_per_node, expected_mem_gb in test_configs:
            with self.subTest(config=f"{cpu_cores}cpu_{memory}_{tasks_per_node}tasks"):
                resource = ResourceShape(
                    cpu_cores=cpu_cores, memory=memory, tasks_per_node=tasks_per_node
                )

                result = json.loads(resource.to_json())

                # Verify all required keys are present
                required_keys = {"cpu_cores", "memory", "tasks_per_node", "mem_gb"}
                self.assertEqual(set(result.keys()), required_keys)

                # Verify values
                self.assertEqual(result["cpu_cores"], cpu_cores)
                self.assertEqual(result["memory"], memory)
                self.assertEqual(result["tasks_per_node"], tasks_per_node)
                self.assertEqual(result["mem_gb"], expected_mem_gb)

    def test_to_sbatch(self):
        """Test to_sbatch format method with various configurations."""
        test_configs = [
            (24, "225G", 1),
            (64, "1T", 2),
            (8, "32G", 1),
            (128, "4T", 1),
            (1, "1G", 1),  # Minimum values
            (256, "16T", 4),  # Large values
        ]

        for cpu_cores, memory, tasks_per_node in test_configs:
            with self.subTest(config=f"{cpu_cores}cpu_{memory}_{tasks_per_node}tasks"):
                resource = ResourceShape(
                    cpu_cores=cpu_cores, memory=memory, tasks_per_node=tasks_per_node
                )

                result = resource.to_sbatch()
                lines = result.split("\n")

                # Verify shebang
                self.assertEqual(lines[0], "#!/bin/bash")

                # Verify SBATCH directives are present
                sbatch_lines = [line for line in lines if line.startswith("#SBATCH")]
                self.assertEqual(len(sbatch_lines), 3)  # cpus, mem, ntasks

                # Verify specific directives
                self.assertIn(f"#SBATCH --cpus-per-task={cpu_cores}", result)
                self.assertIn(f"#SBATCH --mem={memory}", result)
                self.assertIn(f"#SBATCH --ntasks-per-node={tasks_per_node}", result)

    def test_to_srun(self):
        """Test to_srun format method with various configurations."""
        test_configs = [
            (24, "225G", 1),
            (64, "1T", 2),
            (8, "32G", 1),
            (128, "4T", 1),
            (1, "1G", 1),  # Minimum values
            (256, "16T", 4),  # Large values
        ]

        for cpu_cores, memory, tasks_per_node in test_configs:
            with self.subTest(config=f"{cpu_cores}cpu_{memory}_{tasks_per_node}tasks"):
                resource = ResourceShape(
                    cpu_cores=cpu_cores, memory=memory, tasks_per_node=tasks_per_node
                )

                result = resource.to_srun()
                expected_command = f"srun --cpus-per-task={cpu_cores} --mem={memory} --ntasks-per-node={tasks_per_node}"

                self.assertEqual(result, expected_command)

                # Verify command structure
                parts = result.split()
                self.assertEqual(parts[0], "srun")
                self.assertIn(f"--cpus-per-task={cpu_cores}", result)
                self.assertIn(f"--mem={memory}", result)
                self.assertIn(f"--ntasks-per-node={tasks_per_node}", result)

    def test_to_submitit(self):
        """Test to_submitit format method with various configurations."""
        test_configs = [
            # (cpu_cores, memory, tasks_per_node, expected_mem_gb)
            (24, "225G", 1, 225),
            (64, "1T", 2, 1024),
            (8, "32G", 1, 32),
            (128, "4T", 1, 4096),
            (1, "1G", 1, 1),  # Minimum values
            (256, "16T", 4, 16384),  # Large values
        ]

        for cpu_cores, memory, tasks_per_node, expected_mem_gb in test_configs:
            with self.subTest(config=f"{cpu_cores}cpu_{memory}_{tasks_per_node}tasks"):
                resource = ResourceShape(
                    cpu_cores=cpu_cores, memory=memory, tasks_per_node=tasks_per_node
                )

                result = json.loads(resource.to_submitit())

                # Verify all required keys are present
                required_keys = {"cpus_per_task", "mem_gb", "tasks_per_node"}
                self.assertEqual(set(result.keys()), required_keys)

                # Verify values
                self.assertEqual(result["cpus_per_task"], cpu_cores)
                self.assertEqual(result["mem_gb"], expected_mem_gb)
                self.assertEqual(result["tasks_per_node"], tasks_per_node)


if __name__ == "__main__":
    unittest.main()
