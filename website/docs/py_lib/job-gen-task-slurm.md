---
sidebar_position: 2
---

# job_gen_task_slurm()

:::tip

Python LIB version of [`cscope job-gen` command](../CLI/cscope-job-gen.md)

:::

`job_gen_task_slurm()` generates Slurm arguments for [GPU Jobs](#cscope-job-gen-for-gpu-jobs) and [CPU Jobs](#cscope-job-gen-for-cpu-jobs). The user pass a number of Resources (GPUs or CPUs) and a Slurm partition and `job_gen_task_slurm()` calculates the proportionate amount of CPUs and Memory to allocate given what's available in that partition. 


## cscope job-gen for GPU Jobs

Asking for 8 out of 8 GPUs, allocates all the Memory and CPU:
```python
import clusterscope
slurm_args = clusterscope.job_gen_task_slurm(partition="h100", gpus_per_task=8)
print(slurm_args)
# {'cpus_per_task': 192, 'memory': '1999G', 'tasks_per_node': 1, 'slurm_partition': 'h100', 'nodes': 1, 'gpus_per_task': 8, 'mem_gb': 1999}
```


Asking for half of the GPUs, allocates half the Memory and CPU:
```python
import clusterscope
slurm_args = clusterscope.job_gen_task_slurm(partition="h100", gpus_per_task=4)
print(slurm_args)
# {'cpus_per_task': 96, 'memory': '999G', 'tasks_per_node': 1, 'slurm_partition': 'h100', 'nodes': 1, 'gpus_per_task': 4, 'mem_gb': 999}
```

## cscope job-gen for CPU Jobs

Asking for all CPUs, allocates all the Memory:
```python
import clusterscope
slurm_args = clusterscope.job_gen_task_slurm(partition="h100", cpus_per_task=192)
print(slurm_args)
# {'cpus_per_task': 192, 'memory': '1999G', 'tasks_per_node': 1, 'slurm_partition': 'h100', 'nodes': 1, 'mem_gb': 1999}
```

Asking for half of the CPUs, allocates half the Memory:
```python
import clusterscope
slurm_args = clusterscope.job_gen_task_slurm(partition="h100", cpus_per_task=96)
print(slurm_args)
# {'cpus_per_task': 96, 'memory': '999G', 'tasks_per_node': 1, 'slurm_partition': 'h100', 'nodes': 1, 'mem_gb': 999}
```

