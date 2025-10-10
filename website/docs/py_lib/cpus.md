---
sidebar_position: 4
---

# cpus()

`cpus()` shows information about how many cpu cores are available in the System. If its in a Slurm cluster it shows information per partition, if not, it defaults to the local node.

:::tip

if you're interested in the CPUs assigned for your job, check out [get_job()](./get-job.md#get_cpus)

:::

`cpus()` returns a list of `CPUInfo` or a single instance of `CPUInfo`. If on a Slurm Cluster it returns a list of all partitions:

```python
import clusterscope
cpu_info = clusterscope.cpus()
print(cpu_info)
# [
#   CPUInfo(cpu_count=192, partition='cpu'), 
#   CPUInfo(cpu_count=192, partition='h100'), 
#   CPUInfo(cpu_count=192, partition='h200')
# ]
```

If not on a Slurm Cluster it defaults to the local node:

```python
import clusterscope
cpu_info = clusterscope.cpus()
print(cpu_info)
# CPUInfo(cpu_count=16, partition=None)
```

## Slurm Partition Filter

The Optional partition arg limits the output to that partition only:

```python
import clusterscope
cpu_info = clusterscope.cpus(partition='cpu')
print(cpu_info)
# CPUInfo(cpu_count=192, partition='cpu')
```