---
sidebar_position: 5
---

# mem()

`mem()` shows information about how many cpu cores are available in the System. If its in a Slurm cluster it shows information per partition, if not, it defaults to the local node.

`mem()` returns a list of `MemInfo` or a single instance of `MemInfo`. If on a Slurm Cluster it returns a list of all partitions:

```python
import clusterscope
mem_info = clusterscope.mem()
print(mem_info)
# [
#   MemInfo(mem_total_MB=1523799, mem_total_GB=1488, partition='cpu'), 
#   MemInfo(mem_total_MB=2047959, mem_total_GB=1999, partition='h100'), 
#   MemInfo(mem_total_MB=2047959, mem_total_GB=1999, partition='h200')
# ]
```

If not on a Slurm Cluster it defaults to the local node:

```python
import clusterscope
mem_info = clusterscope.mem()
print(mem_info)
# MemInfo(mem_total_MB=65536, mem_total_GB=64, partition=None)
```

## Slurm Partition Filter

The Optional partition arg limits the output to that partition only:

```python
import clusterscope
mem_info = clusterscope.mem(partition='cpu')
print(mem_info)
# MemInfo(mem_total_MB=1523799, mem_total_GB=1488, partition='cpu')
```