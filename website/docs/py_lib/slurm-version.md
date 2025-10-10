---
sidebar_position: 8
---

# slurm_version()

`slurm_version()` returns a tuple of integers with the Slurm Version. Returns `(0, )` if not on a Slurm Cluster.

```python
import clusterscope
slurm_version = clusterscope.slurm_version()
print(slurm_version)
# (24, 11, 5)
```