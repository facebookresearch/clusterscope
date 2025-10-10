---
sidebar_position: 1
---

# cluster()

`cluster()` removes the need from teams to maintain custom logic to identify the cluster when operating accross many clusters. This API retrieves the Slurm cluster name from `scontrol show config` if on a Slurm cluster, otherwise it returns:

- 'github': if running from Github Actions
- 'macos': if running from a Darwin system
- 'local-node' otherwise

It caches the cluster name at `/tmp/clusterscopewhoami`, so as long as files can live in `/tmp` it will not re-run the cluster identification from above. Once cached, this reduces the dependencies of this call to only the filesystem under `/tmp`- i.e. removes the dependency on Slurm (`scontrol show config`).

Sample Usage:

```python
import clusterscope
cluster_name = clusterscope.cluster()

print(cluster_name)
# <your-cluster-name>
```
