---
sidebar_position: 2
---

# cscope check-gpu

`cscope check-gpu` takes a GPU_TYPE argument and prints whether the cluster has that GPU Type available. If on a Slurm cluster it will search for available GPUs in Slurm, if not, it defaults to what's available for the local node.

```shell
$ cscope check-gpu --help
Usage: cscope check-gpu [OPTIONS] GPU_TYPE

  Check if a specific GPU type exists.

  GPU_TYPE: GPU type to check for (e.g., A100, MI300X)

Options:
  --partition TEXT  Slurm partition name to filter queries (optional)
  --help            Show this message and exit.
```

```shell
$ cscope check-gpu h100
GPU type h100 is available in the cluster.
$ cscope check-gpu h300
GPU type h300 is NOT available in the cluster.
```

## Slurm Partition Filter

You can also pass an optional partition arg: `... --partition=<partition-name>`, if partition is passed it limits the check-gpu for only that Slurm partition.

```shell
$ cscope check-gpu h100 --partition=h100
GPU type h100 is available in the cluster.
$ cscope check-gpu h100 --partition=h200
GPU type h100 is NOT available in the cluster.
```