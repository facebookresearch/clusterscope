---
sidebar_position: 7
---

# cscope mem

`cscope mem` shows information about how much memory is availble. If its on a Slurm cluster it shows information about every partition, if not, it defaults to the local node.

## Slurm

```shell
$ cscope mem
Mem total MB, Mem total GB, Slurm Partition:
1523799, 1488, cpu
2047959, 1999, h100
2047959, 1999, h200
```

## Slurm Partition Filter

You can also pass an optional partition arg: `... --partition=<partition-name>`, if partition is passed it limits the `cscope cpus` for only that Slurm partition.

```shell
$ cscope mem --partition=h100
Mem total MB, Mem total GB, Slurm Partition:
2047959, 1999, h100
```

## Local Node

```shell
$ cscope mem
Mem total MB, Mem total GB, Slurm Partition:
65536, 64, None
```
