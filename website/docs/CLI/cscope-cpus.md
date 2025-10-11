---
sidebar_position: 3
---

# cscope cpus

`cscope cpus` shows information about how many cpu cores are available. If its on a Slurm cluster it shows information about every partition, if not, it defaults to the local node.

## Slurm

```shell
$ cscope cpus
CPU Count, Slurm Partition:
192, h100
```

## Slurm Partition Filter

You can also pass an optional partition arg: `... --partition=<partition-name>`, if partition is passed it limits the `cscope cpus` for only that Slurm partition.

```shell
$ cscope cpus --partition=h100
CPU Count, Slurm Partition:
192, h100
```

## Local Node

```shell
$ cscope cpus
CPU Count, Slurm Partition:
16, None
```
