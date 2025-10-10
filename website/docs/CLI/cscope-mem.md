---
sidebar_position: 7
---

# cscope mem

`cscope mem` shows information about how much memory is availble. If its on a Slurm cluster it shows information about every partition, if not, it defaults to the local node.

## Slurm

```shell
$ cscope mem
Mem information:
  partition: cpu, mem_total_MB: 1523799, mem_total_GB: 1488
  partition: h100, mem_total_MB: 2047959, mem_total_GB: 1999
  partition: h200, mem_total_MB: 2047959, mem_total_GB: 1999
```

## Slurm Partition Filter

You can also pass an optional partition arg: `... --partition=<partition-name>`, if partition is passed it limits the `cscope cpus` for only that Slurm partition.

```shell
$ cscope mem --partition=h100
Mem information:
  partition: h100, mem_total_MB: 2047959, mem_total_GB: 1999
```

## Local Node

```shell
➜ cscope mem
Mem information:
  mem_total_MB: 65536, mem_total_GB: 64
```
