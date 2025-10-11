---
sidebar_position: 4
---

# cscope gpus

`cscope gpus` shows information about available gpus. If its on a Slurm cluster it shows information about every partition, if not, it defaults to the local node.

```shell
$ cscope gpus --help
Usage: cscope gpus [OPTIONS]

  Show GPU information.

Options:
  --partition TEXT  Slurm partition name to filter queries (optional)
  --generations     Show only GPU generations
  --counts          Show only GPU counts by type
  --vendor          Show GPU vendor information
  --help            Show this message and exit.
```

## Slurm

```shell
$ cscope gpus
GPU Gen, GPU Count, GPU Vendor, Slurm Partition:
h100, 8, nvidia, h100
h200, 8, nvidia, h200
```

### Slurm gpu generations

```shell
$ cscope gpus --generations
GPU Gen, Slurm Partition:
h100, h100
h200, h200
```

### Slurm gpu counts

```shell
$ cscope gpus --counts
GPU Gen, GPU Count, Slurm Partition:
h100, 8, h100
h200, 8, h200
```

### Slurm gpu vendors

```shell
$ cscope gpus --vendor
GPU Vendors:
nvidia
```

## Slurm Partition Filter

You can also pass an optional partition arg: `... --partition=<partition-name>`, if partition is passed it limits the `cscope gpus` for only that Slurm partition. It works with any of the other `cscope gpus` flags.

```shell
$ cscope gpus --partition=h100
GPU Gen, GPU Count, GPU Vendor, Slurm Partition:
h100, 8, nvidia, h100
```
