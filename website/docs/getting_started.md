---
sidebar_position: 1
---

# Getting Started

## Introduction

Clusterscope is a CLI and python library to extract information from HPC Clusters and Jobs.

## Quick Start Guide

### Installation

```shell
pip install clusterscope
```

Installing clusterscope gives you a CLI and a Python Library.

### CLI

Check out our [CLI Docs](./category/cli---command-line-interface) for more information.

```shell
$ cscope
Usage: cscope [OPTIONS] COMMAND [ARGS]...

  Command-line tool to query Slurm cluster information.

Options:
  --help  Show this message and exit.

Commands:
  aws        Check if running on AWS and show NCCL settings.
  check-gpu  Check if a specific GPU type exists.
  cpus       Show CPU counts per node.
  gpus       Show GPU information.
  info       Show basic cluster information.
  job-gen    Generate job requirements for different job types.
  mem        Show memory information per node.
  version    Show the version of clusterscope.
```

### Python Library

Check out our [Python Library Docs](./category/python-library) for more information.

```shell
$ python
>>> import clusterscope
>>> clusterscope.cluster()
'<your-cluster-name>'
```

## Others

### Community

You can [ask questions](https://github.com/facebookresearch/clusterscope/discussions), [open issues and feature-requests](https://github.com/facebookresearch/clusterscope/issues) in [Github](https://github.com/facebookresearch/clusterscope/).

### Citing Clusterscope
If you use Clusterscope in your research please use the following BibTeX entry:

```
@Misc{
  author =       {Lucca Bertoncini, Kalyan Saladi},
  title =        {Clusterscope - Tooling for extracting HPC Cluster and Jobs information. },
  howpublished = {Github},
  year =         {2025},
  url =          {https://github.com/facebookresearch/clusterscope}
}
```
