---
sidebar_position: 1
---

# cscope aws

`cscope aws` prints whether or not the current env is an AWS Cluster and what are the recommended NCCL settings if its an AWS Cluster:

```shell
$ cscope aws --help
Usage: cscope aws [OPTIONS]

  Check if running on AWS and show NCCL settings.

Options:
  --help  Show this message and exit.
```

```shell
$ cscope aws
This is an AWS cluster.

Recommended NCCL settings:
...
```

```shell
$ cscope aws
This is NOT an AWS cluster.
```
