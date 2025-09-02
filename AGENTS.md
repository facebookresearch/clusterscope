## Overview

Clusterscope is a binary and python library to extract core information from HPC Clusters. It queries Slurm cluster information, local node details, AWSClusters, and Job information when running it from inside a Job. It provides unified access to cluster resources, GPU information, CPU counts, memory details, and AWS-specific configurations.

## Instructions

- the CLI is invoked via `cscope`
- the python library is imported as `import clusterscope`
- make sure all your changes are consistent between the CLI and the library
- DO NOT run pre-commit.
- DO NOT add comments, unless explicitly asked to add.
- Makefile has all the relevant CMDS to build a local environment.
- If you update dependencies, don't forget to update the pyproject.toml file as appropriate. And to keep dependencies in sync with: `make requirements.txt`, `make dev-requirements.txt`
