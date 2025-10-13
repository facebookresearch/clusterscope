# Contributing to clusterscope

## Development Workflow

### Environment setup

Running the below from the root of this repository brings [uv](https://docs.astral.sh/uv/), all required development dependencies, and installs clusterscope in editable mode:

```shell
make install-dev-requirements
```

You can then activate the env:
```shell
source <(make activate)
```

This should get you started with a binary and library available in your local environment:

```shell
$ python
>>> import clusterscope
>>> clusterscope.cluster()
'<your-cluster-name>'
```

```shell
$ cscope
usage: cscope [-h] {info,cpus,gpus,check-gpu,aws} ...
...
```

### `pre-commit`

We have all linters/formatters/typecheckers integrated into pre-commit, these checks are also running as part of github CI. pre-commit automates part of the changes that will be required to land code on the repo. You can run the below to activate pre-commit in your local env:

```
pre-commit install
```

### Requirements

If you update the requirements, make sure to add it to [`pyproject.toml`](./pyproject.toml)'s appropriate section for the dependency. Then you can run the below to update the requirements file:

```
$ make requirements.txt
```

For development dependencies:

```
$ make dev-requirements.txt
```

## Pull Requests
We welcome your pull requests.

1. Fork the repo and create your feature branch from `main`.
1. If you've added code add suitable tests.
1. Ensure the test suite and lint pass.
1. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to clusterscope, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
