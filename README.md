# Clusterscope

Clusterscope is a CLI and python library to extract information from HPC Clusters and Jobs.

Check out our [Website](https://facebookresearch.github.io/clusterscope/) for Getting Started and Documentation.

## Install from pypi

```bash
$ pip install clusterscope
```

You can use it as a python library:

```bash
$ python
>>> import clusterscope
>>> clusterscope.cluster()
'<your-cluster-name>'
```

You can also use it as a CLI:

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

## Contributing

Read our contributing guide to learn about our development process, how to propose bugfixes and feature requests, and how to build your changes.

### [Code of Conduct](https://code.fb.com/codeofconduct)

Facebook has adopted a Code of Conduct that we expect project participants to adhere to. Please read [the full text](https://code.fb.com/codeofconduct) so that you can understand what actions will and will not be tolerated.

## Maintainers

clusterscope is actively maintained by [Lucca Bertoncini](https://github.com/luccabb), and [Kalyan Saladi](https://github.com/skalyan).

## Contributors

[Lucca Bertoncini](https://github.com/luccabb), [Kalyan Saladi](https://github.com/skalyan), [Nikhil Gupta](https://github.com/gunchu), [Misko Dzamba](https://github.com/misko), <Feel free to contribute and add your name>

### License

clusterscope is licensed under the [MIT](./LICENSE) license.

## Citing Clusterscope

```
@Misc{
  author =       {Lucca Bertoncini, and Kalyan Saladi},
  title =        {Clusterscope - Tooling for extracting HPC Cluster and Jobs information. },
  howpublished = {Github},
  year =         {2025},
  url =          {https://github.com/facebookresearch/clusterscope}
}
```
