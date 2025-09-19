# Cluster Scope

clusterscope is a binary and python library to extract core information from HPC Clusters.

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

```bash
$ cscope
usage: cscope [-h] {info,cpus,gpus,check-gpu,aws} ...
...
```

## Contributing

Read our contributing guide to learn about our development process, how to propose bugfixes and feature requests, and how to build your changes.

### [Code of Conduct](https://code.fb.com/codeofconduct)

Facebook has adopted a Code of Conduct that we expect project participants to adhere to. Please read [the full text](https://code.fb.com/codeofconduct) so that you can understand what actions will and will not be tolerated.

## Maintainers

clusterscope is actively maintained by [Lucca Bertoncini](https://github.com/luccabb), and [Kalyan Saladi](https://github.com/skalyan).

### License

clusterscope is licensed under the [MIT](./LICENSE) license.
