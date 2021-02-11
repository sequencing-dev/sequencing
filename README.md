# seQuencing

![seQuencing logo](docs/source/images/sequencing-logo.svg)

``sequencing`` is a Python package for simulating realistic quantum control sequences using [QuTiP](http://qutip.org/docs/latest/index.html). Built for researchers and quantum engineers, ``sequencing`` provides an intuitive framework for constructing models of quantum systems
composed of many modes and generating complex time-dependent control Hamiltonians
for [master equation simulations](http://qutip.org/docs/latest/guide/dynamics/dynamics-master.html).

![PyPI](https://img.shields.io/pypi/v/sequencing) ![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/sequencing-dev/sequencing/lint-and-test/main) [![Documentation Status](https://readthedocs.org/projects/sequencing/badge/?version=latest)](https://sequencing.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/sequencing-dev/sequencing/branch/main/graph/badge.svg?token=LLABAKBJ0C)](https://codecov.io/gh/sequencing-dev/sequencing) ![GitHub](https://img.shields.io/github/license/sequencing-dev/sequencing) [![DOI](https://zenodo.org/badge/334427937.svg)](https://zenodo.org/badge/latestdoi/334427937)

## Documentation

The documentation for `sequencing` is available at:
[sequencing.readthedocs.io](https://sequencing.readthedocs.io/)

## Installation

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sequencing)

```
conda create -n <env-name> python=<3.7, 3.8, or 3.9>
conda activate <env-name>
pip install sequencing
```

`sequencing` requires `python>=3.7` and is tested on `3.7`, `3.8`, and `3.9`. For more details, see the [documentation](https://sequencing.readthedocs.io/en/latest/installation.html).

## Tutorials

The documentation includes a set of [tutorial notebooks](https://sequencing.readthedocs.io/en/latest/tutorials/tutorials.html). Click the badge below to run the notebooks interactively online using [binder](https://mybinder.org/):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sequencing-dev/sequencing/main?filepath=docs%2Fsource%2Fnotebooks)

## Authors

Primary author and maintainer: [@loganbvh](https://github.com/loganbvh/).

## Contributing

Want to contribute to `sequencing`? Check out our [contribution guidelines](CONTRIBUTING.md).

## Acknowledging

If you used `sequencing` for work that was published in a journal article, preprint, blog post, etc., please cite/acknowledge the `sequencing` project using its DOI:

[![DOI](https://zenodo.org/badge/334427937.svg)](https://zenodo.org/badge/latestdoi/334427937)

**Uploading Examples**

So that others may learn from and reproduce the results of your work, please consider uploading a demonstration of the simulations performed for your publication in the form of well-documented Jupyter notebooks or Python files to the [sequencing-examples](https://github.com/sequencing-dev/sequencing-examples) repository.
