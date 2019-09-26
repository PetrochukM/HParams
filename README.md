<h3 align="center">Stop messing around and organize your hyperparameters.</h3>

HParams is a configuration management solution for machine learning projects.

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hparams.svg?style=flat-square)
[![Codecov](https://img.shields.io/codecov/c/github/PetrochukM/HParams/master.svg?style=flat-square)](https://codecov.io/gh/PetrochukM/HParams)
[![Downloads](http://pepy.tech/badge/hparams)](http://pepy.tech/project/hparams)
[![Build Status](https://img.shields.io/travis/PetrochukM/HParams/master.svg?style=flat-square)](https://travis-ci.org/PetrochukM/HParams)

_Logo by [Chloe Yeo](http://www.yeochloe.com/)_

## Installation

Make sure you have Python 3. You can then install `hparams` using `pip`:

    pip install hparams

Or to install the latest code via:

    pip install git+https://github.com/PetrochukM/HParams.git

## What is HParams?

HParams is a configuration management solution for machine learning projects. With this you can
externalize your hyparameters ensuring that they are extensible, accessible and maintainable.

Technically speaking, HParams uses the `@configurable` decorator to inject your hyperparameter
dependencies at runtime from a designated configuration file.

Notable Features:

- HParams is small requiring only one dependency
- `@configurable` adds less than 1e-05 seconds of overhead
- HParams supports Python's notorious `multiprocessing` module

## Basics

Add HParams to your project by following one of the common use cases:

### Configure Training

Configure your training run like so:

```python3
from hparams import configurable, add_config, HParams, HParam

@configurable
def train(batch_size=HParam(int)):
    pass

add_config({ train: HParams(batch_size=32) })
```

HParams supports optional configuration typechecking to help you find bugs. Also you can use
HParams with `json` to support multiple model configurations!

### Set Defaults

Configure PyTorch and Tensorflow defaults to match, enabling reproducibility, like so:

```python3
from torch.nn import BatchNorm1d
from hparams import configurable, add_config, HParams

# NOTE: `momentum=0.01` to match Tensorflow defaults
BatchNorm1d.__init__ = configurable(BatchNorm1d.__init__)
add_config({ 'torch.nn.BatchNorm1d.__init__': HParams(momentum=0.01) })
```

### CLI

Enable rapid command line experimentation, for example:

```bash
$ file.py --torch.optim.adam.Adam.__init__ HParams(lr=0.1,betas=(0.999,0.99))
```

```python3
import sys
from torch.optim import Adam
from hparams import configurable, add_config, parse_hparam_args

Adam.__init__ = configurable(Adam.__init__)
parsed = parse_hparam_args(sys.argv) # Parse command line arguments
add_config(parsed)
```

### Track Hyperparameters

Easily track your hyperparameters using tools like [Comet](comet.ml).

```
from comet_ml import Experiment
from hparams import get_config

experiment = Experiment()
experiment.log_parameters(get_config())
```

### Multiprocessing: Partial Support

Export a Python `functools.partial` to use in another process like so:

```
from hparams import configurable, HParam

@configurable
def func(hparam=HParam(int)):
    pass

partial = func.get_configured_partial()
```

With this approach, you don't have to transfer the entire global state to the new process.

## Contributing

We've released HParams because a lack of hyperparameter management solutions. We hope that
other people can benefit from the project. We are thankful for any contributions from the
community.

### Contributing Guide

Read our [contributing guide](https://github.com/PetrochukM/HParams/blob/master/CONTRIBUTING.md) to
learn about our development process, how to propose bugfixes and improvements, and how to build and
test your changes to hparams.

## Authors

* [Michael Petrochuk](https://github.com/PetrochukM/) — Developer
* [Chloe Yeo](http://www.yeochloe.com/) — Logo Design

## Citing

If you find hparams useful for an academic publication, then please use the following BibTeX to
cite it:

```
@misc{hparams,
  author = {Petrochuk, Michael},
  title = {HParams: Hyperparameter management},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PetrochukM/HParams}},
}
```
