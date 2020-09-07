<p align="center"><img width="544px" src="logo.svg" /></p>

<h3 align="center">Extensible and Fault-Tolerant Hyperparameter Management</h3>

HParams is a thoughtful approach to configuration management for machine learning projects. It
enables you to externalize your hyperparameters into a configuration file. In doing so, you can
reproduce experiments, iterate quickly, and reduce errors.

**Features:**

- Approachable and easy-to-use API
- Battle-tested over three years
- Fast with little to no runtime overhead (< 3e-05 seconds) per configured function
- Robust to most use cases with 100% test coverage and 75 tests
- Lightweight with only one dependency

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hparams.svg?style=flat-square)
[![Codecov](https://img.shields.io/codecov/c/github/PetrochukM/HParams/master.svg?style=flat-square)](https://codecov.io/gh/PetrochukM/HParams)
[![Downloads](http://pepy.tech/badge/hparams)](http://pepy.tech/project/hparams)
[![Build Status](https://img.shields.io/travis/PetrochukM/HParams/master.svg?style=flat-square)](https://travis-ci.org/PetrochukM/HParams)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Twitter: PetrochukM](https://img.shields.io/twitter/follow/MPetrochuk.svg?style=social)](https://twitter.com/MPetrochuk)

_Logo by [Chloe Yeo](http://www.yeochloe.com/), Corporate Sponsorship by [WellSaid Labs](https://wellsaidlabs.com/)_

## Installation

Make sure you have Python 3. You can then install `hparams` using `pip`:

```bash
pip install hparams
```

Install the latest code via:

```bash
pip install git+https://github.com/PetrochukM/HParams.git
```

## Oops 🐛

With HParams, you will avoid common but needless hyperparameter mistakes. It will throw a warning
or error if:

- A hyperparameter is overwritten.
- A hyperparameter is declared but not set.
- A hyperparameter is set but not declared.
- A hyperparameter type is incorrect.

Finally, HParams is built with developer experience in mind. HParams includes 13 errors and 6 warnings
to help catch and resolve issues quickly.

## Examples

Add HParams to your project by following one of these common use cases:

### Configure Training 🤗

Configure your training run, like so:

```python
# main.py
from hparams import configurable, add_config, HParams, HParam
from typing import Union

@configurable
def train(batch_size: int = HParam()):
    pass

class Model():

    @configurable
    def __init__(self, hidden_size=HParam(int), dropout=HParam(float)):
        pass

add_config({ 'main': {
    'train': HParams(batch_size=32),
    'Model.__init__': HParams(hidden_size=1024, dropout=0.25),
}})
```

HParams supports optional configuration typechecking to help you find bugs! 🐛

### Set Defaults

Configure PyTorch and Tensorflow defaults to match via:

```python
from torch.nn import BatchNorm1d
from hparams import configurable, add_config, HParams

# NOTE: `momentum=0.01` to match Tensorflow defaults
BatchNorm1d.__init__ = configurable(BatchNorm1d.__init__)
add_config({ 'torch.nn.BatchNorm1d.__init__': HParams(momentum=0.01) })
```

Configure your random seed globally, like so:

```python
# config.py
import random
from hparams import configurable, add_config, HParams

random.seed = configurable(random.seed)
add_config({'random.seed': HParams(a=123)})
```

```python
# main.py
import config
import random

random.seed()
```

### CLI

Experiment with hyperparameters through your command line, for example:

```console
foo@bar:~$ file.py --torch.optim.adam.Adam.__init__ 'HParams(lr=0.1,betas=(0.999,0.99))'
```

```python
import sys
from torch.optim import Adam
from hparams import configurable, add_config, parse_hparam_args

Adam.__init__ = configurable(Adam.__init__)
parsed = parse_hparam_args(sys.argv[1:])  # Parse command line arguments
add_config(parsed)
```

### Hyperparameter optimization

Hyperparameter optimization is easy to-do, check this out:

```python
import itertools
from torch.optim import Adam
from hparams import configurable, add_config, HParams

Adam.__init__ = configurable(Adam.__init__)

def train():  # Train the model and return the loss.
    pass

for betas in itertools.product([0.999, 0.99, 0.9], [0.999, 0.99, 0.9]):
    add_config({Adam.__init__: HParams(betas=betas)})  # Grid search over the `betas`
    train()
```

### Track Hyperparameters

Easily track your hyperparameters using tools like [Comet](comet.ml).

```python
from comet_ml import Experiment
from hparams import get_config

experiment = Experiment()
experiment.log_parameters(get_config())
```

### Multiprocessing: Partial Support

Export a Python `functools.partial` to use in another process, like so:

```python
from hparams import configurable, HParam

@configurable
def func(hparam=HParam()):
    pass

partial = func.get_configured_partial()
```

With this approach, you don't have to transfer the global state to the new process. To transfer the
global state, you'll want to use `get_config` and `add_config`.

## Docs 📖

The complete documentation for HParams is available [here](./DOCS.md).

Learn more about related projects to HParams [here](./RELATED.md).

## Contributing

We've released HParams because a lack of hyperparameter management solutions. We hope that
other people can benefit from the project. We are thankful for any contributions from the
community.

### Contributing Guide

Read our [contributing guide](https://github.com/PetrochukM/HParams/blob/master/CONTRIBUTING.md) to
learn about our development process, how to propose bugfixes and improvements, and how to build and
test your changes to HParams.

## Authors

- [Michael Petrochuk](https://github.com/PetrochukM/) — Developer
- [Chloe Yeo](http://www.yeochloe.com/) — Logo Design

## Citing

If you find HParams useful for an academic publication, then please use the following BibTeX to
cite it:

```latex
@misc{hparams,
author = {Petrochuk, Michael},
title = {HParams: Hyperparameter management solution},
year = {2019},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/PetrochukM/HParams}},
}
```

