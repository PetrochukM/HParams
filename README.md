<h3 align="center">Get your hyperparameters in a row.</h3>

HParams is a library for machine learning hyperparameter management.

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

## Basics

Add HParams to your project by following one of the common use cases:

### Set Defaults

Configure PyTorch and Tensorflow defaults to match like so:

```python3
from torch import nn
from hparams import configurable, add_config, HParams

# NOTE: `momentum=0.01` to match Tensorflow defaults
nn.modules.batchnorm._BatchNorm.__init__ = configurable(nn.modules.batchnorm._BatchNorm.__init__)
add_config({ nn.modules.batchnorm._BatchNorm.__init__: HParams(momentum=0.01) })
```

### Validate

Validate your hyperparameters like so:

```python3
from hparams import HParam

class Model():
    def __init__(hidden_size=HParam(int), dropout=HParam(float))
        ...
```

`HParam(int)` is syntactic sugar for `Union[int, HParam] = HParam()`.

### CLI

Enable rapid command line experimentation like so:

```bash
$ experiment.py --torch.optim.adam.Adam.__init__ HParams(lr=0.1,betas=(0.999,0.99))
```

```python3
# experiment.py
import torch
import sys
from hparams import configurable, add_config, parse_hparam_args

add_config(parse_hparam_args(sys.argv)) # Parse command line arguments

torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)

# Adam with defaults `lr=0.1` and  `betas=(0.999,0.99)`.
torch.optim.Adam()
```

### Config File(s)

Extend your library to support multiple model configurations like so:

```python3
import torch
import random
from hparams import configurable, add_config, HParams

torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)
random.seed = configurable(random.seed)

add_config({
    # Set the learning rate
    torch.optim.adam.Adam.__init__: HParams(lr=10**-3),

    # Set the random seed
    random.seed: HParams(a=123),

    # Set other source code hyperparameters
    'main': {
        'Model.__init__': HParams(
            hidden_size=256,
            dropout=0.1,
        ),
        'Trainer.__init__': HParams(
            batch_size=32,
            gradient_clipping=0.25,
            optimizer=torch.optim.Adam,
        )
    }
})
```

This example assumes that you have a `Model` and `Trainer` class defined in `main.py`.

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
