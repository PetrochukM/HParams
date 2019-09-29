<h3 align="center">Clean up your code and organize your hyperparameters</h3>

HParams is a configuration management solution for machine learning projects. With this you can
externalize your hyperparameters ensuring that they are extensible, accessible, and maintainable.

Notables:

   - 68 tests and 100% test coverage to cover most use cases.
   - Little to no runtime overhead (< 1e-05 seconds) per configured function.
   - Battle tested over 2 years with two ML teams.
   - Requires only one dependency.

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hparams.svg?style=flat-square)
[![Codecov](https://img.shields.io/codecov/c/github/PetrochukM/HParams/master.svg?style=flat-square)](https://codecov.io/gh/PetrochukM/HParams)
[![Downloads](http://pepy.tech/badge/hparams)](http://pepy.tech/project/hparams)
[![Build Status](https://img.shields.io/travis/PetrochukM/HParams/master.svg?style=flat-square)](https://travis-ci.org/PetrochukM/HParams)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://opensource.org/licenses/MIT)

_Logo by [Chloe Yeo](http://www.yeochloe.com/)_

## Installation

Make sure you have Python 3. You can then install `hparams` using `pip`:

    pip install hparams

Install the latest code via:

    pip install git+https://github.com/PetrochukM/HParams.git

## Examples 🤔

Add HParams to your project by following one of the common use cases:

### Configure Training

Configure your training run like so:

```python
# main.py
from hparams import configurable, add_config, HParams, HParam
from typing import Union

@configurable
def train(batch_size=HParam(int)):
    pass

class Model():
    def __init__(hidden_size: Union[int, HParam] = HParam(),
                 dropout: Union[float, HParam] = HParam()):
          pass

add_config({ 'main': {
  'train': HParams(batch_size=32),
  'Model.__init__': HParams(hidden_size=1024, dropout=0.25),
}})
```

HParams supports optional configuration typechecking to help you find bugs. To ensure correctness,
this throws errors or warnings if a hyperparameter is missing a configuration. Finally, the
decorator adds little to no runtime overhead (< 1e-05 seconds) per function.

### Set Defaults

Configure PyTorch and Tensorflow defaults to match globally, like so:

```python
from torch.nn import BatchNorm1d
from hparams import configurable, add_config, HParams

# NOTE: `momentum=0.01` to match Tensorflow defaults
BatchNorm1d.__init__ = configurable(BatchNorm1d.__init__)
add_config({ 'torch.nn.BatchNorm1d.__init__': HParams(momentum=0.01) })
```

Configure printer formatting globally, like so:

```pycon
>>> import pprint
>>> pprint.pprint([[1, 2]])
[[1, 2]]
>>>
>>> # Configure `pprint` to use a `width` of `2`
>>> pprint.pprint = configurable(pprint.pprint)
>>> add_config({'pprint.pprint': HParams(width=2)})
>>>
>>> pprint.pprint([[1, 2]]) # `pprint` with `width` of `2`
[[1,
  2]]
```

### CLI

Enable rapid command line experimentation, for example:

```console
foo@bar:~$ file.py --torch.optim.adam.Adam.__init__ HParams(lr=0.1,betas=(0.999,0.99))
```

```python
import sys
from torch.optim import Adam
from hparams import configurable, add_config, parse_hparam_args

Adam.__init__ = configurable(Adam.__init__)
parsed = parse_hparam_args(sys.argv) # Parse command line arguments
add_config(parsed)
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

Export a Python `functools.partial` to use in another process like so:

```python
from hparams import configurable, HParam

@configurable
def func(hparam=HParam(int)):
    pass

partial = func.get_configured_partial()
```

With this approach, you don't have to transfer the global state to the new process. To transfer the
global state, you'll want to use `get_config` and `add_config`.

## Docs 📖

The complete documentation for HParams is available [here](./DOCS.md).

## Contributing

We've released HParams because a lack of hyperparameter management solutions. We hope that
other people can benefit from the project. We are thankful for any contributions from the
community.

### Contributing Guide

Read our [contributing guide](https://github.com/PetrochukM/HParams/blob/master/CONTRIBUTING.md) to
learn about our development process, how to propose bugfixes and improvements, and how to build and
test your changes to hparams.

## Authors

This library was initially developed by Michael Petrochuk's from his learnings as a deep learning engineer at Apple
and the Allen Institute for Artificial Intelligence. [Chloe Yeo](http://www.yeochloe.com/) did the logo design.

## Citing

If you find hparams useful for an academic publication, then please use the following BibTeX to
cite it:

```
@misc{hparams,
  author = {Petrochuk, Michael},
  title = {HParams: Hyperparameter management solution},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/PetrochukM/HParams}},
}
```
