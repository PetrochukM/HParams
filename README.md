# Config

This package allows you to configure functions directly.

**Features:**

- Approachable and easy-to-use API
- Battle-tested over many years and many internal projects
- Fast with little to no runtime overhead
- Lightweight with only two dependencies

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pythonic-config.svg?style=flat-square)
[![Codecov](https://img.shields.io/codecov/c/github/PetrochukM/HParams/master.svg?style=flat-square)](https://codecov.io/gh/PetrochukM/HParams)
[![Downloads](http://pepy.tech/badge/pythonic-config)](http://pepy.tech/project/pythonic-config)
[![Build Status](https://img.shields.io/travis/PetrochukM/HParams/master.svg?style=flat-square)](https://travis-ci.org/PetrochukM/HParams)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Twitter: PetrochukM](https://img.shields.io/twitter/follow/MPetrochuk.svg?style=social)](https://twitter.com/MPetrochuk)

## Installation

Make sure you have Python 3. You can then install `pythonic-config` using `pip`:

```bash
pip install pythonic-config
```

Install the latest code via:

```bash
pip install git+https://github.com/PetrochukM/Config.git
```

## Examples

Showing is better than telling.

### Basic

Any function can be configured, and then used anywhere.

```python
import config

# Define function
def do_something_cool(how_many_times: int):
    pass

# Configure function
config.add({do_something_cool: config.Args(how_many_times=5)})

# Use the configured function anywhere! 🎉
do_something_cool(how_many_times=config.get())
```

### Command Line

The configuration can be changed via the command line.

```console
foo@bar:~$ python example.py --sorted 'Args(reverse=True)'
```

```python
import sys
import config

config.add(config.parse_cli_args(sys.argv[1:]))
```

### Multiprocessing

The configuration can be exported to another process.

```python
from multiprocessing import Process
import config

def handler(configs: config.Config):
    config.add(configs)

if __name__ == "__main__":
    process = Process(target=handler, args=(config.export(),))
    process.start()
    process.join()
```

### Logging

The configuration can be logged, easily, to tools like [Comet](https://www.comet.ml/).

```python
from comet_ml import Experiment
import config

experiment = Experiment()
experiment.log_parameters(config.log())
```
