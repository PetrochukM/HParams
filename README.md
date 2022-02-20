# Config

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pythonic-config.svg?style=flat-square)
[![Codecov](https://img.shields.io/codecov/c/github/PetrochukM/HParams/master.svg?style=flat-square)](https://codecov.io/gh/PetrochukM/HParams)
[![Downloads](http://pepy.tech/badge/pythonic-config)](http://pepy.tech/project/pythonic-config)
[![Build Status](https://img.shields.io/travis/PetrochukM/HParams/master.svg?style=flat-square)](https://travis-ci.com/PetrochukM/HParams)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Twitter: PetrochukM](https://img.shields.io/twitter/follow/MPetrochuk.svg?style=social)](https://twitter.com/MPetrochuk)

This package allows you to configure functions directly. You will be able to create an intuitive
configuration file that explicitly sets function arguments, globally.

#### Features

- **Battle-tested over many years** - At WellSaid Labs, we've used this module for years to
  configure our state-of-the-art text-to-speech machine learning scripts.
- **Approachable and easy-to-use API** - This package has a small API with a
  handful of primitives.
- **Fast with little to no runtime overhead** - In addition to being fast, it incorporates caching
  throughout.
- **Lightweight with only two dependencies** - They themselves are lightweight and widely used for
  type checking and introspection.
- **Runtime type checking** - The configuration is type-checked at runtime for correctness.

#### Contents

- [Install](#install)
- [Usage](#usage)
  - [Configuring a function](#configuring-a-function)
  - [Writing a configuration file](#writing-a-configuration-file)
  - [Configuring via the command line](#configuring-via-the-command-line)
  - [Logging the configuration](#logging-the-configuration)
  - [Advanced: Sharing configurations between processes](#advanced-sharing-configurations-between-processes)
- [How does this work?](#how-does-this-work)

## Install

Make sure you have Python 3. You can then install `pythonic-config` using `pip`:

```bash
pip install pythonic-config
```

Install the latest code via:

```bash
pip install git+https://github.com/PetrochukM/Config.git
```

## Usage

### Configuring a function

Any function can be configured, and then used anywhere, see below -

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

This approach is simple but powerful. Now, each configuration can be directly attributed to a
documented function argument.

### Writing a configuration file

The simple example above can be extended to create a configuration file. For example below, it shows
how you might go about configuring a machine learning training run.

With this approach, this configuration file will make it clear which hyperparameters are set and
where. This improves overall readability of the configuration file.

```python
import config
import data
import train

config.add({
  data.get_data: config.Args(
      train_data_path="url_lists/all_train.txt",
      val_data_path="url_lists/all_val.txt"
  ),
  data.source_tokenizer: config.Args(type_="pretrained_transformer", model_name="batch_large"),
  data.dataset_reader: config.Args(
      type_="cnn_dm",
      source_max_tokens=1022,
      target_max_tokens=54,
  ),
  train.make_model: config.Args(type_="bart"),
  train.Trainer.make_optimizer: config.Args(
      type_="huggingface_adamw",
      lr=3e-5,
      betas=[0.9, 0.999],
      eps=1e-8,
      correct_bias=true
  )
  train.Trainer.__init__: config.Args(
      num_epochs=3,
      learning_rate_scheduler="polynomial_decay",
      grad_norm=1.0,
  )
})
```

Also, since the configuration file is written in Python, you can use variables, lambdas, etc to
further modularize the configuration file.

### Configuring via the command line

Additionally, this package supports configuration from the command line, see below -

```console
foo@bar:~$ python example.py --sorted 'Args(reverse=True)'
```

```python
import sys
import config

config.add(config.parse_cli_args(sys.argv[1:]))
```

This is particularly useful if you want to change one configuration at a time.

### Logging the configuration

Lastly, it's useful to track the configuration file by logging it. This package supports that
via `config.log`. In the example below, we log the configuration to
[Comet](https://www.comet.ml/).

```python
from comet_ml import Experiment
import config

experiment = Experiment()
experiment.log_parameters(config.log())
```

### Advanced: Sharing configurations between processes

For software that takes advantage of multiple processes, it may be useful to share the configuration
file between them. In this case, the configuration can be exported to another process and then
subsequently imported, see below -

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

## How does this work?

Our approach is simple, `config` maintains a global configuration mapping each function to its
associated arguments. When a user calls `config.get`, it'll attempt to parse the code to determine
the calling function and associated argument. In most cases, this will be successful and it will
fetch the appropriate value. If it's not successful, it'll `raise` an error that'll help you fix
the issue.
