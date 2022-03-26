# Config

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pythonic-config.svg?style=flat-square)
[![Codecov](https://img.shields.io/codecov/c/github/PetrochukM/HParams/master.svg?style=flat-square)](https://codecov.io/gh/PetrochukM/HParams)
[![Downloads](http://pepy.tech/badge/pythonic-config)](http://pepy.tech/project/pythonic-config)
[![Build Status](https://img.shields.io/travis/PetrochukM/HParams/master.svg?style=flat-square)](https://travis-ci.com/PetrochukM/HParams)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Twitter: PetrochukM](https://img.shields.io/twitter/follow/MPetrochuk.svg?style=social)](https://twitter.com/MPetrochuk)

This package allows you to configure functions explicitly and safely. You will be able to create an
intuitive type-checked configuration file that directly sets function arguments, globally.

This is a lightweight package with only two widely used dependencies and only a couple hundred line
of code.

#### Contents

- [Config](#config)
      - [Contents](#contents)
  - [Install](#install)
  - [Usage 🤗](#usage-)
    - [Configuring a function](#configuring-a-function)
    - [Writing a configuration file](#writing-a-configuration-file)
    - [Configuring via the command line](#configuring-via-the-command-line)
    - [Logging the configuration](#logging-the-configuration)
    - [Advanced: Sharing configurations between processes](#advanced-sharing-configurations-between-processes)
    - [Advanced: Ensuring the configuration is used](#advanced-ensuring-the-configuration-is-used)
    - [Advanced: Find unused configurations](#advanced-find-unused-configurations)

## Install

Make sure you have Python 3, then you can install `pythonic-config` using `pip`:

```bash
pip install pythonic-config
```

Install the latest code via:

```bash
pip install git+https://github.com/PetrochukM/Config.git
```

## Usage 🤗

### Configuring a function

Any function can be configured, and then used anywhere, see below:

```python
import config as cf

# Define function
def do_something_cool(how_many_times: int):
    pass

# Configure function
cf.add({do_something_cool: cf.Args(how_many_times=5)})

# Use the configured function anywhere! 🎉
do_something_cool(how_many_times=cf.get())
```

This approach is simple but powerful. Now, each configuration can be directly attributed to a
documented function argument.

Furthermore, `config` incorporates `typeguard` 💂‍♀️ so every configuration is type checked at runtime.

### Writing a configuration file

The simple example above can be extended to create a configuration file, for example:

```python
import config as cf
import data
import train

cf.add({
  data.get_data: cf.Args(
      train_data_path="url_lists/all_train.txt",
      val_data_path="url_lists/all_val.txt"
  ),
  data.dataset_reader: cf.Args(
      type_="cnn_dm",
      source_max_tokens=1022,
      target_max_tokens=54,
  ),
  train.make_model: cf.Args(type_="bart"),
  train.Trainer.make_optimizer: cf.Args(
      type_="huggingface_adamw",
      lr=3e-5,
      correct_bias=True
  )
  train.Trainer.__init__: cf.Args(
      num_epochs=3,
      learning_rate_scheduler="polynomial_decay",
      grad_norm=1.0,
  )
})
```

With this approach, this configuration file will make it clear which (hyper)parameters are set and
where. This improves overall readability of the configuration file.

🐍 Last but not least, the configuration file is written in Python, you can use variables, lambdas,
etc to further modularize.

### Configuring via the command line

In case you want to change one variable at a time, this package supports configuration from the
command line, for example:

```console
python example.py --sorted='Args(reverse=True)'
```

```python
import sys
import config as cf

cf.add(cf.parse_cli_args(sys.argv[1:]))
```

### Logging the configuration

Lastly, it's useful to track the configuration file by logging it. This package supports that
via `config.log`. In the example below, we log the configuration to
[Comet](https://www.comet.ml/).

```python
from comet_ml import Experiment
import config as cf

experiment = Experiment()
experiment.log_parameters(cf.log())
```

### Advanced: Sharing configurations between processes

In multiprocessing, it may be useful to share the configuration file between processes. In this
case, the configuration can be exported to another process and then subsequently imported, see
below:

```python
from multiprocessing import Process
import config as cf

def handler(configs: cf.Config):
    cf.add(configs)

if __name__ == "__main__":
    process = Process(target=handler, args=(cf.export(),))
    process.start()
    process.join()
```

### Advanced: Ensuring the configuration is used

In a large code base, it might be hard to tell if the configuration has been set for every function
call. In this case, we've exposed `config.trace` which can double check every function call
against the configuration, see below:

```python
import sys
import config as cf

def configured(a=111):
    pass

sys.settrace(cf.trace)
cf.add({configured: cf.Args(a=1)})

configured()  # `cf.trace` issues a WARNING!
configured(a=cf.get())
```

We also have another option for faster tracing with `config.enable_fast_trace`. Instead of a system
wide trace, this traces the configured functions by modifying their code and inserting a trace
function at the beginning of the function definition. This has a MUCH lower overhead; however, it is
still in beta due to the number of edge cases.

### Advanced: Find unused configurations

In a large code base, you may have a lot of configurations, some of which are no longer being used.
`purge` can be run on a process exit, and it'll warn you if configurations were not used.

```python
import atexit
import config as cf

atexit.register(cf.purge)
```
