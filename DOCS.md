# Welcome to HParams’s documentation!

HParams is a configuration management solution for machine learning projects. With this you can
externalize your hyperparameters ensuring that they are extensible, accessible, and maintainable.
It’s open-source software, released under the MIT license.

## Package

### configurable

A light-weight decorator injects configured keyword arguments at runtime. It runs with little to no
runtime overhead (< 1e-05 seconds) per function.

Warnings:

  - Runs `logging.warning` if the function has not been configured.
  - Runs `logging.warning` if the function configuration will be overridden by arguments passed to the
    function. 

Raises:

  - `TypeError` if a keyword argument defaulted to `HParam` for the decorated function is not
    configured. Typechecking is only run during `add_config` for performance.

Side-effects:

  - Adds `get_configured_partial` to the function's attributes. `get_configured_partial` returns a partial
    that's configured.

Returns:

  - (callable): The decorated function.

Example:

```python
from hparams import configurable, add_config, HParams, HParam

@configurable
def train(batch_size=HParam(int)):
    pass

add_config({ train: HParams(batch_size=32) })
```

### add_config

Add a configuration to the global configuration.

Args:

  - config (dict): A nested dictionary such that each key is a `str`, module, or `callable` and
      each value is either a `dict` or an `HParams` object.

Side-effects:

  - The existing global configuration is merged with the new configuration.

Raises:

  - `TypeError` if any path in `config` does not refer to a configurable function.
  - `TypeError` if any path in `config` does not end with an `HParams` object.
  - `TypeError` if a referenced function in `config` cannot be imported.
  - `TypeError` if a referenced function in `config` has duplicate references.
  - `TypeError` or `ValueError` if a referenced function's signature in `config` does not align
      with the related `HParams` object.
  - `TypeError` if any key is not a `str`, module, or `callable`.
  - `TypeError` if any `str` key does not follow Python's dotted module naming conventions.

Example:

```python
# main.py
from hparams import configurable, add_config, HParams, HParam

@configurable
def train(batch_size=HParam(int)):
    pass

class Model():

    @configurable
    def __init__(self, hidden_size=HParam(int)):
        pass

class Optimizer():

    @configurable
    def __init__(self, learning_rate=HParam(float)):
        pass

add_config({
  train: HParams(batch_size=32),
  'main': {
    'Model.__init__': HParam(hidden_size=32),
    'Optimizer.__init__': HParam(learning_rate=32)
  },
})
```

### get_config

Get the current global configuration.

Anti-patterns:

  - Using the return value of this function to set the configuration with `add_config`. 
    That will introduce unnecessary coupling and complexity that this module is designed
    to avoid.

Returns:

  - (dict): Return the current configuration as a dictionary.

### clear_config

Clear the current global configuration.

Side-effects:

  - The global configuration is reset to it's initial state.

### log_config

Log the current global configuration with `pprint` and `logging`.

### parse_hparam_args

Parses CLI arguments like `['--torch.optim.adam.Adam.__init__', 'HParams(lr=0.1)']` to a `dict`
that is compatible with `add_config`.

Args:

  - args (`list`): List of `str` to parse.

Returns:

  - (dict): A dictionary that is compatible with `add_config`.

Example:

```python
import sys
from torch.optim import Adam
from hparams import configurable, add_config, parse_hparam_args

Adam.__init__ = configurable(Adam.__init__)
parsed = parse_hparam_args(sys.argv) # Parse command line arguments
add_config(parsed)
```

### HParams

This is a subclass of `dict`. It is defined simply, like so:

```python
class HParams(dict):
    pass
```

This class is required by `add_config` to parse the configuration.

### HParam

This is a place-holder object indicating that a parameter should be configured.

Args:

  - type_ (typing, optional): The expected type of the hyperparameter.

Raises:

  - `ValueError` if this object is used to execute anything. This object should be overridden by
    a hyperparameter during runtime.
 
