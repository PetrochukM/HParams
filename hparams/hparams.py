from functools import lru_cache
from functools import partial
from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import Any
from typing import get_type_hints

import inspect
import logging
import pprint
import sys

from typeguard import check_type

logger = logging.getLogger(__name__)
pretty_printer = pprint.PrettyPrinter()


class HParams(dict):
    pass


class HParam():
    """ Place-holder object to indicate that a parameter is to be configured. This also,
    ensures that this parameter does have an associated configuration.

    TODO: Given this object is used as a default argument, on its instantiation check if the module
    is `configurable`. We've found that inspecting the instantiation of a default argument does
    not give much information about the module.

    TODO: If we locate the module this was instantiated in, then we can return the correct value
    and avoid decorators all together. This will require some work with AST; unfortunately.

    Args:
        type_ (typing, optional): The HParam type.
    """

    def __init__(self, type_=Any):
        lineno = inspect.stack()[1].lineno  # Ge the caller line number
        filename = inspect.stack()[1].filename
        self.type = type_
        self.error_message = 'The parameter set to `HParam` at %s:%s must be configured.' % (
            filename, lineno)
        # Learn more about special methods:
        # https://stackoverflow.com/questions/21887091/cant-dynamically-bind-repr-str-to-a-class-created-with-type
        # https://stackoverflow.com/questions/1418825/where-is-the-python-documentation-for-the-special-methods-init-new
        for attribute in [
                '__str__', '__repr__', '__contains__', '__hash__', '__len__', '__call__', '__add__',
                '__sub__', '__mul__', '__floordiv__', '__div__', '__mod__', '__pow__', '__lshift__',
                '__rshift__', '__and__', '__xor__', '__or__', '__iadd__', '__isub__', '__imul__',
                '__idiv__', '__ifloordiv__', '__imod__', '__ipow__', '__ilshift__', '__irshift__',
                '__iand__', '__ixor__', '__ior__', '__neg__', '__pos__', '__abs__', '__invert__',
                '__complex__', '__int__', '__long__', '__float__', '__oct__', '__hex__', '__lt__',
                '__le__', '__eq__', '__ne__', '__ge__', '__gt__', '__cmp__', '__round__',
                '__getitem__', '__setitem__', '__delitem__', '__iter__', '__reversed__', '__copy__',
                '__deepcopy__'
        ]:
            setattr(self.__class__, attribute, self._raise)

    def _raise(self, *args, **kwargs):
        raise ValueError(self.error_message)

    def __getattribute__(self, name):
        if name in ['error_message', '_raise', '__dict__', '__class__', 'type']:
            return super().__getattribute__(name)
        self._raise()


@lru_cache()
def _get_function_print_name(func):
    """ Get a name for each function.
    """
    return inspect.getmodule(func).__name__.split('.')[-1] + '.' + func.__qualname__


@lru_cache()
def _get_function_signature(func):
    """ Get a unique signature for each function.
    """
    try:
        absolute_filename = Path(inspect.getfile(func))
        # NOTE: `relative_filename` is the longest filename relative to `sys.path` paths but
        # shorter than a absolute filename.
        relative_filename = None
        for path in sys.path:
            try:
                new_filename = str(absolute_filename.relative_to(Path(path)))
                if relative_filename is None:
                    relative_filename = new_filename
                elif len(new_filename) > len(relative_filename):
                    relative_filename = new_filename
            except ValueError:
                pass
        return relative_filename.replace('/', '.')[:-3] + '.' + func.__qualname__
    except TypeError:
        return '#' + func.__qualname__


@lru_cache()
def _get_function_path(func):
    """ Get an function path that can be resolved by `_resolve_configuration_helper`.
    """
    if hasattr(func, '__qualname__'):
        return inspect.getmodule(func).__name__ + '.' + func.__qualname__
    else:
        return inspect.getmodule(func).__name__


@lru_cache()
def _get_function_parameters(func):
    return inspect.signature(func).parameters


@lru_cache()
def _get_function_hparams(func):
    """ Get all keyword parameters set to `HParam` in func.
    """
    return {
        k: v.default
        for k, v in _get_function_parameters(func).items()
        if v.default is not inspect.Parameter.empty and isinstance(v.default, HParam)
    }


def _function_has_keyword_parameters(func, kwargs):
    """ Raise `TypeError` if `func` does not accept all the keyword arguments in `kwargs`.

    Args:
        func (callable)
        kwargs (dict): Some keyword arguments.
    """
    parameters = _get_function_parameters(func)
    has_var_keyword = any([
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in list(parameters.values())
    ])
    type_hints = get_type_hints(func)

    for kwarg in kwargs.keys():
        if not has_var_keyword and (kwarg not in parameters or
                                    parameters[kwarg].kind == inspect.Parameter.VAR_POSITIONAL):
            raise TypeError('Function `%s` does not accept configured parameter `%s`.' %
                            (_get_function_print_name(func), kwarg))

        try:
            if (kwarg in parameters and parameters[kwarg].default is not inspect.Parameter.empty and
                    isinstance(parameters[kwarg].default, HParam)):
                check_type(kwarg, kwargs[kwarg], parameters[kwarg].default.type)
        except TypeError:
            raise TypeError('Function `%s` requires parameter `%s` to be of type `%s`.' %
                            (_get_function_print_name(func), kwarg, parameters[kwarg].default.type))

        try:
            if kwarg in type_hints:
                check_type(kwarg, kwargs[kwarg], type_hints[kwarg])
        except TypeError:
            raise TypeError('Function `%s` requires parameter `%s` to be of type `%s`.' %
                            (_get_function_print_name(func), kwarg, type_hints[kwarg]))


def _resolve_configuration_helper(dict_, keys):
    """ Recursive helper of `_resolve_configuration`.

    TODO: Test resolution of a module in the `__mp_main__` namespace.

    Args:
        dict_ (dict): Parsed dict from `_parse_configuration`.
        keys (list): List of past nested dictionary keys.

    Raises:
        (TypeError): If any path in `dict_` does not end at some configurable decorated function.
        (TypeError): If any path in `dict_` does not end in an `HParams` object.
        (TypeError): If any path in `dict_` cannot be imported.
        (TypeError): If any path in `dict_` refers to the same function.
        (TypeError or ValueError): If any path in `dict_` has an `HParams` object that does not
            match the function signature.

    Returns:
        dict: Each key is a function signature and each value is an `HParams` object.
    """
    if not isinstance(dict_, HParams) and isinstance(dict_, dict):
        return_ = {}
        if len(dict_) == 0:
            raise TypeError('Failed to find `HParams` object along path `%s`.' % '.'.join(keys))
        for key in dict_:
            resolved = _resolve_configuration_helper(dict_[key], keys[:] + [key])
            if len(set(resolved.keys()) & set(return_.keys())) > 0:
                raise TypeError('Function `%s` was defined twice in configuration.' %
                                '.'.join(keys + [key]))
            return_.update(resolved)
        return return_
    elif not isinstance(dict_, HParams) and not isinstance(dict_, dict):
        raise TypeError('Failed to find `HParams` object along path `%s`.' % '.'.join(keys))

    trace = []
    for i in reversed(range(1, len(keys))):
        try:
            module_path = '.'.join(keys[:i])
            attribute = import_module(module_path)
            for j, key in enumerate(keys[i:]):
                # NOTE: `__qualname__` uses `<locals>` and `<lambdas>` for function naming.
                # Learn more: https://www.python.org/dev/peps/pep-3155/
                if key[0] == '<' and key[-1] == '>':
                    logger.warning('Skipping checks for `%s`, this cannot import `%s`.',
                                   '.'.join(keys), key)
                    signature = (_get_function_signature(attribute) + '.' + '.'.join(keys[i:][j:]))
                    return {signature: dict_}
                else:
                    attribute = getattr(attribute, key)
            if hasattr(attribute, '_configurable'):
                # Check all keyword arguments (`dict_`) are defined in function.
                _function_has_keyword_parameters(attribute.__wrapped__, dict_)

                # Check `HParam` arguments are set.
                for name, hparam in _get_function_hparams(attribute.__wrapped__).items():
                    if name not in dict_:
                        hparam._raise()

                return {_get_function_signature(attribute.__wrapped__): dict_}
            else:
                trace.append('`%s` is not decorated with `configurable`.' % '.'.join(keys))
        except ImportError:
            trace.append('ImportError: Failed to run `import %s`.' % module_path)
        except AttributeError:
            trace.append('AttributeError: `%s` not found in `%s`.' % (key, '.'.join(keys[:i + j])))

    trace.reverse()

    raise TypeError('Failed to find `configurable` decorator along path `%s`.\n' % '.'.join(keys) +
                    'Attempts (most recent attempt last):\n\t%s' % ('\n\t'.join(trace),))


def _resolve_configuration(dict_):
    """ Resolve any relative function paths and check the validity of `dict_`.

    Args:
        dict_ (dict): Parsed dict to resolve and validate.

    Raises:
        (TypeError): If any path in `dict_` does not end at some configurable decorated function.
        (TypeError): If any path in `dict_` does not end in an `HParams` object.
        (TypeError): If any path in `dict_` cannot be imported.
        (TypeError): If any path in `dict_` refers to the same function.
        (TypeError or ValueError): If any path in `dict_` has an `HParams` object that does not
            match the function signature.

    Returns:
        dict: Each key is a function signature and each value is an `HParams` object.
    """
    return _resolve_configuration_helper(dict_, [])


def _parse_configuration_helper(dict_, parsed_dict):
    """ Recursive helper to `_parse_configuration`.

    Args:
        dict_ (dict): Dotted dictionary to parse.
        parsed_dict (dict): Parsed dictionary that is created.

    Raises:
        (TypeError): If any key is not a string, module, or callable.
        (TypeError): If any string key is not formatted like a python dotted module name.
        (TypeError): If any key is duplicated.

    Returns:
        (dict): Parsed dictionary.
    """
    if not isinstance(dict_, dict) or isinstance(dict_, HParams):
        return

    for key in dict_:
        if not (inspect.ismodule(key) or isinstance(key, str) or callable(key)):
            raise TypeError('Key `%s` must be a string, module, or callable.' % key)
        split = (key if isinstance(key, str) else _get_function_path(key)).split('.')
        past_parsed_dict = parsed_dict
        for i, split_key in enumerate(split):
            if split_key == '':
                raise TypeError('Improper key format `%s`.' % key)
            if i == len(split) - 1 and (not isinstance(dict_[key], dict) or
                                        isinstance(dict_[key], HParams)):
                if split_key in parsed_dict:
                    raise TypeError('This key `%s` is a duplicate.' % key)
                parsed_dict[split_key] = dict_[key]
            else:
                if split_key not in parsed_dict:
                    parsed_dict[split_key] = {}
                parsed_dict = parsed_dict[split_key]
        _parse_configuration_helper(dict_[key], parsed_dict)
        parsed_dict = past_parsed_dict

    return parsed_dict


def _parse_configuration(dict_):
    """ Parses `dict_` such that dotted key names are interpreted as multiple keys.

    This configuration parser is intended to replicate python's dotted module names.

    Raises:
        (TypeError): If any key is not a string, module, or callable.
        (TypeError): If any string key is not formatted like a python dotted module name.
        (TypeError): If any key is duplicated.

    Args:
        dict_ (dict): Dotted dictionary to parse

    Returns:
        (dict): Parsed dictionary.

    Example:
        >>> dict = {
        ...   'abc.abc': {
        ...     'cda': 'abc'
        ...   }
        ... }
        >>> _parse_configuration(dict)
        {'abc': {'abc': {'cda': 'abc'}}}
    """
    return _parse_configuration_helper(dict_, {})


_configuration = {}


def add_config(config):
    """ Add configuration to the global configuration.

    Args:
        config (dict): Configuration to add.

    Returns: None

    Side Effects:
        The existing global configuration is merged with the new configuration.

    Raises:
        (TypeError): If any path in `config` does not end at some configurable decorated function.
        (TypeError): If any path in `config` does not end in an `HParams` object.
        (TypeError): If any path in `config` cannot be imported.
        (TypeError): If any path in `config` refers to the same function.
        (TypeError or ValueError): If any path in `config` has an `HParams` object that does not
            match the function signature.
        (TypeError): If any key is not a string, module, or callable.
        (TypeError): If any string key is not formatted like a python dotted module name.

    Example:
        >>> import pprint
        >>> pprint.pprint([[1, 2]])
        [[1, 2]]
        >>>
        >>> # Configure `pprint` to use a `width` of `2`
        >>> pprint.pprint = configurable(pprint.pprint)
        >>> add_config({'pprint.pprint': HParams(width=2)})
        >>> pprint.pprint([[1, 2]])
        [[1,
          2]]
    """
    global _configuration
    parsed = _parse_configuration(config)
    resolved = _resolve_configuration(parsed)
    for key in resolved:
        if key in _configuration:
            _configuration[key].update(resolved[key])
        else:
            _configuration[key] = resolved[key]


def log_config():
    """ Log the current global configuration. """
    logger.info('Global configuration:\n%s', pretty_printer.pformat(_configuration))


def get_config():
    """ Get the current global configuration.

    Anti-Patterns:
        It would be an anti-pattern to use this to set the configuration.

    Returns:
        (dict): The current configuration.
    """
    return _configuration


def clear_config():
    """ Clear the global configuration.

    Side Effects:
        The existing global configuration is reset to it's initial state.
    """
    global _configuration
    _configuration = {}


def _merge_args(parameters, args, kwargs, default_kwargs, print_name, is_first_run):
    """ Merge `func` `args` and `kwargs` with `default_kwargs`.

    The `_merge_args` prefers `kwargs` and `args` over `default_kwargs`.

    Args:
        parameters (list of inspect.Parameter): module that accepts `args` and `kwargs`
        args (list of any): Arguments accepted by `func`.
        kwargs (dict of any): Keyword arguments accepted by `func`.
        default_kwargs (dict of any): Default keyword arguments accepted by `func` to merge.
        print_name (str): Function name to print with warnings.
        is_first_run (bool): If `True` print warnings.

    Returns:
        (dict): kwargs merging `args`, `kwargs`, and `default_kwargs`
    """
    merged_kwargs = default_kwargs.copy()

    # Delete `merged_kwargs` that conflict with `args`.
    # NOTE: Positional arguments must come before keyword arguments.
    for i, arg in enumerate(args):
        if i >= len(parameters):
            raise TypeError('Too many arguments (%d > %d) passed.' % (len(args), len(parameters)))

        if parameters[i].kind == inspect.Parameter.VAR_POSITIONAL:
            break  # NOTE: Rest of the args are absorbed by VAR_POSITIONAL (e.g. `*args`)

        if (parameters[i].kind == inspect.Parameter.POSITIONAL_ONLY or
                parameters[i].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
            if parameters[i].name in merged_kwargs:
                value = merged_kwargs[parameters[i].name]
                if is_first_run and value != arg:
                    logger.warning(
                        '@configurable: Overwriting configured argument `%s=%s` in module `%s` '
                        'with `%s`. This warning will not be repeated in this process.',
                        parameters[i].name, value, print_name, arg)
                del merged_kwargs[parameters[i].name]

    if is_first_run:
        for key, value in kwargs.items():
            if key in merged_kwargs and value != merged_kwargs[key]:
                logger.warning(
                    '@configurable: Overwriting configured argument `%s=%s` in module `%s` '
                    'with `%s`. This warning will not be repeated in this process.', key,
                    merged_kwargs[key], print_name, value)

    merged_kwargs.update(kwargs)
    return args, merged_kwargs


def configurable(function=None):
    """ Decorator enables configuring module arguments.

    Decorator enables one to set the arguments of a module via a global configuration. The decorator
    also stores the parameters the decorated function was called with.

    Args:
        None

    Returns:
        (callable): Decorated function
    """
    if not function:
        return configurable

    function_signature = _get_function_signature(function)
    function_print_name = _get_function_print_name(function)
    function_parameters = list(_get_function_parameters(function).values())
    function_hparams = _get_function_hparams(function).items()
    is_first_run = True

    def _get_configuration():
        return _configuration[function_signature] if function_signature in _configuration else {}

    @wraps(function)
    def decorator(*args, **kwargs):
        global _configuration
        nonlocal is_first_run

        # Get the function configuration
        config = _get_configuration()
        if is_first_run and len(config) == 0:
            logger.warning(
                '@configurable: No config for `%s`. '
                'This warning will not be repeated in this process.', function_print_name)

        # Ensure all `HParam` objects are overridden.
        [h._raise() for n, h in function_hparams if n not in config]

        # NOTE: Skip type checking via `_function_has_keyword_parameters` for performance.

        args, kwargs = _merge_args(function_parameters, args, kwargs, config, function_print_name,
                                   is_first_run)

        if is_first_run:
            is_first_run = False

        return function(*args, **kwargs)

    # USE CASE: `get_configured_partial` can be used to export a function with it's configuration
    # for multiprocessing.
    def get_configured_partial():
        return partial(decorator, **_get_configuration())

    decorator.get_configured_partial = get_configured_partial

    # Add a flag to the func; enabling us to check if a function has the configurable decorator.
    decorator._configurable = True

    return decorator


def parse_hparam_args(args):
    """ Parse CLI arguments like `['--torch.optim.adam.Adam.__init__', 'HParams(lr=0.1)']` to
      :class:`dict`.

    Args:
        args (list of str): List of CLI arguments

    Returns
        (dict): Dictionary of arguments.
    """
    return_ = {}

    while len(args) > 0:
        arg = args.pop(0)

        error = ValueError('The command line argument `%s` is ambiguous. '
                           'The format must be either `--key=value` or `--key value`.')

        try:
            if '--' == arg[:2] and '=' not in arg:
                key = arg
                value = args.pop(0)
            elif '--' == arg[:2] and '=' in arg:
                key, value = tuple(arg.split('=', maxsplit=1))
            else:
                raise error
        except IndexError:
            raise error

        key = key[2:]  # Remove double flags
        return_[key] = eval(value)

        if not isinstance(return_[key], HParams):
            raise ValueError('The command line argument value must be an `HParams` object like so '
                             '`--torch.optim.adam.Adam.__init__=HParams(lr=0.1)`.')

    logger.info('These command line arguments were parsed into:\n%s', return_)

    return return_
