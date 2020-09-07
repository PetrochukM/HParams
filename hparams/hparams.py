from functools import lru_cache
from functools import partial
from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import cast
from typing import get_type_hints

import inspect
import itertools
import logging
import pprint
import sys
import traceback
import typing
import warnings

from typeguard import check_type

logger = logging.getLogger(__name__)
pretty_printer = pprint.PrettyPrinter()


class HParams(dict):
    pass


_HParamReturnType = typing.TypeVar('_HParamReturnType')


class _HParam():
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

    def __init__(self, type_=typing.Any):
        stack = traceback.extract_stack(limit=2)[-2]  # Get the caller line number
        self.type = type_
        self.error_message = 'The parameter set to `HParam` at %s:%s must be configured.' % (
            stack.filename, stack.lineno)
        # Learn more about special methods:
        # https://stackoverflow.com/questions/21887091/cant-dynamically-bind-repr-str-to-a-class-created-with-type
        # https://stackoverflow.com/questions/1418825/where-is-the-python-documentation-for-the-special-methods-init-new
        for attribute in [
                '__contains__', '__hash__', '__len__', '__call__', '__add__', '__sub__', '__mul__',
                '__floordiv__', '__div__', '__mod__', '__pow__', '__lshift__', '__rshift__',
                '__and__', '__xor__', '__or__', '__iadd__', '__isub__', '__imul__', '__idiv__',
                '__ifloordiv__', '__imod__', '__ipow__', '__ilshift__', '__irshift__', '__iand__',
                '__ixor__', '__ior__', '__neg__', '__pos__', '__abs__', '__invert__', '__complex__',
                '__int__', '__long__', '__float__', '__oct__', '__hex__', '__lt__', '__le__',
                '__eq__', '__ne__', '__ge__', '__gt__', '__cmp__', '__round__', '__getitem__',
                '__setitem__', '__delitem__', '__iter__', '__reversed__', '__copy__', '__deepcopy__'
        ]:
            setattr(self.__class__, attribute, self._raise)

    def _raise(self, *args, **kwargs):
        raise ValueError(self.error_message)

    def __getattribute__(self, name):
        if name in ['error_message', '_raise', '__dict__', '__class__', 'type']:
            return super().__getattribute__(name)
        self._raise()


def HParam(type_=typing.Any) -> _HParamReturnType:
    return cast(_HParamReturnType, _HParam(type_=type_))


@lru_cache()
def _get_function_signature(func):
    """ Get a unique signature for each function.
    """
    try:
        # NOTE: Unwrap function decorators because they add indirection to the actual function
        # filename.
        while hasattr(func, '__wrapped__'):
            func = func.__wrapped__

        absolute_filename = Path(inspect.getfile(func))
        # NOTE: `relative_filename` is the longest filename relative to `sys.path` paths but
        # shorter than a absolute filename.
        relative_filename = None
        for path in sys.path:
            try:
                new_filename = str(absolute_filename.relative_to(Path(path).absolute()))
                if relative_filename is None:
                    relative_filename = new_filename
                elif len(new_filename) > len(relative_filename):
                    relative_filename = new_filename
            except ValueError:
                pass
        filename = str(relative_filename if relative_filename is not None else absolute_filename)
        return filename.replace('/', '.')[:-3] + '.' + func.__qualname__
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
def _get_function_default_kwargs(func):
    """ Get all keyword parameters in func.
    """
    return {
        k: v.default
        for k, v in _get_function_parameters(func).items()
        if v.default is not inspect.Parameter.empty
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
                            (_get_function_signature(func), kwarg))

        try:
            if (kwarg in parameters and parameters[kwarg].default is not inspect.Parameter.empty and
                    isinstance(parameters[kwarg].default, _HParam)):
                check_type(kwarg, kwargs[kwarg], parameters[kwarg].default.type)
        except TypeError:
            raise TypeError('Function `%s` requires parameter `%s` to be of type `%s`.' %
                            (_get_function_signature(func), kwarg, parameters[kwarg].default.type))

        try:
            if kwarg in type_hints:
                check_type(kwarg, kwargs[kwarg], type_hints[kwarg])
        except TypeError:
            raise TypeError('Function `%s` requires parameter `%s` to be of type `%s`.' %
                            (_get_function_signature(func), kwarg, type_hints[kwarg]))


_skip_resolution = {}


def _get_skip_resolution():
    """ Helper function for testing `_skip_resolution`. """
    return _skip_resolution


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
    for i in range(1, len(keys)):
        try:
            module_path = '.'.join(keys[:i])

            if _is_lazy_resolution:
                # NOTE: If the module is not already loaded, then skip this resolution for now.
                attribute = sys.modules.get(module_path, None)
                if attribute is None:
                    _skip_resolution[tuple(keys[:])] = dict_
                    return {}
            else:
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
                    # NOTE: This will fail if `key` is a module that has not yet been imported
                    # and `attribute` is a package.
                    attribute = getattr(attribute, key)
            if hasattr(attribute, '_configurable'):
                # Check all keyword arguments (`dict_`) are defined in function.
                _function_has_keyword_parameters(attribute.__wrapped__, dict_)

                # NOTE: Do not check if all `HParam` arguments are set to allow to allow for
                # `add_config` to be called multiples with partial configurations.
                return {_get_function_signature(attribute.__wrapped__): dict_}
            else:
                trace.append('`%s` is not decorated with `configurable`.' % '.'.join(keys))
        # TODO: Instead of the generic `ImportError` consider a the more specific
        # `ModuleNotFoundError` introduced in Python 3.6
        except ImportError:
            trace.append('ImportError: Failed to run `import %s`.' % module_path)
        except AttributeError:
            trace.append('AttributeError: `%s` not found in `%s`.' % (key, '.'.join(keys[:i + j])))

    trace.reverse()

    warnings.warn('Skipping configuration for `%s` because this ' % '.'.join(keys) +
                  'failed to find a `configurable` decorator for that configuration.\n' +
                  'Attempts (most recent attempt last):\n  %s' % ('\n  '.join(trace),))

    return {}


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


def _resolve_skipped():
    """ Invoke resolution for skipped configuration paths.
    """
    # NOTE: `_resolve_configuration_helper` adds more items to `_skip_resolution`, this ensures
    # that those items are ignored.
    global _skip_resolution
    copy_skip_resolution = _skip_resolution.copy()
    _skip_resolution = {}

    for keys, dict_ in copy_skip_resolution.items():
        resolved = _resolve_configuration_helper(dict_, list(keys))
        _add_resolved_config(resolved)


def _add_resolved_config(resolved):
    """ Add to `configuration` a resolved configuration from `_resolve_configuration`.

    Args:
        resolved (dict): Each key is a function signature and each value is an `HParams` object.
    """
    for key in resolved:
        if key in _configuration:
            _configuration[key].update(resolved[key])
        else:
            _configuration[key] = resolved[key]


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
    if len(config) == 0:
        return
    parsed = _parse_configuration(config)
    resolved = _resolve_configuration(parsed)
    _add_resolved_config(resolved)


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
    if _is_lazy_resolution and len(_skip_resolution) > 0:
        logger.warning(
            'There are unresolved configurations because lazy resolution was set to `True`; '
            'therefore, this will only return a partial config.')
    return _configuration


def clear_config():
    """ Clear the global configuration.

    Side Effects:
        The existing global configuration is reset to it's initial state.
    """
    global _configuration
    _configuration = {}


_is_lazy_resolution = False


def set_lazy_resolution(bool_):
    """ Set `HParams` to resolve configurations during when a configured function
    is executed instead of when `add_config` is executed.

    This lazy resolution ensures that no modules are imported that are not.

    Args:
        bool_ (bool): If `True` the configurations are resolved and checked lazily.
    """
    global _is_lazy_resolution
    _is_lazy_resolution = bool_
    _resolve_skipped()


def _merge_args(parameters, args, kwargs, config_kwargs, default_kwargs, print_name):
    """ Merge `func` `args` and `kwargs` with `default_kwargs`.

    The `_merge_args` prefers `kwargs` and `args` over `default_kwargs`.

    Args:
        parameters (list of inspect.Parameter): module that accepts `args` and `kwargs`
        args (list of any): Arguments accepted by `func`.
        kwargs (dict of any): Keyword arguments accepted by `func`.
        config_kwargs (dict of any): Config keyword arguments accepted by `func` to merge.
        default_kwargs (dict of any): Default keyword arguments accepted by `func` to merge.
        print_name (str): Function name to print with warnings.

    Returns:
        (dict): kwargs merging `args`, `kwargs`, and `default_kwargs`
    """
    merged_kwargs = default_kwargs.copy()
    merged_kwargs.update(config_kwargs)

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
                if parameters[i].name in config_kwargs or isinstance(value, _HParam):
                    # NOTE: This uses ``warnings`` based on these guidelines:
                    # https://stackoverflow.com/questions/9595009/python-warnings-warn-vs-logging-warning/14762106
                    warnings.warn(
                        '@configurable: Overwriting configured argument `%s=%s` in module `%s` '
                        'with `%s`.' % (parameters[i].name, str(value), print_name, arg))
                del merged_kwargs[parameters[i].name]

    for key, value in kwargs.items():
        if key in config_kwargs or (key in merged_kwargs and
                                    isinstance(merged_kwargs[key], _HParam)):
            warnings.warn('@configurable: Overwriting configured argument `%s=%s` in module `%s` '
                          'with `%s`.' % (key, str(merged_kwargs[key]), print_name, value))

    merged_kwargs.update(kwargs)
    return args, merged_kwargs


# NOTE: `pytest` coverage module cannot track this code.


def profile_func(frame, event, arg):  # pragma: no cover
    """ This `profile_func` is executed by `sys.setprofile`. It is used to warn the user if
    a configured function is run without the decorator.

    Args:
        See docs for `sys.setprofile`.
    """
    if (event != 'call' or not hasattr(frame, 'f_code') or not hasattr(frame, 'f_back') or
            not hasattr(frame.f_back, 'f_code') or frame.f_code not in _code_to_function):
        return

    function = _code_to_function[frame.f_code]
    last_filename = frame.f_back.f_code.co_filename
    if not (__file__ == last_filename and frame.f_back.f_code.co_name == 'decorator'):
        warnings.warn(
            '@configurable: The decorator was not executed immediately before `%s` at (%s:%s); '
            'therefore, it\'s `HParams` may not have been injected. ' %
            (_get_function_signature(function), last_filename, frame.f_back.f_lineno))


sys.setprofile(profile_func)

_code_to_function = {}  # Reverse lookup from `function.__code__` to `function`.

_ConfiguredFunction = typing.TypeVar('_ConfiguredFunction', bound=typing.Callable[..., typing.Any])


def configurable(function: _ConfiguredFunction = None) -> _ConfiguredFunction:
    """ Decorator enables configuring module arguments.

    Decorator enables one to set the arguments of a module via a global configuration. The decorator
    also stores the parameters the decorated function was called with.

    Args:
        None

    Returns:
        (callable): Decorated function
    """
    # TODO: Add recursive typing after it's supported: https://github.com/python/mypy/issues/731
    if not function:
        return configurable

    # TODO: This may grow in memory if functions are dynamically created and deleted.
    function_signature = _get_function_signature(function)
    function_parameters = list(_get_function_parameters(function).values())
    function_default_kwargs = _get_function_default_kwargs(function)

    def _get_configuration():
        return _configuration[function_signature] if function_signature in _configuration else {}

    @wraps(function)
    def decorator(*args, **kwargs):
        global _configuration

        _resolve_skipped()

        # Get the function configuration
        config = _get_configuration()
        if function_signature not in _configuration:
            warnings.warn('@configurable: No config for `%s`. ' % (function_signature,))

        args, kwargs = _merge_args(function_parameters, args, kwargs, config,
                                   function_default_kwargs, function_signature)

        # Ensure all `HParam` objects are overridden.
        [a._raise() for a in itertools.chain(args, kwargs.values()) if isinstance(a, _HParam)]

        return function(*args, **kwargs)

    # USE CASE: `get_configured_partial` can be used to export a function with it's configuration
    # for multiprocessing.
    def get_configured_partial():
        return partial(decorator, **_get_configuration())

    decorator.get_configured_partial = get_configured_partial

    # Add a flag to the func; enabling us to check if a function has the configurable decorator.
    decorator._configurable = True

    if hasattr(function, '__code__'):
        _code_to_function[function.__code__] = function
    else:
        logger.warning(
            '@configurable: `%s` does not have a `__code__` attribute; '
            'therefore, this cannot verify if `HParams` are injected. '
            'This should only affect Python `builtins`.', function_signature)

    return cast(_ConfiguredFunction, decorator)


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

        error = ValueError('The command line argument `%s` is ambiguous. ' % arg +
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
