from __future__ import annotations

import builtins
import collections
import functools
import inspect
import logging
import sys
import traceback
import types
import typing
import warnings
from collections import defaultdict
from pathlib import Path
from types import CodeType
from typing import get_type_hints

from typeguard import TypeCheckError, check_type

from config.trace import _unwrap, set_trace, unset_trace

logger = logging.getLogger(__name__)


class Args(typing.Dict[str, typing.Any]):
    pass


class DiffArgsWarning(UserWarning):
    pass


class UnusedConfigsWarning(UserWarning):
    pass


class NoFastTraceWarning(UserWarning):
    pass


class SkipTypeCheck(UserWarning):
    pass


ConfigValue = Args
ConfigKey = collections.abc.Callable
Config = typing.Dict[ConfigKey, ConfigValue]
_config: Config = {}
_count: dict[ConfigKey, dict[str, int]] = defaultdict(lambda: defaultdict(int))
_code_to_func: dict[CodeType, ConfigKey] = {}
_fast_trace_enabled: bool = False
# NOTE: These names are unique so they don't interfere with existing attributes.
_orginal_key = "___orginal_key"


class KeyErrorMessage(str):
    # Learn more:
    # https://stackoverflow.com/questions/46892261/new-line-on-error-message-in-keyerror-python-3-3
    def __repr__(self):
        return str(self)


def _is_builtin(func: collections.abc.Callable):
    return any(func is v for v in vars(builtins).values())


def get(func: typing.Optional[ConfigKey], arg: typing.Optional[str] = None) -> typing.Any:
    """Get the configuration for `func` and `arg`.

    Args:
        func: A reference to a function.
        arg: The argument name.

    Raises:
        KeyError: If this cannot find a configuration.

    Returns: If argument is named, then this returns the configured value for the function and
        argument; otherwise, this will return all of the configured values for the function in
        a dictionary.
    """
    global _count

    message = (
        f"`{func.__qualname__}` has not been configured.\n\n"
        "It can be configured like so:\n"
        f'>>> config.add({{{func.__qualname__}: config.Args(placeholder="PLACEHOLDER")}})'
    )
    if func not in _config:
        raise KeyError(KeyErrorMessage(message))

    message = (
        f"`{arg}` for `{func.__qualname__}` has not been configured.\n\n"
        "It can be configured like so:\n"
        f'>>> config.add({{{func.__qualname__}: config.Args({arg}="PLACEHOLDER")}})'
    )
    if arg is not None and arg not in _config[func]:
        raise KeyError(KeyErrorMessage(message))

    if arg is None:
        for key in _config[func].keys():
            _count[func][key] += 1
    else:
        _count[func][arg] += 1

    return _config[func] if arg is None else _config[func][arg]


def purge():
    """Delete the global configuration."""
    global _config, _count, _code_to_func

    unused = []
    for func, args in _config.items():
        for key in args.keys():
            if func not in _count or _count[func][key] == 0:
                unused.append(f"{func.__qualname__}#{key}")
    if len(unused) > 0:
        message = "These configurations were not used:\n" + "\n".join(unused)
        warnings.warn(message, UnusedConfigsWarning)

    [unset_trace(f) for f, _ in _get_funcs_to_trace(_config)]
    [delattr(k, _orginal_key) for k in _config.keys() if hasattr(k, _orginal_key)]
    _config = {}
    _code_to_func = {}
    _call_once.cache_clear()
    _count = defaultdict(lambda: defaultdict(int))


def export() -> Config:
    """Export the global configuration.

    NOTE: It would be an anti-pattern to use this for configuring functions.
    """
    return {
        getattr(k, _orginal_key) if hasattr(k, _orginal_key) else k: v.copy()
        for k, v in _config.items()
    }


def _type_check_args(
    func: ConfigKey, args: ConfigValue, parameters: typing.Dict[str, inspect.Parameter]
):
    # NOTE: Check the `args` type corresponds with the function signature.
    frame = sys._getframe(1)
    while frame.f_code.co_filename == __file__:
        frame = frame.f_back

    try:
        type_hints = get_type_hints(func)
    except (NameError, ModuleNotFoundError) as e:
        warnings.warn(
            f"Skipping type check for `{func.__qualname__}` due to:\n{str(e)}", SkipTypeCheck
        )
        return

    for key, value in args.items():
        if key in parameters and key in type_hints:
            try:
                check_type(value, type_hints[key])
            except (NameError, ModuleNotFoundError) as e:
                warnings.warn(f"Skipping type check for `{key}` due to:\n{str(e)}", SkipTypeCheck)
            except TypeCheckError:
                raise TypeError(f"`{key}` is not instance of `{type_hints[key]}`")


def _check_args(func: ConfigKey, args: ConfigValue):
    """Ensure every argument in `args` exists in `func`."""
    parameters = inspect.signature(func).parameters

    _type_check_args(func, args, parameters)

    # NOTE: Check `args` exist in the function signature.
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters.values()):
        return

    for key in args.keys():
        if key not in parameters:
            raise ValueError(f"`{key}` is not an argument to {func.__qualname__}")


def _get_funcs(func: typing.Callable) -> typing.List[typing.Callable]:
    """Get a list of functions `func` may be referring to."""
    if inspect.isclass(func):
        funcs = []
        if func.__init__ not in (b.__init__ for b in func.__bases__):
            funcs.append(func.__init__)
        if func.__new__ not in (b.__new__ for b in func.__bases__):
            funcs.append(func.__new__)
        if len(funcs) == 0:
            # NOTE: An implicit configuration like this may lead to duplicate configurations.
            # This happens when the user defines a seperate configuration for a child and parent
            # object that share the same initiation. Since there is no difference between their
            # initiation functions, the configuration for them is duplicated.
            raise KeyError(
                f"The initiation for `{func}` is only implicitly defined. "
                "Please use only explicit configurations."
            )
    else:
        funcs = [func]
    return [_unwrap(f) for f in funcs]


def _get_funcs_to_trace(
    config: Config,
) -> typing.List[typing.Tuple[typing.Callable, typing.Callable]]:
    """Get functions to trace along with their corresponding function configuration key."""
    funcs = [k for k in config.keys() if not _is_builtin(k)]
    items = [(k, v) for v in funcs for k in _get_funcs(v)]
    message = "`__code__` isn't unique"
    assert len(set(f.__code__ for f, _ in items)) == len(items), message
    return items


def enable_fast_trace(enable: bool = True):
    """Enable or disable fast tracing."""
    global _fast_trace_enabled
    if enable and not _fast_trace_enabled:
        logger.info(
            "🎉 Fast trace enabled 🎉\n"
            "This traces the configured functions by modifying their code and inserting a trace "
            "function at the beginning of the function definition. This should work most of the "
            "time; however, it can trigger strange `SyntaxError`s or have other weird behaviors."
        )
    _fast_trace_enabled = enable
    _update_trace_globals()


def _update_trace_globals():
    """Update various globals required for tracing."""
    global _code_to_func
    to_trace = _get_funcs_to_trace(_config)
    for func, config in to_trace:
        try:
            set_trace(func, trace) if _fast_trace_enabled else unset_trace(func)
        except SyntaxError:
            message = f"Unable to fast trace `{func}` on behalf of `{config}`."
            warnings.warn(message, NoFastTraceWarning)
    _code_to_func = {k.__code__: v for k, v in to_trace}
    _call_once.cache_clear()


def _is_equal(a: typing.Any, b: typing.Any) -> bool:
    """Generic function for testing equality that can handle various implementations of `__eq__`."""
    if a is b:
        return True

    try:
        return bool(a == b)
    except Exception:
        return False


def add(config: Config, overwrite: bool = False):
    """Add to the global configuration.

    Args:
        config
        overwrite: Iff `True` then configurations can be overwritten.
    """
    global _config
    global _code_to_func

    [_check_args(func, args) for func, args in config.items()]

    for key_, value in config.items():
        key = _unwrap(key_)
        if not _is_builtin(key):
            setattr(key, _orginal_key, key_)
        _get_funcs(key)  # NOTE: Check `key` before adding it
        if key in _config:
            update = Args({**_config[key], **value})
            if not overwrite and len(update) != len(_config[key]) + len(value):
                for arg, val in _config[key].items():
                    if not _is_equal(update[arg], val):
                        message = f"Trying to overwrite `{key.__qualname__}#{arg}` configuration."
                        raise ValueError(message)
            _config[key] = update
        else:
            _config[key] = value.copy()

    _update_trace_globals()


_PartialReturnType = typing.TypeVar("_PartialReturnType")


def partial(
    func: typing.Callable[..., _PartialReturnType], *args, **kwargs
) -> typing.Callable[..., _PartialReturnType]:
    """Get a `partial` for `func` using the global configuration."""
    global _count
    key = _unwrap(func)
    if key not in _config:
        raise KeyError(f"`{key.__qualname__}` has not been configured.")
    # TODO: If `args`, or `kwargs` is sent accross, this won't account that correctly.
    for arg in _config[key].keys():
        _count[key][arg] += 1
    return functools.partial(functools.partial(func, **_config[key]), *args, **kwargs)


def _diff_args_message(func: typing.Callable, arg: str):
    # TODO: Should this be incorperated into the definition of `DiffArgsWarning`?
    name = f"{to_str(func)}#{arg}"
    return f"Argument `{name}` with different arguments than those that were configured."


_CallReturnType = typing.TypeVar("_CallReturnType")


def call(
    func: typing.Callable[..., _CallReturnType],
    *args,
    _overwrite: bool = False,
    **kwargs,
) -> _CallReturnType:
    """Call `func` with it's configured args.

    Args:
        ...
        _overwrite: Iff `True` then ignore `func` configuration during call.
        ...
    """
    with warnings.catch_warnings():
        if _overwrite:
            message = f".*{to_str(func)}.*"
            warnings.filterwarnings("ignore", category=DiffArgsWarning, message=message)
        return partial(func)(*args, **kwargs)


def parse_cli_args(args: typing.List[str]) -> Config:
    """Parse CLI arguments like `['--sorted', 'Args(reverse=True)']` to
    `{sorted: Args(reverse=True)}`.

    Args:
        args: List of CLI arguments.

    Returns: Configuration that can be used.
    """
    return_ = {}

    while len(args) > 0:
        arg = args.pop(0)

        error = ValueError(
            f"Unable to parse the command line argument `{arg}`. "
            "The format must be either `--key=value` or `--key value`."
        )

        try:
            if "--" == arg[:2] and "=" not in arg:
                key = arg
                value = args.pop(0)
            elif "--" == arg[:2] and "=" in arg:
                key, value = tuple(arg.split("=", maxsplit=1))
            else:
                raise error
        except IndexError:
            raise error

        key = key[2:]  # NOTE: Remove double flags
        funcs = [func for func in _config.keys() if func.__qualname__ == key]
        if len(funcs) > 1:
            raise NotImplementedError(
                "Unable to disambiguate between multiple functions with the same `__qualname__`."
            )
        elif len(funcs) == 0:
            raise ValueError(f"Unable to find function '{key}' in configuration.")

        return_[funcs[0]] = eval(value)

        if not isinstance(return_[funcs[0]], Args):
            raise ValueError(
                "The command line argument value must be an `Args` object like so "
                "`--sorted=Args(reverse=True)`."
            )

    return return_


@functools.lru_cache(maxsize=None)
def to_str(func: ConfigKey):
    """Get a unique string for each function.

    Example:
        >>> to_str(to_str)
        'config.config.to_str'
        >>> import random
        >>> to_str(random.randint)
        'random.Random.randint'
    """
    try:
        func = _unwrap(func)
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
        return filename.replace("/", ".")[:-3] + "." + func.__qualname__
    except TypeError:
        return "#" + func.__qualname__


def log(repr_: typing.Callable[[str], str] = repr) -> typing.Dict[str, str]:
    """Get a loggable flat dictionary of the configuration."""
    return {f"{to_str(f)}.{k}": repr_(v) for f, a in _config.items() for k, v in a.items()}


_CallOnceReturnType = typing.TypeVar("_CallOnceReturnType")


@functools.lru_cache(maxsize=None)
def _call_once(
    callable_: typing.Callable[..., _CallOnceReturnType], *args, **kwargs
) -> _CallOnceReturnType:
    """Call `callable_` only once with `args` and `kwargs` within the same process."""
    return callable_(*args, **kwargs)


@functools.lru_cache(maxsize=None)
def _get_var_keyword(func: typing.Callable, co_name: str) -> typing.Optional[str]:
    if inspect.isclass(func):
        params = inspect.signature(getattr(func, co_name)).parameters
    else:
        params = inspect.signature(func).parameters
    return next((k for k, v in params.items() if v.kind == inspect.Parameter.VAR_KEYWORD), None)


def _diff_args_warn(func: typing.Callable, arg: str, frame: types.FrameType, limit: int = 5):
    traceback_ = "".join(traceback.format_stack(f=frame, limit=limit))
    message = f"{_diff_args_message(func, arg)}\n\nTraceback\n{traceback_}"
    _call_once(warnings.warn, message, DiffArgsWarning)


def trace(frame: types.FrameType, event: str, arg, limit: int = 5):  # pragma: no cover
    """Warn the user if a function is run without it's configured arguments.

    Usage:
        >>> sys.settrace(trace)

    Args:
        See docs for `sys.settrace`.
    """
    frame.f_trace_lines = False
    frame.f_trace_opcodes = False

    if (
        event != "call"
        or not hasattr(frame, "f_code")
        or not hasattr(frame, "f_back")
        or frame.f_code.co_name == "<module>"
        or not hasattr(frame.f_back, "f_code")
        or frame.f_code not in _code_to_func
    ):
        return trace

    func = _code_to_func[frame.f_code]
    items = _config[func].items()
    f_locals = frame.f_locals
    var = _get_var_keyword(func, frame.f_code.co_name)
    if var is None:
        for key, value in items:
            if not _is_equal(f_locals[key], value):
                _diff_args_warn(func, key, frame, limit)
        return

    kwargs = f_locals[var]
    for key, value in items:
        if key in kwargs:
            if not _is_equal(kwargs[key], value):
                _diff_args_warn(func, key, frame, limit)
        elif key in f_locals:
            if not _is_equal(f_locals[key], value):
                _diff_args_warn(func, key, frame, limit)
        else:
            _diff_args_warn(func, key, frame, limit)

    return trace
