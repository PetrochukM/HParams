from __future__ import annotations

import ast
import atexit
import collections
import functools
import inspect
import sys
import textwrap
import types
import typing
import warnings
from collections import defaultdict
from pathlib import Path
from typing import get_type_hints

import executing
from typeguard import check_type


class Args(typing.Dict[str, typing.Any]):
    pass


ConfigValue = Args
ConfigKey = collections.abc.Callable
Config = typing.Dict[ConfigKey, ConfigValue]
_config: Config = {}
_count: dict[ConfigKey, dict[str, int]] = defaultdict(lambda: defaultdict(int))


class KeyErrorMessage(str):
    # Learn more:
    # https://stackoverflow.com/questions/46892261/new-line-on-error-message-in-keyerror-python-3-3
    def __repr__(self):
        return str(self)


def _get_child_to_parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
    """Get a map from child nodes to parent nodes.

    As seen in: https://stackoverflow.com/questions/34570992/getting-parent-of-ast-node-in-python
    """
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _find_object(frame: types.FrameType, name: str) -> typing.Any:
    """Look up a Python object in `frame`."""
    if name in frame.f_locals:
        return frame.f_locals[name]
    elif name in frame.f_globals:
        return frame.f_globals[name]
    return frame.f_builtins[name]


def _resolve_attributes(frame: types.FrameType, attr: ast.AST) -> typing.Any:
    """Resolve a chain of attributes to a Python object."""
    attrs = [attr.attr]
    while isinstance(attr.value, ast.Attribute):
        attr = attr.value
        attrs.append(attr.attr)
    if isinstance(attr.value, ast.Call) and isinstance(attr.value.func, ast.Name):
        obj = _find_object(frame, attr.value.func.id)
        if not inspect.isclass(obj):
            raise SyntaxError("Object is anonymous.")
    elif not isinstance(attr.value, ast.Name):
        raise SyntaxError("Object is anonymous.")
    else:
        obj = _find_object(frame, attr.value.id)
    for attr in reversed(attrs):
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            raise SyntaxError("Unable to resolve attribute.")
    return obj


def _resolve_func(
    frame: types.FrameType, node: typing.Union[ast.Attribute, ast.Name]
) -> typing.Any:
    """Resolve a `Attribute` or `Name` node to a Python object."""
    if isinstance(node, ast.Attribute):
        return _resolve_attributes(frame, node)
    elif not hasattr(node, "id"):
        raise SyntaxError("Object is anonymous.")
    return _find_object(frame, node.id)


def _get_func_and_arg(
    arg: typing.Optional[str] = None,
    func: typing.Optional[ConfigKey] = None,
    stack: int = 1,
) -> tuple[ConfigKey, str]:
    """Get the calling function and argument that executes this code to get it's input.

    NOTE: This may not work with PyTest, learn more:
    https://github.com/alexmojaki/executing/issues/2

    NOTE: Use `executing` until Python 3.11, learn more:
    https://github.com/alexmojaki/executing/issues/24
    https://www.python.org/dev/peps/pep-0657/

    TODO: Cache the function similar to `executing_cache` in the `executing` package.
    """
    if func is not None and arg is not None:
        return func, arg

    frame = sys._getframe(stack)
    exec_ = executing.Source.executing(frame)
    if frame.f_code.co_filename == "<stdin>":
        raise NotImplementedError("REPL is not supported.")
    assert len(exec_.statements) == 1, "Invariant failure."
    tree = next(iter(exec_.statements))
    parents = _get_child_to_parent_map(tree)
    parent = parents[exec_.node]

    if arg is None:
        if isinstance(parent, ast.keyword):
            arg = parent.arg
            parent = parents[parent]

    if func is None:
        if not isinstance(parent, ast.Call):
            raise SyntaxError("Unable to find calling function.")
        func = _resolve_func(frame, parent.func)
        if func == functools.partial:
            if len(parent.args) == 0:
                raise SyntaxError("Partial doesn't have arguments.")
            func = _resolve_func(frame, parent.args[0])
        # NOTE: `builtins` like `enumerate` are triggered like a class.
        if inspect.isclass(func) and not any(func is v for v in frame.f_builtins.values()):
            func = func.__init__

    return func, arg


def get(arg: typing.Optional[str] = None, func: typing.Optional[ConfigKey] = None) -> typing.Any:
    """Get the configuration for `func` and `arg`.

    NOTE: If `arg` and `func` isn't passed in, then this will attempt to automatically determine
          them by parsing the code with Python AST. For this to succeed, the function and argument
          must be explicitly named in the code base. Here are a couple examples for reference...

          ✅ function(arg=get())
          ✅ function(get('arg'))
          ✅ arg = get('arg', function)
             function(arg=arg)
          ✅ function(get()) # NOTE: All arguments for `function` are returned.
          ❌ func = lambda: function\n"
              func()(arg=get()) # NOTE: Function isn't named.

    Args:
        arg: The argument name.
        func: A reference to a function.

    Raises:
        SyntaxError: If this is unable to determine the argument name or function reference.
        KeyError: If this cannot find a configuration.

    Returns: If argument is named, then this returns the configured value for the function and
        argument; otherwise, this will return all of the configured values for the function in
        a dictionary.
    """

    global _count

    message = (
        "Unable to determine the calling function and argument name.\n\n"
        + textwrap.fill(
            "This uses Python AST to parse the code and retrieve the calling function and argument "
            "name. In order to do so, they need to be both explicitly named, like so:"
        )
        + (
            "\n"
            "✅ function(arg=get())\n"
            "✅ function(get('arg'))\n"
            "✅ arg = get('arg', function)\n"
            "   function(arg=arg)\n"
            "✅ function(get()) # NOTE: All arguments for `function` are returned.\n"
            "❌ func = lambda: function\n"
            "   func()(arg=get()) # NOTE: Function isn't named.\n"
        )
    )

    try:
        func, arg = _get_func_and_arg(arg, func, stack=2)
    except SyntaxError as e:
        raise SyntaxError(message) from e

    message = (
        f"`{arg}` for `{func.__qualname__}` has not been configured.\n\n"
        "It can be configured like so:\n"
        f'>>> config.add({{{func.__qualname__}: config.Args({arg}="PLACEHOLDER")}})'
    )

    if func not in _config or (arg is not None and arg not in _config[func]):
        raise KeyError(KeyErrorMessage(message))

    if arg is None:
        for key in _config[func].keys():
            _count[func][key] += 1
    else:
        _count[func][arg] += 1

    return _config[func] if arg is None else _config[func][arg]


def purge():
    """Delete the global configuration."""
    global _config, _count

    unused = []
    for func, args in _config.items():
        for key in args.keys():
            if func not in _count or _count[func][key] == 0:
                unused.append(f"{func.__qualname__}#{key}")
    if len(unused) > 0:
        warnings.warn("These configurations were not used:\n" + "\n".join(unused))

    _config = {}
    _count = defaultdict(lambda: defaultdict(int))


atexit.register(purge)


def export() -> Config:
    """Export the global configuration.

    NOTE: It would be an anti-pattern to use this for configuring functions.
    """
    return {k: v.copy() for k, v in _config.items()}


def _check_args(func: ConfigKey, args: ConfigValue):
    """Ensure every argument in `args` exists in `func`."""
    parameters = inspect.signature(func).parameters

    # NOTE: Check the `args` type corresponds with the function signature.
    frame = sys._getframe(1)
    while frame.f_code.co_filename == __file__:
        frame = frame.f_back
    context = dict(globals=frame.f_globals, locals=frame.f_locals)
    type_hints = get_type_hints(func)
    for key, value in args.items():
        if key in parameters and key in type_hints:
            check_type(key, value, type_hints[key], **context)

    # NOTE: Check `args` exist in the function signature.
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters.values()):
        return

    for key in args.keys():
        if key not in parameters:
            raise ValueError(f"{key} is not any argument in {func.__qualname__}")


def add(config: Config):
    """Add to the global configuration."""
    global _config
    [_check_args(func, args) for func, args in config.items()]
    _config = {**{k: v.copy() for k, v in config.items()}, **_config}


def partial(func: ConfigKey, *args, **kwargs) -> ConfigKey:
    """Get a `partial` for `func` using the global configuration."""
    return functools.partial(func, *args, **kwargs, **_config[func])


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
        # NOTE: Unwrap function decorators because they add indirection to the actual function
        # filename.
        while hasattr(func, "__wrapped__"):
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
        return filename.replace("/", ".")[:-3] + "." + func.__qualname__
    except TypeError:
        return "#" + func.__qualname__


def log() -> typing.Dict[str, str]:
    """Get a loggable flat dictionary of the configuration."""
    return {f"{to_str(f)}.{k}": repr(v) for f, a in _config.items() for k, v in a.items()}
