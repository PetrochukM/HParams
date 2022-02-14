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

import executing


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
    obj = _find_object(frame, attr.value.id)
    for attr in reversed(attrs):
        obj = getattr(obj, attr)
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
    func: typing.Optional[collections.abc.Callable] = None,
    stack: int = 1,
) -> tuple[collections.abc.Callable, str]:
    """Get the calling function that executes this code to get it's input.

    NOTE: This may not work with PyTest, learn more:
    https://github.com/alexmojaki/executing/issues/2

    NOTE: Use `executing` until Python 3.11, learn more:
    https://github.com/alexmojaki/executing/issues/24
    https://www.python.org/dev/peps/pep-0657/

    TODO: Cache the function similar to `executing_cache` in the `executing` package.

    For example:
        >>> def func(a):
        ...      pass
        ...
        >>> func((caller := get_calling_func()))
        >>> caller
        func
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

    return func, arg


class Params(dict[str, typing.Any]):
    pass


_config: dict[collections.abc.Callable, Params] = {}
_count: dict[collections.abc.Callable, dict[str, int]] = defaultdict(lambda: defaultdict(int))


def fill(
    func: typing.Optional[collections.abc.Callable] = None, arg: typing.Optional[str] = None
) -> typing.Any:
    global _count

    message = (
        "Unable to determine the calling function and argument name.\n\n"
        + textwrap.fill(
            "This uses Python AST to parse the code and retrieve the calling function and argument "
            "name. In order to do so, they need to be both explicitly named, like so:"
        )
        + (
            "\n"
            "✅ function(arg=get_calling_func())\n"
            "❌ function(get_calling_func())\n"
            "❌ func = lambda: function\n"
            "   func()(arg=get_calling_func())\n"
        )
    )

    try:
        func, arg = _get_func_and_arg(arg, func, stack=2)
    except SyntaxError as e:
        raise SyntaxError(message) from e

    message = (
        f"`{arg}` for `{func.__qualname__}` has not been configured.\n\n"
        "It can be configured like so:"
        f'>>> config.add({{{func.__qualname__}: config.Params({arg}="PLACEHOLDER")}})'
    )

    if func not in _config or (arg is not None and arg not in _config[func]):
        raise KeyError(message)

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
    for func, params in _config.items():
        for key in params.keys():
            if func not in _count or _count[func][key] == 0:
                unused.append(f"{func.__qualname__}#{key}")
    if len(unused) > 0:
        warnings.warn("These configurations were not used:\n" + "\n".join(unused))

    _config = {}
    _count = defaultdict(lambda: defaultdict(int))


atexit.register(purge)


def get() -> dict[collections.abc.Callable, Params]:
    """Get the global configuration.

    NOTE: It would be an anti-pattern to use this for configuring functions.
    """
    return {k: v.copy() for k, v in _config.items()}


def _check_params(func: collections.abc.Callable, params: Params):
    """Ensure every argument in `params` exists in `func`."""
    parameters = inspect.signature(func).parameters
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in parameters.values()):
        return True

    for key in params.keys():
        if key not in parameters:
            raise ValueError(f"{key} is not any argument in {func.__qualname__}")


def add(config: dict[collections.abc.Callable, Params]):
    """Add to the global configuration."""
    global _config
    for func, params in config.items():
        _check_params(func, params)
    _config = {**{k: v.copy() for k, v in config.items()}, **_config}


def partial(func: collections.abc.Callable) -> collections.abc.Callable:
    """Get a `partial` for `func` using the global configuration."""
    return functools.partial(func, **_config[func])
