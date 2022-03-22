""" This module implements a localized and low-overhead tracer.

It's based on these posts:
https://hardenedapple.github.io/stories/computers/python_function_override/
https://stackoverflow.com/questions/59088671/hooking-every-function-call-in-python
https://stackoverflow.com/questions/27671553/modifying-code-of-function-at-runtime
https://stackoverflow.com/questions/71574980/low-overhead-tracing-function-in-python-by-modify-the-code-object/71576012#71576012
"""

import inspect
import sys
import types
import typing
from functools import partial

# NOTE: These names are unique so they don't interfere with existing globals.
_original_code_attribute_name = "___trace_orig_code"
_closure_fn_name = "___closure"
_get_frame_fn_name = "___get_frame"


def _unwrap(func: typing.Callable):
    """Unwrap decorated and bounded function."""
    if hasattr(func, "__func__"):
        func = func.__func__
    func = inspect.unwrap(func)
    return func


def _get_trace_fn_name(fn):
    """Get a unique tracing function name per function."""
    name = fn.__qualname__.replace(".", "_").replace("<", "_").replace(">", "_")
    return f"___trace_{name}"


def _make_code(fn: typing.Callable, trace_fn_name: str) -> types.CodeType:
    """Create Code object that runs `trace_fn_name` on the first line."""
    lines = inspect.getsourcelines(fn)[0]

    # Remove extra indentation
    init_indent = len(lines[0]) - len(lines[0].lstrip())
    lines = [l[init_indent:] for l in lines]

    offset = next(i for i, l in enumerate(lines) if "@" != l[0])

    # Add indentation for template code below
    lines = ["    " + l for l in lines]

    # NOTE: To ensure `co_lnotab` isn't affected, the trace function is added to the first line
    # along with the original fist line.
    whitespace = lines[offset + 1][: len(lines[offset + 1]) - len(lines[offset + 1].lstrip())]
    lines[offset + 1] = (
        f"{whitespace}{trace_fn_name}({_get_frame_fn_name}()); " f"{lines[offset + 1].strip()}\n"
    )

    # Create closures
    free_vars = " ".join([f"    {var} = None;" for var in fn.__code__.co_freevars])

    code = f"""
def {_closure_fn_name}():
{free_vars}

{"".join(lines)}

    return {fn.__name__}
"""
    module = fn.__globals__.copy()
    try:
        exec(code, module)
    except SyntaxError:
        raise SyntaxError("Unable to add `___trace` to function definition.")
    new: typing.Callable = _unwrap(module[_closure_fn_name]())

    return types.CodeType(
        fn.__code__.co_argcount,
        fn.__code__.co_posonlyargcount,
        fn.__code__.co_kwonlyargcount,
        fn.__code__.co_nlocals,
        fn.__code__.co_stacksize,
        fn.__code__.co_flags,
        new.__code__.co_code,
        fn.__code__.co_consts,
        tuple([trace_fn_name, _get_frame_fn_name] + list(fn.__code__.co_names)),
        fn.__code__.co_varnames,
        fn.__code__.co_filename,
        fn.__code__.co_name,
        fn.__code__.co_firstlineno,
        new.__code__.co_lnotab,
        fn.__code__.co_freevars,
        fn.__code__.co_cellvars,
    )


def _update_globals(
    globals_: typing.Dict, key: str, value: typing.Any, cmp_: typing.Callable = lambda x: x
):
    """Add `value` to `globals_` under `key`."""
    if key in globals_:
        if cmp_(globals_[key]) is not cmp_(value):
            raise RuntimeError(f"`{key}` has already been set.")
    else:
        globals_[key] = value


def set_trace(func: typing.Callable, trace_func: typing.Callable):
    """Set `trace_func` to be called at the beginning of `func`."""
    func = _unwrap(func)
    trace_fn_name = _get_trace_fn_name(func)
    trace_func = partial(trace_func, event="call", arg=None)
    _update_globals(func.__globals__, trace_fn_name, trace_func, cmp_=lambda p: p.func)
    if not hasattr(func, _original_code_attribute_name):
        setattr(func, _original_code_attribute_name, func.__code__)
        func.__code__ = _make_code(func, trace_fn_name)
        _update_globals(func.__globals__, trace_fn_name, trace_func)
        _update_globals(func.__globals__, _get_frame_fn_name, sys._getframe)


def unset_trace(func: typing.Callable):
    """Remove trace function from `func`."""
    if hasattr(func, _original_code_attribute_name):
        func.__code__ = getattr(func, _original_code_attribute_name)
        delattr(func, _original_code_attribute_name)
        del func.__globals__[_get_trace_fn_name(func)]
