""" This module implements a localized and low-overhead tracer.

It's based on these posts:
https://hardenedapple.github.io/stories/computers/python_function_override/
https://stackoverflow.com/questions/59088671/hooking-every-function-call-in-python
https://stackoverflow.com/questions/27671553/modifying-code-of-function-at-runtime
https://stackoverflow.com/questions/71574980/low-overhead-tracing-function-in-python-by-modify-the-code-object/71576012#71576012
"""

import functools
import inspect
import io
import sys
import tokenize
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


@functools.lru_cache(maxsize=None)
def _make_code(fn: typing.Callable, trace_fn_name: str) -> types.CodeType:
    """Create Code object that runs `trace_fn_name` on the first line."""
    lines = inspect.getsourcelines(fn)[0]

    # Remove extra indentation
    init_indent = len(lines[0]) - len(lines[0].lstrip())
    lines = [l[init_indent:] if len(l.strip()) > 0 else l for l in lines]

    # Get first line and col index of body
    tokens = list(tokenize.generate_tokens(io.StringIO("".join(lines)).readline))
    # NOTE: Find a ")" followed by either ":" or "->"
    idx = next(
        i
        for i, (p, n) in enumerate(zip(tokens, tokens[1:]))
        if (p.type == tokenize.OP and p.string == ")")
        and (n.type == tokenize.OP and (n.string == ":" or n.string == "->"))
    )
    # NOTE: Afterwards, find the next ":" if we haven't found it yet.
    idx += next(i for i, t in enumerate(tokens[idx:]) if t.type == tokenize.OP and t.string == ":")
    idx += 2
    # NOTE: Lastly, skip over, any new lines or indents, until the first operation.
    whitespaces = (tokenize.NEWLINE, tokenize.INDENT)
    idx += next(i for i, t in enumerate(tokens[idx:]) if t.type not in whitespaces)
    offset = tokens[idx].start[0] - 1

    # Insert trace function there
    # NOTE: To ensure `co_lnotab` isn't affected, the trace function is added to the first line
    # along with the original fist line.
    insert = f"{trace_fn_name}({_get_frame_fn_name}()); "
    line = lines[offset]
    lines[offset] = line[: tokens[idx].start[1]] + insert + line[tokens[idx].start[1] :]

    is_closure = len(fn.__code__.co_freevars) != 0
    if is_closure:
        code = f"""
def {_closure_fn_name}():
{" ".join([f"    {var} = None;" for var in fn.__code__.co_freevars])}

{"".join(["    " + l for l in lines])}

    return {fn.__name__}
"""
    else:
        code = "".join(lines)
    module = fn.__globals__.copy()
    try:
        exec(code, module)
    except SyntaxError:
        raise SyntaxError(f"Unable to add `___trace` to `{fn.__qualname__}` definition.")
    new: typing.Callable = module[_closure_fn_name]() if is_closure else module[fn.__name__]
    new = _unwrap(new)

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
