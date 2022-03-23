import functools
import traceback
import types
import typing
import warnings

import pytest

from config.trace import set_trace, unset_trace


def helper(*args, **kwargs):
    pass


def trace_func(frame: types.FrameType, event: str, arg):
    warnings.warn(f"{frame.f_code.co_name}:{frame.f_lineno}")


def test_trace():
    """Test `set_trace` handles basic cases."""
    set_trace(helper, trace_func)
    with pytest.warns(UserWarning, match="^helper:13$"):
        helper()
    unset_trace(helper)
    unset_trace(helper)
    set_trace(helper, trace_func)
    with pytest.warns(UserWarning, match="^helper:13$"):
        helper()


def helper_closure():
    variable = "blah"

    def func():
        return variable

    assert func.__code__.co_freevars == ("variable",)  # Ensure this is a closure
    assert func() == variable
    set_trace(func, trace_func)
    assert func() == variable


def test_trace__closure():
    """Test `set_trace` handles closures."""
    with pytest.warns(UserWarning, match="^func:36$"):
        helper_closure()


def helper_globals(a: typing.List, b: typing.List):
    pytest.raises


def test_trace__globals():
    """Test `set_trace` handles globals."""
    set_trace(helper_globals, trace_func)
    with pytest.warns(UserWarning, match="^helper_globals:51$"):
        helper_globals([], [])


def helper_traceback():
    return "".join(traceback.format_stack(limit=2))


def test_trace__traceback():
    """Test `set_trace` handles a traceback."""
    set_trace(helper_traceback, trace_func)
    with pytest.warns(UserWarning, match="^helper_traceback:62$"):
        expected = [
            'tests/test_trace.py", line 75, in test_trace__traceback',
            'result = [l.strip() for l in helper_traceback().split("\\n") if len(l.strip()) > 0]',
            'tests/test_trace.py", line 62, in helper_traceback',
            'return "".join(traceback.format_stack(limit=2))',
        ]
        result = [l.strip() for l in helper_traceback().split("\n") if len(l.strip()) > 0]
        assert all(e in r for r, e in zip(result, expected))


def helper_cellvars():
    a = 10

    def func():
        return a

    return func()


def test_trace__cellvars():
    """Test `set_trace` handles cellvars."""
    assert helper_cellvars.__code__.co_cellvars == ("a",)
    set_trace(helper_cellvars, trace_func)
    with pytest.warns(UserWarning, match="^helper_cellvars:80$"):
        helper_cellvars()


@functools.lru_cache()
@functools.lru_cache()
def helper_decorator():
    pass


def test_trace__decorator():
    """Test `set_trace` handles decorators."""
    set_trace(helper_decorator, trace_func)
    with pytest.warns(UserWarning, match="^helper_decorator:99$"):
        helper_decorator()


class HelperObject:
    def __init__(self):
        pass


def test_trace__object():
    """Test `set_trace` handles objects."""
    set_trace(HelperObject.__init__, trace_func)
    with pytest.warns(UserWarning, match="^__init__:111$"):
        HelperObject()


def helper_funky_first_line():
    def func():
        return


def test_trace__funky_first_line():
    """Test `set_trace` handles a incomatible first line."""
    with pytest.raises(SyntaxError):
        set_trace(helper_funky_first_line, trace_func)


@functools.lru_cache()
def helper_multiline(
    a: str = "a",
    b: str = "b",
):
    """This is a multi-line...
    comment!"""
    pass


# fmt: off

def helper_multiline_one(  # noqa: E704
    a=10,
    b=100): print(100); print(200);  # noqa: E702, E703


def helper_multiline_two(  # noqa: E704
    a=(""),
    b=100): print(100); print(200);  # noqa: E702, E703


def func_one_liner(): print(200); print("a", "b", "c")  # noqa: E702, E703, E704


def helper_multiline_three(  # noqa: E704
    a: str = 'x', b: int = 5 + 6, c: list = []
    ) -> max(2, 9): print(100); print(200)  # noqa: E702, E703, E704, E123


def helper_multiline_four(  # noqa: E704
    a: str = 'x', b: int = 5 + 6, c: list = []
    ) -> None: print(100); print(200)  # noqa: E702, E703, E704, E123

# fmt: on


def test_trace__multiline():
    """Test `set_trace` handles a multiline definition."""
    for funcs, lineno in [
        (helper_multiline, 137),
        (helper_multiline_one, 146),
        (helper_multiline_two, 151),
        (func_one_liner, 154),
        (helper_multiline_three, 159),
        (helper_multiline_four, 164),
    ]:
        set_trace(funcs, trace_func)
        with pytest.warns(UserWarning, match=f"^{funcs.__name__}:{lineno}$"):
            funcs()
