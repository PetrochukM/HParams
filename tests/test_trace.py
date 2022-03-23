import functools
import traceback
import types
import typing
import warnings
from unittest import mock

import pytest

from config.trace import set_trace, unset_trace


def helper(*args, **kwargs):
    pass


def trace_func(frame: types.FrameType, event: str, arg):
    warnings.warn(f"{frame.f_code.co_name}:{frame.f_lineno}")


def test_trace():
    """Test `set_trace` handles basic cases."""
    set_trace(helper, trace_func)
    with pytest.warns(UserWarning, match="^helper:14$"):
        helper()
    unset_trace(helper)
    unset_trace(helper)
    set_trace(helper, trace_func)
    with pytest.warns(UserWarning, match="^helper:14$"):
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
    with pytest.warns(UserWarning, match="^func:37$"):
        helper_closure()


def helper_globals(a: typing.List, b: typing.List):
    pytest.raises


def test_trace__globals():
    """Test `set_trace` handles globals."""
    set_trace(helper_globals, trace_func)
    with pytest.warns(UserWarning, match="^helper_globals:52$"):
        helper_globals([], [])


def helper_traceback():
    return "".join(traceback.format_stack(limit=2))


def test_trace__traceback():
    """Test `set_trace` handles a traceback."""
    set_trace(helper_traceback, trace_func)
    with pytest.warns(UserWarning, match="^helper_traceback:63$"):
        expected = [
            'tests/test_trace.py", line 76, in test_trace__traceback',
            'result = [l.strip() for l in helper_traceback().split("\\n") if len(l.strip()) > 0]',
            'tests/test_trace.py", line 63, in helper_traceback',
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
    with pytest.warns(UserWarning, match="^helper_cellvars:81$"):
        helper_cellvars()


@functools.lru_cache()
@functools.lru_cache()
def helper_decorator():
    pass


def test_trace__decorator():
    """Test `set_trace` handles decorators."""
    set_trace(helper_decorator, trace_func)
    with pytest.warns(UserWarning, match="^helper_decorator:100$"):
        helper_decorator()


class HelperObject:
    def __init__(self):
        pass


def test_trace__object():
    """Test `set_trace` handles objects."""
    set_trace(HelperObject.__init__, trace_func)
    with pytest.warns(UserWarning, match="^__init__:112$"):
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
    b=100): pass; pass;  # noqa: E702, E703


def helper_multiline_two(  # noqa: E704
    a=(""),
    b=100): pass; pass;  # noqa: E702, E703


def func_one_liner(): pass; pass  # noqa: E702, E703, E704


def helper_multiline_three(  # noqa: E704
    a: str = 'x', b: int = 5 + 6, c: list = []
    ) -> max(2, 9): pass; pass  # noqa: E702, E703, E704, E123


def helper_multiline_four(  # noqa: E704
    a: str = 'x', b: int = 5 + 6, c: list = []
    ) -> None: pass; pass  # noqa: E702, E703, E704, E123

# fmt: on


def helper_multiline_five(a="a"):
    helper_multiline_five.a = a


def helper_multiline_six(a="a"):
    def helper_multiline_five():
        helper_multiline_five.a = a

    return helper_multiline_five


class HelperMultilineObjectOne:
    """Blah Blah"""

    def __init__(self) -> None:
        super().__init__()

        self.a = "a"


def helper_multiline_seven():
    """Hello"""

    try:
        pass
    except Exception:
        pass


def helper_multiline_eight():
    """Hello"""
    if True:
        pass


def helper_multiline_nine():
    # This is a comment
    if True:
        pass

    pass


class HelperMultilineObjectTwo:
    """Blah Blah"""

    def __init__(self) -> None:
        self.a = [i for i in range(10)]


def test_trace__multiline():
    """Test `set_trace` handles a multiline definition."""
    for funcs, lineno in [
        (helper_multiline, 139),
        (helper_multiline_one, 147),
        (helper_multiline_two, 152),
        (func_one_liner, 155),
        (helper_multiline_three, 160),
        (helper_multiline_four, 165),
        (helper_multiline_five, 171),
        (helper_multiline_six(), 176),
        (helper_multiline_seven, 191),
        (helper_multiline_eight, 200),
        (helper_multiline_nine, 206),
    ]:
        set_trace(funcs, trace_func)
        with pytest.warns(UserWarning, match=f"^{funcs.__name__}:{lineno}$"):
            funcs()

    for class_, lineno in [(HelperMultilineObjectOne, 185), (HelperMultilineObjectTwo, 217)]:
        set_trace(class_.__init__, trace_func)
        with pytest.warns(UserWarning, match=f"^__init__:{lineno}$"):
            class_()


name = "helper_multiline_funky_one"
helper_multiline_funky_one_code = f"""def {name}(
    a: str = "a",
    b: str = "b",
):
\t\t
\t\tpass"""
module = {}
exec(helper_multiline_funky_one_code, module)
helper_multiline_funky_one = module[name]


@mock.patch("trace.inspect.getsourcelines")
def test_trace__multiline_no_formatting(mock_getsourcelines):
    for funcs, lineno, code in [
        (helper_multiline_funky_one, 5, helper_multiline_funky_one_code),
    ]:
        mock_getsourcelines.side_effect = lambda fn: [[l + "\n" for l in code.split("\n")]]
        set_trace(funcs, trace_func)
        with pytest.warns(UserWarning, match=f"^{funcs.__name__}:{lineno}$"):
            funcs()


def other_trace_func(frame: types.FrameType, event: str, arg):
    warnings.warn(f"Other: {frame.f_code.co_name}:{frame.f_lineno}")


def test_set_trace__another():
    """Test `set_trace` handles case where it's set again."""
    set_trace(helper, trace_func)
    set_trace(helper, trace_func)
    with pytest.warns(UserWarning, match="^helper:14$"):
        helper()
    with pytest.raises(ValueError):
        set_trace(helper, other_trace_func)
    unset_trace(helper)
    set_trace(helper, other_trace_func)
    with pytest.warns(UserWarning, match="^Other: helper:14$"):
        helper()


def helper_zero_type():
    assert type(0) == int
    int(0)


def test_set_trace__pytest_assert():
    """Test `set_trace` fails if PyTest `assert` is encountered. PyTest also does fancy code
    manipulation."""
    with pytest.raises(SyntaxError, match="Unable to add trace to `helper_zero_type` definition."):
        set_trace(helper_zero_type, trace_func)
