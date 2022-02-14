import functools

import pytest

from config.config import Params, _get_func_and_arg, add, fill, get, purge


def _func(*a, **k):
    return (a, k)


def _other_func(*a, **k):
    return _func(*a, **k)


_func.attr = _other_func
_func.attr.bttr = _other_func


def test__get_func_and_arg():
    """Test `_get_func_and_arg` can handle the basic case."""
    result = _func(a=_get_func_and_arg())
    assert result == (tuple(), {"a": (_func, "a")})


def test__get_func_and_arg__defined_arg():
    """Test `_get_func_and_arg` can handle if argument name is passed in."""
    result = _func(_get_func_and_arg("a"))
    assert result == (((_func, "a"),), {})


def test__get_func_and_arg__not_defined_arg():
    """Test `_get_func_and_arg` can handle when arg isn't defined."""
    result = _func(_get_func_and_arg())
    assert result == (((_func, None),), {})


def test__get_func_and_arg__defined_func():
    """Test `_get_func_and_arg` can handle if the function is passed in."""
    result = (lambda: _func)()(a=_get_func_and_arg(func=_func))
    assert result == (tuple(), {"a": (_func, "a")})


def test__get_func_and_arg__not_defined_func():
    """Test `_get_func_and_arg` can handle when func isn't defined."""
    with pytest.raises(SyntaxError):
        _get_func_and_arg("a")


def test__get_func_and_arg__defined_func_and_arg():
    """Test `_get_func_and_arg` can handle if the function and argument is passed in."""
    assert _get_func_and_arg("a", _func) == (_func, "a")


def test__get_func_and_arg__attributes():
    """Test `_get_func_and_arg` can parse attributes."""
    result = _func.attr.bttr(a=_get_func_and_arg())
    assert result == (tuple(), {"a": (_func.attr.bttr, "a")})


def test__get_func_and_arg__anonymous_funcs():
    """Test `_get_func_and_arg` errors for anonymous functions."""
    with pytest.raises(SyntaxError):
        (lambda: _func)()(a=_get_func_and_arg())

    with pytest.raises(SyntaxError):
        functools.partial(lambda *a, **k: (a, k), a=_get_func_and_arg())


def test__get_func_and_arg__partial():
    """Test `_get_func_and_arg` can resolve partials."""
    partial = functools.partial(_func, a=_get_func_and_arg())
    assert partial() == (tuple(), {"a": (_func, "a")})


def test__get_func_and_arg__partial_with_attributes():
    """Test `_get_func_and_arg` can resolve partials with attributes."""
    partial = functools.partial(_func.attr.bttr, a=_get_func_and_arg())
    assert partial() == (tuple(), {"a": (_func.attr.bttr, "a")})


def test__get_func_and_arg__partial_malformed():
    """Test `_get_func_and_arg` errors if partial only has keyword args."""
    with pytest.raises(SyntaxError):
        functools.partial(a=_get_func_and_arg())


def test__get_func_and_arg__builtin():
    """Test `_get_func_and_arg` errors if partial only has keyword args."""
    with pytest.raises(SyntaxError):
        functools.partial(a=_get_func_and_arg())


def test_config():
    """Test `config` operations can handle the basic case."""
    config = {enumerate: Params(start=1)}
    assert get() == {}
    add(config)
    result = list(enumerate(range(3), start=fill()))
    assert result == [(1, 0), (2, 1), (3, 2)]
    result = list(enumerate(range(3), **fill()))
    assert result == [(1, 0), (2, 1), (3, 2)]
    assert get() == config
    purge()
    assert get() == {}
    with pytest.raises(KeyError):
        enumerate(range(3), start=fill())


def test_config__unused_func():
    """Test `config.purge` warns if a func configuration isn't used."""
    add({enumerate: Params(start=1)})
    message = "^These configurations were not used:\nenumerate#start$"
    with pytest.warns(UserWarning, match=message):
        purge()


def test_config__unused_arg():
    """Test `config.purge` warns if a func argument configuration isn't used."""
    add({sorted: Params(reverse=False, key=None)})
    sorted([], reverse=fill())
    message = "^These configurations were not used:\nsorted#key$"
    with pytest.warns(UserWarning, match=message):
        purge()


def test_config__incorrect_arg():
    """Test `config.add` errors if `Params` has non existant arguments."""
    with pytest.raises(ValueError):
        add({sorted: Params(does_not_exist=False)})


def test_config__variable_args():
    """Test `config.add` handles variable parameters."""
    add({_func: Params(does_not_exist=False)})


def test_config__change():
    """Test `config.get` and `config.add` use copies to prevent side-effects."""
    config = {sorted: Params(reverse=False, key=None)}
    excepted = {sorted: Params(reverse=False, key=None)}
    add(config)
    config[sorted]["reverse"] = True
    assert get() == excepted
    config[sorted] = Params()
    assert get() == excepted
    gotten = get()
    gotten[sorted]["reverse"] = True
    assert get() == excepted
    gotten[sorted] = Params()
    assert get() == excepted
