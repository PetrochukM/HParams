import functools
import sys
import warnings

import pytest

from config.config import (
    Args,
    _get_func_and_arg,
    add,
    export,
    get,
    log,
    parse_cli_args,
    partial,
    profile,
    purge,
    to_str,
)


def _func(*a, **k):
    return (a, k)


def _other_func(*a, **k):
    return _func(*a, **k)


_func.attr = _other_func
_func.attr.bttr = _other_func


class Obj:
    def __init__(self, *a, **k) -> None:
        self.results = (a, k)

    def func(self, *a, **k):
        return _func(*a, **k)

    def __call__(self, *a, **k):
        return _func(*a, **k)

    def new(self, *a, **k):
        return Obj(*a, **k)


class OtherObj(Obj):

    static_obj = Obj()

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.obj = super().new(*a, **k)


def test__get_func_and_arg():
    """Test `_get_func_and_arg` can handle the basic case."""
    result = _func(a=_get_func_and_arg())
    assert result == (tuple(), {"a": (_func, "a")})


def test__get_func_and_arg__class_init():
    """Test `_get_func_and_arg` can handle a class instantiation."""
    result = Obj(a=_get_func_and_arg()).results
    assert result == (tuple(), {"a": (Obj.__init__, "a")})


def test__get_func_and_arg__class_func():
    """Test `_get_func_and_arg` can handle a class func."""
    result = Obj().func(a=_get_func_and_arg())
    assert result == (tuple(), {"a": (Obj.func, "a")})


def test__get_func_and_arg__class_attribute():
    """Test `_get_func_and_arg` can handle instantiated attributes."""
    with pytest.raises(SyntaxError):
        OtherObj().obj.new(a=_get_func_and_arg())


def test__get_func_and_arg__class_static_attribute():
    """Test `_get_func_and_arg` can handle static attributes."""
    result = OtherObj().static_obj.new(a=_get_func_and_arg()).results
    assert result == (tuple(), {"a": (OtherObj.static_obj.new, "a")})


def test__get_func_and_arg__class_special():
    """Test `_get_func_and_arg` can handle a special class func."""
    with pytest.raises(SyntaxError):
        Obj()(a=_get_func_and_arg())


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
    config = {enumerate: Args(start=1)}
    assert export() == {}
    add(config)
    result = list(enumerate(range(3), start=get()))
    assert result == [(1, 0), (2, 1), (3, 2)]
    result = list(enumerate(range(3), **get()))
    assert result == [(1, 0), (2, 1), (3, 2)]
    assert export() == config
    purge()
    assert export() == {}
    with pytest.raises(KeyError):
        enumerate(range(3), start=get())


def test_config__repeated():
    """Test `config` operations can handle if a kwarg is already defined."""
    config = {enumerate: Args(start=1)}
    add(config)
    with pytest.raises(TypeError):
        enumerate(range(3), start=2, **get())


def test_config__unused_func():
    """Test `config.purge` warns if a func configuration isn't used."""
    add({enumerate: Args(start=1)})
    message = "^These configurations were not used:\nenumerate#start$"
    with pytest.warns(UserWarning, match=message):
        purge()


def test_config__unused_arg():
    """Test `config.purge` warns if a func argument configuration isn't used."""
    add({sorted: Args(reverse=False, key=None)})
    sorted([], reverse=get())
    message = "^These configurations were not used:\nsorted#key$"
    with pytest.warns(UserWarning, match=message):
        purge()


def test_config__incorrect_arg():
    """Test `config.add` errors if `Args` has non existant arguments."""
    with pytest.raises(ValueError):
        add({sorted: Args(does_not_exist=False)})


def test_config__variable_args():
    """Test `config.add` handles variable parameters."""
    add({_func: Args(does_not_exist=False)})


def test_config__incorrect_type():
    """Test `config.add` handles incorrectly typed arguments."""

    def func(a: str):
        pass

    with pytest.raises(TypeError):
        add({func: Args(a=False)})


def test_config__partial():
    """Test `config.partial` configures a partial."""
    add({enumerate: Args(start=1)})
    result = list(partial(enumerate)(range(3)))
    assert result == [(1, 0), (2, 1), (3, 2)]


def test_config__change():
    """Test `config.get` and `config.add` use copies to prevent side-effects."""
    config = {sorted: Args(reverse=False, key=None)}
    excepted = {sorted: Args(reverse=False, key=None)}
    add(config)
    config[sorted]["reverse"] = True
    assert export() == excepted
    config[sorted] = Args()
    assert export() == excepted
    gotten = export()
    gotten[sorted]["reverse"] = True
    assert export() == excepted
    gotten[sorted] = Args()
    assert export() == excepted


def test_parse_cli_args():
    """Test `config.parse_cli_args` on a basic case."""
    add({sorted: Args(reverse=False)})
    cli_args = ["--sorted", "Args(reverse=True)"]
    assert parse_cli_args(cli_args) == {sorted: Args(reverse=True)}


def test_parse_cli_args__no_config():
    """Test `config.parse_cli_args` errors when the configuration doesn't exist."""
    cli_args = ["--sorted", "Args(reverse=True)"]
    with pytest.raises(ValueError, match="Unable to find function 'sorted' in configuration."):
        parse_cli_args(cli_args)


def test_parse_cli_args__single_flag():
    """Test `config.parse_cli_args` errors when a single flag is used."""
    add({sorted: Args(reverse=False)})
    cli_args = ["-sorted", "Args(reverse=True)"]
    with pytest.raises(ValueError, match="Unable to parse the command line argument `-sorted`."):
        parse_cli_args(cli_args)


def test_parse_cli_args__no_value():
    """Test `config.parse_cli_args` errors when a value isn't passed in."""
    cli_args = ["--sorted"]
    with pytest.raises(ValueError, match="Unable to parse the command line argument `--sorted`."):
        parse_cli_args(cli_args)


def test_parse_cli_args__no_args():
    """Test `config.parse_cli_args` errors when `Args` isn't passed in."""
    add({sorted: Args(reverse=False)})
    cli_args = ["--sorted", "True"]
    with pytest.raises(ValueError, match="argument value must be an `Args` object"):
        parse_cli_args(cli_args)


def test_parse_cli_args__invalid_eval_expression():
    """Test `config.parse_cli_args` errors when an invalid expression is passed in."""
    add({sorted: Args(reverse=False)})
    cli_args = ["--sorted", "reverse=True"]
    with pytest.raises(SyntaxError):
        parse_cli_args(cli_args)


def test_to_str():
    """Test `config.to_str` can handle a basic case."""
    assert to_str(test_to_str) == "tests.test_config.test_to_str"


def test_to_str__no_sys_path():
    """Test to ensure that `to_str` can handle an absolute path."""
    original = sys.path
    sys.path = []
    assert "tests.test_config.test_to_str__no_sys_path" in (to_str(test_to_str__no_sys_path))
    sys.path = original


def test_to_str__relative_sys_path():
    """Test to ensure that `to_str` can handle an relative path."""
    original = sys.path
    sys.path = [""]
    expected = "tests.test_config.test_to_str__relative_sys_path"
    assert expected == (to_str(test_to_str__relative_sys_path))
    sys.path = original


def test_log():
    """Test `config.log` can handle a basic case."""
    add({enumerate: Args(start=1)})
    assert log() == {"#enumerate.start": "1"}


def test_profile():
    """Test `config.profile` can handle a basic case."""
    profile_ = sys.getprofile()
    sys.setprofile(profile)

    def configured(a=5):
        pass

    add({configured: Args(a=1)})
    with pytest.warns(UserWarning):
        configured()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        configured(a=get())

    sys.setprofile(profile_)
