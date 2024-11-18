import functools
import pickle
import sys
import typing
import warnings

import pytest

from config.config import (
    Args,
    DiffArgsWarning,
    SkipTypeCheck,
    UnusedConfigsWarning,
    _diff_args_message,
    _orginal_key,
    add,
    call,
    enable_fast_trace,
    export,
    get,
    log,
    parse_cli_args,
    partial,
    purge,
    to_str,
    trace,
)


@pytest.fixture(autouse=True)
def run_before_test():
    enable_fast_trace(True)

    yield

    # Reset the global state after every test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        purge()


def _func(*a, **k):
    return (a, k)


def _other_func(*a, **k):
    return _func(*a, **k)


@functools.lru_cache()
def _dec_func(*a, **k):
    return _func(*a, **k)


@functools.lru_cache()
@functools.lru_cache()
def _dec_other_func(*a, **k):
    return _func(*a, **k)


_func.attr = _other_func
_func.attr.bttr = _other_func


class Obj:
    def __new__(cls, *a, **k):
        return super().__new__(cls)

    def __init__(self, *a, **k):
        self.results = (a, k)

    def func(self, *a, **k):
        return _func(*a, **k)

    @functools.lru_cache()
    def dec_func(self, *a, **k):
        return _func(*a, **k)

    def __call__(self, *a, **k):
        return _func(*a, **k)

    def new(self, *a, **k):
        return Obj(*a, **k)


class OtherObj(Obj):
    static_obj = Obj()

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.obj = super().new(*a, **k)


class DecObj(Obj):
    @functools.lru_cache()
    @functools.lru_cache()
    def __init__(self, *a, **k):
        super().__init__(*a, **k)


class NewObj:
    def __new__(cls, *_, **__):
        return super().__new__(cls)

    def __init__(self, *a, **k):
        self.results = (a, k)


class NoInitObj(Obj):
    pass


def test_config():
    """Test `config` operations can handle the basic case."""
    config = {enumerate: Args(start=1)}
    assert export() == {}
    add(config)
    result = list(partial(enumerate)(range(3)))
    assert result == [(1, 0), (2, 1), (3, 2)]
    assert export() == config
    purge()
    assert export() == {}
    with pytest.raises(KeyError):
        partial(enumerate)(range(3))


def test_config__export():
    """Test `config` pickle the exported config."""
    add({_dec_func: Args(start=1), enumerate: Args(start=1)})
    assert hasattr(_dec_func.__wrapped__, _orginal_key)
    pickle.dumps(export())
    with pytest.warns(UnusedConfigsWarning):
        purge()
    assert not hasattr(_dec_func.__wrapped__, _orginal_key)


def test_config__forward_ref():
    """Test `config` can handle another global space where an object may not be defined."""

    def _typed_func(a: typing.ForwardRef("AnotherObj")):  # noqa: F821
        return a

    with pytest.warns(SkipTypeCheck, match=_typed_func.__qualname__):
        add({_typed_func: Args(a="str")})

    def _typed_func(a: "typing.ForwardRef('AnotherObj')"):  # noqa: F821
        return a

    with pytest.warns(SkipTypeCheck, match="AnotherObj"):
        add({_typed_func: Args(a="str")})


def test_config__cache():
    """Test `config` operations are cached."""
    add({sorted: Args(reverse=True, key=lambda k: 10 if k == 0 else k)})
    for _ in range(500):
        result = list(partial(sorted)(range(3)))
        assert result == [0, 2, 1]


def test_config__repeated():
    """Test `config` operations can handle if a kwarg is already defined."""

    def func(a):
        return a

    config = {func: Args(a=1)}
    add(config)
    with pytest.warns(DiffArgsWarning):
        partial(func)(a=2)


def test_config__unused_func():
    """Test `config.purge` warns if a func configuration isn't used."""
    add({enumerate: Args(start=1)})
    message = "^These configurations were not used:\nenumerate#start$"
    with pytest.warns(UnusedConfigsWarning, match=message):
        purge()


def test_config__unused_arg():
    """Test `config.purge` warns if a func argument configuration isn't used."""
    add({sorted: Args(reverse=False, key=None)})
    sorted([])
    message = "^These configurations were not used:\nsorted#reverse\nsorted#key"
    with pytest.warns(UnusedConfigsWarning, match=message):
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


def test_config__class():
    """Test `config` can handle a class and class functions."""
    add({Obj: Args(a=1), Obj.func: Args(b=2)})
    obj = partial(Obj)()
    assert obj.results == (tuple(), {"a": 1})

    obj = partial(Obj)()
    assert obj.results == (tuple(), {"a": 1})

    result = partial(obj.func)()
    assert result == (tuple(), {"b": 2})

    result = partial(obj.func)()
    assert result == (tuple(), {"b": 2})


def test_config__class_init():
    """Test `config` errors if unbounded method `__init__` method is used."""
    add({Obj.__init__: Args(a=1)})
    with pytest.raises(KeyError):
        partial(Obj)()


def test_config__class_no_init():
    """Test `config` errors if object has no initiation methods."""
    with pytest.raises(KeyError):
        add({NoInitObj: Args(a=1)})


def test_config__decorators():
    """Test `config` unwraps decorators."""
    add({_dec_func: Args(a=1), _dec_other_func: Args(b=2)})
    result = partial(_dec_func)()
    assert result == (tuple(), {"a": 1})
    result = partial(_dec_other_func)()
    assert result == (tuple(), {"b": 2})

    add({_dec_func.__wrapped__: Args(a=3), _dec_other_func.__wrapped__: Args(b=4)}, overwrite=True)
    result = partial(_dec_func)()
    assert result == (tuple(), {"a": 3})
    result = partial(_dec_other_func)()
    assert result == (tuple(), {"b": 4})

    add({_dec_other_func.__wrapped__.__wrapped__: Args(b=5)}, overwrite=True)
    result = partial(_dec_other_func)()
    assert result == (tuple(), {"b": 5})

    with pytest.warns(DiffArgsWarning, match=_diff_args_message(_dec_other_func, "b")):
        assert _dec_other_func(b=6) == (tuple(), {"b": 6})

    assert partial(_dec_other_func)() == (tuple(), {"b": 5})

    add({Obj.dec_func: Args(c=7)})
    obj = Obj()
    result = partial(obj.dec_func)()
    assert result == (tuple(), {"c": 7})


def test_config__dec_class():
    """Test `config` can handle decorated class init."""
    add({DecObj: Args(a=1)})
    obj = partial(DecObj)()
    assert obj.results == (tuple(), {"a": 1})
    assert partial(DecObj)().results == (tuple(), {"a": 1})
    with pytest.warns(DiffArgsWarning, match=_diff_args_message(DecObj, "a")):
        assert DecObj(a=2).results == (tuple(), {"a": 2})


def test_config__new_class():
    """Test `config` can handle class with `__new__` implemented."""
    add({NewObj: Args(a=1, k=2)})
    obj = partial(NewObj)()
    assert obj.results == (tuple(), {"a": 1, "k": 2})
    assert partial(NewObj)().results == (tuple(), {"a": 1, "k": 2})
    with pytest.warns(DiffArgsWarning, match=_diff_args_message(NewObj, "a")):
        assert NewObj(a=3).results == (tuple(), {"a": 3})


def test_config__subclass():
    """Test `config` can handle class with the same `__new__` method implemented because of
    subclassing."""
    config = {OtherObj.func: Args(a=1, k=2), DecObj.func: Args(a=3, k=4)}
    assert len(config) == 1  # NOTE: This a really confusing edge case

    add({OtherObj.func: Args(a=1, k=2)})
    with pytest.raises(ValueError):
        add({DecObj.func: Args(a=3, k=4)})

    results = partial(OtherObj().func)()
    assert results == (tuple(), {"a": 1, "k": 2})
    results = partial(DecObj().func)()
    assert results == (tuple(), {"a": 1, "k": 2})

    config = {OtherObj.__new__: Args(a=1, k=2), DecObj.__new__: Args(a=3, k=4)}
    assert len(config) == 1  # NOTE: This a really confusing edge case
    add({OtherObj: Args(a=1, k=2), DecObj: Args(a=3, k=4)})
    results = partial(OtherObj)().results
    assert results == (tuple(), {"a": 1, "k": 2})
    results = partial(DecObj)().results
    assert results == (tuple(), {"a": 3, "k": 4})


def test_config__var_kwargs():
    """Test `config` can handle if the configured argument isn't passed into a variable key word
    parameter."""
    add({_func: Args(b=1)})
    with pytest.warns(DiffArgsWarning, match=_diff_args_message(_func, "b")):
        assert _func() == (tuple(), {})


def test_config__different_args():
    """Test `config` reports different args and ignores them."""
    add({_func: Args(b=1)})
    with pytest.warns(DiffArgsWarning, match=_diff_args_message(_func, "b")):
        assert _func() == (tuple(), {})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert call(_func, b=2, _overwrite=True) == (tuple(), {"b": 2})

    with pytest.warns(DiffArgsWarning, match=_diff_args_message(_func, "b")):
        assert call(_func, b=2, _overwrite=False) == (tuple(), {"b": 2})


def test_config__different_args__large_numbers():
    """Test `config` handles large numbers that are the same."""
    num = 1000000000000000000000024
    other_num = int("1000000000000000000000024")
    assert id(num) != id(other_num)
    assert num is not other_num
    assert num == other_num

    add({_func: Args(b=num)})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _func(b=other_num) == (tuple(), {"b": other_num})


def test_config__call_inner():
    """Test `config` silences the correct error."""

    def inner(b):
        return b

    def func(a, b):
        return a, inner(b)

    add({func: Args(a=1), inner: Args(b=2)})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert func(1, 2) == (1, 2)

    with pytest.warns(DiffArgsWarning) as record:
        assert func(2, 3) == (2, 3)
        assert len(record) == 2

    with pytest.warns(DiffArgsWarning, match=_diff_args_message(func, "a")) as record:
        assert func(2, 2) == (2, 2)
        assert len(record) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert call(func, a=2, b=2, _overwrite=True) == (2, 2)

    with pytest.warns(DiffArgsWarning, match=_diff_args_message(inner, "b")) as record:
        assert call(func, a=2, b=3, _overwrite=True) == (2, 3)


def test_config__merge_configs():
    """Test `config` merges configs correctly."""
    add({Obj: Args(a=1)})
    add({Obj: Args(b=2)})
    result = get(Obj)
    assert result == {"a": 1, "b": 2}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        add({Obj: Args(a=3)}, overwrite=True)
    result = get(Obj)
    assert result == {"a": 3, "b": 2}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        add({Obj: Args(a=3)}, overwrite=False)

    with pytest.raises(ValueError, match="Trying to overwrite `Obj#a` configuration."):
        add({Obj: Args(a=2)}, overwrite=False)


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


def test_trace():
    """Test `config.trace` can handle a basic case."""
    add({_func: Args(a=1)})
    with pytest.warns(DiffArgsWarning):
        _func()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        partial(_func)()
        purge()
        _func()


def test_trace__repeated_warning():
    """Test `config.trace` doesn't throw repeated warnings."""
    add({_func: Args(a=1)})
    with pytest.warns(DiffArgsWarning) as record:
        for _ in range(10):
            _func()
        assert len(record) == 1

    with pytest.warns(DiffArgsWarning):
        _func()


def test_trace__sys():
    """Test `config.trace` can handle a basic case using sys."""
    enable_fast_trace(False)
    trace_ = sys.gettrace()
    sys.settrace(trace)

    add({_func: Args(a=1)})
    with pytest.warns(DiffArgsWarning):
        _func()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        partial(_func)()
        purge()
        _func()

    sys.settrace(trace_)
