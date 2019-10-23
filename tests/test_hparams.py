from functools import wraps
from typing import Union
from unittest import mock

import builtins
import gc
import inspect
import itertools
import os
import sys
import time

import pytest
import _pytest

from hparams.hparams import _function_has_keyword_parameters
from hparams.hparams import _get_function_default_kwargs
from hparams.hparams import _get_function_path
from hparams.hparams import _get_function_signature
from hparams.hparams import _merge_args
from hparams.hparams import _parse_configuration
from hparams.hparams import _resolve_configuration
from hparams.hparams import add_config
from hparams.hparams import clear_config
from hparams.hparams import configurable
from hparams.hparams import get_config
from hparams.hparams import HParam
from hparams.hparams import HParams
from hparams.hparams import log_config
from hparams.hparams import parse_hparam_args

import hparams


def test_hparam():
    """ Test that `HParam` throws an error at runtime if not overridden. """
    arg = HParam()

    with pytest.raises(ValueError):
        str(arg)

    with pytest.raises(ValueError):
        arg.test

    with pytest.raises(ValueError):
        arg == arg

    with pytest.raises(ValueError):
        '' in arg

    with pytest.raises(ValueError):
        {arg: ''}

    with pytest.raises(ValueError):
        arg()

    with pytest.raises(ValueError):
        len(arg)

    with pytest.raises(ValueError):
        repr(arg)

    with pytest.raises(ValueError):
        arg + 1

    with pytest.raises(ValueError):
        arg * 1

    with pytest.raises(ValueError):
        arg - 1


def test__get_function_signature():
    assert _get_function_signature(
        test__get_function_signature) == 'tests.test_hparams.test__get_function_signature'


def test__get_function_signature__no_sys_path():
    """ Test to ensure that `_get_function_signature` can handle an absolute path. """
    original = sys.path
    sys.path = []
    assert 'tests.test_hparams.test__get_function_signature' in _get_function_signature(
        test__get_function_signature__no_sys_path)
    sys.path = original


def test__get_function_path__module():
    assert _get_function_path(hparams.hparams) == 'hparams.hparams'


def test__get_function_path__function():
    assert (_get_function_path(test__get_function_path__function) ==
            'tests.test_hparams.test__get_function_path__function')


def test__get_function_path__local_function():

    def func(arg):
        pass

    assert (_get_function_path(func) ==
            'tests.test_hparams.test__get_function_path__local_function.<locals>.func')


def test__get_function_path__package():
    assert _get_function_path(hparams) == 'hparams'


def test__parse_configuration():
    """ Basic test for `_parse_configuration`. """
    parsed = _parse_configuration({'abc.abc': {'cda': 'abc'}})
    assert parsed == {'abc': {'abc': {'cda': 'abc'}}}


def test__parse_configuration__module():
    """ Test for `_parse_configuration` to resolve a module. """
    parsed = _parse_configuration({hparams.hparams: {'cda': 'abc'}})
    assert parsed == {'hparams': {'hparams': {'cda': 'abc'}}}


def test__parse_configuration__package():
    """ Test for `_parse_configuration` to resolve a package. """
    parsed = _parse_configuration({hparams: {'cda': 'abc'}})
    assert parsed == {'hparams': {'cda': 'abc'}}


def test__parse_configuration__function():
    """ Test for `_parse_configuration` to resolve a function. """
    parsed = _parse_configuration({test__parse_configuration__function: {'cda': 'abc'}})
    assert parsed == {
        'tests': {
            'test_hparams': {
                'test__parse_configuration__function': {
                    'cda': 'abc'
                }
            }
        }
    }


def test__parse_configuration__hparams():
    """ Test if `_parse_configuration` respects `HParams`. """
    parsed = _parse_configuration({'abc': HParams()})
    assert parsed == {'abc': HParams()}
    assert isinstance(parsed['abc'], HParams)


def test__parse_configuration__wrong_type():
    """ Test if the key is not a string, module, or callable a `TypeError` is raised. """
    with pytest.raises(TypeError):
        _parse_configuration({None: 'abc'})


def test__parse_configuration__improper_format():
    """ Test if the key is improperly formatted a `TypeError` is raised. """
    with pytest.raises(TypeError):
        _parse_configuration({'abc..abc': 'abc'})


def test__parse_configuration__improper_format_two():
    """ Test if the key is improperly formatted a `TypeError` is raised. """
    with pytest.raises(TypeError):
        _parse_configuration({'abc.abc.': 'abc'})


def test__parse_configuration__improper_format_three():
    """ Test if the key is improperly formatted a `TypeError` is raised. """
    with pytest.raises(TypeError):
        _parse_configuration({'.abc.abc': 'abc'})


def test__parse_configuration__improper_format_four():
    """ Test if the key is improperly formatted a `TypeError` is raised. """
    with pytest.raises(TypeError):
        _parse_configuration({'.': 'abc'})


def test__parse_configuration__duplicate_key():
    """ Test if the key is duplicated a `TypeError` is raised. """
    with pytest.raises(TypeError):
        _parse_configuration({'abc.abc': 'abc', 'abc': {'abc': 'xyz'}})


def test__resolve_configuration__internal_function():
    """ Test resolution for an internal function. """

    @configurable
    def configured(arg):
        pass

    function_name = ('tests.test_hparams.test__resolve_configuration__internal_function'
                     '.<locals>.configured')
    parsed = _parse_configuration({function_name: HParams()})
    function_signature = _get_function_signature(configured.__wrapped__)
    assert isinstance(_resolve_configuration(parsed)[function_signature], HParams)


def test__resolve_configuration__lambda_function():
    """ Test resolution for an lambda function. """
    configured = configurable(lambda: None)
    function_name = ('tests.test_hparams.test__resolve_configuration__lambda_function'
                     '.<locals>.<lambda>')
    parsed = _parse_configuration({function_name: HParams()})
    function_signature = _get_function_signature(configured.__wrapped__)
    assert isinstance(_resolve_configuration(parsed)[function_signature], HParams)


@configurable
def _test__resolve_configuration__no_sys_path():
    pass


def test__resolve_configuration__no_sys_path():
    """ Test resolution with no `sys` path. """
    original = sys.path
    sys.path = []
    parsed = _parse_configuration(
        {_test__resolve_configuration__no_sys_path.__wrapped__: HParams()})
    function_signature = _get_function_signature(
        _test__resolve_configuration__no_sys_path.__wrapped__)
    assert isinstance(_resolve_configuration(parsed)[function_signature], HParams)
    sys.path = original


@configurable
def _test__resolve_configuration__multiple_sys_path():
    pass


def test__resolve_configuration__multiple_sys_path():
    """ Test resolution for multiple `sys` path. """
    sys.path = [os.path.dirname(__file__)] + sys.path
    function_name = 'test_hparams._test__resolve_configuration__multiple_sys_path'
    parsed = _parse_configuration({function_name: HParams()})
    function_signature = _get_function_signature(
        _test__resolve_configuration__multiple_sys_path.__wrapped__)
    assert isinstance(_resolve_configuration(parsed)[function_signature], HParams)
    sys.path.pop(0)


def test__resolve_configuration__built_in_function():
    """ Test resolution for an built in function. """
    builtins.open = configurable(builtins.open)
    function_name = 'builtins.open'
    parsed = _parse_configuration({function_name: HParams()})
    function_signature = _get_function_signature(builtins.open.__wrapped__)
    assert isinstance(_resolve_configuration(parsed)[function_signature], HParams)


def test__resolve_configuration__internal_class():
    """ Test resolution for an internal class. """

    class Configured:

        @configurable
        def __init__(self):
            pass

    function_name = ('tests.test_hparams.test__resolve_configuration__internal_class.'
                     '<locals>.Configured.__init__')
    parsed = _parse_configuration({function_name: HParams()})
    function_signature = _get_function_signature(Configured.__init__.__wrapped__)
    assert isinstance(_resolve_configuration(parsed)[function_signature], HParams)


def test__resolve_configuration__multiple_configurations():
    """ Test resolution for with multiple configurations. """

    @configurable
    def configured(arg):
        pass

    class Configured:

        @configurable
        def __init__(self):
            pass

    parsed = _parse_configuration({
        'tests.test_hparams.test__resolve_configuration__multiple_configurations': {
            '<locals>.Configured.__init__': HParams(),
            '<locals>.configured': HParams()
        }
    })
    resolved = _resolve_configuration(parsed)

    assert isinstance(resolved[_get_function_signature(Configured.__init__.__wrapped__)], HParams)
    assert isinstance(resolved[_get_function_signature(configured.__wrapped__)], HParams)
    assert len(resolved) == 2


def test__resolve_configuration__external_library():
    """ Test resolution for an external library and with a none-empty `HParams`. """
    pytest.approx = configurable(pytest.approx)
    parsed = _parse_configuration({'pytest.approx': HParams(expected='')})
    function_signature = _get_function_signature(pytest.approx.__wrapped__)
    assert isinstance(_resolve_configuration(parsed)[function_signature], HParams)


def test__resolve_configuration__external_internal_library():
    """ Test resolution for an internal api of an external library. """
    _pytest.python_api.approx = configurable(_pytest.python_api.approx)
    parsed = _parse_configuration({'pytest.approx': HParams(expected='')})
    function_signature = _get_function_signature(pytest.approx.__wrapped__)
    assert isinstance(_resolve_configuration(parsed)[function_signature], HParams)


@configurable
def _test__resolve_configuration__configured_hparam(arg=HParam()):
    pass


def test__resolve_configuration__configured_hparam():
    """ Test resolution for a configured `HParam`. """
    parsed = _parse_configuration(
        {_test__resolve_configuration__configured_hparam: HParams(arg=None)})
    function_signature = _get_function_signature(
        _test__resolve_configuration__configured_hparam.__wrapped__)
    assert isinstance(_resolve_configuration(parsed)[function_signature], HParams)


@configurable
def _test__resolve_configuration__duplicate():
    pass


def test__resolve_configuration__duplicate():
    """ Test resolution for multiple sys path allowing for duplicate configuration. """
    sys.path = [os.path.dirname(__file__)] + sys.path
    parsed = _parse_configuration({
        'test_hparams._test__resolve_configuration__duplicate': HParams(),
        'tests.test_hparams._test__resolve_configuration__duplicate': HParams(),
    })
    with pytest.raises(TypeError):
        _resolve_configuration(parsed)
    sys.path.pop(0)


def test__resolve_configuration__attribute_error():
    """ Test resolution for an none-existant function in an existing module. """
    parsed = _parse_configuration({'pytest.abcdefghijklmnopqrstuvwxyz': HParams()})
    with pytest.raises(TypeError):
        _resolve_configuration(parsed)


def test__resolve_configuration__wrong_arguments():
    """ Test resolution with the a none-existant function argument. """
    parsed = _parse_configuration({'pytest.approx': HParams(abcdefghijklmnopqrstuvwxyz='')})
    with pytest.raises(TypeError):
        _resolve_configuration(parsed)


def test__resolve_configuration__import_error():
    """ Test resolution for an none-existant module. """
    parsed = _parse_configuration({'abcdefghijklmnopqrstuvwxyz': HParams()})
    with pytest.raises(TypeError):
        _resolve_configuration(parsed)


def test__resolve_configuration__no_decorator():
    """ Test resolution for a function that is not decorated. """
    parsed = _parse_configuration({test__resolve_configuration__no_decorator: HParams()})
    with pytest.raises(TypeError):
        _resolve_configuration(parsed)


def test__resolve_configuration__no_configuration():
    """ Test resolution for a function that is not configured. """
    parsed = _parse_configuration({test__resolve_configuration__no_configuration: None})
    with pytest.raises(TypeError):
        _resolve_configuration(parsed)


def test__resolve_configuration__empty():
    """ Test resolution for a function that is not configured. """
    parsed = _parse_configuration({test__resolve_configuration__empty: {}})
    with pytest.raises(TypeError):
        _resolve_configuration(parsed)


def test__function_has_keyword_parameters__empty():
    """ Test if `_function_has_keyword_parameters` handles no argument cases. """

    def func():
        pass

    with pytest.raises(TypeError):
        _function_has_keyword_parameters(func, {'arg': None})

    _function_has_keyword_parameters(func, {})


def test__function_has_keyword_parameters__variable_keyword():
    """ Test if `_function_has_keyword_parameters` handles variable keyword cases. """

    def func(**kwargs):
        pass

    _function_has_keyword_parameters(func, {'arg': None})
    # Ensure that keyword argument `kwargs` is allowed despite variable keyword argument `kwargs`.
    _function_has_keyword_parameters(func, {'kwargs': None})
    _function_has_keyword_parameters(func, {})


def test__function_has_keyword_parameters__variable_positional():
    """ Test if `_function_has_keyword_parameters` handles variable positional cases. """

    def func(*args):
        pass

    with pytest.raises(TypeError):
        _function_has_keyword_parameters(func, {'arg': None})
    with pytest.raises(TypeError):
        # Ensure that keyword argument `args` is not mixed up with variable argument `args`.
        _function_has_keyword_parameters(func, {'args': None})
    _function_has_keyword_parameters(func, {})


def test__function_has_keyword_parameters__none_variable():
    """ Test if `_function_has_keyword_parameters` handles none variable argument cases. """

    def func(arg, kwarg=''):
        pass

    _function_has_keyword_parameters(func, {'arg': None})
    _function_has_keyword_parameters(func, {'kwarg': None})
    _function_has_keyword_parameters(func, {'arg': None, 'kwarg': None})
    _function_has_keyword_parameters(func, {})
    with pytest.raises(TypeError):
        _function_has_keyword_parameters(func, {'other': None})
    with pytest.raises(TypeError):
        _function_has_keyword_parameters(func, {'arg': None, 'kwarg': None, 'other': None})


def test__function_has_keyword_parameters__type_hints():
    """ Test if `_function_has_keyword_parameters` handles checks validates type hints. """

    def func(arg: str, kwarg: Union[str, HParam] = HParam()) -> None:
        pass

    _function_has_keyword_parameters(func, {'arg': ''})
    _function_has_keyword_parameters(func, {'kwarg': ''})
    with pytest.raises(TypeError):
        _function_has_keyword_parameters(func, {'arg': None})

    with pytest.raises(TypeError):
        _function_has_keyword_parameters(func, {'kwarg': None})


def test__function_has_keyword_parameters__less_verbose_type_hints():
    """ Test if `_function_has_keyword_parameters` handles checks validates less verbose type hints.
    """

    def func(kwarg=HParam(str)) -> None:
        pass

    _function_has_keyword_parameters(func, {'kwarg': ''})

    with pytest.raises(TypeError):
        _function_has_keyword_parameters(func, {'kwarg': None})


def test__get_function_default_kwargs__empty():
    """ Test if `_get_function_default_kwargs` handles an empty function. """

    def func():
        pass

    assert list(_get_function_default_kwargs(func).keys()) == []


def test__get_function_default_kwargs__kwarg():
    """ Test if `_get_function_default_kwargs` handles a single kwarg. """

    def func(kwarg=HParam()):
        pass

    assert list(_get_function_default_kwargs(func).keys()) == ['kwarg']
    assert all([isinstance(v, HParam) for v in _get_function_default_kwargs(func).values()])


def test__get_function_default_kwargs__arg_kwarg():
    """ Test if `_get_function_default_kwargs` handles a kwarg, arg, args, and kwargs. """

    def func(arg, *args, kwarg=None, **kwargs):
        pass

    assert list(_get_function_default_kwargs(func).keys()) == ['kwarg']


def test_config_operators():
    """ Test the `log_config`, `clear_config`, `add_config` and `get_config` together. It's
    difficult to test them alone.
    """

    @configurable
    def configured(arg):
        pass

    clear_config()
    assert len(get_config()) == 0
    add_config({configured: HParams()})
    log_config()  # Smoke test
    assert len(get_config()) == 1
    clear_config()
    assert len(get_config()) == 0


def test_merge_configs():
    """ Test the merging of two configurations via `add_config`.
    """

    @configurable
    def configured(arg, arg_two):
        pass

    @configurable
    def other_configured(arg):
        pass

    clear_config()
    add_config({configured: HParams(arg='arg', arg_two='arg_two')})
    add_config({other_configured: HParams()})
    assert len(get_config()) == 2
    assert get_config()[_get_function_signature(configured.__wrapped__)]['arg'] == 'arg'
    add_config({configured: HParams(arg='gra')})
    assert len(get_config()) == 2
    assert get_config()[_get_function_signature(configured.__wrapped__)]['arg'] == 'gra'
    assert get_config()[_get_function_signature(configured.__wrapped__)]['arg_two'] == 'arg_two'
    clear_config()


def test_parse_hparam_args__decimal():
    hparam_args = ['--foo', 'HParams(boo=0.01)']
    assert parse_hparam_args(hparam_args) == {'foo': HParams(boo=0.01)}


def test_parse_hparam_args__string():
    hparam_args = ['--foo', 'HParams(boo="WaveNet")']
    assert parse_hparam_args(hparam_args) == {'foo': HParams(boo='WaveNet')}


def test_parse_hparam_args__equals():
    hparam_args = ['--foo=HParams(boo=1)']
    assert parse_hparam_args(hparam_args) == {'foo': HParams(boo=1)}


def test_parse_hparam_args__nesting():
    hparam_args = ['--moo.foo', 'HParams(boo=1)']
    assert parse_hparam_args(hparam_args) == {'moo.foo': HParams(boo=1)}


def test_parse_hparam_args__exponent():
    hparam_args = ['--foo', 'HParams(boo=10**-6)']
    assert parse_hparam_args(hparam_args) == {'foo': HParams(boo=10**-6)}


def test_parse_hparam_args__list():
    hparam_args = ['--foo', 'HParams(boo=[1,2])']
    assert parse_hparam_args(hparam_args) == {'foo': HParams(boo=[1, 2])}


def test_parse_hparam_args__single_flag():
    hparam_args = ['-foo', 'HParams(boo=10**-6)']
    with pytest.raises(ValueError):
        parse_hparam_args(hparam_args)


def test_parse_hparam_args__no_value():
    hparam_args = ['--foo']
    with pytest.raises(ValueError):
        parse_hparam_args(hparam_args)


def test_parse_hparam_args__no_hparams():
    hparam_args = ['--foo', '0.01']
    with pytest.raises(ValueError):
        parse_hparam_args(hparam_args)


@mock.patch('hparams.hparams.logger')
def test_merge_arg_kwarg(logger_mock):
    """ Test `_merge_args` under the basic case with one argument and one keyword argument.
    """
    lambda_ = lambda a, b='abc': (a, b)
    parameters = list(inspect.signature(lambda_).parameters.values())

    # Prefer `args` over `other_kwargs`
    merged = _merge_args(parameters, ['a', 'abc'], {}, {'b': 'xyz'}, '', True)
    assert merged == (['a', 'abc'], {})
    assert logger_mock.warning.call_count == 1
    logger_mock.reset_mock()

    # Prefer `kwargs` over `other_kwargs`
    merged = _merge_args(parameters, ['a'], {'b': 'abc'}, {'b': 'xyz'}, '', True)
    assert merged == (['a'], {'b': 'abc'})
    assert logger_mock.warning.call_count == 1
    logger_mock.reset_mock()

    # Prefer `other_kwargs` over default argument
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz'}, '', True)
    assert merged == (['a'], {'b': 'xyz'})
    logger_mock.warning.assert_not_called()


@mock.patch('hparams.hparams.logger')
def test_merge_arg_variable(logger_mock):
    """ For arguments, order matters; therefore, unless we are able to abstract everything into a
    keyword argument, we have to keep the `args` the same.

    The case where we are unable to shift everything to `args` is when there exists a `*args`.
    For example (a, b) cannot be flipped with kwarg:
    >>> lambda_ = lambda a, b='abc': (a, b)
    >>> lambda_('b', a='a')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: <lambda>() got multiple values for argument 'a'
    """
    lambda_ = lambda a, *args, b='abc': (a, args, b)
    parameters = list(inspect.signature(lambda_).parameters.values())

    merged = _merge_args(parameters, ['a', 'b', 'c'], {}, {'b': 'xyz'}, '', True)
    assert merged == (['a', 'b', 'c'], {'b': 'xyz'})
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()

    merged = _merge_args(parameters, ['a', 'b', 'c'], {}, {'a': 'xyz'}, '', True)
    assert merged == (['a', 'b', 'c'], {})
    assert logger_mock.warning.call_count == 1
    logger_mock.reset_mock()

    # More arguments than parameters
    merged = _merge_args(parameters, ['a', 'b', 'c', 'd', 'e', 'g'], {}, {'a': 'xyz'}, '', True)
    assert merged == (['a', 'b', 'c', 'd', 'e', 'g'], {})
    assert logger_mock.warning.call_count == 1
    logger_mock.reset_mock()


@mock.patch('hparams.hparams.logger')
def test_merge_kwarg_variable(logger_mock):
    """ Test `_merge_args` under the basic case with a variable keyword argument.
    """
    lambda_ = lambda a, b, **kwargs: (a, b, kwargs)
    parameters = list(inspect.signature(lambda_).parameters.values())

    merged = _merge_args(parameters, ['a', 'b'], {}, {'b': 'xyz'}, '', True)
    assert merged == (['a', 'b'], {})
    assert logger_mock.warning.call_count == 1
    logger_mock.reset_mock()

    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz'}, '', True)
    assert merged == (['a'], {'b': 'xyz'})
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()

    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz', 'c': 'abc'}, '', True)
    assert merged == (['a'], {'b': 'xyz', 'c': 'abc'})
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()


def test_merge_args__too_many_args():
    lambda_ = lambda a, b='abc': (a, b)
    parameters = list(inspect.signature(lambda_).parameters.values())

    # Test too many arguments passed
    with pytest.raises(TypeError):
        _merge_args(parameters, ['a', 'abc', 'one too many'], {}, {'b': 'xyz'}, '', True)


@mock.patch('hparams.hparams.logger')
def test_configurable__no_config(logger_mock):
    """ Test `@configurable` if there is a warning for a missing configuration. """

    @configurable
    def configured():
        pass

    configured()
    assert logger_mock.warning.call_count == 1
    logger_mock.reset_mock()

    # Test that the the warning is only called the first time.
    configured()
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()


@mock.patch('hparams.hparams.logger')
def test_configurable__override(logger_mock):
    """ Test if `@configurable` throws an error on a override of an `HParam` argument that's not
    configured.
    """

    @configurable
    def configured(arg=HParam()):
        return arg

    add_config({configured: HParams()})

    logger_mock.reset_mock()
    configured('a')
    assert logger_mock.warning.call_count == 1

    # It shouldn't throw a second warning
    configured('a')
    assert logger_mock.warning.call_count == 1


@mock.patch('hparams.hparams.logger')
def test_configurable__empty_configuration_warnings(logger_mock):
    """ Test if `@configurable` throws warnings for an empty configuration.
    """

    @configurable
    def configured():
        pass

    add_config({configured: HParams()})

    logger_mock.reset_mock()
    configured()
    assert logger_mock.warning.call_count == 0


def test_configurable__get_partial():
    """ Test if `@configurable#get_configured_partial` is able to export a partial with expected
    configuration.
    """

    @configurable
    def configured(arg):
        return arg

    expected = ''
    add_config({configured: HParams(arg=expected)})
    partial = configured.get_configured_partial()
    assert expected == partial()


def test_configurable__double_invoke():
    """ Test if `@configurable` can be called with no arguments. """

    @configurable()
    def configured(arg):
        return arg

    expected = ''
    add_config({configured: HParams(arg=expected)})
    assert configured() == expected


def test_configurable__unused_hparam():
    """ Test if `@configurable` errors if `HParam` is unused. """

    @configurable
    def configured(arg=HParam()):
        return arg

    with pytest.raises(ValueError):
        configured()

    # Ensure it continues to throw this error
    with pytest.raises(ValueError):
        configured()


def test_configurable__unwrap():
    """ Test if `@configurable` works with the original `configured` path. """

    @configurable()
    def configured(arg):
        return arg

    expected = ''
    add_config({configured.__wrapped__: HParams(arg=expected)})
    assert configured() == expected


def test_add_config__empty():
    """ Test if `add_config` works with an empty config. """
    clear_config()
    add_config({})
    assert {} == get_config()


class __test__resolve_configuration__super_class:

    def __init__(self):
        pass


class _test__resolve_configuration__super_class(__test__resolve_configuration__super_class):
    pass


def test__configurable__regression():
    """ Test if `@configurable` fails with a none-overridden `__init__` function for a global class.

    TODO: Support this use case. Curiously, this works for none-global classes though.
    """

    _test__resolve_configuration__super_class.__init__ = configurable(
        _test__resolve_configuration__super_class.__init__)

    with pytest.raises(TypeError):
        add_config({_test__resolve_configuration__super_class.__init__: HParams()})


def test_configurable__benchmark():
    """ Test if `@configurable` is within the ballpark of a native decorator in performance. """

    def baseline(function):

        @wraps(function)
        def decorator(*args, **kwargs):
            return function(*args, **kwargs)

        return decorator

    lambda_ = lambda arg: arg
    native = baseline(lambda_)
    configured = configurable(lambda_)

    add_config({configured: HParams(arg='')})

    samples = 10000

    # NOTE: The timer decisions are inspired by the native `timeit` module.
    gc.disable()
    iterator = itertools.repeat(None, samples)
    start = time.perf_counter()
    for i in iterator:
        native('')
    native_elapsed = time.perf_counter() - start

    iterator = itertools.repeat(None, samples)
    start = time.perf_counter()
    for i in iterator:
        configured('')
    configured_elapsed = time.perf_counter() - start
    gc.enable()

    assert (configured_elapsed / samples) - (native_elapsed / samples) < 1e-05
