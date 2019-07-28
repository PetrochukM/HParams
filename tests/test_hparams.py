import pytest
import _pytest
import inspect

from unittest import mock

from hparams.hparams import _check_configuration
from hparams.hparams import _merge_args
from hparams.hparams import _parse_configuration
from hparams.hparams import add_config
from hparams.hparams import clear_config
from hparams.hparams import configurable
from hparams.hparams import ConfiguredArg
from hparams.hparams import get_config
from hparams.hparams import log_config


def test_parse_configuration_example():
    # Test a simple case
    parsed = _parse_configuration({'abc.abc': {'cda': 'abc'}})
    assert parsed == {'abc': {'abc': {'cda': 'abc'}}}


def test_parse_configuration_improper_format():
    # Test if the key is improperly formatted, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'abc..abc': 'abc'})


def test_parse_configuration_improper_format_2():
    # Test if the key is improperly formatted, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'abc.abc.': 'abc'})


def test_parse_configuration_improper_format_3():
    # Test if the key is improperly formatted, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'.abc.abc': 'abc'})


def test_parse_configuration_improper_format_4():
    # Test if the key is improperly formatted, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'.': 'abc'})


def test_parse_configuration_duplicate_key():
    # Test if the key is duplicated, TypeError is raised
    with pytest.raises(TypeError):
        _parse_configuration({'abc.abc': 'abc', 'abc': {'abc': 'xyz'}})


@configurable
def mock_configurable(*args, **kwargs):
    # Mock function with configurable
    return kwargs


@configurable
def mock_configurable_limited_args(arg, **kwargs):
    # Mock function with configurable
    return kwargs


def mock_without_configurable(**kwargs):
    # Mock function without configurable
    return kwargs


def test_mock_attributes():
    # Test the attributes mock is give, if it's ``@configurable``
    assert hasattr(mock_configurable, '_configurable')
    assert not hasattr(mock_without_configurable, '_configurable')


_pytest.python_api.approx = configurable(_pytest.python_api.approx)


class Mock(object):

    def __init__(self):
        pass


class MockConfigurable(object):

    @configurable
    def __init__(self):
        pass


def test_mock_configurable_limited_args():
    # Check if TypeError on too many args
    with pytest.raises(TypeError):
        mock_configurable_limited_args('abc', 'abc')


def test_check_configuration_external_libraries():
    # Test that check configuration can check ``configurable`` on external libraries
    _check_configuration({'_pytest': {'python_api': {'approx': {'rel': None}}}})


def test_check_configuration_internal_libraries():
    # Test that check configuration can check ``configurable`` on internal libraries
    _check_configuration({'tests': {'test_hparams': {'mock_configurable': {'kwarg': None}}}})


def test_check_configuration_error_internal_libraries():
    # Test that check configuration fails on internal libraries
    with pytest.raises(TypeError):
        _check_configuration({'tests': {'test_hparams': {'mock': {'kwarg': None}}}})


def test_check_configuration_error_external_libraries():
    # Test that check configuration fails on internal libraries
    with pytest.raises(TypeError):
        _check_configuration({'random': {'seed': {'a': 1}}})


def test_check_configuration_class():
    # Test that check configuration works for classes
    _check_configuration(
        {'tests': {
            'test_hparams': {
                'MockConfigurable': {
                    '__init__': {
                        'kwarg': None
                    }
                }
            }
        }})


def test_check_configuration_error_class():
    # Test that check configuration works for classes
    with pytest.raises(TypeError):
        _check_configuration({'tests': {'test_hparams': {'Mock': {'__init__': {'kwarg': None}}}}})


def test_add_config_and_arguments():
    # Check that a function can be configured
    kwargs = {'xyz': 'xyz'}
    add_config({'tests.test_hparams.mock_configurable': kwargs})
    assert mock_configurable() == kwargs

    # Reset
    clear_config()

    # Check reset worked
    assert mock_configurable() == {}


def test_log_config():
    log_config()


def test_get_config():
    # Check that a function can be configured
    add_config({'tests.test_hparams.mock_configurable': {'xyz': 'xyz'}})
    assert len(get_config())

    # Reset
    clear_config()

    assert len(get_config()) == 0


def mock_configurable_2(arg=ConfiguredArg()):
    pass


@mock.patch('hparams.hparams.logger')
def test_configured_arg_error(logger_mock):
    import logging
    logging.basicConfig(level=logging.INFO)
    global mock_configurable_2

    mock_configurable_2 = configurable(mock_configurable_2)

    # Check the ``ConfiguredArg`` parameter
    logger_mock.reset_mock()
    mock_configurable_2()
    assert logger_mock.warning.call_count == 2

    add_config({'tests.test_hparams.mock_configurable_2.arg': ''})
    # Does not raise error
    mock_configurable_2()

    clear_config()  # Reset config for other tests


def test_configured_arg():
    # Test that configured arg throws an error if you try anything with it
    arg = ConfiguredArg()

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


# Mocks that have never been run before `test_no_config_warning`
@configurable
def mock_configurable_test_no_config_warning(*args, **kwargs):
    return kwargs


class MockConfigurableTestNoConfigWarning(object):

    @configurable
    def __init__(self):
        pass


@mock.patch('hparams.hparams.logger')
def test_no_config_warning(logger_mock):
    clear_config()

    MockConfigurableTestNoConfigWarning()
    logger_mock.warning.assert_called_once()  # No config, warning
    logger_mock.reset_mock()

    mock_configurable_test_no_config_warning()
    logger_mock.warning.assert_called_once()  # No config, warning
    logger_mock.reset_mock()


def test_merge_args_too_many_args():
    arg_kwarg = lambda a, b='abc': (a, b)
    parameters = list(inspect.signature(arg_kwarg).parameters.values())

    # Test too many arguments passed
    with pytest.raises(TypeError):
        _merge_args(parameters, ['a', 'abc', 'one too many'], {}, {'b': 'xyz'})


@mock.patch('hparams.hparams.logger')
def test_merge_arg_kwarg(logger_mock):
    arg_kwarg = lambda a, b='abc': (a, b)
    parameters = list(inspect.signature(arg_kwarg).parameters.values())

    # Prefer ``args`` over ``other_kwargs``
    merged = _merge_args(parameters, ['a', 'abc'], {}, {'b': 'xyz'}, is_first_run=True)
    assert merged == (['a', 'abc'], {})
    logger_mock.warning.assert_called_once()
    logger_mock.reset_mock()

    # Prefer ``kwargs`` over ``other_kwargs``
    merged = _merge_args(parameters, ['a'], {'b': 'abc'}, {'b': 'xyz'}, is_first_run=True)
    assert merged == (['a'], {'b': 'abc'})
    logger_mock.warning.assert_called_once()
    logger_mock.reset_mock()

    # Prefer ``other_kwargs`` over default argument
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz'}, is_first_run=True)
    assert merged == (['a'], {'b': 'xyz'})
    logger_mock.warning.assert_not_called()


@mock.patch('hparams.hparams.logger')
def test_merge_arg_variable(logger_mock):
    """
    For arguments, order matters; therefore, unless we are able to abstract everything into a
    key word argument, we have to keep the ``args`` the same.

    The case where we are unable to shift everything to ``args`` is when there exists a ``*args``.

    For example (a, b) cannot be flipped with kwarg:
    >>> arg_kwarg = lambda a, b='abc': (a, b)
    >>> arg_kwarg('b', a='a')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: <lambda>() got multiple values for argument 'a'
    """
    arg_kwarg = lambda a, *args, b='abc': (a, args, b)
    parameters = list(inspect.signature(arg_kwarg).parameters.values())

    # Handling of variable ``*args``
    merged = _merge_args(parameters, ['a', 'b', 'c'], {}, {'b': 'xyz'}, is_first_run=True)
    assert merged == (['a', 'b', 'c'], {'b': 'xyz'})
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()

    # Handling of variable ``*args``
    merged = _merge_args(parameters, ['a', 'b', 'c'], {}, {'a': 'xyz'}, is_first_run=True)
    assert merged == (['a', 'b', 'c'], {})
    logger_mock.warning.assert_called_once()
    logger_mock.reset_mock()


@mock.patch('hparams.hparams.logger')
def test_merge_kwarg_variable(logger_mock):
    """
    If there exists a ``**kwargs``, then
    """
    arg_kwarg = lambda a, b, **kwargs: (a, b, kwargs)
    parameters = list(inspect.signature(arg_kwarg).parameters.values())

    # Handling of variable ``**kwargs``
    merged = _merge_args(parameters, ['a', 'b'], {}, {'b': 'xyz'}, is_first_run=True)
    assert merged == (['a', 'b'], {})
    logger_mock.warning.assert_called_once()
    logger_mock.reset_mock()

    # Handling of variable ``**kwargs``
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz'}, is_first_run=True)
    assert merged == (['a'], {'b': 'xyz'})
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()

    # Handling of variable ``**kwargs``
    merged = _merge_args(parameters, ['a'], {}, {'b': 'xyz', 'c': 'abc'}, is_first_run=True)
    assert merged == (['a'], {'b': 'xyz', 'c': 'abc'})
    logger_mock.warning.assert_not_called()
    logger_mock.reset_mock()
