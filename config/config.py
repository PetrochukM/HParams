import builtins
import functools
import inspect
import operator
import traceback
import types
import typing
import warnings


class Params(dict):
    pass


# Learn more about special methods:
# https://stackoverflow.com/questions/21887091/cant-dynamically-bind-repr-str-to-a-class-created-with-type
# https://stackoverflow.com/questions/1418825/where-is-the-python-documentation-for-the-special-methods-init-new
builtin_types = [
    getattr(builtins, d) for d in dir(builtins) if isinstance(getattr(builtins, d), type)
]
SPECIAL_METHODS = set(m for t in builtin_types + [operator] for m in dir(t) if m[:2] == "__")
_OTHER_SPECIAL_METHODS = {
    "__div__",
    "__copy__",
    "__long__",
    "__deepcopy__",
    "__complex__",
    "__cmp__",
    "__oct__",
    "__hex__",
    "__idiv__",
}
assert len(SPECIAL_METHODS.intersection(_OTHER_SPECIAL_METHODS)) == 0
SPECIAL_METHODS = SPECIAL_METHODS.union(_OTHER_SPECIAL_METHODS)


class Placeholder:
    """Temporary object that serves as a placeholder for another object. If this object is used in
    any way, it will error.

    Args:
        type_ (typing, optional): The object type this is placeholding for.
    """

    def __init__(self, type_: type = typing.Any):
        self.type = type_

        stack = traceback.extract_stack(limit=2)[-2]
        self._lineno = f"{stack.filename}:{stack.lineno}"

        for attribute in SPECIAL_METHODS - {"__getattribute__"}:
            try:
                partial = functools.partial(self._raise, attribute=attribute)
                setattr(self.__class__, attribute, partial)
            except (TypeError, AttributeError):
                continue

    def _raise(self, *_, attribute: str, **__):
        raise ValueError(
            f"Oops. The `{self.__class__.__name__}` object attribute `{attribute}` was called. This"
            "object should only be used as a placeholder for a different object likely defined at"
            f"{self._lineno}."
        )

    def __getattribute__(self, name: str):
        if name in ["_lineno", "_raise", "type", "__dict__", "__class__"]:
            return super().__getattribute__(name)
        self._raise(attribute=name)


_config: typing.Dict[typing.Callable, Params] = {}


def clear():
    """Clear the global configuration.

    Side Effects:
        The existing global configuration is reset to it's initial state.
    """
    global _config
    _config = {}


def get() -> typing.Dict[typing.Callable, Params]:
    """Get the current global configuration.

    Anti-Patterns:
        It would be an anti-pattern to use this to set the configuration.

    Returns:
        (dict): The current configuration.
    """
    return _config


def _merge_args(
    func: typing.Callable,
    args: typing.List,
    kwargs: typing.Dict,
    config_kwargs: typing.Dict,
    default_kwargs: typing.Dict,
) -> typing.Dict:
    """Merge `args`, `kwargs`, `config_kwargs`, and `default_kwargs` with special handling for
    `Placeholder` objects.

    Args:
        func: Function.
        args: Arguments for function.
        kwargs: Keyword arguments for function.
        config_kwargs: Additional keyword arguments for function.
        default_kwargs: Default keyword arguments for function.

    Returns:
        Keyword arguments that merge `args`, `kwargs`, `config_kwargs`, and `default_kwargs`.
    """
    POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL

    params = list(inspect.signature(func).parameters)
    merged_kwargs = default_kwargs.copy()
    merged_kwargs.update(config_kwargs)

    # NOTE: Delete `merged_kwargs` that conflict with `args`.
    # NOTE: Positional arguments must come before keyword arguments.
    for i, arg in enumerate(args):
        if i >= len(params):
            raise TypeError(f"Too many arguments ({len(args)} > {len(params)}) passed.")

        param = params[i]

        if param.kind == VAR_POSITIONAL:
            break  # NOTE: Rest of the args are absorbed by VAR_POSITIONAL (e.g. `*args`)

        is_positional = param.kind == POSITIONAL_ONLY or param.kind == POSITIONAL_OR_KEYWORD

        if (
            is_positional
            and param.name in merged_kwargs
            and (param.name in config_kwargs or isinstance(merged_kwargs[param.name], Placeholder))
        ):
            # NOTE: This uses ``warnings`` based on these guidelines:
            # https://stackoverflow.com/questions/9595009/python-warnings-warn-vs-logging-warning/14762106
            message = (
                f"Overwriting configured argument `{param.name}={str(merged_kwargs[param.name])}` "
            )
            warnings.warn(message + f"in module `{func.__qualname__}` with `{arg}`.")
        del merged_kwargs[param.name]

    for key, value in kwargs.items():
        if key in config_kwargs or (
            key in merged_kwargs and isinstance(merged_kwargs[key], Placeholder)
        ):
            message = f"Overwriting configured argument `{key}={str(merged_kwargs[key])}` "
            warnings.warn(message + f"in module `{func.__qualname__}` with `{value}`.")

    merged_kwargs.update(kwargs)
    return args, merged_kwargs


def _copy_func(func: typing.Callable) -> typing.Callable:
    """Create an new instance of `func`.

    Based on:
    https://stackoverflow.com/questions/13503079/how-to-create-a-copy-of-a-python-function
    """
    copy_ = types.FunctionType(
        func.__code__,
        func.__globals__,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )
    copy_ = functools.update_wrapper(copy_, func)
    copy_.__kwdefaults__ = func.__kwdefaults__
    return copy_


def _wrapper(func: typing.Callable):
    """Decorator enables configuring module arguments.

    Decorator enables one to set the arguments of a module via a global configuration. The decorator
    also stores the parameters the decorated function was called with.

    Args:
        None

    Returns:
        (callable): Decorated function
    """

    @functools.wraps(func)
    def decorator(
        *args,
        ___func: typing.Callable = func,
        ___func_copy: typing.Callable = _copy_func(func),
        ___config: typing.Dict[typing.Callable, Params] = _config,
        ___Placeholder=Placeholder,
        **kwargs,
    ):
        import warnings

        if ___func not in ___config:
            warnings.warn("@configurable: No config for `%s`. " % (___func.__qualname__,))

        args, kwargs = _merge_args(
            params, args, kwargs, ___config[___func], function_default_kwargs, function_signature
        )

        # Ensure all `Placeholder` objects are overridden.
        for arg in [a for v in [args, kwargs.values()] for a in v]:
            if isinstance(arg, ___Placeholder):
                arg._raise()

        return ___func_copy(*args, **kwargs)

    # Add a flag to the func; enabling us to check if a function has the configurable decorator.
    decorator._wrapped = True

    return decorator


def _wrap_func_in_place(func: typing.Callable) -> typing.Callable:
    if hasattr(func, "_wrapped"):
        return func

    wrapped = _wrapper(func)
    func.__code__ = wrapped.__code__


def add(config: typing.Dict[typing.Callable, Params]):
    """Configure."""
    # TODO: Why do we need __code__?
    # We could check that all `Placeholders` are overwritten with `__defaults__`
    # We can get the function configurations...
    # We DON'T KNOW if a default is overwritten, which is fine, maybe? So, if, a configuration
    # which is intended to be used, is overwritten, we'd have no idea...
    # We can add a partial export
    # We can add a _configurable flag
    # Let's use __code__
    # TODO: How do we make this work with multiple processes? Will we still need a unique function
    # signature? That's fine... I just don't want to parse strings, and import them... I guess, I'll
    # still need to...
    # Okay, we need to figure out, can we pickle a refernece, and how do we reinstantiate the config
    # in another process...

    # NOTE: We're looking to dramatically simplify this code base, basically, the configuration
    # is a map from functions to parameters. The function isn't wrapped with a decorator, instead,
    # we actually change the code of the function to do a look up in the _config dictionary.
    # NOTE: We want to get a quick read on if this concept works before we flesh it out,
    # so one of the last things I was working on is testing the multiprocessing aspects of it?
    # - The worry with multiprocessing is that it creates a new version of the code base,
    # so we'd need to reinstantiate the configuration module, the issues are...
    # Can we pickle the config file if it's not stored as strings?
    # NOTE: Does it all matter? I think, if we can't pickle it, than we can recreate it, just
    # fine.
    # TODO: Let's just start cleaning up all this, writing tests, documentation, etc. We'll
    # 100% leave room for multiprocessing tests
    global _config
    [_wrap_func_in_place(k) for k in config]
    _config = {**_config, **config}
