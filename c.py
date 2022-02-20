import ast
import functools
import sys
import textwrap

import executing


def _get_child_to_parent_map(root):
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


message = (
    "Unable to determine the calling function and argument name."
    + "\n"
    + "\n"
    + textwrap.fill(
        "This uses Python AST to parse the code and retrieve the calling function and argument "
        "name. In order to do so, they need to be both explicitly named, like so:"
    )
    + "\n"
    + "✅ function(arg=get_calling_func())\n"
    + "❌ function(get_calling_func())\n"
    + "❌ func = lambda: function\n"
    + "   func()(arg=get_calling_func())\n"
)


def get_calling_func(stack=1):
    """Get the calling function that executes this code to get it's input.

    For example:
        >>> def func(a):
        ...      pass
        ...
        >>> func((caller := get_calling_func()))
        >>> caller
        func
    """
    frame = sys._getframe(stack)
    executing_ = executing.Source.executing(frame)
    assert len(executing_.statements) == 1
    root = next(iter(executing_.statements))
    parents = _get_child_to_parent_map(root)
    parent = parents[executing_.node]
    if not isinstance(parent, ast.keyword):
        raise SyntaxError("Unable to find keyword.")
    parent = parents[parent]
    if not isinstance(parent, ast.Call):
        raise SyntaxError("Unable to find calling function.")
    if not hasattr(parent.func, "id"):
        raise SyntaxError("Calling function is anonymous.")
    name = parent.func.id
    func = frame.f_globals[name]
    if func == functools.partial:
        pass
    return func


def _identity(*a, **kw):
    return (a, kw)


def _other_identity(*a, **kw):
    return _identity(*a, **kw)


class Identity:
    def __init__(self, *a, **kw) -> None:
        self.init_result = _identity(*a, **kw)

    def identity(self, *a, **kw):
        return _identity(*a, **kw)

    def __call__(self, *a, **kw):
        return _identity(*a, **kw)

    def __str__(self, *a, **kw):
        self.str_result = _identity(*a, **kw)
        return ""


# TODO: Instead of relying on name, what if we rely on lineno? and then name also?
# print("parent.func", ast.dump(parent, indent=2))
# TODO: We also need to figure out... which parameter, not just, which function, shit.
# NOTE: This would be so much easier with `sys.profile`

# TODO: Test Lambda
# TODO: Test built in
# TODO: Test external library
# TODO: Test partial
# TODO: Test decorator
# TODO: Test function attribute, similar to class?
# TODO: What about a function which returns another function?
# TODO: I think the solution is a combination of both, basically, we pass in a CONFIGURED argument
# to a function, and then a decorator finds the arguments, and replaces them appropriately?
# NOTE: Even the decorater, doesn't know the parameter names, because they haven't been assigned
# yet, except the kwargs?
# TODO: What if we make it simple, and explict, so we don't support fancy use cases? It has
# to use kwargs, it'd be hard without partials, but we can use lambdas.
# - We could replace
# TODO: Another option, which is robust, is just get some sort of `Pyright` or `mypy` code analysis
# going.
# TODO: And if someone really needs to, they can just pass the func and argument name, and it'll
# grab the right one?

# OPTIONS
# - Set `setprofile` so that we know when a function is entered, and we can quickly exchange
# the placeholders
#    - This is slow
#    - This is the cleanest approach
# - Inject a decorator on top of the function, parse the *args and **kwargs, and exchange them
#    - This requires some funky code injection
#    - This requires some funky arguement parsing
#    - This could be a bit slow because we are injecting code around functions
#         - We could do more explicit decorators but that creates a lot of coding overhead
# - Use AST to parse the code, and find the correct argument
#    - This requires parsing the entire code base, unless, we just implement this in a limited
#      context, which is potentially really nice, explicit, fast. Unfortunately, we won't cover
#      allow our bases. Basically, we are requiring explicit function calls + kwargs.
# - We could... combine the decorator approach + the set profile approach together, where basically
#   we `settrace` with our injected code, haha, and then turn it off. This might still be slow.
#   The implementation is embarassing to explain, and it seems like magic, because I'm passing in
#   some sort of global, which is then automagically replaced in the frame or by the magic decorator
#   Let's just do AST, and, we'll be in a good spot.
# https://stackoverflow.com/questions/44758862/get-types-in-python-expression-using-mypy-as-library
# TODO: We can't always know what function is returned or called, because, it might be the
# result of funky if statements...


def test_get_calling_func():
    assert _identity(get_calling_func()) == ((_identity,), {})


def test_get_calling_func__multiple_lines_and_assignment():
    result = _identity(
        get_calling_func(),
        get_calling_func(),
        a=get_calling_func(),
    )
    assert result == ((_identity, _identity), {"a": _identity})


def test_get_calling_func__semicolons():
    # fmt: off
    result = _identity(get_calling_func()); other_result = _other_identity(get_calling_func())
    # fmt: on
    assert result == ((_identity,), {})
    assert other_result == ((_other_identity,), {})


# def test_get_calling_func__return_func():
#     def func():
#         return _identity

#     assert func()(get_calling_func()) == ((_identity,), {})


# def test_get_calling_func__object():
#     assert Identity(get_calling_func()).result == ((Identity,), {})


# def test_get_calling_func__object_call():
#     assert Identity(None)(get_calling_func()) == ((Identity.__call__,), {})


if __name__ == "__main__":
    items = list(locals().items())
    for key, value in items:
        if callable(value) and value.__module__ == __name__:
            if key.startswith("test_"):
                print(f">>> Running test: {key}...")
                value()
                print(f">>> Passed `{key}`! 🎉")
