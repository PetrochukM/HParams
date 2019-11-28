""" A module outside of `tests` used for testing `hparams`. """

from hparams import configurable
from hparams import HParam


@configurable
def configured(arg=HParam()):
    return arg
