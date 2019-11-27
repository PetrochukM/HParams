from hparams.hparams import clear_config
from hparams.hparams import set_lazy_resolution

import pytest


@pytest.fixture(autouse=True)
def run_before_test():
    # Reset the global state before every test
    clear_config()
    set_lazy_resolution(False)
    yield
