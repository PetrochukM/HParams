import warnings

import pytest

from config.config import purge


@pytest.fixture(autouse=True)
def run_before_test():
    yield

    # Reset the global state after every test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        purge()
