#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# Exit immediately if a command exits with a non-zero status.
set -e

export PYTHONPATH=.

python --version

if [[ "$RUN_FLAKE8" == "true" ]]; then
    flake8 hparams/
    flake8 tests/
fi

run_tests() {
    python -m pytest tests/ hparams/ --verbose --durations=20 --cov=hparams --doctest-modules
}

run_tests
