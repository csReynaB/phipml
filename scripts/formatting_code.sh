#!/bin/bash

python -m isort src tests
python -m black src tests
python -m isort --check-only --diff src tests
python -m black --check src tests
python -m pytest -q
