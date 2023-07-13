#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run unit tests

python3 -m unit_tests.envs.base_test
python3 -m unit_tests.envs.gomoku_test
python3 -m unit_tests.envs.go_test
python3 -m unit_tests.transformation_test
