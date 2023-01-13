#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run unit tests

python3 -m tests.games.boardgame_test
python3 -m tests.games.gomoku_test
python3 -m tests.games.tictactoe_test
python3 -m tests.mcts_v1_test
python3 -m tests.mcts_v2_test
python3 -m tests.data_processing_test
python3 -m tests.rating_test
