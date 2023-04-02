# Copyright 2022 Michael Hu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for games.gomoku.py."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from alpha_zero.games.gomoku import GomokuEnv


class GomokuEnvTest(parameterized.TestCase):
    # player id: 1 - black, 2 - white
    @parameterized.named_parameters(
        ('diagonal_top_left_00_black', 1, [0, 8, 16, 24, 32]),
        ('diagonal_top_left_00_white', 2, [0, 8, 16, 24, 32]),
        ('diagonal_top_left_12_black', 1, [9, 17, 25, 33, 41]),
        ('diagonal_top_left_12_white', 2, [9, 17, 25, 33, 41]),
        ('diagonal_top_right_06_black', 1, [6, 12, 18, 24, 30]),
        ('diagonal_top_right_06_white', 2, [6, 12, 18, 24, 30]),
        ('diagonal_top_right_26_black', 1, [20, 26, 32, 38, 44]),
        ('diagonal_top_right_26_white', 2, [20, 26, 32, 38, 44]),
        ('vertical_00_black', 1, [0, 7, 14, 21, 28]),
        ('vertical_00_white', 2, [0, 7, 14, 21, 28]),
        ('vertical_11_black', 1, [8, 15, 22, 29, 36]),
        ('vertical_11_white', 2, [8, 15, 22, 29, 36]),
        ('horizontal_00_black', 1, [0, 1, 2, 3, 4]),
        ('horizontal_00_white', 2, [0, 1, 2, 3, 4]),
        ('horizontal_52_black', 1, [37, 38, 39, 40, 41]),
        ('horizontal_52_white', 2, [37, 38, 39, 40, 41]),
    )
    # diagonal_top_left_00:
    # [1 0 0 0 0 0 0]
    # [0 1 0 0 0 0 0]
    # [0 0 1 0 0 0 0]
    # [0 0 0 1 0 0 0]
    # [0 0 0 0 1 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]

    # diagonal_top_left_12:
    # [0 0 0 0 0 0 0]
    # [0 0 1 0 0 0 0]
    # [0 0 0 1 0 0 0]
    # [0 0 0 0 1 0 0]
    # [0 0 0 0 0 1 0]
    # [0 0 0 0 0 0 1]
    # [0 0 0 0 0 0 0]

    # diagonal_top_right_06:
    # [0 0 0 0 0 0 1]
    # [0 0 0 0 0 1 0]
    # [0 0 0 0 1 0 0]
    # [0 0 0 1 0 0 0]
    # [0 0 1 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]

    # diagonal_top_right_26:
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 1]
    # [0 0 0 0 0 1 0]
    # [0 0 0 0 1 0 0]
    # [0 0 0 1 0 0 0]
    # [0 0 1 0 0 0 0]

    # vertical_00:
    # [1 0 0 0 0 0 0]
    # [1 0 0 0 0 0 0]
    # [1 0 0 0 0 0 0]
    # [1 0 0 0 0 0 0]
    # [1 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]

    # vertical_11:
    # [0 0 0 0 0 0 0]
    # [0 1 0 0 0 0 0]
    # [0 1 0 0 0 0 0]
    # [0 1 0 0 0 0 0]
    # [0 1 0 0 0 0 0]
    # [0 1 0 0 0 0 0]
    # [0 0 0 0 0 0 0]

    # horizontal_00:
    # [1 1 1 1 1 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]

    # horizontal_52:
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0]
    # [0 0 1 1 1 1 1]
    # [0 0 0 0 0 0 0]

    def test_score_and_winner(self, winner_id, win_actions):
        env = GomokuEnv(board_size=7)
        obs = env.reset()

        done = False
        winner_steps = 0

        while not done:
            if env.current_player == winner_id:
                action = win_actions[winner_steps]
                winner_steps += 1
            else:
                # Opponent should not take win_actions
                legit_actions = np.flatnonzero(env.legal_actions)
                opponent_actions = list(set(legit_actions) - set(win_actions))
                action = np.random.choice(opponent_actions, 1).item()

            obs, reward, done, _ = env.step(action)

        self.assertEqual(winner_steps, 5)
        self.assertEqual(env.winner, winner_id)
        self.assertEqual(reward, 1.0)

    @parameterized.named_parameters(
        ('num_3_black', 1, 3),
        ('num_3_white', 2, 3),
        ('num_4_black', 1, 4),
        ('num_4_white', 2, 4),
    )
    def test_num_to_win(self, winner_id, num_to_win):
        env = GomokuEnv(board_size=7, num_to_win=num_to_win)
        obs = env.reset()

        win_actions = [0, 8, 16, 24, 32]
        winner_steps = 0
        opponent_steps = 0
        done = False

        while not done:
            if env.current_player == winner_id:
                action = win_actions[winner_steps]
                winner_steps += 1
            else:
                # Opponent should not take win_actions
                legit_actions = np.flatnonzero(env.legal_actions)
                opponent_actions = list(set(legit_actions) - set(win_actions))

                action = np.random.choice(opponent_actions, 1).item()
                opponent_steps += 1

            obs, reward, done, _ = env.step(action)

        self.assertEqual(winner_steps, num_to_win)
        self.assertEqual(env.winner, winner_id)
        self.assertEqual(reward, 1.0)

if __name__ == '__main__':
    absltest.main()
