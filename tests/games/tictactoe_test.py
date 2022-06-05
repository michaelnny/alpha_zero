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
"""Tests for games.tictactoe.py."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from alpha_zero.games.tictactoe import TicTacToeEnv


class TicTacToeEnvTest(parameterized.TestCase):
    # player id: 1 - black, 2 - white
    @parameterized.named_parameters(
        ('diagonal_top_left_00_black', 1, [0, 4, 8]),
        ('diagonal_top_left_00_white', 2, [0, 4, 8]),
        ('diagonal_top_right_02_black', 1, [2, 4, 6]),
        ('diagonal_top_right_02_white', 2, [2, 4, 6]),
        ('vertical_00_black', 1, [0, 3, 6]),
        ('vertical_00_white', 2, [0, 3, 6]),
        ('horizontal_00_black', 1, [0, 1, 2]),
        ('horizontal_00_white', 2, [0, 1, 2]),
    )
    # diagonal_top_left_00:
    # [1 0 0]
    # [0 1 0]
    # [0 0 1]

    # diagonal_top_right_02:
    # [0 0 1]
    # [0 1 0]
    # [1 0 0]

    # vertical_00:
    # [1 0 0]
    # [1 0 0]
    # [1 0 0]

    # horizontal_00:
    # [1 1 1]
    # [0 0 0]
    # [0 0 0]

    def test_score_and_winner(self, winner_id, win_actions):
        env = TicTacToeEnv()
        obs = env.reset()

        done = False
        winner_steps = 0

        while not done:
            if env.current_player == winner_id:
                action = win_actions[winner_steps]
                winner_steps += 1
            else:
                # Opponent should not take win_actions and resign action
                legit_actions = np.flatnonzero(env.actions_mask)
                opponent_actions = list(set(legit_actions) - set(win_actions) - set([env.resign_action]))

                action = np.random.choice(opponent_actions, 1).item()

            obs, reward, done, _ = env.step(action)

        self.assertEqual(env.winner, winner_id)
        self.assertEqual(reward, 1.0)


if __name__ == '__main__':
    absltest.main()
