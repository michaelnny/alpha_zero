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
                legit_actions = np.flatnonzero(env.actions_mask)
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
                legit_actions = np.flatnonzero(env.actions_mask)
                opponent_actions = list(set(legit_actions) - set(win_actions))

                action = np.random.choice(opponent_actions, 1).item()
                opponent_steps += 1

            obs, reward, done, _ = env.step(action)

        self.assertEqual(winner_steps, num_to_win)
        self.assertEqual(env.winner, winner_id)
        self.assertEqual(reward, 1.0)


class GomokuStackHistoryTest(parameterized.TestCase):
    @parameterized.named_parameters(('black_to_player', 0), ('white_to_player', 1))
    def test_observation_state_tensor_history(self, player_id):
        black_actions = [0, 1, 7, 8, 21, 22, 28, 29]
        white_actions = [2, 3, 9, 10, 23, 24, 30, 31]

        env = GomokuEnv(board_size=7, stack_history=8)

        # Overwrite current player.
        env.current_player = player_id

        obs = env.reset()

        while env.steps < env.stack_history * 2:
            if env.current_player == env.black_player:
                action = black_actions[env.steps // 2]
            else:
                action = white_actions[env.steps // 2]
            obs, reward, done, _ = env.step(action)

        # Compute expected planes for both black and white players
        expected_black_planes = []
        expected_white_planes = []

        for action in black_actions:
            if len(expected_black_planes) > 0:
                plane = np.copy(expected_black_planes[-1])
            else:
                plane = np.zeros((7, 7), dtype=np.int8)
            x, y = env.action_to_coords(action)
            plane[x, y] = 1
            expected_black_planes.append(plane)

        for action in white_actions:
            if len(expected_white_planes) > 0:
                plane = np.copy(expected_white_planes[-1])
            else:
                plane = np.zeros((7, 7), dtype=np.int8)
            x, y = env.action_to_coords(action)
            plane[x, y] = 1
            expected_white_planes.append(plane)

        # Note the newest state is one top, so we reverse before stack.
        expected_black_planes.reverse()
        expected_white_planes.reverse()

        if env.current_player == env.black_player:
            planes = []
            for t in range(env.stack_history):
                planes.append(expected_black_planes[t])
                planes.append(expected_white_planes[t])

            color_to_play = np.ones((1, 7, 7), dtype=np.int8)
        else:
            planes = []
            for t in range(env.stack_history):
                planes.append(expected_white_planes[t])
                planes.append(expected_black_planes[t])

            color_to_play = np.zeros((1, 7, 7), dtype=np.int8)

        planes = np.stack(planes, axis=0)
        features = np.concatenate([planes, color_to_play], axis=0)

        np.testing.assert_equal(obs, features)


if __name__ == '__main__':
    absltest.main()
