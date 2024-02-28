# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


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
"""Tests for games.env.py."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from alpha_zero.envs.base import BoardGameEnv


class BoardGameEnvTest(parameterized.TestCase):
    @parameterized.named_parameters(('board_size_7', 7), ('board_size_9', 9))
    def test_env_setup(self, board_size):
        env = BoardGameEnv(board_size)
        env.reset()

        np.testing.assert_equal(env.board, np.zeros((board_size, board_size), dtype=np.uint8))
        self.assertEqual(env.to_play, env.black_player)  # black
        self.assertEqual(env.opponent_player, env.white_player)  # white

    @parameterized.named_parameters(('action_-1', -1), ('action_50', 50))
    def test_invalid_action_out_of_range(self, action):
        env = BoardGameEnv(board_size=7)
        env.reset()

        with self.assertRaisesRegex(ValueError, 'Invalid action'):
            env.step(action)

    @parameterized.named_parameters(('action_7', 7), ('action_13', 13))
    def test_invalid_action_already_taken(self, action):
        env = BoardGameEnv(board_size=7)
        env.reset()

        env.step(action)
        with self.assertRaisesRegex(ValueError, 'Illegal action'):
            env.step(action)

    @parameterized.named_parameters(
        ('onboard_(0,0)', 7, (0, 0), True),
        ('onboard_(0,6)', 7, (0, 6), True),
        ('onboard_(6,6)', 7, (6, 6), True),
        ('onboard_(6,0)', 7, (6, 0), True),
        ('onboard_(3,4)', 7, (3, 4), True),
        ('not_onboard_(-1,6)', 7, (-1, 6), False),
        ('not_onboard_(0,7)', 7, (0, 7), False),
        ('onboard_(0,7)', 9, (0, 7), True),
    )
    def test_is_coords_on_board(self, board_size, coords, expected_result):
        env = BoardGameEnv(board_size=board_size)
        env.reset()

        self.assertEqual(env.is_coords_on_board(coords), expected_result)

    def test_action_to_coords(self):
        env = BoardGameEnv(board_size=7)
        env.reset()

        actions = [0, 3, 6, 42, 48]
        coords = [(0, 0), (0, 3), (0, 6), (6, 0), (6, 6)]

        for action, coord in zip(actions, coords):
            result_coords = env.action_to_coords(action)
            self.assertEqual(coord, result_coords)

    def test_coords_to_action(self):
        env = BoardGameEnv(board_size=7)
        env.reset()

        actions = [0, 3, 6, 42, 48]
        coords = [(0, 0), (0, 3), (0, 6), (6, 0), (6, 6)]

        for action, coord in zip(actions, coords):
            result_action = env.coords_to_action(coord)
            self.assertEqual(result_action, action)

    @parameterized.named_parameters(
        ('gtp_9_A9', 9, 'A9', 0),
        ('gtp_9_J9', 9, 'J9', 8),
        ('gtp_9_A1', 9, 'A1', 72),
        ('gtp_9_J1', 9, 'J1', 80),
        ('gtp_9_C8', 9, 'C8', 11),
        ('gtp_19_A19', 19, 'A19', 0),
        ('gtp_19_T19', 19, 'T19', 18),
        ('gtp_19_A1', 19, 'A1', 342),
        ('gtp_19_T1', 19, 'T1', 360),
        ('gtp_19_C18', 19, 'C18', 21),
    )
    def test_gtp_to_action(self, board_size, gtpc, expected_action):
        env = BoardGameEnv(board_size=board_size)
        env.reset()

        self.assertEqual(env.gtp_to_action(gtpc), expected_action)

    def test_env_render(self):
        env = BoardGameEnv(board_size=7)
        env.reset()

        for i in range(0, 5):
            env.render()
            obs, reward, done, _ = env.step(i)
            if done:
                env.render()
                break


class StackHistoryTest(parameterized.TestCase):
    @parameterized.named_parameters(('stack_4', 4), ('stack_8', 8))
    def test_env_initial_state(self, num_stack):
        board_size = 7
        env = BoardGameEnv(board_size=board_size, num_stack=num_stack)
        obs = env.reset()

        zero_planes = np.zeros((num_stack * 2, board_size, board_size), dtype=np.uint8)
        player_plane = np.ones((1, board_size, board_size), dtype=np.uint8)  # Black plays first.
        np.testing.assert_equal(obs, np.concatenate([zero_planes, player_plane]))

    def test_env_stacked_state(self):
        env = BoardGameEnv(num_stack=8)
        obs = env.reset()

        empty_board = np.copy(env.board)
        np.testing.assert_equal(empty_board, 0)

        # Define moves
        black_moves = [0, 1, 2, 3]
        white_moves = [5, 6, 7, 8]

        # This is really a mess
        expected = np.zeros((8 * 2, env.board_size, env.board_size), dtype=np.int8)

        for i in range(len(black_moves)):
            j = 0 if i == 0 else 4 * i
            for move in black_moves[: -i or None]:  # Get all elements if i =0
                _coords = env.action_to_coords(move)
                expected[j][_coords] = 1
                expected[j + 2][_coords] = 1

        for i in range(len(white_moves)):
            j = 1 if i == 0 else 4 * i - 1
            for move in white_moves[: -i or None]:  # Get all elements if i =0
                _coords = env.action_to_coords(move)
                expected[j][_coords] = 1
                # White is one step behind
                if j > 1:
                    expected[j + 2][_coords] = 1

        # Black to play
        color_to_play = np.ones((1, env.board_size, env.board_size), dtype=np.int8)
        expected = np.concatenate([expected, color_to_play], axis=0)

        # Let each player makes some moves
        for b_move, w_move in zip(black_moves, white_moves):
            obs, _, _, _ = env.step(b_move)
            obs, _, _, _ = env.step(w_move)

        self.assertTrue(np.array_equal(obs, expected))


if __name__ == '__main__':
    absltest.main()
