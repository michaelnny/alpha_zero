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

from alpha_zero.games.env import BoardGameEnv


class BoardGameEnvTest(parameterized.TestCase):
    @parameterized.named_parameters(('board_size_7', 7), ('board_size_9', 9))
    def test_env_setup(self, board_size):
        env = BoardGameEnv(board_size)
        env.reset()

        np.testing.assert_equal(env.board, np.zeros((board_size, board_size), dtype=np.uint8))
        self.assertEqual(env.current_player, env.black_player_id)  # black
        self.assertEqual(env.opponent_player, env.white_player_id)  # white

    @parameterized.named_parameters(('action_-1', -1), ('action_50', 50))
    def test_invalid_action_out_of_range(self, action):
        env = BoardGameEnv(board_size=7)
        env.reset()

        with self.assertRaisesRegex(ValueError, 'Invalid action'):
            env.step(action)

    @parameterized.named_parameters(('board_size_7', 7), ('board_size_9', 9))
    def test_resign_action(self, board_size):
        env = BoardGameEnv(board_size=board_size)

        resign_action = int(board_size**2)
        env.reset()
        obs, reward, done, _ = env.step(resign_action)

        self.assertEqual(resign_action, env.resign_action)
        self.assertTrue(done)
        self.assertEqual(reward, -1.0)

        self.assertEqual(env.loser, env.black_player_id)  # black
        self.assertEqual(env.winner, env.white_player_id)  # white

    @parameterized.named_parameters(('action_7', 7), ('action_13', 13))
    def test_invalid_action_already_taken(self, action):
        env = BoardGameEnv(board_size=7)
        env.reset()

        env.step(action)
        with self.assertRaisesRegex(ValueError, 'Invalid action'):
            env.step(action)

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
    @parameterized.named_parameters(('stack_4', 4), ('stack_9', 9))
    def test_env_tensor_state(self, stack_history):
        board_size = 7
        env = BoardGameEnv(board_size=board_size, stack_history=stack_history)
        obs = env.reset()

        zero_planes = np.zeros((stack_history * 2, board_size, board_size), dtype=np.uint8)
        player_plane = np.ones((1, board_size, board_size), dtype=np.uint8)  # Black plays first.
        np.testing.assert_equal(obs, np.concatenate([zero_planes, player_plane]))


if __name__ == '__main__':
    absltest.main()
