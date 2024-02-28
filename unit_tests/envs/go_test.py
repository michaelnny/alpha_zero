# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Tests for cleargo.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import os

BOARD_SIZE = 19
STACK_HISTORY = 8
os.environ['BOARD_SIZE'] = str(BOARD_SIZE)


from alpha_zero.envs.go import GoEnv
import alpha_zero.envs.go_engine as go


class RunGoEnvTest(parameterized.TestCase):
    def setUp(self):
        self.expected_board_size = BOARD_SIZE
        self.expected_action_dim = self.expected_board_size**2 + 1
        self.expected_board_shape = (self.expected_board_size, self.expected_board_size)
        self.expected_state_shape = (
            STACK_HISTORY * 2 + 1,
            self.expected_board_size,
            self.expected_board_size,
        )

        return super().setUp()

    def test_can_set_board_size(self):
        env = GoEnv(num_stack=STACK_HISTORY)
        obs = env.reset()

        self.assertEqual(env.action_space.n, self.expected_action_dim)
        self.assertEqual(env.observation_space.shape, self.expected_state_shape)
        self.assertEqual(obs.shape, self.expected_state_shape)

        self.assertEqual(env.board_size, self.expected_board_size)
        self.assertEqual(env.board.shape, self.expected_board_shape)

    @parameterized.named_parameters(
        ('action_A19', 'A19', 0),  # upper-left
        ('action_T19', 'T19', BOARD_SIZE - 1),  # upper-right
        ('action_A1', 'A1', BOARD_SIZE * (BOARD_SIZE - 1)),  # lower-left
        ('action_T1', 'T1', BOARD_SIZE**2 - 1),  # lower-right
        ('action_C3', 'C3', BOARD_SIZE * (BOARD_SIZE - 3) + 3 - 1),
        ('action_E7', 'E7', BOARD_SIZE * (BOARD_SIZE - 7) + 5 - 1),
        ('action_PASS', 'PASS', BOARD_SIZE**2),
    )
    def test_gtp_to_action(self, gtpc, expected):
        env = GoEnv(num_stack=STACK_HISTORY)
        action = env.gtp_to_action(gtpc)

        self.assertEqual(action, expected)

    @parameterized.named_parameters(('action_500', 500), ('action_plus2', BOARD_SIZE**2 + 2), ('action_999', 999))
    def test_illegal_move_out_of_action_space(self, action):
        env = GoEnv(num_stack=STACK_HISTORY)
        env.reset()

        with self.assertRaisesRegex(ValueError, 'Invalid action'):
            env.step(action)

    @parameterized.named_parameters(('action_A1', 'A1'), ('action_C7', 'C7'))
    def test_illegal_move_already_taken(self, gtpc):
        env = GoEnv(num_stack=STACK_HISTORY)
        env.reset()

        env.step(env.gtp_to_action(gtpc, check_illegal=False))
        with self.assertRaisesRegex(ValueError, 'Illegal action'):
            env.step(env.gtp_to_action(gtpc, check_illegal=False))

    @parameterized.named_parameters(
        ('action_B1', ('A3', 'A2', 'B2', 'A1', 'C1'), 'B1'),
        (
            'action_F4',
            (
                'D3',
                'A1',
                'D4',
                'A2',
                'D5',
                'A3',
                'E3',
                'A4',
                'E5',
                'A5',
                'F3',
                'A6',
                'F5',
                'E4',
                'G4',
            ),
            'F4',
        ),
    )
    def test_illegal_move_suicidal(self, moves, illegal_move):
        env = GoEnv(num_stack=STACK_HISTORY)
        env.reset()

        for gtpc in moves:
            env.step(env.gtp_to_action(gtpc, check_illegal=False))

        with self.assertRaisesRegex(ValueError, 'Illegal action'):
            env.step(env.gtp_to_action(illegal_move, check_illegal=False))

    def test_illegal_move_ko(self):
        env = GoEnv(num_stack=STACK_HISTORY)
        env.reset()

        black_moves = ['A4', 'B4', 'C3', 'C1', 'D2']
        white_moves = ['A2', 'A3', 'B1', 'B3', 'C2']

        for b_move, w_move in zip(black_moves, white_moves):
            env.step(env.gtp_to_action(b_move, check_illegal=False))
            env.step(env.gtp_to_action(w_move, check_illegal=False))

        env.step(env.gtp_to_action('B2'))
        with self.assertRaisesRegex(ValueError, 'Illegal action'):
            env.step(env.gtp_to_action('C2', check_illegal=False))

    def test_game_over_by_resign(self):
        env = GoEnv(num_stack=STACK_HISTORY)
        env.reset()

        for i in range(4):
            env.step(i)

        env.step(env.resign_move)
        with self.assertRaisesRegex(RuntimeError, 'Game is over'):
            env.step(6)

    def test_game_over_by_pass(self):
        env = GoEnv(num_stack=STACK_HISTORY)
        env.reset()

        for i in range(4):
            env.step(i)

        env.step(env.pass_move)
        env.step(env.pass_move)

        with self.assertRaisesRegex(RuntimeError, 'Game is over'):
            env.step(6)

    def test_pass_move_steps(self):
        env = GoEnv(num_stack=STACK_HISTORY)
        env.reset()

        for i in range(4):
            env.step(i)
            env.step(env.pass_move)

        self.assertEqual(env.steps, 8)
        self.assertFalse(env.is_game_over())

    @parameterized.named_parameters(('steps_31', 31), ('stack_101', 101))
    def test_pass_game_over_max_steps(self, max_steps):
        env = GoEnv(max_steps=max_steps)
        env.reset()

        for i in range(max_steps):
            env.step(i)

        with self.assertRaisesRegex(RuntimeError, 'Game is over'):
            env.step(int(max_steps + 1))

    @parameterized.named_parameters(
        ('BLACK_won', ('C1', 'A1', 'B2', 'A2', 'A3', 'PASS', 'PASS'), go.BLACK, 1.0),
        (
            'WHITE_won',
            (
                'A1',
                'D2',
                'A2',
                'C3',
                'A3',
                'C4',
                'B1',
                'D5',
                'D3',
                'E4',
                'D4',
                'E3',
                'PASS',
                'PASS',
            ),
            go.WHITE,
            1.0,
        ),
    )
    def test_score_basic(self, moves, expected_winner, expected_reward):
        env = GoEnv(num_stack=STACK_HISTORY)
        env.reset()

        for gtpc in moves:
            _, reward, done, _ = env.step(env.gtp_to_action(gtpc))

            if done:
                self.assertEqual(env.winner, expected_winner)
                self.assertEqual(reward, expected_reward)
                break

    @parameterized.named_parameters(('white_won', 6, go.WHITE), ('black_won', 9, go.BLACK))
    def test_won_by_resign(self, num_steps, expected_winner):
        env = GoEnv(num_stack=STACK_HISTORY)
        env.reset()

        for i in range(num_steps):
            env.step(i)

        env.step(env.resign_move)
        self.assertEqual(env.winner, expected_winner)

    @parameterized.named_parameters(('stack_4', 4), ('stack_8', 8))
    def test_stacked_env_state_empty(self, num_stack):
        env = GoEnv(num_stack=num_stack)
        obs = env.reset()

        zero_planes = np.zeros(
            (num_stack * 2, self.expected_board_size, self.expected_board_size),
            dtype=np.uint8,
        )
        player_plane = np.ones((1, self.expected_board_size, self.expected_board_size), dtype=np.uint8)  # Black plays first.

        expected = np.concatenate([zero_planes, player_plane])
        np.testing.assert_equal(obs, expected)

    def test_stacked_env_state(self):
        env = GoEnv(num_stack=8)
        obs = env.reset()

        empty_board = np.copy(env.board)
        np.testing.assert_equal(empty_board, 0)

        # Define moves
        black_moves = ['B2', 'C3', 'C1', 'B3']
        white_moves = ['A3', 'A1', 'C2', 'B1']

        # This is really a mess
        expected = np.zeros((8 * 2, env.board_size, env.board_size), dtype=np.int8)

        for i in range(len(black_moves)):
            j = 0 if i == 0 else 4 * i
            for move in black_moves[: -i or None]:  # Get all elements if i =0
                _coords = env.cc.from_gtp(move)
                expected[j][_coords] = 1
                expected[j + 2][_coords] = 1

        for i in range(len(white_moves)):
            j = 1 if i == 0 else 4 * i - 1
            for move in white_moves[: -i or None]:  # Get all elements if i =0
                _coords = env.cc.from_gtp(move)
                expected[j][_coords] = 1
                # White is one step behind
                if j > 1:
                    expected[j + 2][_coords] = 1

        # Black to play
        color_to_play = np.ones((1, env.board_size, env.board_size), dtype=np.int8)
        expected = np.concatenate([expected, color_to_play], axis=0)

        # Let each player makes num_steps moves
        for b_move, w_move in zip(black_moves, white_moves):
            obs, _, _, _ = env.step(env.gtp_to_action(b_move))
            obs, _, _, _ = env.step(env.gtp_to_action(w_move))

        # self.assertTrue(np.array_equal(expected, env.minigo_stacked_features()))
        self.assertTrue(np.array_equal(obs, expected))

        # for i in range(len(obs)):
        #     target = expected[i, ...]
        #     pred = obs[i, ...]
        #     if not np.array_equal(pred, target):
        #         print(f"{i}-th index:")
        #         print(target)
        #         print(pred)
        #         print("\n")


if __name__ == '__main__':
    absltest.main()
