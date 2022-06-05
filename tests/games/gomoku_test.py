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

from alpha_zero.games.gomoku import (
    GomokuEnv,
    check_openness,
    check_open_and_semiopen_seq,
    full_scan_for_open_and_semiopen_seq,
    evaluate_and_score,
)


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
                # Opponent should not take win_actions and resign action
                legit_actions = np.flatnonzero(env.actions_mask)
                opponent_actions = list(set(legit_actions) - set(win_actions) - set([env.resign_action]))

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
                # Opponent should not take win_actions and resign action
                legit_actions = np.flatnonzero(env.actions_mask)
                opponent_actions = list(set(legit_actions) - set(win_actions) - set([env.resign_action]))

                action = np.random.choice(opponent_actions, 1).item()
                opponent_steps += 1

            obs, reward, done, _ = env.step(action)

        self.assertEqual(winner_steps, num_to_win)
        self.assertEqual(env.winner, winner_id)
        self.assertEqual(reward, 1.0)


class GomokuEvaluationFunctionsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.env = GomokuEnv(board_size=9)
        self.env.reset()
        self.board = self.env.board

        self.black_color = self.env.black_color
        self.white_color = self.env.white_color

        # Row 1
        self.board[0, 1] = self.white_color
        self.board[0, 2] = self.white_color
        self.board[0, 3] = self.white_color
        self.board[0, 5] = self.white_color
        self.board[0, 6] = self.white_color

        # Row 2
        self.board[1, 2] = self.white_color
        self.board[1, 3] = self.white_color
        self.board[1, 4] = self.white_color
        self.board[1, 6] = self.white_color
        self.board[1, 7] = self.white_color

        # Row 3
        self.board[2, 3] = self.white_color

        # Row 4
        self.board[3, 2] = self.black_color
        self.board[3, 3] = self.black_color
        self.board[3, 4] = self.black_color
        self.board[3, 5] = self.black_color
        self.board[3, 6] = self.black_color

        # Row 5
        self.board[4, 5] = self.black_color
        self.board[4, 6] = self.black_color

        # Row 6
        self.board[5, 2] = self.white_color
        self.board[5, 3] = self.white_color

        # Row 7
        self.board[6, 2] = self.white_color
        self.board[6, 3] = self.white_color

        # Row 8
        self.board[7, 2] = self.white_color

        # Expected board:
        # [
        # [0 1 1 1 0 1 1 0 0]
        # [0 0 1 1 1 0 1 1 0]
        # [0 0 0 1 0 0 0 0 0]
        # [0 0 2 2 2 2 2 0 0]
        # [0 0 0 0 0 2 2 0 0]
        # [0 0 1 1 0 0 0 0 0]
        # [0 0 1 1 0 0 0 0 0]
        # [0 0 1 0 0 0 0 0 0]
        # [0 0 0 0 0 0 0 0 0]
        # ]

    def test_check_openness_horizontal(self):
        status_1 = check_openness(self.board, 0, 1, 0, 3, 0, 1)
        status_2 = check_openness(self.board, 0, 2, 0, 3, 0, 1)
        status_3 = check_openness(self.board, 0, 3, 0, 5, 0, 1)

        self.assertEqual(status_1, 'OPEN')
        self.assertEqual(status_2, 'SEMIOPEN')
        self.assertEqual(status_3, 'CLOSED')

    def test_check_openness_vertical(self):
        status_1 = check_openness(self.board, 0, 3, 2, 3, 1, 0)
        status_2 = check_openness(self.board, 2, 5, 5, 5, 1, 0)
        status_3 = check_openness(self.board, 6, 2, 7, 2, 1, 0)

        self.assertEqual(status_1, 'CLOSED')
        self.assertEqual(status_2, 'OPEN')
        self.assertEqual(status_3, 'SEMIOPEN')

    def test_check_openness_digonal_top_left_to_bottom_right(self):
        status_1 = check_openness(self.board, 0, 1, 3, 4, 1, 1)
        status_2 = check_openness(self.board, 0, 3, 1, 4, 1, 1)
        status_3 = check_openness(self.board, 3, 5, 5, 7, 1, 1)
        status_4 = check_openness(self.board, 0, 5, 2, 7, 1, 1)

        self.assertEqual(status_1, 'CLOSED')
        self.assertEqual(status_2, 'SEMIOPEN')
        self.assertEqual(status_3, 'OPEN')
        self.assertEqual(status_4, 'SEMIOPEN')

    def test_check_openness_digonal_top_right_to_bottom_left(self):
        status_1 = check_openness(self.board, 0, 5, 2, 3, 1, -1)
        status_2 = check_openness(self.board, 0, 3, 1, 2, 1, -1)
        status_3 = check_openness(self.board, 2, 7, 4, 5, 1, -1)

        self.assertEqual(status_1, 'CLOSED')
        self.assertEqual(status_2, 'SEMIOPEN')
        self.assertEqual(status_3, 'OPEN')

    def test_check_openness_off_board(self):
        status_1 = check_openness(self.board, -1, -2, 55, 66, -1, -1)
        self.assertEqual(status_1, 'CLOSED')

    def test_check_open_and_semiopen_sequence_horizontal(self):
        # White stones
        open_count_w_1, semi_count_w_1 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 0, 3, 0, 1)
        self.assertEqual(open_count_w_1, 1)
        self.assertEqual(semi_count_w_1, 0)

        open_count_w_2, semi_count_w_2 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 0, 2, 0, 1)
        self.assertEqual(open_count_w_2, 1)
        self.assertEqual(semi_count_w_2, 0)

        # Black stones
        open_count_b_1, semi_count_b_1 = check_open_and_semiopen_seq(self.board, self.black_color, 3, 0, 5, 0, 1)
        self.assertEqual(open_count_b_1, 1)
        self.assertEqual(semi_count_b_1, 0)

        open_count_b_2, semi_count_b_2 = check_open_and_semiopen_seq(self.board, self.black_color, 4, 0, 2, 0, 1)
        self.assertEqual(open_count_b_2, 1)
        self.assertEqual(semi_count_b_2, 0)

    def test_check_open_and_semiopen_sequence_vertical(self):
        # White stones
        open_count_w_1, semi_count_w_1 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 2, 3, 1, 0)
        self.assertEqual(open_count_w_1, 1)
        self.assertEqual(semi_count_w_1, 0)

        open_count_w_2, semi_count_w_2 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 2, 2, 1, 0)
        self.assertEqual(open_count_w_2, 0)
        self.assertEqual(semi_count_w_2, 1)

        # Black stones
        open_count_b_1, semi_count_b_1 = check_open_and_semiopen_seq(self.board, self.black_color, 0, 5, 2, 1, 0)
        self.assertEqual(open_count_b_1, 1)
        self.assertEqual(semi_count_b_1, 0)

    def test_check_open_and_semiopen_sequence_diagonal_top_left_to_bottom_right(self):
        # White stones
        open_count_w_1, semi_count_w_1 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 1, 2, 1, 1)
        self.assertEqual(open_count_w_1, 0)
        self.assertEqual(semi_count_w_1, 0)

        open_count_w_2, semi_count_w_2 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 2, 2, 1, 1)
        self.assertEqual(open_count_w_2, 0)
        self.assertEqual(semi_count_w_2, 1)

        open_count_w_3, semi_count_w_3 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 3, 2, 1, 1)
        self.assertEqual(open_count_w_3, 0)
        self.assertEqual(semi_count_w_3, 1)

        # Black stones
        open_count_b_1, semi_count_b_1 = check_open_and_semiopen_seq(self.board, self.black_color, 0, 1, 2, 1, 1)
        self.assertEqual(open_count_b_1, 0)
        self.assertEqual(semi_count_b_1, 1)

        open_count_b_2, semi_count_b_2 = check_open_and_semiopen_seq(self.board, self.black_color, 0, 2, 2, 1, 1)
        self.assertEqual(open_count_b_2, 1)
        self.assertEqual(semi_count_b_2, 0)

    def test_check_open_and_semiopen_sequence_diagonal_top_right_to_bottom_left(self):
        # White stones
        open_count_w_1, semi_count_w_1 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 5, 3, 1, -1)
        self.assertEqual(open_count_w_1, 0)
        self.assertEqual(semi_count_w_1, 0)

        open_count_w_2, semi_count_w_2 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 8, 2, 1, -1)
        self.assertEqual(open_count_w_2, 1)
        self.assertEqual(semi_count_w_2, 0)

        open_count_w_3, semi_count_w_3 = check_open_and_semiopen_seq(self.board, self.white_color, 0, 3, 2, 1, -1)
        self.assertEqual(open_count_w_3, 0)
        self.assertEqual(semi_count_w_3, 1)

        # Black stones
        open_count_b_1, semi_count_b_1 = check_open_and_semiopen_seq(self.board, self.black_color, 1, 8, 2, 1, -1)
        self.assertEqual(open_count_b_1, 1)
        self.assertEqual(semi_count_b_1, 0)

    def test_check_special_open_seq(self):
        # White stones
        count_w_1, _ = check_open_and_semiopen_seq(self.board, self.white_color, 0, 0, 4, 0, 1, True, 1)
        self.assertEqual(count_w_1, 1)

        count_w_2, _ = check_open_and_semiopen_seq(self.board, self.white_color, 1, 0, 4, 0, 1, True, 1)
        self.assertEqual(count_w_2, 1)

        # Black stones
        self.board[3, 4] = 0
        self.board[6, 5] = self.black_color
        self.board[7, 5] = self.black_color

        count_b_1, _ = check_open_and_semiopen_seq(self.board, self.black_color, 3, 0, 4, 0, 1, True, 1)
        self.assertEqual(count_b_1, 1)

        count_b_2, _ = check_open_and_semiopen_seq(self.board, self.black_color, 0, 5, 4, 1, 0, True, 1)
        self.assertEqual(count_b_2, 1)

    def test_full_scan_for_open_and_semiopen_seq(self):
        # White stones
        open_count_w_1, semi_count_w_1 = full_scan_for_open_and_semiopen_seq(self.board, self.white_color, 2)
        self.assertEqual(open_count_w_1, 8)
        self.assertEqual(semi_count_w_1, 7)

        open_count_w_2, semi_count_w_2 = full_scan_for_open_and_semiopen_seq(self.board, self.white_color, 3)
        self.assertEqual(open_count_w_2, 3)
        self.assertEqual(semi_count_w_2, 0)

        # Black stones
        open_count_b_1, semi_count_b_1 = full_scan_for_open_and_semiopen_seq(self.board, self.black_color, 2)
        self.assertEqual(open_count_b_1, 5)
        self.assertEqual(semi_count_b_1, 1)

        open_count_b_2, semi_count_b_2 = full_scan_for_open_and_semiopen_seq(self.board, self.black_color, 5)
        self.assertEqual(open_count_b_2, 1)
        self.assertEqual(semi_count_b_2, 0)

    def test_full_scan_for_open_and_semiopen_seq_special_case(self):
        # White stones
        open_count_w_1, _ = full_scan_for_open_and_semiopen_seq(self.board, self.white_color, 5, True, 1)
        self.assertEqual(open_count_w_1, 2)

        # Black stones
        self.board[3, 4] = 0
        open_count_b_1, _ = full_scan_for_open_and_semiopen_seq(self.board, self.black_color, 4, True, 1)
        self.assertEqual(open_count_b_1, 1)

    def test_evaluate_and_score_black_win(self):
        black_score = evaluate_and_score(self.board, self.black_color, self.white_color, self.black_color, max_score=1)
        white_score = evaluate_and_score(self.board, self.black_color, self.white_color, self.white_color, max_score=1)
        self.assertEqual(black_score, 1)
        self.assertEqual(white_score, -1)

    def test_evaluate_and_score_white_win(self):
        self.board[3, 2] = self.white_color
        self.board[4, 2] = self.white_color

        white_score = evaluate_and_score(self.board, self.black_color, self.white_color, self.white_color, max_score=1)
        black_score = evaluate_and_score(self.board, self.black_color, self.white_color, self.black_color, max_score=1)
        self.assertEqual(white_score, 1)
        self.assertEqual(black_score, -1)

    def test_evaluate_and_score_white_has_upper_hand(self):
        self.board[3, 3] = 0
        score = evaluate_and_score(self.board, self.black_color, self.white_color, self.white_color)
        self.assertGreater(score, 0)

    def test_evaluate_and_score_black_has_upper_hand(self):
        self.board[0, 3] = 0
        self.board[1, 4] = 0
        self.board[7, 2] = 0

        self.board[3, 2] = 0

        score = evaluate_and_score(self.board, self.black_color, self.white_color, self.black_color)
        self.assertGreater(score, 0)

    def test_evaluate_position_current_player_black(self):

        self.board[0, 3] = 0
        self.board[1, 4] = 0
        self.board[7, 2] = 0

        self.board[3, 2] = 0

        score = self.env.evaluate_position()
        self.assertGreater(score, 0)

    def test_evaluate_position_current_player_white(self):
        self.env.current_player = self.env.opponent_player

        self.board[0, 3] = 0
        self.board[1, 4] = 0
        self.board[7, 2] = 0

        self.board[3, 2] = 0

        score = self.env.evaluate_position()
        self.assertLess(score, 0)


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
            if env.current_player == env.black_player_id:
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

        if env.current_player == env.black_player_id:
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
