# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Gomoku env class."""
from typing import Tuple
import numpy as np
from copy import copy

from alpha_zero.envs.base import BoardGameEnv
from alpha_zero.utils import sgf_wrapper
from alpha_zero.utils.util import get_time_stamp


class GomokuEnv(BoardGameEnv):
    """Free-style Gomoku Environment with OpenAI Gym api.

    Free-style Gomoku has no restrictions on either player
    and allows a player to win by creating a line of 5 or more stones,
    with each player alternating turns placing one stone at a time.

    """

    def __init__(self, board_size: int = 15, num_to_win: int = 5, num_stack: int = 8) -> None:
        """
        Args:
            board_size: board size, default 15.
            num_to_win: number of connected stones to win, default 5.
            num_stack: stack last N history states, default 8.
        """

        # Gomoku has no pass move and resign move
        super().__init__(
            id='Freestyle Gomoku',
            board_size=board_size,
            num_stack=num_stack,
            has_pass_move=False,
            has_resign_move=False,
        )

        self.num_to_win = num_to_win

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Plays one move."""
        if self.is_game_over():
            raise RuntimeError('Game is over, call reset before using step method.')
        if action is not None and action != self.resign_move and not 0 <= int(action) <= self.action_space.n - 1:
            raise ValueError(f'Invalid action. The action {action} is out of bound.')
        if action is not None and action != self.resign_move and self.legal_actions[int(action)] != 1:
            raise ValueError(f'Illegal action {action}.')

        self.last_move = copy(int(action))
        self.last_player = copy(self.to_play)
        self.steps += 1

        self.add_to_history(self.last_player, self.last_move)

        reward = 0.0

        # Make sure the action is illegal from now on.
        self.legal_actions[action] = 0

        # Update board state.
        row_index, col_index = self.action_to_coords(action)
        self.board[row_index, col_index] = self.to_play

        # Make sure the latest board position is always at index 0
        self.board_deltas.appendleft(np.copy(self.board))

        # The reward is always computed from last player's perspective
        # which follows standard MDP practice, where reward function r_t = R(s_t, a_t)
        if self.is_current_player_won():
            reward = 1.0
            self.winner = self.to_play

        done = self.is_game_over()

        # Switch next player
        self.to_play = self.opponent_player

        return self.observation(), reward, done, {}

    def is_current_player_won(self) -> bool:
        """This is a simple and quick way to check N connected sequence of stones,
        by starting from the last postion, without doing a full scan of the board."""
        # Less than num_to_win steps for each player.
        if self.steps < (self.num_to_win - 1) * 2:
            return False

        x_last, y_last = self.action_to_coords(self.last_move)
        color = self.to_play

        board = np.copy(self.board)

        # Check vertical
        vertical_dirs = (
            (0, -1),  # left
            (0, 1),  # right
        )
        if count_sequence_length_on_dir(board, x_last, y_last, color, vertical_dirs) >= self.num_to_win:
            return True

        # Check horizontal
        horizontal_dirs = (
            (-1, 0),  # up
            (1, 0),  # down
        )
        if count_sequence_length_on_dir(board, x_last, y_last, color, horizontal_dirs) >= self.num_to_win:
            return True

        # Check diagonal top-left to bottom-right
        diagonal_dirs_1 = (
            (-1, -1),  # to top-left
            (1, 1),  # to down-right
        )
        if count_sequence_length_on_dir(board, x_last, y_last, color, diagonal_dirs_1) >= self.num_to_win:
            return True

        # Check diagonal top-right to bottom-left
        diagonal_dirs_2 = (
            (-1, 1),  # to top-right
            (1, -1),  # to down-left
        )
        if count_sequence_length_on_dir(board, x_last, y_last, color, diagonal_dirs_2) >= self.num_to_win:
            return True

        return False

    def is_game_over(self) -> bool:
        if self.winner is not None:
            return True
        if self.is_board_full():
            return True
        return False

    def get_result_string(self) -> str:
        if not self.is_game_over():
            return ''

        if self.winner == self.black_player:
            return 'B+1.0'
        elif self.winner == self.white_player:
            return 'W+1.0'
        else:
            return 'DRAW'

    def to_sgf(self) -> str:
        return sgf_wrapper.make_sgf(
            board_size=self.board_size,
            move_history=self.history,
            result_string=self.get_result_string(),
            ruleset='',
            komi='',
            date=get_time_stamp(),
        )


# Extra functions for evaluation board positions and calculate score.
def is_bounded(board: np.ndarray, x: int, y: int) -> bool:
    """Returns whether the point in the format of (x, y) is on board.

    Args:
        board: a 2D numpy.array representing the board.
        x: row index to check.
        y: column index to check.

    Returns
        Bool indicte where the point is on board.

    Raises:
        ValueError:
            if board is not a 2D numpy.array.
            if any one of the {x, y} input arguments is not a integer.
    """
    if not isinstance(board, np.ndarray) or len(board.shape) != 2:
        raise ValueError('Expect input arguments board to be a 2D numpy.array.')
    if not isinstance(x, int):
        raise ValueError('Expect input arguments x to be integer.')
    if not isinstance(y, int):
        raise ValueError('Expect input arguments y to be integer.')

    board_size = board.shape[0]
    return (max(x, y) < board_size) and (min(x, y) >= 0)


def count_sequence_length_on_dir(
    board: np.ndarray,
    x_start: int,
    y_start: int,
    color: int,
    dirs: Tuple[Tuple[int, int]],
) -> int:
    """Give a start position and moving direction represented by a tuple of (d_x, d_y),
    count the sequence length of same color stones.

    Args:
        board: a 2D numpy.array representing the board.
        x_start: the row index for start position.
        y_start: the column index for start position.
        color: stone color we want to match.
        dirs: a Tuple (or list of Tuples) contains a pair of (d_x, d_y) indicate the moving direction.

    Returns:
        number of connected same color stones.

    Raises:
        ValueError:
            if board is not a 2D numpy.array.
            if any one of the {x_start, y_start, color} input arguments is not a integer.
            if the input argument dir is not a tuple or length greater than 2.
    """
    if not isinstance(board, np.ndarray) or len(board.shape) != 2:
        raise ValueError('Expect input arguments board to be a 2D numpy.array.')
    if not isinstance(x_start, int):
        raise ValueError('Expect input arguments x_start to be integer.')
    if not isinstance(y_start, int):
        raise ValueError('Expect input arguments y_start to be integer.')
    if not isinstance(color, int):
        raise ValueError('Expect input arguments color to be integer.')
    if not isinstance(dirs, tuple) or len(dirs) > 2:
        raise ValueError('Expect input arguments dirs to be tuple, and max length to be 2.')

    c = sum([count_same_color_stones(board, x_start, y_start, color, d_x, d_y) for d_x, d_y in dirs])

    if len(dirs) == 2:
        # Minus one because we double count the start position.
        return c - 1
    return c


def count_same_color_stones(
    board: np.ndarray,
    x_start: int,
    y_start: int,
    color: int,
    d_x: int,
    d_y: int,
    max_empty: int = 0,
) -> int:
    """Give a start position (x_start, y_start), and a moving direction (d_x, d_y),
    count connected stones that matches a stone color, does not include the start position.

    Examples for (d_x, d_y):
        up: (-1, 0)
        down: (1, 0)
        left: (0, -1)
        right: (0, 1)

    Args:
        board: a 2D numpy.array representing the board.
        x_start: the row index for start position.
        y_start: the column index for start position.
        color: stone color we want to match.
        d_x: moving x from start position.
        d_y: moving y from start position.
        max_empty: allowed maximum empty points in the sequence, default 0.

    Returns:
        number of connected stones.

    Raises:
        ValueError:
            if board is not a 2D numpy.array.
            if any one of the {x_start, y_start, color, d_x, d_y} input arguments is not a integer.
    """
    if not isinstance(board, np.ndarray) or len(board.shape) != 2:
        raise ValueError('Expect input arguments board to be a 2D numpy.array.')
    if not isinstance(x_start, int):
        raise ValueError('Expect input arguments x_start to be integer.')
    if not isinstance(y_start, int):
        raise ValueError('Expect input arguments y_start to be integer.')
    if not isinstance(color, int):
        raise ValueError('Expect input arguments color to be integer.')
    if not isinstance(d_x, int):
        raise ValueError('Expect input arguments d_x to be integer.')
    if not isinstance(d_y, int):
        raise ValueError('Expect input arguments d_y to be integer.')

    if not is_bounded(board, x_start, y_start):
        return 0

    if board[x_start, y_start] != color:
        return 0

    count = 1
    empty = 0
    x, y = x_start, y_start

    while is_bounded(board, x + d_x, y + d_y):
        if board[x + d_x, y + d_y] == color:
            count += 1
            x += d_x
            y += d_y
        elif max_empty > 0 and empty < max_empty and count > 1 and board[x + d_x, y + d_y] == 0:
            x += d_x
            y += d_y
            empty += 1
        else:
            break

    return count
