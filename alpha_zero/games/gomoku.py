"""Gomoku env class."""
from typing import Tuple
import numpy as np

from alpha_zero.games.env import BoardGameEnv


class GomokuEnv(BoardGameEnv):
    """Free-style Gomoku Environment with openAI Gym api.

    Free-style Gomoku has no restrictions on either player
    and allows a player to win by creating a line of 5 or more stones,
    with each player alternating turns placing one stone at a time.

    """

    def __init__(
        self,
        board_size: int = 15,
        num_to_win: int = 5,
        stack_history: int = 8,
    ) -> None:
        """
        Args:
            board_size: board size, default 15.
            num_to_win: number of connected stones to win, default 5.
            stack_history: stack last N history states, default 8.
        """
        super().__init__(board_size=board_size, stack_history=stack_history, name='Free-style Gomoku')
        self.num_to_win = num_to_win

    def evaluate_position(self) -> float:
        # Resign is always a loss, no matter how good the board position is.
        if self.last_actions[self.current_player] == self.resign_action:
            return -1.0
        if self.last_actions[self.opponent_player] == self.resign_action:
            return 1.0

        # Win or loss.
        if self.winner is not None:
            if self.winner == self.current_player:
                return 1.0
            else:
                return -1.0

        # Doing multiple full scans of the board in order to compute the estimated score.
        score = evaluate_and_score(
            self.board.copy(),
            self.black_color,
            self.white_color,
            self.current_player_color,
            self.num_to_win,
            max_score=1,
        )

        return score

    def is_current_player_won(self) -> bool:
        """This is a simple and quick way to check N connected sequence of stones,
        by starting from the last postion, without doing a full scan of the board."""
        # Less than num_to_win steps for each player.
        if self.steps < (self.num_to_win - 1) * 2:
            return False

        x_last, y_last = self.action_to_coords(self.last_actions[self.current_player])
        color = self.current_player_color

        board = self.board.copy()

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


# The following functions and design ideas are copied and adapted from the course:
# CSC180: Introduction to Computer Programming
# https://www.cs.toronto.edu/~guerzhoy/180/proj/proj2/gomoku.py
# https://github.com/d-hasan/Gomoku-CSC/blob/master/gomoku.py


def check_openness(
    board: np.ndarray,
    x_start: int,
    y_start: int,
    x_end: int,
    y_end: int,
    d_x: int,
    d_y: int,
) -> str:
    """Give a start point (x_start, y_start), a end point (x_end, y_end), and a moving direction (d_x, d_y),
    checks if the sequence is open or closed.

    It's done by checking one point 'before' the start point, and one point 'after' the end point.
    If both ends are empty, then it's 'OPEN',
    if one end is not empty or off the board, then it's 'SEMIOPEN',
    otherwise it's 'CLOSED'.

    Args:
        board: a 2D numpy.array representing the board.
        x_start: the row index for start point.
        y_start: the column index for start point.
        x_end: the row index for end point.
        y_end: the column index for end point.
        d_x: the changing row index indicate moving direction.
        d_y: the changing column index indicate moving direction.

    Returns:
        a string in one of {'CLOSED', 'OPEN', 'SEMIOPEN'}

     Raises:
        ValueError:
            if board is not a 2D numpy.array.
            if any one of the {x_start, y_start, x_end, y_end, d_x, d_y} input arguments is not a integer.
    """
    if not isinstance(board, np.ndarray) or len(board.shape) != 2:
        raise ValueError('Expect input arguments board to be a 2D numpy.array.')
    if not isinstance(x_start, int):
        raise ValueError('Expect input arguments x_start to be integer.')
    if not isinstance(y_start, int):
        raise ValueError('Expect input arguments y_start to be integer.')
    if not isinstance(x_end, int):
        raise ValueError('Expect input arguments x_end to be integer.')
    if not isinstance(y_end, int):
        raise ValueError('Expect input arguments y_end to be integer.')
    if not isinstance(d_x, int):
        raise ValueError('Expect input arguments d_x to be integer.')
    if not isinstance(d_y, int):
        raise ValueError('Expect input arguments d_y to be integer.')

    end_status = 'CLOSED'
    start_status = 'CLOSED'

    # Check that end point is on board
    if not is_bounded(board, x_end, y_end):
        return 'CLOSED'

    # Check the one step offset of end point is on board
    if not is_bounded(board, x_end + d_x, y_end + d_y):
        end_status = 'CLOSED'
    elif board[x_end + d_x, y_end + d_y] == 0:
        end_status = 'OPEN'

    # Check the one step offset of start point is on board
    if not is_bounded(board, x_start - d_x, y_start - d_y):
        start_status = 'CLOSED'
    elif board[x_start - d_x, y_start - d_y] == 0:
        start_status = 'OPEN'

    # Compare two ends status.
    if end_status != start_status:
        return 'SEMIOPEN'
    elif start_status == end_status == 'OPEN':
        return 'OPEN'
    else:
        return 'CLOSED'


def check_open_and_semiopen_seq(
    board: np.ndarray,
    color: int,
    x_start: int,
    y_start: int,
    length: int,
    d_x: int,
    d_y: int,
    special_case: bool = False,
    max_empty: int = 0,
) -> Tuple[int, int]:
    """
    Detect number of open and semi-open sequences stones matching the given color and length in a row.

    A note on handle special case where three or more same colored stones are in a row,
    but has one empty spaces between them, we count it as it is without count the empty point.
    For example:
        [0 1 1 0 1 1 0]
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0]
    When check start from point (0, 0) and moving direction (0, 1), the above can detect
    both a sequence length of 3, 4.

    Args:
        board: a 2D numpy.array representing the board.
        color: stone color  we want to match in a row on the board.
        x_start: row index of start position.
        y_start: column index of start position.
        length: the connected stone length we want to check.
        d_x: delta of row index indicate a moving direction.
        d_y: delta of column index indicate a moving direction.
        special_case: check on special open sequence with empty point in between,
            when set to true, will not check the status, and assume it's open,
            default off.
        max_empty: allowed maximum empty points in the sequence, default 0.

    Returns:
        a tuple of the number of open and semi-open sequences of color that matches length.

    Raises:
        ValueError:
            if board is not a 2D numpy.array.
            if any one of the {color, x_start, y_start, length, d_x, d_y} input arguments is not a integer.
    """

    if not isinstance(board, np.ndarray) or len(board.shape) != 2:
        raise ValueError('Expect input arguments board to be a 2D numpy.array.')
    if not isinstance(color, int):
        raise ValueError('Expect input arguments color to be integer.')
    if not isinstance(x_start, int):
        raise ValueError('Expect input arguments x_start to be integer.')
    if not isinstance(y_start, int):
        raise ValueError('Expect input arguments y_start to be integer.')
    if not isinstance(length, int):
        raise ValueError('Expect input arguments length to be integer.')
    if not isinstance(d_x, int):
        raise ValueError('Expect input arguments d_x to be integer.')
    if not isinstance(d_y, int):
        raise ValueError('Expect input arguments d_y to be integer.')

    open_seq_count = 0
    semi_open_seq_count = 0
    cur_length = 0

    board_size = board.shape[0]

    for i in range(board_size + 1):
        if not is_bounded(board, x_start + d_x, y_start + d_y):
            return open_seq_count, semi_open_seq_count
        elif board[x_start, y_start] == color:
            cur_length = count_same_color_stones(board, x_start, y_start, color, d_x, d_y, max_empty)
            # Special open sequence without checking on openness.
            if special_case:
                if cur_length >= length:
                    open_seq_count += 1
            elif length == cur_length:
                status = check_openness(
                    board,
                    x_start,
                    y_start,
                    x_start + ((length - 1) * d_x),
                    y_start + ((length - 1) * d_y),
                    d_x,
                    d_y,
                )
                if status == 'OPEN':
                    open_seq_count += 1
                if status == 'SEMIOPEN':
                    semi_open_seq_count += 1

            x_start += (cur_length - 1) * d_x
            y_start += (cur_length - 1) * d_y

        x_start += d_x
        y_start += d_y


def full_scan_for_open_and_semiopen_seq(
    board: np.ndarray,
    color: int,
    length: int,
    spepcial_case: bool = False,
    max_empty: int = 0,
) -> Tuple[int, int]:
    """
    Doing a full scan of the board for open and semi-open sequences which matches the given length and color.

    Args:
        board: a 2D numpy.array representing the board.
        color: stone color we want to match in a row on the board.
        length: the connected stone length we want to check.
        special_case: check on special open sequence with empty point in between,
            default off.
        max_empty: allowed maximum empty points in the sequence, default 0.

    Returns:
        a tuple of the number of open and semi-open sequences of color that matches length.

    Raises:
        ValueError:
            if board is not a 2D numpy.array.
            if any one of the {color, length} input arguments is not a integer.
    """
    if not isinstance(board, np.ndarray) or len(board.shape) != 2:
        raise ValueError('Expect input arguments board to be a 2D numpy.array.')
    if not isinstance(color, int):
        raise ValueError('Expect input arguments color to be integer.')
    if not isinstance(length, int):
        raise ValueError('Expect input arguments length to be integer.')

    open_seq_count, semi_open_seq_count = 0, 0

    board_size = board.shape[0]

    # check rows
    for row in range(board_size):
        count_tuple = check_open_and_semiopen_seq(board, color, 0, row, length, 1, 0, spepcial_case, max_empty)
        open_seq_count += count_tuple[0]
        semi_open_seq_count += count_tuple[1]

    # check columns
    for column in range(board_size):
        count_tuple = check_open_and_semiopen_seq(board, color, column, 0, length, 0, 1, spepcial_case, max_empty)
        open_seq_count += count_tuple[0]
        semi_open_seq_count += count_tuple[1]

    # check diagonals
    for diagonal in range(board_size - 1):  # the "- 1" prevents it from double counting the corner diagonals
        # top row
        for dir in (1, -1):
            count_tuple = check_open_and_semiopen_seq(board, color, diagonal, 0, length, dir, 1, spepcial_case, max_empty)
            open_seq_count += count_tuple[0]
            semi_open_seq_count += count_tuple[1]
            # bottom row
            count_tuple = check_open_and_semiopen_seq(
                board, color, diagonal, board_size - 1, length, dir, -1, spepcial_case, max_empty
            )
            open_seq_count += count_tuple[0]
            semi_open_seq_count += count_tuple[1]

    return open_seq_count, semi_open_seq_count


def evaluate_and_score(
    board: np.ndarray,
    black_color: int,
    white_color: int,
    current_player_color: int,
    num_to_win: int = 5,
    max_score: int = 100000,
) -> float:
    """Evaluate the board position and score from current player's perspective
    by use different weighting of the number of open, semi-open sequences.


    General idea for compute the score:
    * if one player has more than one sequence length equals to num_to_win, then the player is assigned max_score,
        it's opponent player gets -max_score score.
    * else compute the score based on number of open and semi-open sequences for each player,
        we rescal current player's score weights to avoid beeing too optimistic.

    Args:
        board: a 2D numpy.array representing the board.
        black_color: stone color for black player on the board.
        white_color: stone color for white player on the board.
        current_player_color: stone color for current player.
        num_to_win: the number of same color stones in a row to win.
        max_score: the max score for the game, default 1000000.

    Returns:
        evaluated score from current player's perspective.

    Raises:
        ValueError:
            if board is not a 2D numpy.array.
            if any one of the {black_color, white_color, max_score} input arguments is not a integer.
    """
    if not isinstance(board, np.ndarray) or len(board.shape) != 2:
        raise ValueError('Expect input arguments board to be a 2D numpy.array.')
    if not isinstance(black_color, int):
        raise ValueError('Expect input arguments black_color to be integer.')
    if not isinstance(white_color, int):
        raise ValueError('Expect input arguments white_color to be integer.')
    if not isinstance(max_score, int):
        raise ValueError('Expect input arguments max_score to be integer.')

    # Sequence lengths to check.
    # An example for the case five to win, we only checks:
    # * five-in-a-row
    # * four-in-a-row
    # * three-in-a-row
    seq_lengths = [num_to_win, num_to_win - 1, num_to_win - 2]

    open_black = {}
    semi_open_black = {}
    open_white = {}
    semi_open_white = {}

    # Checks standard open and semi-open sequences.
    for i in seq_lengths:
        open_black[i], semi_open_black[i] = full_scan_for_open_and_semiopen_seq(board, black_color, i)
        open_white[i], semi_open_white[i] = full_scan_for_open_and_semiopen_seq(board, white_color, i)

    # Checks for special sequence with one empty point allowed,
    # where by palying at this empty point is going to lead to a win.
    special_seq_length = num_to_win - 1
    special_open_b, _ = full_scan_for_open_and_semiopen_seq(board, black_color, special_seq_length, True, 1)
    special_open_w, _ = full_scan_for_open_and_semiopen_seq(board, white_color, special_seq_length, True, 1)
    open_black[special_seq_length] += special_open_b
    open_white[special_seq_length] += special_open_w

    # Needs to consider who's going to make the next move.
    if open_black[num_to_win] >= 1 or semi_open_black[num_to_win] >= 1:
        if current_player_color == black_color:
            return max_score
        else:
            return -max_score

    if open_white[num_to_win] >= 1 or semi_open_white[num_to_win] >= 1:
        if current_player_color == white_color:
            return max_score
        else:
            return -max_score

    my_open = open_black
    my_semi_open = semi_open_black
    opp_open = open_white
    opp_semi_open = semi_open_white
    if current_player_color == white_color:
        my_open = open_white
        my_semi_open = semi_open_white
        opp_open = open_black
        opp_semi_open = semi_open_black

    # TODO, find proof the way to assign these weights to the sequence length is correct or appropriate.
    # All weights are order from: [num_to_win, num_to_win - 1, num_to_win - 2]
    # General design:
    # * 1.0 when both open and semi-open sequence length already win
    # * 0.1 when need one more stone for both open and semi-open sequence which will lead to a win
    # * 0.01 when need two more stones in the open sequence which will lead to a win
    # * 0.0001 when need two more stones in the semi-open sequence which will lead to a win
    open_weights = [1.0, 0.1, 0.01]
    semi_open_weights = [1.0, 0.1, 0.0001]

    # A rescaling factor for current player to avoid being too optimistic.
    rescal_c = 0.8

    score = 0.0
    for open_w, semi_open_w, k in zip(open_weights, semi_open_weights, seq_lengths):
        # Compute opponent's score.
        opp_score = (open_w * opp_open[k] + semi_open_w * opp_semi_open[k]) * max_score

        # Compute current player's score.
        curr_score = (open_w * my_open[k] + semi_open_w * my_semi_open[k]) * rescal_c * max_score

        # The final score is using current player's score minus opponent's score.
        score += curr_score - opp_score

    return score
