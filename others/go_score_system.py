# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""
This module demonstrate the problem of how dead stones,
and with our current implement of the area scoring system,
it often produces inaccurate scores.
"""
import numpy as np

N = 9

WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)

ALL_COORDS = [(i, j) for i in range(N) for j in range(N)]


def _check_bounds(c):
    return 0 <= c[0] < N and 0 <= c[1] < N


NEIGHBORS = {(x, y): list(filter(_check_bounds, [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])) for x, y in ALL_COORDS}


def place_stones(board, color, stones):
    for s in stones:
        board[s] = color


def find_reached(board, c):
    color = board[c]
    chain = set([c])
    reached = set()
    frontier = [c]
    while frontier:
        current = frontier.pop()
        chain.add(current)
        for n in NEIGHBORS[current]:
            if board[n] == color and n not in chain:
                frontier.append(n)
            elif board[n] != color:
                reached.add(n)
    return chain, reached


def area_score(board):
    """Return the area scores for both players following the simplifier Chinese rules.

    Note, this does not handle dead stones, so for some complex games, the score will be incorrect.
    """
    board = np.copy(board)

    # Need to remove dead stones from the board first

    while EMPTY in board:
        unassigned_spaces = np.where(board == EMPTY)
        c = unassigned_spaces[0][0], unassigned_spaces[1][0]
        territory, borders = find_reached(board, c)
        border_colors = set(board[b] for b in borders)
        X_border = BLACK in border_colors
        O_border = WHITE in border_colors
        if X_border and not O_border:
            territory_color = BLACK
        elif O_border and not X_border:
            territory_color = WHITE
        else:
            territory_color = UNKNOWN  # dame, or seki
        place_stones(board, territory_color, territory)

    black_score = np.count_nonzero(board == BLACK)
    white_score = np.count_nonzero(board == WHITE)
    return black_score, white_score


# Test cases
B, W = BLACK, WHITE


def run_test_on_board(board, komi, actual_black_score, actual_white_score):
    print(f'\n Board: \n {board}')

    black_score, white_score = area_score(board)
    white_score += komi

    print(f'Computed - black score: {black_score}, white score: {white_score}')
    print(f'Expected - black score: {actual_black_score}, white score: {actual_white_score}')


# Game 1 - incorrect score
board = np.array(
    [
        [0, 0, B, W, W, 0, 0, 0, 0],
        [0, B, B, W, 0, W, 0, 0, 0],
        [0, B, 0, B, W, 0, W, 0, 0],
        [0, 0, B, B, W, W, 0, 0, 0],
        [0, 0, 0, W, B, B, W, W, 0],
        [0, 0, B, 0, B, B, B, W, 0],
        [0, 0, 0, 0, 0, B, W, 0, W],
        [0, 0, 0, B, 0, B, W, W, 0],
        [0, 0, 0, 0, B, W, W, 0, 0],
    ]
)
komi = 7.5
actual_black_score = 44
actual_white_score = 44.5

run_test_on_board(board, komi, actual_black_score, actual_white_score)


# Game 2 - incorrect score
board = np.array(
    [
        [0, 0, 0, 0, B, B, B, 0, 0],
        [0, B, B, B, B, W, B, B, B],
        [B, 0, 0, B, W, W, W, W, W],
        [B, B, B, B, W, W, W, W, 0],
        [B, W, B, W, B, B, B, 0, W],
        [W, W, B, W, W, W, W, W, 0],
        [W, W, W, 0, W, 0, W, W, W],
        [0, W, 0, W, B, B, B, W, B],
        [W, 0, W, B, B, B, B, B, B],
    ]
)
komi = 7.5
actual_black_score = 28
actual_white_score = 60.5

run_test_on_board(board, komi, actual_black_score, actual_white_score)


# Game 3 - incorrect score
board = np.array(
    [
        [0, 0, 0, 0, B, B, W, W, W],
        [0, 0, 0, 0, B, W, W, 0, W],
        [0, 0, 0, 0, B, B, W, W, W],
        [B, B, B, 0, 0, B, W, W, 0],
        [B, W, W, B, B, B, W, W, W],
        [B, B, W, B, B, B, B, W, W],
        [B, B, W, W, B, W, W, 0, W],
        [W, W, W, W, B, W, W, B, W],
        [0, W, 0, W, W, W, W, B, W],
    ]
)
komi = 7.5
actual_black_score = 37
actual_white_score = 51.5

run_test_on_board(board, komi, actual_black_score, actual_white_score)


# Game 4 - incorrect winner
board = np.array(
    [
        [0, 0, B, W, W, 0, 0, 0, 0],
        [B, B, 0, B, W, 0, W, W, 0],
        [B, 0, B, B, W, W, B, W, 0],
        [0, B, W, W, W, B, B, B, W],
        [B, W, B, B, W, B, B, W, 0],
        [0, 0, B, B, B, B, W, W, 0],
        [0, 0, 0, W, 0, B, W, 0, 0],
        [0, 0, 0, 0, B, B, W, 0, 0],
        [0, 0, 0, 0, B, W, W, 0, 0],
    ]
)
komi = 7
actual_black_score = 46
actual_white_score = 42

run_test_on_board(board, komi, actual_black_score, actual_white_score)


# Game 5 - incorrect winner
board = np.array(
    [
        [0, 0, 0, 0, 0, B, B, W, 0],
        [0, B, 0, 0, 0, B, W, 0, W],
        [B, 0, 0, 0, W, B, W, 0, 0],
        [W, B, B, 0, 0, B, W, 0, 0],
        [W, W, B, 0, B, B, W, W, 0],
        [0, 0, W, B, 0, B, W, W, W],
        [0, W, W, W, B, 0, B, B, B],
        [W, 0, W, B, 0, B, 0, 0, 0],
        [0, 0, W, B, B, 0, 0, 0, 0],
    ]
)

komi = 7
actual_black_score = 48
actual_white_score = 40

run_test_on_board(board, komi, actual_black_score, actual_white_score)


# Game 6 - incorrect winner
board = np.array(
    [
        [0, W, B, 0, 0, 0, 0, 0, 0],
        [B, W, B, B, 0, 0, B, 0, B],
        [B, W, W, B, 0, 0, 0, B, W],
        [B, 0, W, W, B, B, B, W, W],
        [W, W, W, B, B, W, W, W, 0],
        [W, B, B, W, W, W, B, 0, 0],
        [B, B, 0, B, W, W, 0, 0, 0],
        [0, 0, B, B, B, W, 0, 0, 0],
        [0, 0, 0, B, W, W, 0, 0, 0],
    ]
)

komi = 7
actual_black_score = 39
actual_white_score = 49

run_test_on_board(board, komi, actual_black_score, actual_white_score)


# Game 7 - incorrect winner
board = np.array(
    [
        [0, W, B, B, 0, 0, B, 0, 0],
        [W, W, W, B, 0, B, 0, B, 0],
        [0, W, B, B, B, B, B, 0, 0],
        [0, 0, W, B, B, W, B, 0, 0],
        [0, 0, W, W, B, W, W, B, 0],
        [0, 0, 0, W, W, W, W, B, B],
        [0, B, W, 0, W, W, B, 0, 0],
        [0, W, 0, W, B, W, B, B, B],
        [W, 0, W, B, B, B, B, W, 0],
    ]
)

komi = 7
actual_black_score = 43
actual_white_score = 45

run_test_on_board(board, komi, actual_black_score, actual_white_score)


board = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
