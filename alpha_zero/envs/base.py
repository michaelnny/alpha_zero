# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Iterable, Tuple, Mapping, Text
from collections import deque, namedtuple
import os
import sys
from copy import copy
from six import StringIO
import numpy as np
import gym
from gym.spaces import Box, Discrete

from alpha_zero.envs.coords import CoordsConvertor


class PlayerMove(namedtuple('PlayerMove', ['color', 'move'])):
    """Record game history moves"""

    pass


class BoardGameEnv(gym.Env):
    """Basic board game environment implemented using OpenAI Gym api."""

    def __init__(
        self,
        board_size: int = 15,
        num_stack: int = 8,
        black_player_id: int = 1,
        white_player_id: int = 2,
        has_pass_move: bool = False,
        has_resign_move: bool = False,
        id: str = '',
    ) -> None:
        """
        Args:
            board_size: board size, default 15.
            num_stack: stack last N history states, default 8.
            black_player_id: id and the stone color for black player, default 1.
            white_player_id: id and the stone color for white player, default 2.
            has_pass_move: the game has pass move, default off.
            id: environment id or name.
        """
        assert black_player_id != white_player_id != 0, 'player ids can not be the same, and can not be zero'

        self.id = id
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.num_stack = num_stack

        self.black_player = black_player_id  # black player id as well as stone color on the board
        self.white_player = white_player_id  # white player id as well as stone color on the board

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(self.num_stack * 2 + 1, self.board_size, self.board_size),
            dtype=np.int8,
        )

        self.has_pass_move = has_pass_move
        self.has_resign_move = has_resign_move

        self.action_dim = self.board_size**2 + 1 if self.has_pass_move else self.board_size**2
        self.action_space = Discrete(self.action_dim)

        self.pass_move = self.action_space.n - 1 if self.has_pass_move else None
        self.resign_move = -1 if self.has_resign_move else None

        # Legal actions mask, where '1' represents a legal action and '0' represents a illegal action
        self.legal_actions = np.ones(self.action_dim, dtype=np.int8).flatten()

        self.to_play = self.black_player

        self.steps = 0
        self.winner = None
        self.last_player = None
        self.last_move = None

        # Save last N board, so we can stack history planes
        self.board_deltas = self.get_empty_queue()

        self.history: Iterable[PlayerMove] = []

        self.gtp_columns = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'
        self.gtp_rows = [str(i) for i in range(self.board_size, -1, -1)]

        self.cc = CoordsConvertor(self.board_size)

    def reset(self, **kwargs) -> np.ndarray:
        """Reset game to initial state."""
        super().reset(**kwargs)

        self.board = np.zeros_like(self.board)
        self.legal_actions = np.ones_like(self.legal_actions, dtype=np.int8).flatten()

        self.to_play = self.black_player

        self.steps = 0
        self.winner = None
        self.last_player = None
        self.last_move = None

        self.board_deltas = self.get_empty_queue()

        del self.history[:]

        return self.observation()

    def render(self, mode='terminal'):
        """Prints out the board to terminal or ansi."""
        board = np.copy(self.board)
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        if mode == 'human':
            # Clearing the Screen
            if os.name == 'posix':  # posix is os name for Linux or mac
                os.system('clear')
            else:  # else screen will be cleared for windows
                os.system('cls')

        black_stone = 'X'
        white_stone = 'O'

        # Head information
        outfile.write(f'{self.id} ({self.board_size}x{self.board_size})')
        outfile.write('\n')
        outfile.write(f'Black: {black_stone}, White: {white_stone}')
        outfile.write('\n')
        outfile.write('\n')

        game_over_label = 'Yes' if self.is_game_over() else 'No'
        outfile.write(f'Game over: {game_over_label}, Result: {self.get_result_string()}')
        outfile.write('\n')
        outfile.write(
            f'Steps: {self.steps}, Current player: {black_stone if self.to_play == self.black_player else white_stone}'
        )
        outfile.write('\n')
        outfile.write('\n')

        self.render_additional_header(outfile, black_stone, white_stone)

        # Add top column label
        outfile.write('     ')
        for y in range(self.board_size):
            outfile.write('{0:3}'.format(self.gtp_columns[y]))
        outfile.write('\n')
        outfile.write('   +' + '-' * self.board_size * 3 + '+\n')

        # Each row
        for r in range(self.board_size):
            # Add left row label
            outfile.write('{0:2} |'.format(self.gtp_rows[r]))
            # Each column
            for c in range(0, self.board_size):
                # Single cell.
                our_str = '.'
                if board[r, c] == self.black_player:
                    our_str = f'{black_stone}'
                elif board[r, c] == self.white_player:
                    our_str = f'{white_stone}'
                if (r, c) == self.action_to_coords(self.last_move):
                    our_str = f'({our_str})'
                outfile.write(f'{our_str}'.center(3))
            # Add right row label
            outfile.write('| {0:2}'.format(self.gtp_rows[r]))
            outfile.write('\r\n')

        # Add bottom column label
        outfile.write('   +' + '-' * self.board_size * 3 + '+\n')
        outfile.write('     ')
        for y in range(self.board_size):
            outfile.write('{0:3}'.format(self.gtp_columns[y]))

        outfile.write('\n\n')
        return outfile

    def render_additional_header(self, outfile, black_stone, white_stone):
        pass

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """IMPORTANT: this is just a example, replace it inside each individual game."""

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

        # Handle actual game logic
        # Make sure the action is illegal from now on.
        self.legal_actions[action] = 0

        # Update board state.
        row_index, col_index = self.action_to_coords(action)
        self.board[row_index, col_index] = self.to_play

        # Make sure the latest board position is always at index 0
        self.board_deltas.appendleft(np.copy(self.board))

        # Switch next player
        self.to_play = self.opponent_player

        return self.observation(), 0, False, {}

    def close(self):
        """Clean up deques"""
        self.board_deltas.clear()
        del self.history[:]

        return super().close()

    def add_to_history(self, player_id, move):
        if move != self.resign_move:
            self.history.append(PlayerMove(color=self.get_player_name_by_id(player_id), move=move))

    def observation(self) -> np.ndarray:
        """Stack N history of feature planes and one plane represent the color to play.

        Specifics:
            Xt is for current player
            Yt is for opponent player
            C is the color to play, 1 if black to play, 0 if white to play.

            The stack order is
            [Xt, Yt, Xt-1, Yt-1, Xt-2, Yt-2, ..., C]

        Returns a 3D tensor with the dimension [N, board_size, board_size],
            where N = 2 x num_stack + 1
        """
        # Create an empty array to hold the stacked planes, with shape (16, 19, 19)
        features = np.zeros((self.num_stack * 2, self.board_size, self.board_size), dtype=np.int8)

        deltas = np.array(self.board_deltas)

        # Current player first, then the opponent
        features[::2] = deltas == self.to_play
        features[1::2] = deltas == self.opponent_player

        # Color to play is a plane with all zeros for white, ones for black.
        color_to_play = np.zeros((1, self.board_size, self.board_size), dtype=np.int8)
        if self.to_play == self.black_player:
            color_to_play += 1

        # Using [C, H, W] channel first for PyTorch
        stacked_obs = np.concatenate([features, color_to_play], axis=0)

        return stacked_obs

    def get_empty_queue(self) -> deque:
        """Returns empty queue with stack_N * all zeros planes."""
        return deque(
            [np.zeros((self.board_size, self.board_size))] * self.num_stack,
            maxlen=self.num_stack,
        )

    def is_board_full(self) -> bool:
        return np.all(self.board != 0)

    def is_pass_move(self, move: int) -> bool:
        """Returns bool state to indicate given move is pass move or not."""
        if not self.has_pass_move:
            return False

        return move == self.pass_move

    def is_resign_move(self, move: int) -> bool:
        """Returns bool state to indicate given move is resign move or not."""
        if not self.has_resign_move:
            return False

        return move == self.has_resign_move

    def is_legal_move(self, move: int) -> bool:
        """Returns bool state to indicate given move is valid or not."""
        if move is None:
            return False
        elif move < 0 or move > self.action_dim - 1:
            return False
        else:
            return self.legal_actions[move] == 1

    def is_coords_on_board(self, coords: Tuple[int, int]) -> int:
        """Check whether the coords in the format of (x, y) is on board."""
        x, y = coords
        return (max(x, y) < self.board_size) and (min(x, y) >= 0)

    def action_to_coords(self, action: int) -> Tuple[int, int]:
        """Convert action index into coords in the format of (row_index, column_index)"""
        # Return dummy coords
        if action is None:
            return (-1, -1)

        return self.cc.from_flat(action)

    def action_to_gtp(self, action: int) -> Tuple[int, int]:
        """Convert action index into coords in the GTP format e.g D4"""
        try:
            return self.cc.to_gtp(self.cc.from_flat(action))
        except Exception:
            return None

    def coords_to_action(self, coords: Tuple[int, int]) -> int:
        """Convert coords in the format of (row_index, column_index) into action index"""
        try:
            if self.is_coords_on_board(coords):
                return self.cc.to_flat(coords)
            else:
                return None
        except Exception:
            return None

    def gtp_to_action(self, gtpc: str, check_illegal: bool = True) -> int:
        """Convert GTP action (e.g. 'D4') into flat action"""
        try:
            action = self.cc.to_flat(self.cc.from_gtp(gtpc))
            # Invalid GTP input
            if action < 0 or action >= self.action_dim:
                return None
            elif check_illegal and self.legal_actions[action] != 1:
                return None
            return action
        except Exception:
            return None

    def get_player_name_by_id(self, id) -> str:
        if id == self.black_player:
            return 'B'
        elif id == self.white_player:
            return 'W'
        else:
            return None

    def is_game_over(self) -> bool:
        return False

    @property
    def opponent_player(self) -> int:
        if self.to_play == self.black_player:
            return self.white_player
        return self.black_player

    def get_captures(self) -> Mapping[Text, int]:
        """Number of captures for the players, this is only for game of Go."""
        return {self.black_player: 0, self.white_player: 0}

    def get_result_string(self) -> str:
        """Game results for sgf, where B+ indicates black won, R+ white won."""
        return ''

    def to_sgf(self) -> str:
        """Game record to sgf content"""
        return
