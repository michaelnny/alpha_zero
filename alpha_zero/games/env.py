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
"""Base board game class."""
from typing import Union, Tuple, Mapping, Text
from collections import deque
import os
import sys
from six import StringIO
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete


class BoardGameEnv(Env):
    """General board game environment like Go and Gomoku.

    NOTE:
    1. This implementation does not include the score and evaluation for each game,
    you should extend the functions `is_current_player_won` for the specific game.

    2. We don't check who should play next move in our internal env, so it's very important
    the program that calls the step() method should do the check.
    To help with the check, we expose the current_player information in the observation.

    """

    metadata = {'render.modes': ['terminal'], 'players': ['black', 'white']}

    def __init__(
        self, board_size: int = 15, stack_history: int = 8, black_player_id: int = 1, white_player_id: int = 2, name: str = ''
    ) -> None:
        """
        Args:
            board_size: board size, default 15.
            stack_history: stack last N history states, the final state is a image contains N x 2 + 1 binary planes, default 8.
            black_player_id: id and the color for black player, default 1.
            white_player_id: id and the color for white player, default 2.
            name: name of the game, default ''.
        """

        assert black_player_id != white_player_id != 0, 'player ids can not be the same, and can not be zero'

        super().__init__()

        self.name = name
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.stack_history = stack_history

        self.black_player = black_player_id  # black player id as well as stone color on the board
        self.white_player = white_player_id  # white player id as well as stone color on the board

        self.observation_space = Box(
            low=0, high=2, shape=(self.stack_history * 2 + 1, self.board_size, self.board_size), dtype=np.int8
        )

        self.num_actions = self.board_size**2
        self.action_space = Discrete(self.num_actions)

        # Legal actions mask, where '1' represents a legal action and '0' represents a illegal action
        self.legal_actions = np.ones(self.num_actions, dtype=np.int8).flatten()

        # The player to move at current time step, if game is over this is the player who made the last move and won/loss the game.
        self.current_player = self.black_player

        self.steps = 0

        self.winner = None

        self.last_player = None
        self.last_action = None

        # Save last N board, so we can stack history planes
        self.board_deltas = self.get_empty_queue()

        self.gtp_columns = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'
        self.gtp_rows = [str(i) for i in range(self.board_size, -1, -1)]

    def reset(self, **kwargs) -> np.ndarray:
        """Reset game to initial state."""
        super().reset(**kwargs)

        self.board = np.zeros_like(self.board)
        self.legal_actions = np.ones_like(self.legal_actions, dtype=np.int8).flatten()

        self.current_player = self.black_player

        self.steps = 0

        self.winner = None

        self.last_player = None
        self.last_action = None

        self.board_deltas = self.get_empty_queue()

        return self.observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Plays one move."""
        if not 0 <= action <= self.action_space.n - 1:
            raise ValueError(f'Invalid action. Expect action to be in range [0, {self.action_space.n}], got {action}')
        if not self.legal_actions[action]:
            raise ValueError(f'Invalid action. The action {action} has already been taken.')
        if self.is_game_over:
            raise RuntimeError('Game is over, call reset before using step method.')

        action = int(action)
        reward = 0.0

        # Make sure the action is illegal from now on.
        self.legal_actions[action] = 0
        self.last_action = action

        # Update board state.
        row_index, col_index = self.action_to_coords(action)
        self.board[row_index, col_index] = self.current_player

        # Make sure the latest board position is always at index 0
        self.board_deltas.appendleft(np.copy(self.board))

        if self.is_current_player_won():
            reward = 1.0
            self.winner = self.current_player

        # The reward is always computed from last player's perspective
        self.last_player = self.current_player

        done = self.is_game_over
        self.steps += 1

        self.current_player = self.opponent_player

        return self.observation(), reward, done, {}

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
        outfile.write(f'{self.name} ({self.board_size}x{self.board_size})')
        outfile.write('\n')
        outfile.write(f'Black: {black_stone}, White: {white_stone}')
        outfile.write('\n')
        outfile.write('\n')

        game_over_label = 'Yes' if self.is_game_over else 'No'
        outfile.write(f'Steps: {self.steps}, Game over: {game_over_label}, Winner: {self.winner_name}')
        outfile.write('\n')
        outfile.write(f'Current player: {self.current_player_name}')
        outfile.write('\n')
        outfile.write('\n')

        # Each row
        for r in range(self.board_size):
            # Add row label
            outfile.write('{0:2} | '.format(self.gtp_rows[r]))
            # Each column
            for c in range(0, self.board_size):
                # Single cell.
                our_str = '.'
                if board[r, c] == self.black_player:
                    our_str = f'{black_stone}'
                elif board[r, c] == self.white_player:
                    our_str = f'{white_stone}'
                if (r, c) == self.action_to_coords(self.last_action):
                    our_str = f'({our_str})'
                outfile.write(f'{our_str}'.center(3))
            outfile.write('\r\n')

        # Add column label
        outfile.write('    ' + '_' * self.board_size * 3)
        outfile.write('\r\n')
        outfile.write('      ')
        for y in range(self.board_size):
            outfile.write('{0:3}'.format(self.gtp_columns[y]))

        outfile.write('\n')
        return outfile

    def close(self):
        """Clean up deques"""
        self.board_deltas.clear()

        return super().close()

    def action_to_coords(self, action: int) -> Tuple[int, int]:
        """Convert action index into coords in the format of (row_index, column_index)"""
        # Returns dummy coords that is not on the board.
        if action is None:
            return (-1, -1)
        if not 0 <= action <= self.action_space.n - 1:
            return (-1, -1)

        row_index, col_index = action // self.board_size, action % self.board_size
        return (row_index, col_index)

    def coords_to_action(self, coords: Tuple[int, int]) -> int:
        """Convert coords in the format of (row_index, column_index) into action index"""
        row_index, col_index = coords
        action = row_index * self.board_size + col_index
        return action

    def observation(self) -> np.ndarray:
        """Stack N history of feature planes and one plane represent the color to play.

        Specifics:
            Xt is for current player
            Yt is for opponent player
            C is the color to play, 1 if black to play, 0 if white to play.

            The stack order is
            [Xt, Yt, Xt-1, Yt-1, Xt-2, Yt-2, ..., C]

        Returns a 3D tensor with the dimension [N, board_size, board_size],
            where N = 2 x stack_history + 1
        """
        # Create an empty array to hold the stacked planes, with shape (16, 19, 19)
        features = np.zeros((self.stack_history * 2, self.board_size, self.board_size), dtype=np.int8)

        deltas = np.array(self.board_deltas)

        # Current player first, then the opponent
        features[::2] = deltas == self.current_player
        features[1::2] = deltas == self.opponent_player

        # Color to play is a plane with all zeros for white, ones for black.
        color_to_play = np.zeros((1, self.board_size, self.board_size), dtype=np.int8)
        if self.current_player == self.black_player:
            color_to_play += 1

        # Using [C, H, W] channel first for PyTorch
        stacked_obs = np.concatenate([features, color_to_play], axis=0)

        return stacked_obs

    def is_current_player_won(self) -> bool:
        """Checks if the current player just won the game during play."""
        raise NotImplementedError()

    def get_empty_queue(self) -> deque:
        """Returns empty queue with stack_N * all zeros planes."""
        return deque([np.zeros((self.board_size, self.board_size))] * self.stack_history, maxlen=self.stack_history)

    def is_action_valid(self, action: int) -> bool:
        """Returns bool state to indicate given action is valid or not."""
        if action is None:
            return False
        if not 0 <= action <= self.action_space.n - 1:
            return False

        return self.legal_actions[action] == 1

    @property
    def opponent_player(self) -> int:
        if self.current_player == self.black_player:
            return self.white_player
        return self.black_player

    @property
    def current_player_name(self) -> str:
        if self.current_player == self.black_player:
            return 'black'
        return 'white'

    @property
    def opponent_player_name(self) -> str:
        if self.opponent_player == self.black_player:
            return 'black'
        return 'white'

    @property
    def is_board_full(self) -> bool:
        return np.all(self.board != 0)

    @property
    def is_game_over(self) -> bool:
        if self.winner is not None:
            return True
        if self.is_board_full:
            return True
        return False

    @property
    def winner_name(self) -> Union[None, str]:
        if self.winner == self.black_player:
            return 'black'
        elif self.winner == self.white_player:
            return 'white'
        else:
            return None
