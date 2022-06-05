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
"""Board game class."""
from typing import Union, Tuple, Mapping, Text
from collections import deque
import sys
from six import StringIO
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete


class BoardGameEnv(Env):
    """General board game environment like Go and Gomoku.

    NOTE:
    1. This implementation does not include the score and evaluation for each game,
    you should extend the functions `is_current_player_won` and `evaluate_position` for the specific game.

    2. We don't check who should play next move in our iternal env, so it's very important
    the program that calls the step() method should do the check.
    To help with the check, we expose the current_player information in the observation.

    """

    metadata = {'render.modes': ['terminal'], 'players': ['black', 'white']}

    def __init__(
        self,
        board_size: int = 19,
        stack_history: int = 8,
        black_player_id: int = 1,
        white_player_id: int = 2,
        name: str = '',
    ) -> None:
        """
        Args:
            board_size: board size, default 19.
            stack_history: stack last N history states, default 8.
            black_player_id: black player id, default 1.
            white_player_id: white player id, default 2.
            name: name of the game, default ''.
        """
        super().__init__()

        self.name: str = name
        self.board_size: int = board_size
        self.board: np.ndarray = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.stack_history: int = stack_history

        self.black_player_id: int = black_player_id
        self.white_player_id: int = white_player_id

        # Note color values can't be zeros.
        self.black_color: int = 1
        self.white_color: int = 2

        self.observation_space: Box = Box(
            low=0, high=2, shape=(self.stack_history * 2 + 1, self.board_size, self.board_size), dtype=np.int8
        )

        # Plus resign action.
        self.num_actions = self.board_size**2 + 1
        self.action_space: Discrete = Discrete(self.num_actions)

        # Legal actions mask.
        self.actions_mask: np.ndarray = np.ones(self.num_actions, dtype=np.bool8).flatten()

        # Last action is resign.
        self.resign_action: int = self.action_space.n - 1

        # The player to move at current time step, if game is over this is the player made the last move to won/loss the game.
        self.current_player: int = self.black_player_id

        self.steps: int = 0

        self.winner: Union[None, int] = None

        # Stores last action from each player.
        self.last_actions: Mapping[int, int] = {self.black_player_id: None, self.white_player_id: None}

        # History planes are FIFO queues, with the most recent state at index 0.
        self.feature_planes: Mapping[int, deque] = self._get_empty_queue_dict()

    def reset(self, **kwargs) -> np.ndarray:
        """Reset game to initial state."""
        super().reset(**kwargs)

        self.board: np.ndarray = np.zeros_like(self.board)
        self.actions_mask: np.ndarray = np.ones_like(self.actions_mask, dtype=np.bool8).flatten()

        self.current_player: int = self.black_player_id

        self.steps: int = 0

        self.winner: Union[None, int] = None

        self.last_actions: Mapping[int, int] = {self.black_player_id: None, self.white_player_id: None}

        self.feature_planes: Mapping[int, deque] = self._get_empty_queue_dict()

        return self.observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Plays one move."""
        if not 0 <= action <= self.action_space.n - 1:
            raise ValueError(f'Invalid action. Expect action to be in range [0, {self.action_space.n}], got {action}')
        if not self.actions_mask[action]:
            raise ValueError(f'Invalid action. The action {action} has alread been taken.')
        if self.is_game_over:
            raise RuntimeError('Game is over, call reset before using step method.')

        action = int(action)
        reward = 0.0

        # Make sure the action is illegal from now on.
        self.actions_mask[action] = False
        self.last_actions[self.current_player] = action

        # Resign always is a loss for current player.
        if action == self.resign_action:
            reward = -1.0
            self.winner = self.opponent_player
        else:
            # Update board state.
            row_index, col_index = self.action_to_coords(action)
            self.board[row_index, col_index] = self.current_player_color

            self._update_feature_planes()

            if self.is_current_player_won():
                reward = 1.0
                self.winner = self.current_player

        done = self.is_game_over
        # Switch next player.
        if not done:
            self.current_player = self.opponent_player

        self.steps += 1
        return self.observation(), reward, done, {}

    def render(self, mode='terminal'):
        """Prints out the board to terminal or ansi."""
        board = np.copy(self.board)
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        if mode == 'human':
            # Clear terminal screen
            # https://stackoverflow.com/questions/517970/how-to-clear-the-interpreter-console
            outfile.write('\033[H\033[J')

        black_stone = 'X'
        white_stone = 'O'

        game_over = 'Yes' if self.is_game_over else 'No'
        last_action_coords = [self.action_to_coords(action) for action in self.last_actions.values()]

        outfile.write(f'{self.name} ({self.board_size}x{self.board_size})')
        outfile.write('\n')
        outfile.write(f'Black: {black_stone}, White: {white_stone}')
        outfile.write('\n')
        outfile.write('\n')

        # Head information
        outfile.write(f'Steps: {self.steps}, Game over: {game_over}, Winner: {self.winner_name}')
        outfile.write('\n')
        outfile.write(f'Current player: {self.current_player_name}')
        outfile.write('\n')
        outfile.write('\n')

        # Column numbers
        for y in range(self.board_size):
            if y == 0:
                outfile.write('{0:6}'.format(y))
            else:
                outfile.write('{0:4}'.format(y))
        outfile.write('\n')
        # Top border
        outfile.write('    ' + '_' * self.board_size * 4)
        outfile.write('\r\n')
        # Content table, start with rows.
        for r in range(0, self.board_size):
            outfile.write('{0:2d} |'.format(r))
            # Columns.
            for c in range(0, self.board_size):
                # Single cell.
                our_str = '.'
                if board[r, c] == self.black_color:
                    our_str = f'{black_stone}'
                elif board[r, c] == self.white_color:
                    our_str = f'{white_stone}'
                if (r, c) in last_action_coords:
                    our_str = f'({our_str})'
                outfile.write(f'{our_str}'.center(4))
            outfile.write('|\r\n')
        # Bottom border
        outfile.write('    ' + '_' * self.board_size * 4)
        outfile.write('\n')
        outfile.write('\n')
        return outfile

    def close(self):
        """Clean up deques"""
        for q in self.feature_planes.values():
            q.clear()

        return super().close()

    def action_to_coords(self, action: int) -> Tuple[int, int]:
        """Convert action index into coords in the format of (row_index, column_index)"""
        # Returns dummy coords that is not on the board.
        if action is None:
            return (-1, -1)
        if not 0 <= action <= self.action_space.n - 1:
            return (-1, -1)
        if action == self.resign_action:
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

        """
        # Stack feature planes from t, t-1, t-2, ...
        feature_planes = []
        for t in range(self.stack_history):
            current_t = self.feature_planes[self.current_player][t]
            opponent_t = self.feature_planes[self.opponent_player][t]
            feature_planes.append(current_t)
            feature_planes.append(opponent_t)

        feature_planes = np.array(feature_planes, dtype=np.int8)

        # Color to play is a plane with all zeros for white, ones for black.
        color_to_play = np.zeros((1, self.board_size, self.board_size), dtype=np.int8)
        if self.current_player == self.black_player_id:
            color_to_play += 1

        stacked_obs = np.concatenate([feature_planes, color_to_play], axis=0)

        return stacked_obs

    def hash_board(self) -> str:
        """Returns a hash board state."""
        return self.board.data.tobytes()

    def is_action_valid(self, action: int) -> bool:
        """Returns bool state to indicate given action is valid or not."""
        if action is None:
            return False
        if not 0 <= action <= self.action_space.n - 1:
            return False

        return self.actions_mask[action]

    def is_current_player_won(self) -> bool:
        """Checks if the current player just won the game during play."""
        return

    def evaluate_position(self) -> float:
        """Returns the evaluated score of the game from current player's perspective."""
        raise NotImplementedError

    def _update_feature_planes(self) -> None:
        board = self.board.copy()
        if self.current_player == self.black_player_id:
            feature_plane = np.where(board == self.black_color, 1, 0)
        else:
            feature_plane = np.where(board == self.white_color, 1, 0)

        # History planes is a FIFO queue, the most recent state is always at index 0.
        self.feature_planes[self.current_player].appendleft(feature_plane)

    def _get_empty_queue_dict(self):
        """Returns empty queue with stack_history * all zeros planes."""
        player_ids = [self.black_player_id, self.white_player_id]
        return {
            id: deque([np.zeros((self.board_size, self.board_size))] * self.stack_history, maxlen=self.stack_history)
            for id in player_ids
        }

    @property
    def opponent_player(self) -> int:
        if self.current_player == self.black_player_id:
            return self.white_player_id
        return self.black_player_id

    @property
    def current_player_name(self) -> str:
        if self.current_player == self.black_player_id:
            return 'black'
        return 'white'

    @property
    def opponent_player_name(self) -> str:
        if self.opponent_player == self.black_player_id:
            return 'black'
        return 'white'

    @property
    def current_player_color(self) -> int:
        """Returns current player stone color."""
        if self.current_player == self.black_player_id:
            return self.black_color
        return self.white_color

    @property
    def opponent_player_color(self) -> int:
        """Returns opponent player stone color."""
        if self.opponent_player == self.black_player_id:
            return self.black_color
        return self.white_color

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
    def loser(self) -> Union[None, int]:
        if self.winner == self.black_player_id:
            return self.white_player_id
        elif self.winner == self.white_player_id:
            return self.black_player_id
        else:
            return None

    @property
    def winner_name(self) -> Union[None, str]:
        if self.winner == self.black_player_id:
            return 'black'
        elif self.winner == self.white_player_id:
            return 'white'
        else:
            return None

    @property
    def loser_name(self) -> Union[None, str]:
        if self.loser == self.black_player_id:
            return 'black'
        elif self.loser == self.white_player_id:
            return 'white'
        else:
            return None
