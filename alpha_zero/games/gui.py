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
"""A very basic GUI for board game, there's no plan to optimize this messy code."""
from sys import platform
import tkinter as tk
import itertools
from typing import Callable, Any

from alpha_zero.games.env import BoardGameEnv


class Colors:
    BOARD_BG = '#DBA14B'
    PANEL_BG = '#ffffff'
    INFO_BOX_BG = '#367588'
    ACTIONS_BG = '#1b1b1b'
    BUTTON_BG = '#f2f3f4'
    BLACK = '#000000'
    WHITE = '#ffffff'
    LINE = '#1b1b1b'
    LABEL = '#121212'
    TEXT = '#EDEDED'
    INFO = '#DBA14B'


class BoardGameGui:

    """A simple GUI for board game like Gomoku or go.
    This does not implement any game rules or scoring function.

    Supports:
    * Human vs. AlphaZero
    * AlphaZero vs. AlphaZero


    """

    def __init__(
        self,
        env: BoardGameEnv,
        black_player: Callable[[Any], int],
        white_player: Callable[[Any], int],
        show_step: bool = True,
        caption: str = 'Free-style Gomoku',
    ) -> None:
        """
        Args:
            env: the board game environment.
            black_player: a callable function which should return a valid action, or 'human' in the case human vs. AlphaZero.
            white_player: a callable function which should return a valid action, or 'human' in the case human vs. AlphaZero.
            show_step: if true, show step number of pieces, default on.
            caption: the caption on the GUI window.
        """
        if black_player is None:
            raise ValueError('Invalid black_player')
        if white_player is None:
            raise ValueError('Invalid white_player')

        self.env = env
        self.black_player = black_player
        self.white_player = white_player
        self.show_step = show_step

        self.human_player = None
        if self.black_player == 'human':
            self.human_player = 'black'
        elif self.white_player == 'human':
            self.human_player = 'white'

        self.num_rows = self.env.board_size
        self.num_cols = self.env.board_size

        self.played_games = 0
        self.black_won_games = 0
        self.white_won_games = 0

        self.last_move = None
        self.game_loop = None
        self.delay_time = 500  # delays 0.5 seconds per move

        self.col_labels = 'ABCDEFGHIJKLMNOPQRS'
        self.row_labels = [str(i) for i in range(1, self.num_rows + 1)]
        self.stone_colors = {self.env.white_player: Colors.WHITE, self.env.black_player: Colors.BLACK}

        # UI element sizes
        self.cell_size = 46
        self.piece_size = 38
        self.panel_w = 380
        self.dot_size = 8
        self.padding = 40

        self.font_family = 'Arial'
        self.font_size = 16
        self.info_font_size = 12
        self.title_font_size = 22

        if platform == "linux":
            scale = 1.8
            self.cell_size *= scale
            self.piece_size *= scale
            self.panel_w *= scale
            self.dot_size *= scale
            self.padding *= scale
            self.font_size *= scale
            self.info_font_size *= scale
            self.title_font_size *= scale

        self.half_size = self.cell_size // 2
        self.board_size = self.env.board_size * self.cell_size
        self.window_w = self.board_size + self.panel_w + self.padding * 2
        self.window_h = self.board_size + self.padding * 2

        self.root = tk.Tk()
        self.root.title(caption)
        self.root.resizable(0, 0)

        self.canvas = tk.Canvas(self.root, width=self.window_w, height=self.window_h, bg=Colors.BOARD_BG)
        self.canvas.pack()

        # Game board
        self.board = tk.Canvas(
            self.canvas,
            width=self.board_size,
            height=self.board_size,
            bg=Colors.BOARD_BG,
            highlightthickness=0,
        )

        # Right side panel
        self.panel = tk.Canvas(
            self.root,
            width=self.panel_w,
            height=self.window_h,
            bg=Colors.PANEL_BG,
            highlightthickness=0,
        )

        # We need to add column and row labels besides the board, so we create new canvas with some padding.
        self.canvas.create_window(
            self.padding,
            self.padding,
            anchor=tk.NW,
            window=self.board,
        )
        self.canvas.create_window(
            self.window_w - self.panel_w + 2,
            2,
            anchor=tk.NW,
            window=self.panel,
        )

        # Variables to update UI at runtime
        self.title_var = tk.StringVar(value=self.get_games_title())  # GAME N
        self.scores_var = tk.StringVar(value=self.get_match_scores())  # Won games: Black vs White
        self.info_var = tk.StringVar()  # Who's move and who won

        self.env.reset()

        self.initialize()

        # Register click event
        if self.human_player is not None:
            self.board.bind('<Button-1>', self.click_on_board)

    def initialize(self):
        self.initialize_board()
        self.draw_board_label()
        self.initialize_panel()
        self.update_game_info()

    def initialize_board(self):
        def _row_lines():
            for y in range(self.half_size, self.board_size - self.half_size + self.cell_size, self.cell_size):
                yield (self.half_size, y), (self.board_size - self.half_size, y)

        def _col_lines():
            for x in range(self.half_size, self.board_size - self.half_size + self.cell_size, self.cell_size):
                yield (x, self.half_size), (x, self.board_size - self.half_size)

        def _guide_dots():
            guide_dots = [
                (self.num_rows // 2, self.num_cols // 2),
            ]

            if self.num_cols > 9:
                guide_dots.extend(
                    [
                        (3, 3),
                        (3, self.num_cols // 2),
                        (3, self.num_cols - 4),
                        (self.num_rows // 2, 3),
                        (self.num_rows // 2, self.num_cols - 4),
                        (self.num_rows - 4, 3),
                        (self.num_rows - 4, self.num_cols // 2),
                        (self.num_cols - 4, self.num_cols - 4),
                    ]
                )

            return guide_dots

        # Clean up
        self.board.delete('all')

        # Grid lines.
        for start, end in itertools.chain(_col_lines(), _row_lines()):
            self.board.create_line(start[0], start[1], end[0], end[1], fill=Colors.LINE, width=2)

        # Guiding dots.
        for dot in _guide_dots():
            pos = self.env_coords_to_board_position(dot)
            self.draw_dot(pos, Colors.LINE, self.dot_size)

    def draw_board_label(self):
        # Row labels
        for j in range(self.num_rows):
            label = tk.Label(
                self.canvas,
                font=(self.font_family, self.font_size, 'bold'),
                text=self.row_labels[j],
                background=Colors.BOARD_BG,
                foreground=Colors.LABEL,
            )
            x = self.padding * 0.2
            y = j * self.cell_size + self.half_size + self.padding - self.font_size * 0.8
            label.place(x=x, y=y, anchor='nw')

        # Column labels
        for i in range(self.num_cols):
            label = tk.Label(
                self.canvas,
                font=(self.font_family, self.font_size, 'bold'),
                text=self.col_labels[i],
                anchor="center",
                background=Colors.BOARD_BG,
                foreground=Colors.LABEL,
            )
            x = i * self.cell_size + self.half_size + self.padding - self.font_size / 2
            y = self.padding + self.board_size + self.padding * 0.2
            label.place(x=x, y=y, anchor='nw')

    def initialize_panel(self):
        game_info_box = tk.Canvas(
            self.panel,
            width=self.panel_w,
            height=self.window_h * 0.25 - 1,
            background=Colors.INFO_BOX_BG,
            highlightthickness=0,
        )
        game_info_box.place(x=0, y=0, anchor='nw')

        player_info_box = tk.Canvas(
            self.panel,
            width=self.panel_w,
            height=self.window_h * 0.25 - 1,
            background=Colors.INFO_BOX_BG,
            highlightthickness=0,
        )
        player_info_box.place(x=0, y=self.window_h * 0.25, anchor='nw')

        actions_box = tk.Canvas(
            self.panel,
            width=self.panel_w,
            height=self.window_h * 0.5,
            background=Colors.ACTIONS_BG,
            highlightthickness=0,
        )
        actions_box.place(x=0, y=self.window_h * 0.5, anchor='nw')

        # Game info
        game_label = tk.Label(
            game_info_box,
            font=(self.font_family, self.title_font_size, 'bold'),
            textvariable=self.title_var,
            background=Colors.INFO_BOX_BG,
            foreground=Colors.TEXT,
        )
        game_label.place(relx=0.5, y=self.padding, anchor='center')

        info_label = tk.Label(
            game_info_box,
            font=(self.font_family, self.font_size, 'bold'),
            textvariable=self.info_var,
            background=Colors.INFO_BOX_BG,
            foreground=Colors.INFO,
        )
        info_label.place(relx=0.5, y=self.padding * 2.5, anchor='center')

        # Players info
        for i, t, c in zip([0, 1], [self.black_player_name, self.white_player_name], [Colors.BLACK, Colors.WHITE]):
            if i == 0:
                offset_x = self.padding
            else:
                offset_x = self.padding * 0.8 + self.panel_w * 0.5
            pos = (offset_x, self.padding)
            x0, y0, x1, y1 = self.get_circle_from_pos_and_size(pos, self.piece_size * 0.7)
            player_info_box.create_oval(x0, y0, x1, y1, fill=c, width=0, state=tk.DISABLED)

            # Title
            title = tk.Label(
                player_info_box,
                font=(self.font_family, self.font_size, 'bold'),
                text=t,
                background=Colors.INFO_BOX_BG,
                foreground=Colors.TEXT,
            )
            title.place(x=offset_x + self.piece_size * 0.5, y=self.padding // 2 + 6, anchor='nw')

        # Match scores
        results_label = tk.Label(
            player_info_box,
            font=(self.font_family, self.font_size, 'bold'),
            textvariable=self.scores_var,
            background=Colors.INFO_BOX_BG,
            foreground=Colors.TEXT,
        )
        results_label.place(relx=0.5, y=self.padding * 2 + self.info_font_size, anchor='center')

        # Action buttons
        new_game_btn = tk.Label(
            actions_box,
            text='New Game',
            font=(self.font_family, self.font_size),
            background=Colors.BUTTON_BG,
            foreground=Colors.BLACK,
            padx=12,
            pady=4,
        )
        new_game_btn.place(relx=0.5, y=self.padding * 2, anchor='center')

        exit_btn = tk.Label(
            actions_box,
            text='Exit',
            font=(self.font_family, self.font_size),
            background=Colors.BUTTON_BG,
            foreground=Colors.BLACK,
            padx=12,
            pady=4,
        )
        exit_btn.place(relx=0.5, y=self.padding * 4, anchor='center')

        new_game_btn.bind('<Button-1>', self.click_on_new_game)
        exit_btn.bind('<Button-1>', self.click_on_exit)

    def env_coords_to_board_position(self, coords):
        row, col = coords
        pos = (
            col * self.cell_size + self.half_size,
            row * self.cell_size + self.half_size,
        )
        return pos

    def pos_to_env_coords(self, pos):
        # Screen pos (x, y) is no the same as in row major numpy.array.
        x, y = pos
        row = y // self.cell_size
        col = x // self.cell_size
        return (row, col)

    def get_circle_from_pos_and_size(self, pos, size):
        half = size / 2
        x0 = pos[0] - half
        y0 = pos[1] - half
        x1 = pos[0] + half
        y1 = pos[1] + half
        return (x0, y0, x1, y1)

    def draw_dot(self, pos, color, size):
        x0, y0, x1, y1 = self.get_circle_from_pos_and_size(pos, size)
        self.board.create_oval(x0, y0, x1, y1, fill=color, width=0, state=tk.DISABLED)

    def draw_open_circle(self, pos, color, size):
        x0, y0, x1, y1 = self.get_circle_from_pos_and_size(pos, self.piece_size * 0.6)
        self.board.create_oval(x0, y0, x1, y1, fill=None, width=2, outline=color, state=tk.DISABLED)

    def draw_stone(self, pos, color, text_color, text, show_indicator=False):
        self.draw_dot(pos, color, self.piece_size)

        if show_indicator:
            self.draw_open_circle(pos, text_color, self.piece_size * 0.6)

        if self.show_step:
            self.board.create_text(pos[0], pos[1], text=text, fill=text_color)

    def draw_piece_on_board(self, coords):
        color = self.stone_colors[self.env.current_player]
        text_color = self.stone_colors[self.env.opponent_player]

        # Re-draw last move to remove border
        if self.last_move is not None:
            last_pos, last_color, last_text_color = self.last_move
            self.draw_stone(last_pos, last_color, last_text_color, self.env.steps)

        pos = self.env_coords_to_board_position(coords)
        self.draw_stone(pos, color, text_color, self.env.steps + 1, True)
        self.last_move = (pos, color, text_color)

    def click_on_board(self, event):
        if self.env.is_game_over or self.human_player is None or self.env.current_player_name != self.human_player:
            return

        coords = self.pos_to_env_coords((event.x, event.y))
        action = self.env.coords_to_action(coords)

        self.make_move(action)

    def click_on_new_game(self, event):
        self.reset()

    def click_on_exit(self, event):
        self.close()

    def get_current_player(self):
        if self.env.current_player == self.env.black_player:
            return self.black_player
        elif self.env.current_player == self.env.white_player:
            return self.white_player

    def make_move(self, action):
        # Avoid makes the same move repeatedly.
        if not self.env.is_action_valid(action):
            return

        coords = self.env.action_to_coords(action)
        self.draw_piece_on_board(coords)

        self.env.step(action)
        self.update_game_info()

    def play(self):
        # Let AlphaZero make a move.
        if not self.env.is_game_over and self.env.current_player_name != self.human_player:
            player = self.get_current_player()
            action = player(self.env)
            self.make_move(action)

        if not self.env.is_game_over:
            self.set_loop()
        else:
            self.clear_loop()
            self.update_match_results_info()

    def update_game_info(self):
        info_text = ''
        if self.env.is_game_over:
            if self.env.winner == self.env.black_player:
                info_text = 'Black won'
            elif self.env.winner == self.env.white_player:
                info_text = 'White won'
            else:
                info_text = 'Draw'
        else:
            if self.env.current_player == self.env.black_player:
                info_text = 'Black to move'
            else:
                info_text = 'White to move'
        self.info_var.set(info_text)

    def update_match_results_info(self):
        if self.env.winner_name == 'black':
            self.black_won_games += 1
        elif self.env.winner_name == 'white':
            self.white_won_games += 1
        self.scores_var.set(self.get_match_scores())

    def get_games_title(self):
        return f'GAME {self.played_games + 1}'

    def get_match_scores(self):
        return f'{self.black_won_games}            vs            {self.white_won_games}'

    def start(self):
        self.root.eval('tk::PlaceWindow . center')
        self.set_loop()
        self.root.mainloop()

    def set_loop(self):
        # Call the play function after some delay.
        self.game_loop = self.root.after(self.delay_time, self.play)

    def close(self):
        self.clear_loop()
        self.env.close()
        self.root.destroy()

    def clear_loop(self):
        if self.game_loop is not None:
            self.root.after_cancel(self.game_loop)

    def reset(self):
        self.clear_loop()

        self.played_games += 1
        self.title_var.set(self.get_games_title())

        self.env.reset()
        self.last_move = None
        self.game_loop = None

        self.initialize_board()
        self.update_game_info()
        self.set_loop()

    @property
    def black_player_name(self):
        if self.black_player is not None and self.black_player == 'human':
            return 'Human'
        else:
            return 'AlphaZero'

    @property
    def white_player_name(self):
        if self.white_player is not None and self.white_player == 'human':
            return 'Human'
        else:
            return 'AlphaZero'
