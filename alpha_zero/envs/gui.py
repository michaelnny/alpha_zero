# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""A very basic GUI program for board games, note there's no plan to optimize this messy code."""
from sys import platform
import tkinter as tk
import tkinter.messagebox as msgbox
from tkinter.filedialog import asksaveasfilename
import itertools
from typing import Callable, Any
import numpy as np
from copy import deepcopy

from alpha_zero.envs.base import BoardGameEnv


class Colors:
    BOARD_BG = '#D8A757'
    PANEL_BG = '#295150'
    BUTTON_BG = '#f2f3f4'
    BLACK = '#100c08'
    WHITE = '#f5f5f5'
    LINE = '#1b1b1b'
    LABEL = '#121212'
    TEXT = '#EDEDED'
    INFO = '#E2B24E'


class BoardGameGui:
    """A simple GUI for board game like Gomoku or Go.
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
        show_steps: bool = True,
        delay: int = 1000,
    ) -> None:
        """
        Args:
            env: the board game environment.
            black_player: a callable function which should return a valid action, or 'human' in the case human vs. AlphaZero.
            white_player: a callable function which should return a valid action, or 'human' in the case human vs. AlphaZero.
            show_steps: if true, show step number of pieces, default on.
            delay: add some delay (in milliseconds) for each move.
        """
        if black_player is None:
            raise ValueError('Invalid black_player')
        if white_player is None:
            raise ValueError('Invalid white_player')
        assert delay >= 1000

        self.env = env
        self.black_player = black_player
        self.white_player = white_player
        self.show_steps = show_steps

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
        self.delay_time = delay

        self.stone_colors = {
            self.env.white_player: Colors.WHITE,
            self.env.black_player: Colors.BLACK,
        }
        self.inverted_colors = {
            self.env.white_player: Colors.BLACK,
            self.env.black_player: Colors.WHITE,
        }

        # UI element sizes
        self.cell_size = 46
        self.piece_size = 38
        self.panel_w = 420
        self.dot_size = 8
        self.padding = 40

        self.font_family = 'Arial'
        self.font_size = 16
        self.title_font_size = 18
        self.heading_font_size = 22

        self.is_os_linux = platform == 'linux'

        if self.is_os_linux:
            self.scale = 2.0
        else:
            self.scale = 1.0

        self.cell_size = int(self.cell_size * self.scale)
        self.piece_size = int(self.piece_size * self.scale)
        self.panel_w = int(self.panel_w * self.scale)
        self.dot_size = int(self.dot_size * self.scale)
        self.padding = int(self.padding * self.scale)

        self.half_size = self.cell_size // 2
        self.board_size = self.env.board_size * self.cell_size
        self.window_w = self.board_size + self.panel_w + self.padding * 2
        self.window_h = self.board_size + self.padding * 2

        self.root = tk.Tk()
        self.root.title(env.id)
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
            3,
            anchor=tk.NW,
            window=self.panel,
        )

        # Variables to update UI at runtime
        self.game_title_var = tk.StringVar(value=self.get_games_title())
        self.black_title_var = tk.StringVar(value=self.get_player_title(self.env.black_player))
        self.white_title_var = tk.StringVar(value=self.get_player_title(self.env.white_player))
        self.black_last_move_var = tk.StringVar(value='Last move:')
        self.white_last_move_var = tk.StringVar(value='Last move:')
        self.black_to_move_var = tk.StringVar()
        self.white_to_move_var = tk.StringVar()
        self.black_info_var = tk.StringVar()
        self.white_info_var = tk.StringVar()

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
            for y in range(
                self.half_size,
                self.board_size - self.half_size + self.cell_size,
                self.cell_size,
            ):
                yield (self.half_size, y), (self.board_size - self.half_size, y)

        def _col_lines():
            for x in range(
                self.half_size,
                self.board_size - self.half_size + self.cell_size,
                self.cell_size,
            ):
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
            self.draw_dot(pos, Colors.LINE, self.dot_size, 'guiding-dot')

    def draw_board_label(self):
        scale = 0.75
        if self.is_os_linux:
            scale = 1.0

        # Row labels
        for j in range(self.num_rows):
            label = tk.Label(
                self.canvas,
                font=(self.font_family, self.font_size, 'bold'),
                text=self.env.gtp_rows[j],
                width=2,
                background=Colors.BOARD_BG,
                foreground=Colors.LABEL,
            )
            x = self.padding * 0.2
            y = j * self.cell_size + self.padding + self.font_size * scale
            label.place(x=x, y=y, anchor='nw')

            label = tk.Label(
                self.canvas,
                font=(self.font_family, self.font_size, 'bold'),
                text=self.env.gtp_rows[j],
                width=2,
                background=Colors.BOARD_BG,
                foreground=Colors.LABEL,
            )
            x = self.board_size + self.padding * 1.2
            y = j * self.cell_size + self.padding + self.font_size * scale
            label.place(x=x, y=y, anchor='nw')

        # Column labels
        for i in range(self.num_cols):
            label = tk.Label(
                self.canvas,
                font=(self.font_family, self.font_size, 'bold'),
                text=self.env.gtp_columns[i],
                width=2,
                background=Colors.BOARD_BG,
                foreground=Colors.LABEL,
            )
            x = i * self.cell_size + self.padding + self.font_size * scale
            y = self.padding + self.board_size
            label.place(x=x, y=y, anchor='nw')

            label = tk.Label(
                self.canvas,
                font=(self.font_family, self.font_size, 'bold'),
                text=self.env.gtp_columns[i],
                width=2,
                background=Colors.BOARD_BG,
                foreground=Colors.LABEL,
            )
            x = i * self.cell_size + self.padding + self.font_size * scale
            y = self.padding * 0.2
            label.place(x=x, y=y, anchor='nw')

    def initialize_panel(self):
        block_h = self.window_h * 0.55
        # if self.is_os_linux:
        #     block_h = self.window_h * 0.6

        top_block = tk.Canvas(
            self.panel,
            width=self.panel_w,
            height=block_h,
            background=Colors.PANEL_BG,
            highlightthickness=0,
        )
        top_block.place(x=0, y=0, anchor='nw')

        # Game title
        game_title = tk.Label(
            top_block,
            font=(self.font_family, self.heading_font_size, 'bold'),
            textvariable=self.game_title_var,
            background=Colors.PANEL_BG,
            foreground=Colors.TEXT,
        )
        game_title.place(relx=0.5, y=self.padding, anchor='center')

        # Player info
        for i, title_var, c, last_move_var, to_move_var, info_var in zip(
            [0, 1],
            [self.black_title_var, self.white_title_var],
            [Colors.BLACK, Colors.WHITE],
            [self.black_last_move_var, self.white_last_move_var],
            [self.black_to_move_var, self.white_to_move_var],
            [self.black_info_var, self.white_info_var],
        ):

            if i == 0:
                offset_x = self.padding
            else:
                offset_x = self.padding + self.panel_w * 0.5
            pos = (offset_x, 50 + self.padding + self.piece_size * 0.5)
            x0, y0, x1, y1 = self.get_circle_from_pos_and_size(pos, self.piece_size * 0.8)
            top_block.create_oval(x0, y0, x1, y1, fill=c, width=0, state=tk.DISABLED)

            player_name = tk.Label(
                top_block,
                font=(self.font_family, self.title_font_size, 'bold'),
                textvariable=title_var,
                background=Colors.PANEL_BG,
                foreground=Colors.TEXT,
            )
            player_name.place(
                x=offset_x + self.piece_size * 0.75,
                y=50 + self.padding + self.title_font_size * 0.25,
                anchor='nw',
            )

            player_last_move = tk.Label(
                top_block,
                font=(self.font_family, self.font_size),
                textvariable=last_move_var,
                background=Colors.PANEL_BG,
                foreground=Colors.TEXT,
            )
            player_last_move.place(x=offset_x, y=self.padding + 120 * self.scale, anchor='nw')

            info = tk.Label(
                top_block,
                font=(self.font_family, self.font_size),
                textvariable=info_var,
                background=Colors.PANEL_BG,
                foreground=Colors.TEXT,
            )
            info.place(x=offset_x, y=self.padding + 145 * self.scale, anchor='nw')

            to_move = tk.Label(
                top_block,
                font=(self.font_family, self.font_size),
                textvariable=to_move_var,
                background=Colors.PANEL_BG,
                foreground=Colors.INFO,
            )
            to_move.place(x=offset_x, y=self.padding + 170 * self.scale, anchor='nw')

        # Action buttons
        action_block = tk.Canvas(
            self.panel,
            width=self.panel_w,
            height=self.window_h - block_h,
            background=Colors.PANEL_BG,
            highlightthickness=0,
        )
        action_block.place(x=0, y=block_h, anchor='nw')

        if self.is_go() and self.human_player is not None:
            pass_btn = tk.Label(
                action_block,
                text='Pass',
                font=(self.font_family, self.font_size),
                background=Colors.BUTTON_BG,
                foreground=Colors.BLACK,
                padx=8,
                pady=2,
            )

            pass_btn.place(relx=0.5, y=self.padding * 1, anchor='center')

            pass_btn.bind('<Button-1>', self.click_on_pass)

        save_game_btn = tk.Label(
            action_block,
            text='Save Game',
            font=(self.font_family, self.font_size),
            background=Colors.BUTTON_BG,
            foreground=Colors.BLACK,
            padx=8,
            pady=2,
        )
        save_game_btn.place(relx=0.5, y=self.padding * 2.25, anchor='center')

        new_game_btn = tk.Label(
            action_block,
            text='New Game',
            font=(self.font_family, self.font_size),
            background=Colors.BUTTON_BG,
            foreground=Colors.BLACK,
            padx=8,
            pady=2,
        )
        new_game_btn.place(relx=0.5, y=self.padding * 3.5, anchor='center')

        exit_btn = tk.Label(
            action_block,
            text='Exit',
            font=(self.font_family, self.font_size),
            background=Colors.BUTTON_BG,
            foreground=Colors.BLACK,
            padx=8,
            pady=2,
        )
        exit_btn.place(relx=0.5, y=self.padding * 4.75, anchor='center')

        save_game_btn.bind('<Button-1>', self.click_on_save_game)
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

    def draw_dot(self, pos, color, size, tags='stone'):
        x0, y0, x1, y1 = self.get_circle_from_pos_and_size(pos, size)
        self.board.create_oval(x0, y0, x1, y1, fill=color, width=0, state=tk.DISABLED, tags=tags)

    def draw_open_circle(self, pos, color, size, tags='stone'):
        x0, y0, x1, y1 = self.get_circle_from_pos_and_size(pos, size)
        self.board.create_oval(
            x0,
            y0,
            x1,
            y1,
            fill=None,
            width=2,
            outline=color,
            state=tk.DISABLED,
            tags=tags,
        )

    def draw_stone(self, pos, color, invert_color, text, show_indicator=False, tags='stone'):
        self.draw_dot(pos, color, self.piece_size)

        if show_indicator:
            self.draw_open_circle(pos, invert_color, self.piece_size * 0.6)

        if self.show_steps:
            self.board.create_text(pos[0], pos[1], text=text, fill=invert_color, tags=tags)

    def draw_piece_on_board(self, move, step, color, invert_color, is_last_move):
        coords = self.env.action_to_coords(move)
        pos = self.env_coords_to_board_position(coords)
        self.draw_stone(pos, color, invert_color, step, is_last_move)

    def redraw_board(self, board, history):
        # Clean up board
        self.board.delete('stone')

        board = np.copy(board)
        for i, item in enumerate(history):
            if self.env.is_pass_move(item.move):
                continue
            coords = self.env.action_to_coords(item.move)
            if board[coords[0], coords[1]] == 0:
                continue

            id = self.env.white_player if item.color == 'W' else self.env.black_player
            color = self.stone_colors[id]
            invt_color = self.inverted_colors[id]

            self.draw_piece_on_board(item.move, i + 1, color, invt_color, i == len(history) - 1)

    def click_on_board(self, event):
        if self.env.is_game_over() or self.human_player is None or not self.is_human_to_play():
            return

        coords = self.pos_to_env_coords((event.x, event.y))
        action = self.env.coords_to_action(coords)

        self.make_move(action)

    def click_on_pass(self, event):
        if self.is_go() and self.human_player is not None:
            if self.is_human_to_play():
                self.make_move(self.env.pass_move)

    def click_on_save_game(self, event):
        try:
            sgf_content = self.env.to_sgf()
            file_name = asksaveasfilename(
                title='Save game as sgf file',
                filetypes=(('sgf files', '*.sgf'), ('all files', '*.*')),
            )
            with open(file_name, 'w') as f:
                f.write(sgf_content)
                f.close()
            msgbox.showinfo('Success', f'Game saved to {file_name}')
        except Exception:
            pass

    def click_on_new_game(self, event):
        self.reset()

    def click_on_exit(self, event):
        self.close()

    def get_current_player(self):
        if self.env.is_game_over():
            return
        if self.env.to_play == self.env.black_player:
            return self.black_player
        else:
            return self.white_player

    def is_human_to_play(self):
        if self.env.to_play == self.env.black_player and self.human_player == 'black':
            return True
        elif self.env.to_play == self.env.white_player and self.human_player == 'white':
            return True
        else:
            return False

    def is_go(self):
        return self.env.id == 'Go'

    def make_move(self, action):
        # Avoid making illegal move.
        if not self.env.is_legal_move(action):
            player = 'Black' if self.env.to_play == self.env.black_player else 'White'
            msgbox.showwarning('Warning', f'{player} move {self.env.action_to_gtp(action)} is illegal')
            return

        self.env.step(action)

        self.redraw_board(np.copy(self.env.board), deepcopy(self.env.history))

        self.update_game_info()

    def play(self):
        # Let AI make a move.
        if not self.env.is_game_over() and not self.is_human_to_play():
            player = self.get_current_player()
            if player:
                action = player(self.env)
                self.make_move(action)
        if not self.env.is_game_over():
            self.set_loop()
        else:
            self.clear_loop()
            self.update_match_results_info()

    def update_game_info(self):
        last_move_var = None
        last_move_gtp = self.env.action_to_gtp(self.env.last_move)
        if self.env.last_player == self.env.black_player:
            last_move_var = self.black_last_move_var
        elif self.env.last_player == self.env.white_player:
            last_move_var = self.white_last_move_var

        if last_move_var:
            last_move_var.set(f'Last move: {last_move_gtp}')

        if self.env.is_game_over():
            self.black_to_move_var.set('')
            self.white_to_move_var.set('')
        elif self.env.to_play == self.env.black_player:
            self.black_to_move_var.set('To move')
            self.white_to_move_var.set('')
        else:
            self.white_to_move_var.set('To move')
            self.black_to_move_var.set('')

        if self.is_go():
            caps = self.env.get_captures()

            self.black_info_var.set(f'Captures: {caps[self.env.black_player]}')
            self.white_info_var.set(f'Captures: {caps[self.env.white_player]}')

    def update_match_results_info(self):
        if self.env.winner == self.env.black_player:
            self.black_won_games += 1
        elif self.env.winner == self.env.white_player:
            self.white_won_games += 1

        self.game_title_var.set(self.get_games_title())

        self.black_title_var.set(self.get_player_title(self.env.black_player))
        self.white_title_var.set(self.get_player_title(self.env.white_player))

    def get_games_title(self):
        txt = f'GAME {self.played_games + 1}'
        if self.env.is_game_over():
            if self.env.winner is not None:
                result_str = self.env.get_result_string()
                if self.env.winner == self.env.black_player:
                    txt += ' - black won'
                else:
                    txt += ' - white won'
                txt += f' ({result_str})'
            else:
                txt += ' - draw'
        return txt.upper()

    def get_player_title(self, player_id):
        if player_id == self.env.black_player:
            if self.black_won_games > 0:
                return f'{self.black_player_name} + {self.black_won_games}'
            return self.black_player_name
        elif player_id == self.env.white_player:
            if self.white_won_games > 0:
                return f'{self.white_player_name} + {self.white_won_games}'
            return self.white_player_name
        else:
            return ''

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
        self.game_title_var.set(self.get_games_title())

        self.env.reset()
        self.last_move = None
        self.game_loop = None

        self.initialize_board()
        self.update_game_info()
        self.update_match_results_info()
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
