"""A simple GUI for board game."""
import tkinter as tk
import itertools
from typing import Callable, Any

from alpha_zero.games.env import BoardGameEnv


class Colors:
    BACKGROUND = '#2D292E'
    BLACK = '#000000'
    WHITE = '#ffffff'
    BOARD = '#DBA14B'
    LINE = '#1b1b1b'
    LABEL = '#121212'
    RED = '#EF0107'
    BOX = '#4F4C51'
    TEXT = '#EDEDED'
    BUTTON = '#f2f3f4'


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
        cell_size: int = 46,
        piece_size: int = 38,
        show_step: bool = True,
        caption: str = 'Free-style Gomoku',
    ) -> None:
        """
        Args:
            env: the board bame environment.
            black_player: a callable function which should return a valid action, or 'human' in the case human vs. AlphaZero.
            white_player: a callable function which should return a valid action, or 'human' in the case human vs. AlphaZero.
            cell_size: board cell size.
            piece_size: stone size.
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

        self.cell_size = cell_size
        self.piece_size = piece_size
        self.half_size = cell_size // 2
        self.dot_size = 8
        self.num_rows = self.env.board_size
        self.num_cols = self.env.board_size
        self.padding = 30

        self.board_size = self.env.board_size * self.cell_size

        self.panel_width = 340
        self.panel_height = self.board_size + self.padding * 2

        self.window_w = self.board_size + self.panel_width + self.padding * 2 + 8
        self.window_h = self.board_size + self.padding * 2

        self.window = tk.Tk()
        self.window.title(caption)
        self.window.resizable(0, 0)

        self.canvas = tk.Canvas(self.window, width=self.window_w, height=self.window_h, bg=Colors.BACKGROUND)
        self.canvas.pack()

        # We add column and row labels on main canvas.
        self.main = tk.Canvas(
            self.canvas,
            width=self.board_size + self.padding * 2,
            height=self.board_size + self.padding * 2,
            bg=Colors.BOARD,
            # highlightthickness=0,
        )
        self.board = tk.Canvas(self.main, width=self.board_size, height=self.board_size, bg=Colors.BOARD, highlightthickness=0)

        self.panel = tk.Canvas(
            self.canvas, width=self.panel_width, height=self.board_size, bg=Colors.BACKGROUND, highlightthickness=0
        )

        self.main.create_window(self.padding, self.padding, anchor=tk.NW, window=self.board)
        self.canvas.create_window(0, 0, anchor=tk.NW, window=self.main)
        self.canvas.create_window(self.padding * 2 + self.board_size + 8, self.padding, anchor=tk.NW, window=self.panel)

        if self.human_player is not None:
            self.board.bind('<Button-1>', self.click_on_board)

        self.col_labels = 'ABCDEFGHIJKLMNOPQRS'
        self.row_labels = [str(i) for i in range(1, 20)]

        self.played_games = 0
        self.black_won_games = 0
        self.white_won_games = 0
        self.black_won_var = tk.StringVar()
        self.white_won_var = tk.StringVar()

        self.black_last_move_var = tk.StringVar()
        self.white_last_move_var = tk.StringVar()
        self.black_info_var = tk.StringVar()
        self.white_info_var = tk.StringVar()

        self.added_board_label = False

        self.initialize_board()

        self.initialize_panel()

        self.update_info()

        self.player_colors = {'white': Colors.WHITE, 'black': Colors.BLACK}

        self.last_move = None
        self.game_loop = None

        self.won_count_updated = False
        self.loop_time = 1000

        self.env.reset()

        self.current_player = self.env.current_player_name

    def get_row_lines(self):
        half = self.half_size

        for y in range(half, self.board_size - half + self.cell_size, self.cell_size):
            yield (half, y), (self.board_size - half, y)

    def get_col_lines(self):
        half = self.half_size

        for x in range(half, self.board_size - half + self.cell_size, self.cell_size):
            yield (x, half), (x, self.board_size - half)

    def get_guide_dots(self):
        # Guide dots
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

    def initialize_board(self):
        # Grid lines.
        lines = itertools.chain(self.get_col_lines(), self.get_row_lines())

        for start, end in lines:
            self.board.create_line(start[0], start[1], end[0], end[1], fill=Colors.LINE, width=2)

        # Guiding dots.
        dots = self.get_guide_dots()

        for dot in dots:
            pos = self.env_coords_to_board_position(dot)
            self.draw_dot_on_board(pos, Colors.LINE, self.dot_size)

        # Column and row labels.
        if not self.added_board_label:
            self.draw_board_label()
            self.added_board_label = True

    def draw_board_label(self):
        font_size = 16
        for i in range(self.num_cols):
            label = tk.Label(
                self.main,
                font=('Helvetica', font_size, 'bold'),
                text=self.col_labels[i],
                background=Colors.BOARD,
                foreground=Colors.LABEL,
            )
            x = i * self.cell_size + self.half_size - 8 + self.padding
            y = self.padding * 0.25
            label.place(x=x, y=y, anchor='nw')

        for j in range(self.num_rows):
            label = tk.Label(
                self.main,
                font=('Helvetica', font_size, 'bold'),
                text=self.row_labels[j],
                background=Colors.BOARD,
                foreground=Colors.LABEL,
            )
            x = self.padding * 0.25
            y = j * self.cell_size + self.half_size - 15 + self.padding
            label.place(x=x, y=y, anchor='nw')

    def initialize_panel(self):
        box_w = int(self.panel_width * 0.85)
        box_h = 140
        offset_x = (self.panel_width - box_w) // 2
        offset_y = offset_x
        padding = 15

        # Black player info
        self.create_player_info_box(
            box_w,
            box_h,
            (offset_x, 0),
            padding,
            self.black_player_name,
            Colors.BLACK,
            self.black_won_var,
            self.black_last_move_var,
            self.black_info_var,
        )

        # White player info
        self.create_player_info_box(
            box_w,
            box_h,
            (offset_x, offset_y + box_h),
            padding,
            self.white_player_name,
            Colors.WHITE,
            self.white_won_var,
            self.white_last_move_var,
            self.white_info_var,
        )

        # Actions
        buttons_canvas = tk.Canvas(
            self.panel, width=box_w, height=self.panel_height - box_h * 2, bg=Colors.BACKGROUND, highlightthickness=0
        )

        buttons_canvas.place(x=offset_x, y=box_h * 2 + offset_y * 2, anchor='nw')

        button_font_size = 12
        new_game_btn = tk.Label(
            buttons_canvas,
            text='New Game',
            font=('Helvetica', button_font_size),
            background=Colors.BUTTON,
            foreground=Colors.BLACK,
            padx=12,
            pady=2,
        )
        new_game_btn.place(relx=0.5, y=15, anchor='center')

        exit_btn = tk.Label(
            buttons_canvas,
            text='Exit',
            font=('Helvetica', button_font_size),
            background=Colors.BUTTON,
            foreground=Colors.BLACK,
            padx=12,
            pady=2,
        )
        exit_btn.place(relx=0.5, y=60, anchor='center')

        new_game_btn.bind('<Button-1>', self.click_on_new_game)
        exit_btn.bind('<Button-1>', self.click_on_exit)

    def create_player_info_box(self, width, height, offset, padding, title, piece_color, win_var, last_move_var, info_var):
        # Box
        player_box = tk.Canvas(self.panel, width=width, height=height, background=Colors.BOX, highlightthickness=0)
        player_box.place(x=offset[0], y=offset[1], anchor='nw')

        # Symbol
        pos = (padding + self.piece_size // 2, padding + self.piece_size // 2)
        half_size = self.piece_size // 2
        start_x = pos[0] - half_size
        start_y = pos[1] - half_size
        end_x = pos[0] + half_size
        end_y = pos[1] + half_size
        player_box.create_oval(start_x, start_y, end_x, end_y, fill=piece_color, width=0)

        # Title
        title = tk.Label(player_box, font=('Helvetica', 28, 'bold'), text=title, background=Colors.BOX, foreground=Colors.TEXT)
        title.place(x=padding * 2 + self.piece_size, y=padding * 0.95, anchor='nw')

        # Win counter
        counter = tk.Label(
            player_box, font=('Helvetica', 17, 'bold'), textvariable=win_var, background=Colors.BOX, foreground=Colors.TEXT
        )
        counter.place(x=padding * 2 + self.piece_size, y=padding * 2 + 30, anchor='nw')

        # Additional info
        last_move = tk.Label(
            player_box, font=('Helvetica', 17), textvariable=last_move_var, background=Colors.BOX, foreground=Colors.TEXT
        )
        last_move.place(x=padding * 2 + self.piece_size, y=padding * 2 + 55, anchor='nw')

        info = tk.Label(
            player_box, font=('Helvetica', 17), textvariable=info_var, background=Colors.BOX, foreground=Colors.RED
        )
        info.place(x=padding * 2 + self.piece_size, y=padding * 2 + 80, anchor='nw')

    def env_coords_to_board_position(self, coords):
        row, col = coords
        pos = (
            col * self.cell_size + self.half_size,
            row * self.cell_size + self.half_size,
        )
        return pos

    def board_position_to_env_coords(self, pos):
        # Screen pos (x, y) is no the same as in row major numpy.array.
        x, y = pos
        row = y // self.cell_size
        col = x // self.cell_size
        return (row, col)

    def env_coords_to_human_label(self, coords):
        x, y = coords
        return self.col_labels[y] + self.row_labels[x]

    def draw_dot_on_board(self, pos, color, size):
        half_size = size / 2
        start_x = pos[0] - half_size
        start_y = pos[1] - half_size
        end_x = pos[0] + half_size
        end_y = pos[1] + half_size
        self.board.create_oval(start_x, start_y, end_x, end_y, fill=color, width=0, state=tk.DISABLED)

    def draw_circle_on_board(self, pos, color, size):
        half_size = size / 2
        start_x = pos[0] - half_size
        start_y = pos[1] - half_size
        end_x = pos[0] + half_size
        end_y = pos[1] + half_size

        self.board.create_oval(start_x, start_y, end_x, end_y, fill=None, width=2, outline=color, state=tk.DISABLED)

    def draw_piece_on_board(self, pos, color, text_color, size, text, show_indicator=False):
        self.draw_dot_on_board(pos, color, size)
        if show_indicator:
            self.draw_circle_on_board(pos, text_color, size * 0.6)
        if self.show_step:
            self.board.create_text(pos[0], pos[1], text=text, fill=text_color)

    def draw_piece(self, coords):
        color = self.player_colors[self.env.current_player_name]
        text_color = self.player_colors[self.env.opponent_player_name]

        # Re-draw last move to remove border
        if self.last_move is not None:
            last_pos, last_color, last_text_color = self.last_move
            self.draw_piece_on_board(last_pos, last_color, last_text_color, self.piece_size, self.env.steps)

        pos = self.env_coords_to_board_position(coords)
        self.draw_piece_on_board(pos, color, text_color, self.piece_size, self.env.steps + 1, True)
        self.last_move = (pos, color, text_color)

    def click_on_board(self, event):
        if self.env.is_game_over or self.human_player is None or self.env.current_player_name != self.human_player:
            return

        pos = (event.x, event.y)
        coords = self.board_position_to_env_coords(pos)

        action = self.env.coords_to_action(coords)

        # Avoid repeated clicks on same interesection.
        if not self.env.is_action_valid(action):
            return

        self.draw_piece(coords)
        self.update_last_move(action)

        self.env.step(action)

        self.current_player = self.env.current_player_name
        self.update_info()

        if self.env.is_game_over and not self.won_count_updated:
            self.update_win_counter()
            self.won_count_updated = True

    def click_on_new_game(self, event):
        self.reset()

    def click_on_exit(self, event):
        self.close()

    def update_info(self):
        if self.env.is_game_over:
            if self.env.winner == self.env.black_player_id:
                self.black_info_var.set('Won')
                self.white_info_var.set('Loss')
            elif self.env.winner == self.env.white_player_id:
                self.black_info_var.set('Loss')
                self.white_info_var.set('Won')
            else:
                self.black_info_var.set('Draw')
                self.white_info_var.set('Draw')
        else:
            if self.env.current_player == self.env.black_player_id:
                self.black_info_var.set('To move')
                self.white_info_var.set('')
            else:
                self.black_info_var.set('')
                self.white_info_var.set('To move')

    def update_last_move(self, action):
        if action == self.env.resign_action:
            label = 'resign'
        else:
            coords = self.env.action_to_coords(action)
            label = self.env_coords_to_human_label(coords)
        if self.env.current_player == self.env.black_player_id:
            self.black_last_move_var.set(f'Last move: {label}')
        else:
            self.white_last_move_var.set(f'Last move: {label}')

    def get_next_player(self):
        if self.env.current_player_name == 'black':
            return self.black_player
        elif self.env.current_player_name == 'white':
            return self.white_player

    def make_move(self, action):
        # Avoid repeated clicks on same interesection.
        if not self.env.is_action_valid(action):
            return

        coords = self.env.action_to_coords(action)
        self.draw_piece(coords)

        self.update_last_move(action)

        self.env.step(action)

        self.update_info()

    def play(self):
        # Let AlphaZero make a move.
        if (
            not self.env.is_game_over
            and self.current_player == self.env.current_player_name
            and self.current_player != self.human_player
        ):
            current_player = self.get_next_player()
            action = current_player(self.env)
            self.make_move(action)
            self.current_player = self.env.current_player_name

        if not self.env.is_game_over:
            self.set_loop()
        else:
            if not self.won_count_updated:
                self.update_win_counter()
                self.won_count_updated = True

            self.clear_loop()

    def update_win_counter(self):

        self.played_games += 1
        if self.env.winner_name == 'black':
            self.black_won_games += 1
        elif self.env.winner_name == 'white':
            self.white_won_games += 1

        self.black_won_var.set(f'Won: {self.black_won_games}/{self.played_games}')
        self.white_won_var.set(f'Won: {self.white_won_games}/{self.played_games}')

    def start(self):
        self.window.eval('tk::PlaceWindow . center')

        self.set_loop()

        self.window.mainloop()

    def set_loop(self):
        # Call the play function after 1 sec.
        self.game_loop = self.window.after(self.loop_time, self.play)

    def close(self):
        self.clear_loop()
        self.env.close()
        self.window.destroy()

    def clear_loop(self):
        if self.game_loop is not None:
            self.window.after_cancel(self.game_loop)

    def reset(self):
        self.clear_loop()

        if not self.env.is_game_over:
            self.update_win_counter()

        self.env.reset()
        self.last_move = None
        self.current_player = self.env.current_player_name

        self.game_loop = None
        self.won_count_updated = False

        self.board.delete('all')
        self.black_last_move_var.set('')
        self.white_last_move_var.set('')

        self.initialize_board()
        self.update_info()

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
