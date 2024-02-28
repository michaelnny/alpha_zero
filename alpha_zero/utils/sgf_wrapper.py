# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to extract a series of positions + their next moves from an SGF.

Most of the complexity here is dealing with two features of SGF:
- Stones can be added via "play move" or "add move", the latter being used
  to configure L+D puzzles, but also for initial handicap placement.
- Plays don't necessarily alternate colors; they can be repeated B or W moves
  This feature is used to handle free handicap placement.
"""
import itertools
import re
import sgf

from alpha_zero.envs.coords import CoordsConvertor
from alpha_zero.envs import go_engine as go


SGF_TEMPLATE = """(;\nCA[UTF-8]\nAP[AlphaZeroMini_sgfgenerator]\nRU[{ruleset}]
PB[{black_name}]\nBR[{black_rank}]\nPW[{white_name}]\nWR[{white_rank}]
KM[{komi}]\nRE[{result}]\nDT[{date}]\nSZ[{boardsize}]\n
{game_moves})"""


def translate_sgf_move(cc, player_move, comment):
    if player_move.color not in ('B', 'W'):
        raise ValueError("Can't translate color %s to sgf" % player_move.color)
    if comment is not None:
        comment = comment.replace(']', r'\]')
        comment_node = 'C[{}]'.format(comment)
    else:
        comment_node = ''
    return ';{color}[{coords}]{comment_node}'.format(
        color=player_move.color,
        coords=cc.to_sgf(cc.from_flat(player_move.move)),
        comment_node=comment_node,
    )


def make_sgf(
    board_size,
    move_history,
    result_string,
    ruleset='Chinese',
    komi=7.5,
    white_name='AlphaZeroMini',
    white_rank='',
    black_name='AlphaZeroMini',
    black_rank='',
    date='',
    comments=[],
):
    """Turn a game into SGF.

    Doesn't handle handicap games or positions with incomplete history.

    Args:
        board_size: board size
        move_history: iterable of PlayerMoves
        result_string: "B+R", "W+0.5", etc.
        comments: iterable of string/None. Will be zipped with move_history.
    """
    boardsize = board_size
    cc = CoordsConvertor(board_size)
    game_moves = [translate_sgf_move(cc, *z) for z in itertools.zip_longest(move_history, comments)]

    # Preprocess the list by adding a newline character after every 10th element
    game_moves = [game_moves[i] + '\n' if (i + 1) % 10 == 0 else game_moves[i] for i in range(len(game_moves))]

    game_moves = ''.join(game_moves)
    result = result_string
    return SGF_TEMPLATE.format(**locals())


def sgf_prop(value_list):
    'Converts raw sgf library output to sensible value'
    if value_list is None:
        return None
    if len(value_list) == 1:
        return value_list[0]
    else:
        return value_list


def get_sgf_root_node(sgf_contents):
    collection = sgf.parse(sgf_contents)
    game = collection.children[0]
    return game.root


def parse_game_result(result):
    'Parse an SGF result string into value target.'
    if re.match(r'[bB]\+', result):
        return go.BLACK
    if re.match(r'[wW]\+', result):
        return go.WHITE
    return None
