# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Implements the functions to build a evaluation dataset by loading Go game (in sgf format)"""
import os
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset

from alpha_zero.envs.go import GoEnv
from alpha_zero.utils import sgf_wrapper
from alpha_zero.utils.util import create_logger

# keep a history of player names so we can avoid adding duplicates
GAME_COUNTS = {}
MATCHES = []
MISMATCH_GAMES = {
    'winner_mismatch': 0,
    'score_mismatch': 0,
    'score_mismatch_le_1': 0,
    'score_mismatch_gt_1_le_2': 0,
    'score_mismatch_gt_2_le_4': 0,
    'score_mismatch_gt_4': 0,
}


def _one_hot(index: int, action_dim: int) -> np.ndarray:
    onehot = np.zeros([action_dim], dtype=np.float32)
    onehot[index] = 1.0
    return onehot


def _get_player_str(player: str) -> str:
    player = re.sub(r'\([^)]*\)', '', player)  # Remove parentheses and everything inside them
    player = re.sub(r'[^a-zA-Z0-9 ]', '', player)  # Remove special characters except spaces
    return player.strip()  # Remove leading/trailing spaces


def get_sgf_files(games_dir):
    results = []

    # Recursively search for .sgf files in the directory and its subdirectories
    if os.path.exists(games_dir):
        for root, dirnames, filenames in os.walk(games_dir):
            for f in filenames:
                if f.endswith('.sgf'):
                    results.append(os.path.join(root, f))

    return results


def _extract_ratings(black_player, white_player, black_rank, white_rank):
    ratings = []
    # Case 1 - elo rating is located inside another property like 'WR[2345]' or 'BR[2345]'
    # needs to skip the case where ranks are like '9d' or '10p'
    if all(
        rank is not None and rank != '' and 'k' not in rank and 'd' not in rank and 'p' not in rank
        for rank in (black_rank, white_rank)
    ):
        for rank in (black_rank, white_rank):
            try:
                rank = re.sub(r'[^0-9]', '', rank)
                ratings.append(int(rank))
            except Exception:
                pass
    # Case 2 - elo rating is located inside player name like 'PW[James (2435)]'
    elif all('(' in player_id and ')' in player_id for player_id in (black_player, white_player)):
        for player_id in (black_player, white_player):
            elo = re.search(r'\((\d+)\)', player_id)
            if elo:
                ratings.append(int(elo.group(1)))
    return ratings


# A elo of 2100 is roughly the level of amateur 1 dan
def replay_sgf(sgf_file, num_stack, logger, skip_n=0, min_elo=2100, max_games_per_player=200):  # noqa: C901
    """Replay a game in sgf format and return the transitions tuple (states, target_pi, target_v) for every move in the game."""
    sgf_content = None

    try:
        with open(sgf_file) as f:
            sgf_content = f.read()
            f.close()
        root_node = sgf_wrapper.get_sgf_root_node(sgf_content)
    except Exception:
        return None

    props = root_node.properties

    board_size = sgf_wrapper.sgf_prop(props.get('SZ', ''))
    if board_size is None or board_size == '' or int(board_size) != int(os.environ['BOARD_SIZE']):
        logger.debug(f'Game "{sgf_file}" board size mismatch')
        return None

    result_str = sgf_wrapper.sgf_prop(props.get('RE', ''))
    if result_str is None or result_str == '' or len(result_str) < 3:
        logger.debug(f'Game "{sgf_file}" has no result property')
        return None
    elif re.search(r'\+T', result_str):  # Skip won by timeout
        logger.debug(f'Game "{sgf_file}" with result {result_str} does not have a natural winner')
        return None

    black_player = sgf_wrapper.sgf_prop(props.get('PB', ''))
    white_player = sgf_wrapper.sgf_prop(props.get('PW', ''))

    # For those online games, it's difficult to assert the player's strengthen or rank, as there's no such recordings, only elo ratings.
    black_rank = sgf_wrapper.sgf_prop(props.get('BR', ''))
    white_rank = sgf_wrapper.sgf_prop(props.get('WR', ''))

    elo_ratings = _extract_ratings(black_player, white_player, black_rank, white_rank)
    if len(elo_ratings) > 0 and any(v < min_elo for v in elo_ratings):
        logger.info(f'Game "{sgf_file}" with player ratings {elo_ratings} is too weak')
        return None

    # Avoid potential duplicates
    black_id = _get_player_str(black_player)
    white_id = _get_player_str(white_player)

    # Find all move sequences
    move_sequences = re.findall(r';[BW]\[[a-z]{0,2}\]', sgf_content)

    # Count the number of move sequences
    num_moves = len(move_sequences)

    match_str = f'{black_id}-{white_id}-{num_moves}-{result_str}'
    if match_str in MATCHES:
        logger.info(f'Game "{sgf_file}" might be duplicate')
        return None
    MATCHES.append(match_str)

    # Avoid too much games from the same player
    for id in [black_id, white_id]:
        if id in GAME_COUNTS:
            if GAME_COUNTS[id] > max_games_per_player:
                logger.info(f'Too many games from player {id}')
                return None
            GAME_COUNTS[id] += 1
        else:
            GAME_COUNTS[id] = 1

    komi = 0
    if props.get('KM') is not None:
        komi = float(sgf_wrapper.sgf_prop(props.get('KM')))

    env = GoEnv(komi=komi, num_stack=num_stack)
    obs = env.reset()

    winner = None
    if re.match(r'B\+', result_str, re.IGNORECASE):
        winner = env.black_player
    elif re.match(r'W\+', result_str, re.IGNORECASE):
        winner = env.white_player

    node = root_node
    assert node.first

    history = []

    # Replay the game, check for end move, also exclude 'TW' and 'TB', which are territory markup
    while node.next is not None and 'TW' not in node.next.properties and 'TB' not in node.next.properties:
        next_player = None
        next_move = None

        props = node.next.properties
        if 'W' in props:
            next_player = env.white_player
            next_move = env.cc.from_sgf(props['W'][0])
        elif 'B' in props:
            next_player = env.black_player
            next_move = env.cc.from_sgf(props['B'][0])

        if next_player is None:
            return

        next_move = env.cc.to_flat(next_move)

        if not env.is_legal_move(next_move):
            return None

        if next_player is not None and env.to_play != next_player:
            # Game might have handicap moves
            return None

        value = 0.0
        if winner is not None and winner in [env.black_player, env.white_player]:
            if winner == next_player:
                value = 1.0
            else:
                value = -1.0

        if env.steps > skip_n:
            history.append((obs, _one_hot(next_move, env.action_dim), value))

        try:
            obs, _, _, _ = env.step(next_move)
            node = node.next
        except Exception:
            logger.debug(f"Skipping game '{sgf_file}', as move {node.next.properties} at step {env.steps} is illegal")
            return None

    if env.steps != num_moves:
        return None

    # Additional check to see how many games have mismatching results
    env_result_str = env.get_result_string()
    env_result_str = env_result_str.upper()
    result_str = result_str.upper()
    if not re.search(r'\+T', result_str, re.IGNORECASE) and not re.search(r'\+R', result_str, re.IGNORECASE):
        is_mismatch = False
        if env_result_str[:2] != result_str[:2]:
            is_mismatch = True
            MISMATCH_GAMES['winner_mismatch'] += 1
        else:
            sgf_score = re.findall(r'[-+]?\d*\.\d+|\d+', result_str)
            env_score = re.findall(r'[-+]?\d*\.\d+|\d+', env_result_str)
            if sgf_score:
                sgf_score = float(sgf_score[0])
            if env_score:
                env_score = float(env_score[0])
            if sgf_score != env_score:
                is_mismatch = True
                MISMATCH_GAMES['score_mismatch'] += 1
                delta = abs(sgf_score - env_score)
                if delta <= 1:
                    MISMATCH_GAMES['score_mismatch_le_1'] += 1
                elif 1 < delta <= 2:
                    MISMATCH_GAMES['score_mismatch_gt_1_le_2'] += 1
                elif 2 < delta <= 4:
                    MISMATCH_GAMES['score_mismatch_gt_2_le_4'] += 1
                else:
                    MISMATCH_GAMES['score_mismatch_gt_4'] += 1

        if is_mismatch:
            logger.debug(f'Game "{sgf_file}" has mismatching result, env result: {env_result_str}, SGF result: {result_str}')

    return history


def build_eval_dataset(games_dir, num_stack, logger=None) -> TensorDataset:
    if logger is None:
        logger = create_logger()

    logger.info('Building evaluation dataset...')

    sgf_files = get_sgf_files(games_dir)

    states = []
    target_pi = []
    target_v = []

    valid_games = 0
    for sgf_file in sgf_files:
        history = replay_sgf(sgf_file, num_stack, logger)
        if history is None:
            continue
        valid_games += 1
        for transition in history:
            states.append(transition[0])
            target_pi.append(transition[1])
            target_v.append(transition[2])

    states = torch.from_numpy(np.stack(states, axis=0)).to(dtype=torch.float32)
    target_pi = torch.from_numpy(np.stack(target_pi, axis=0)).to(dtype=torch.float32)
    target_v = torch.from_numpy(np.stack(target_v, axis=0)).to(dtype=torch.float32)

    eval_dataset = TensorDataset(states, target_pi, target_v)

    logger.warning(f'Number of games with mismatched results: {MISMATCH_GAMES}')
    sorted_game_counts = dict(sorted(GAME_COUNTS.items(), key=lambda x: x[1], reverse=True))
    logger.debug(f'Number of games by player: {sorted_game_counts}')

    logger.info(f'Finished loading {len(eval_dataset)} positions from {valid_games} games')
    return eval_dataset
