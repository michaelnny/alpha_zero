# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


import os
import shutil
import re
import sgf_wrapper

from util import create_logger

# keep a history of player names so we can avoid adding duplicates
GAME_COUNTS = {}
MATCHES = []


def _get_player_str(player):
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


def is_valid_game(sgf_file, size, min_elo, max_games_per_player):
    sgf_content = None

    try:
        with open(sgf_file) as f:
            sgf_content = f.read()
            f.close()
        root_node = sgf_wrapper.get_sgf_root_node(sgf_content)
    except:
        return False

    props = root_node.properties

    board_size = sgf_wrapper.sgf_prop(props.get('SZ', ''))
    if board_size is None or board_size == '' or int(board_size) != int(size):
        return False

    result_str = sgf_wrapper.sgf_prop(props.get('RE', ''))
    if result_str is None or result_str == '' or len(result_str) < 3:
        # logger.info(f'Game "{sgf_file}" has no valid result property')
        return False
    elif re.search(r'\+T', result_str):  # Skip won by timeout
        # logger.info(f'Game "{sgf_file}" with result {result_str} does not have a natural winner')
        return False

    black_player = sgf_wrapper.sgf_prop(props.get('PB', ''))
    white_player = sgf_wrapper.sgf_prop(props.get('PW', ''))

    # For those online games, it's difficult to assert the player's strengthen or rank, as there's no such recordings, only elo ratings.
    ranks = []
    black_rank = sgf_wrapper.sgf_prop(props.get('BR', ''))
    white_rank = sgf_wrapper.sgf_prop(props.get('WR', ''))

    # Case 1 - elo rating is located inside another property like 'WR[2345]' or 'BR[2345]'
    # needs to skip the case where ranks are like '9d' or '10p'
    if all(
        rank is not None and rank != '' and 'k' not in rank and 'd' not in rank and 'p' not in rank
        for rank in (black_rank, white_rank)
    ):
        for rank in (black_rank, white_rank):
            try:
                rank = re.sub(r'[^0-9]', '', rank)
                ranks.append(int(rank))
            except:
                logger.info(f'Game "{sgf_file}" rank: {rank}')
    # Case 2 - elo rating is located inside player name like 'PW[James (2435)]'
    elif all('(' in player_id and ')' in player_id for player_id in (black_player, white_player)):
        for player_id in (black_player, white_player):
            elo = re.search(r'\((\d+)\)', player_id)
            if elo:
                ranks.append(int(elo.group(1)))

    if len(ranks) > 0 and any(v < min_elo for v in ranks):
        logger.info(f'Game "{sgf_file}" with player ranks {ranks} is too weak')
        return False

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
        return False
    MATCHES.append(match_str)

    # Avoid too much games from the same player
    for id in [black_id, white_id]:
        if id in GAME_COUNTS:
            if GAME_COUNTS[id] > max_games_per_player:
                logger.info(f'Too many games from player {id}')
                return False
            GAME_COUNTS[id] += 1
        else:
            GAME_COUNTS[id] = 1

    return True


if __name__ == '__main__':
    """Find all games in sgf format from the source directory,
    and filter out those by board size, minimum elo ratings.
    Additionally, avoid adding too much games from the same player.
    After than, clean up the target directory and then copy the matched games to the target directory.
    """
    source_dir = '/Users/michael/Downloads/go_games'
    target_dir = './pro_games/go/9x9'

    board_size = 9
    min_elo = 2100  # A elo of 2100 is roughly the level of amateur 1 dan
    max_games_per_player = 200

    assert os.path.exists(source_dir) and os.path.isdir(source_dir)
    assert os.path.exists(target_dir) and os.path.isdir(target_dir)

    logger = create_logger()

    games = []
    sgf_files = get_sgf_files(source_dir)

    logger.info(f'Found {len(sgf_files)} sgf files inside "{source_dir}", going to filter them out, this may take sometime...')

    for i, f in enumerate(sgf_files):
        if is_valid_game(f, board_size, min_elo, max_games_per_player):
            games.append(f)

        if i % 10000 == 0:
            logger.info(
                f'Sor far found {len(games)} games from "{source_dir}" that matches board size {board_size} and minimum elo rating {min_elo}'
            )

    logger.info(
        f'Found {len(games)} games from "{source_dir}" that matches board size {board_size} and minimum elo rating {min_elo}'
    )
    sorted_game_counts = dict(sorted(GAME_COUNTS.items(), key=lambda x: x[1], reverse=True))
    logger.info(f'Number of games by player: {sorted_game_counts}')

    # Clean up target
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        # Check if the current item is a file
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)

    for f in games:
        shutil.copy(f, target_dir)
