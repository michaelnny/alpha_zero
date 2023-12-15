# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Functions to plot statistics from csv log files."""
from absl import app, flags
import os
import math
import re
import pandas as pd


FLAGS = flags.FLAGS
flags.DEFINE_string('logs_dir', './9x9_matches', '')


def extract_info_from_filename(filename):
    # Define the regular expression pattern
    pattern = r'\/(\d+x\d+)_(\d+b\d+)\/training_steps_(\d+)\.ckpt'

    # Use the pattern to search for matches in the filename
    matches = re.search(pattern, filename)

    if matches:
        # Extract the desired parts from the matches
        board_size = matches.group(1)
        model_info = matches.group(2)
        training_steps = matches.group(3)

        # Concatenate the extracted parts to form the desired output
        output = f'{model_info}_{training_steps}'
        return output
    else:
        return None


def main(argv):  # noqa: C901
    print('Loading log files, this may take few minutes...')

    log_files = []

    target_csv_file = None
    if os.path.exists(FLAGS.logs_dir):
        target_csv_file = os.path.join(FLAGS.logs_dir, 'logs_aggregated.csv')

        if os.path.exists(target_csv_file):
            os.remove(target_csv_file)

        for root, dirnames, filenames in os.walk(FLAGS.logs_dir):
            for f in filenames:
                if f.endswith('.csv'):
                    log_files.append(os.path.join(root, f))

    if len(log_files) == 0:
        print(f'No log files have been found at "{FLAGS.logs_dir}"')
        return

    df = pd.concat([pd.read_csv(f) for f in log_files], sort=True)

    # Derive new columns from existing data
    df['total_games'] = 1  # every single row is a game

    df['black_id'] = df.apply(
        lambda row: extract_info_from_filename(row['black']) if row['black'] else None,
        axis=1,
    )
    df['white_id'] = df.apply(
        lambda row: extract_info_from_filename(row['white']) if row['white'] else None,
        axis=1,
    )

    df['black_won_games'] = df.apply(
        lambda row: 1 if re.match(r'B\+', row['game_result'], re.IGNORECASE) else 0,
        axis=1,
    )
    df['white_won_games'] = df.apply(
        lambda row: 1 if re.match(r'W\+', row['game_result'], re.IGNORECASE) else 0,
        axis=1,
    )

    # Group data by training steps
    grouped_df = df.groupby(['black_id', 'white_id']).sum(numeric_only=True)
    grouped_df = grouped_df.reset_index()

    # Remove unnecessary columns ('game')
    # grouped_df = grouped_df.drop('game', axis=1)

    # Keep only specific columns
    columns_to_keep = [
        'black_id',
        'white_id',
        'total_games',
        'black_won_games',
        'white_won_games',
    ]
    grouped_df = grouped_df.loc[:, columns_to_keep]

    print(grouped_df.sort_values(['white_won_games', 'white_id'], ascending=False).head(20))

    if target_csv_file:
        columns_to_keep = [
            'black_id',
            'white_id',
            'total_games',
            'black_won_games',
            'white_won_games',
        ]
        grouped_df.to_csv(target_csv_file, columns=columns_to_keep, index=False)


if __name__ == '__main__':
    app.run(main)
