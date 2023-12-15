# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Functions to plot statistics from csv log files."""
from absl import app, flags
import os
import re
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('logs_dir', './logs/go/9x9', '')


def get_selfplay_dataframe(logs_dir):
    actor_csv_logs = []
    if os.path.exists(logs_dir):
        for root, dirnames, filenames in os.walk(logs_dir):
            for f in filenames:
                if f.startswith('actor') and f.endswith('.csv'):
                    actor_csv_logs.append(os.path.join(root, f))

    # Load actor statistics files
    df = pd.concat([pd.read_csv(f) for f in actor_csv_logs], sort=True) if len(actor_csv_logs) > 0 else None

    grouped_df = None
    if df is not None:
        # Derive new columns from existing data
        df['game_count'] = 1  # every single row is a game

        df['game_lt_60_step_count'] = df.apply(lambda row: 1 if row['game_length'] < 60 else 0, axis=1)
        df['game_60_to_100_step_count'] = df.apply(
            lambda row: 1 if row['game_length'] >= 60 and row['game_length'] <= 100 else 0,
            axis=1,
        )
        df['game_gt_100_step_count'] = df.apply(lambda row: 1 if row['game_length'] > 100 else 0, axis=1)

        df['black_won_count'] = df.apply(
            lambda row: 1 if re.match(r'B\+', row['game_result'], re.IGNORECASE) else 0,
            axis=1,
        )
        df['white_won_count'] = df.apply(
            lambda row: 1 if re.match(r'W\+', row['game_result'], re.IGNORECASE) else 0,
            axis=1,
        )

        df['black_resign_count'] = df.apply(lambda row: 1 if row['game_result'] == 'W+R' else 0, axis=1)
        df['white_resign_count'] = df.apply(lambda row: 1 if row['game_result'] == 'B+R' else 0, axis=1)
        df['resign_disabled_count'] = df.apply(lambda row: 1 if row['is_resign_disabled'] else 0, axis=1)

        # Games marked for resign by a player, where resign move is never played since resignation is disabled
        df['marked_resign_count'] = df.apply(
            lambda row: 1 if row['is_resign_disabled'] and row['is_marked_for_resign'] else 0,
            axis=1,
        )

        # Games marked for resign but ended the marked resign player won, where resign move is never played since resignation is disabled
        df['marked_resign_could_won_count'] = df.apply(
            lambda row: 1 if row['is_resign_disabled'] and row['is_marked_for_resign'] and row['is_could_won'] else 0,
            axis=1,
        )

        # Group data by hours
        # df['datetime'] = pd.to_datetime(df['datetime'])  # if not already as datetime object
        # grouped_df = df.groupby(pd.Grouper(key='datetime', axis=0, freq='H')).sum(numeric_only=True)
        # grouped_df = grouped_df.reset_index()
        # grouped_df['hours'] = grouped_df.index

        # Group data by training steps
        grouped_df = df.groupby(['training_steps']).sum(numeric_only=True)
        grouped_df = grouped_df.reset_index()

        grouped_df['avg_steps_per_game'] = grouped_df['game_length'].cumsum() / grouped_df['game_count'].cumsum()
        grouped_df['avg_passes_per_game'] = grouped_df['num_passes'].cumsum() / grouped_df['game_count'].cumsum()

        # Compute accumulative sum
        grouped_df['total_games'] = grouped_df['game_count'].cumsum()
        grouped_df['games_lt_60_steps'] = grouped_df['game_lt_60_step_count'].cumsum()
        grouped_df['games_60_to_100_steps'] = grouped_df['game_60_to_100_step_count'].cumsum()
        grouped_df['games_gt_100_steps'] = grouped_df['game_gt_100_step_count'].cumsum()

        grouped_df['black_won_games'] = grouped_df['black_won_count'].cumsum()
        grouped_df['white_won_games'] = grouped_df['white_won_count'].cumsum()

        grouped_df['black_resign_games'] = grouped_df['black_resign_count'].cumsum()
        grouped_df['white_resign_games'] = grouped_df['white_resign_count'].cumsum()

        grouped_df['resign_disabled_games'] = grouped_df['resign_disabled_count'].cumsum()

        grouped_df['marked_resign_games'] = grouped_df['marked_resign_count']  # .cumsum()
        grouped_df['marked_resign_could_won_games'] = grouped_df['marked_resign_could_won_count']  # .cumsum()

        # Compute estimated resignation false positive ratio using marked resign games from those 10% where resign is disabled
        grouped_df['resign_false_positive_rate'] = (
            grouped_df['marked_resign_could_won_games'] / grouped_df['marked_resign_games']
        )

        # Other ratio
        grouped_df['steps_lt_60_rate'] = grouped_df['games_lt_60_steps'] / grouped_df['total_games']
        grouped_df['steps_60_to_100_rate'] = grouped_df['games_60_to_100_steps'] / grouped_df['total_games']
        grouped_df['steps_gt_100_rate'] = grouped_df['games_gt_100_steps'] / grouped_df['total_games']
        grouped_df['black_won_rate'] = grouped_df['black_won_games'] / grouped_df['total_games']
        grouped_df['white_won_rate'] = grouped_df['white_won_games'] / grouped_df['total_games']
        grouped_df['black_resign_rate'] = grouped_df['black_resign_games'] / grouped_df['total_games']
        grouped_df['white_resign_rate'] = grouped_df['white_resign_games'] / grouped_df['total_games']
        grouped_df['resign_disabled_rate'] = grouped_df['resign_disabled_games'] / grouped_df['total_games']

    return grouped_df


def main(argv):  # noqa: C901
    print('Loading log files, this may take few minutes...')

    selfplay_df = get_selfplay_dataframe(FLAGS.logs_dir)

    target_csv_file = './az_selfplay_aggregated.csv'

    columns_to_keep = [
        'training_steps',
        'total_games',
        'avg_steps_per_game',
        'avg_passes_per_game',
        'black_won_games',
        'white_won_games',
        'black_resign_games',
        'white_resign_games',
        'resign_disabled_games',
        'resign_false_positive_rate',
        'black_won_rate',
        'white_won_rate',
        'black_resign_rate',
        'white_resign_rate',
        'steps_lt_60_rate',
        'steps_60_to_100_rate',
        'steps_gt_100_rate',
    ]
    selfplay_df.to_csv(target_csv_file, columns=columns_to_keep, index=False)


if __name__ == '__main__':
    app.run(main)
