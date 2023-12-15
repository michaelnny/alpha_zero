# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Functions to plot statistics from csv log files."""
from absl import app, flags
import logging
import os
import math
import re
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

FLAGS = flags.FLAGS
flags.DEFINE_string('logs_dir', './logs/gomoku/13x13', '')
flags.DEFINE_float('line_width', 1, '')


def shorten(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else math.floor(math.log10(abs(value)) / 3)
    value = round(value / 1000**num_thousands, 2)
    return f'{value:g}' + ' KMGTPEZY'[num_thousands]


def get_selfplay_dataframe(logs_dir):
    actor_csv_logs = []
    if os.path.exists(logs_dir):
        for root, dirnames, filenames in os.walk(logs_dir):
            for f in filenames:
                if f.startswith('actor') and f.endswith('.csv'):
                    actor_csv_logs.append(os.path.join(root, f))

    # Load actor statistics files
    if len(actor_csv_logs) == 0:
        logging.warning(f'No log files have been found at "{logs_dir}"')
        return

    df = pd.concat([pd.read_csv(f) for f in actor_csv_logs], sort=True)

    # Derive new columns from existing data
    df['game_count'] = 1  # every single row is a game

    df['game_lt_15_step_count'] = df.apply(lambda row: 1 if row['game_length'] < 15 else 0, axis=1)
    df['game_15_to_30_step_count'] = df.apply(
        lambda row: 1 if row['game_length'] >= 15 and row['game_length'] <= 30 else 0,
        axis=1,
    )
    df['game_gt_30_step_count'] = df.apply(lambda row: 1 if row['game_length'] > 30 else 0, axis=1)

    df['black_won_count'] = df.apply(
        lambda row: 1 if re.match(r'B\+', row['game_result'], re.IGNORECASE) else 0,
        axis=1,
    )
    df['white_won_count'] = df.apply(
        lambda row: 1 if re.match(r'W\+', row['game_result'], re.IGNORECASE) else 0,
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

    # Compute accumulative sum
    grouped_df['total_games'] = grouped_df['game_count'].cumsum()
    grouped_df['games_lt_15_steps'] = grouped_df['game_lt_15_step_count'].cumsum()
    grouped_df['games_15_to_30_steps'] = grouped_df['game_15_to_30_step_count'].cumsum()
    grouped_df['games_gt_30_steps'] = grouped_df['game_gt_30_step_count'].cumsum()

    grouped_df['black_won_games'] = grouped_df['black_won_count'].cumsum()
    grouped_df['white_won_games'] = grouped_df['white_won_count'].cumsum()

    # Other ratio
    grouped_df['steps_lt_15_rate'] = grouped_df['games_lt_15_steps'] / grouped_df['total_games']
    grouped_df['steps_15_to_30_rate'] = grouped_df['games_15_to_30_steps'] / grouped_df['total_games']
    grouped_df['steps_gt_30_rate'] = grouped_df['games_gt_30_steps'] / grouped_df['total_games']
    grouped_df['black_won_rate'] = grouped_df['black_won_games'] / grouped_df['total_games']
    grouped_df['white_won_rate'] = grouped_df['white_won_games'] / grouped_df['total_games']

    return grouped_df


def get_dataframe(csv_file):
    if not os.path.exists(csv_file) or not os.path.isfile(csv_file):
        logging.warning(f'No log files have been found at "{csv_file}"')
        return

    df = pd.read_csv(csv_file)
    # Derive hours from datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    start_time = df['datetime'].iloc[0]
    df['hours'] = (df['datetime'] - start_time).dt.total_seconds() / 3600

    return df


def main(argv):  # noqa: C901
    logging.info('Loading log files, this may take few minutes...')

    train_df = get_dataframe(os.path.join(FLAGS.logs_dir, 'training.csv'))
    eval_df = get_dataframe(os.path.join(FLAGS.logs_dir, 'evaluation.csv'))
    selfplay_df = get_selfplay_dataframe(FLAGS.logs_dir)

    fig = plt.figure(layout='constrained', figsize=(16, 9))

    # Three columns
    subfigs = fig.subfigures(1, 3, wspace=0.04)

    for subfig in subfigs:
        subfig.set_facecolor('0.95')

    subfigs[0].suptitle('Self-play', fontsize='x-large')
    subfigs[1].suptitle('Training', fontsize='x-large')
    subfigs[2].suptitle('Evaluation', fontsize='x-large')

    axs_selfplay = subfigs[0].subplots(4, 1, sharex=True)
    axs_train = subfigs[1].subplots(5, 1, sharex=True)
    axs_eval = subfigs[2].subplots(2, 1, sharex=True)

    # Self-play statistics
    for i, ax in enumerate(axs_selfplay):
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # if group by hours
        ax.xaxis.set_major_formatter(FuncFormatter(shorten))
        if i == 0:
            if selfplay_df is not None:
                plot_selfplay_num_games(selfplay_df, ax)
        elif i == 1:
            if selfplay_df is not None:
                plot_selfplay_game_length(selfplay_df, ax)
        elif i == 2:
            if selfplay_df is not None:
                plot_selfplay_games_winrate(selfplay_df, ax)
        elif i == 3:
            if selfplay_df is not None:
                plot_selfplay_games_precentage(selfplay_df, ax)
            ax.set_xlabel('Training steps', fontsize='large')

    # Training statistics
    for i, ax in enumerate(axs_train):
        ax.xaxis.set_major_formatter(FuncFormatter(shorten))
        if i == 0:
            if train_df is not None:
                plot_training_policy_loss(train_df, ax)
        elif i == 1:
            if train_df is not None:
                plot_training_value_loss(train_df, ax)
        elif i == 2:
            if train_df is not None:
                plot_training_lr(train_df, ax)
        elif i == 3:
            if train_df is not None:
                plot_training_samples(train_df, ax)
        elif i == 4:
            if train_df is not None:
                plot_training_time(train_df, ax)
            ax.set_xlabel('Training steps', fontsize='large')

    # Evaluation statistics
    for i, ax in enumerate(axs_eval):
        ax.xaxis.set_major_formatter(FuncFormatter(shorten))
        if i == 0:
            if eval_df is not None:
                plot_eval_game_length(eval_df, ax)
        elif i == 1:
            if eval_df is not None:
                plot_eval_elo_rating(eval_df, ax)
            ax.set_xlabel('Training steps', fontsize='large')

    plt.show()


def plot_selfplay_games_precentage(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.steps_lt_15_rate * 100,
            color='steelblue',
            linewidth=FLAGS.line_width,
            label='< 15 steps',
        )
        ax.plot(
            df.training_steps,
            df.steps_15_to_30_rate * 100,
            color='purple',
            linewidth=FLAGS.line_width,
            label='15 - 30 steps',
        )
        ax.plot(
            df.training_steps,
            df.steps_gt_30_rate * 100,
            color='orange',
            linewidth=FLAGS.line_width,
            label='> 30 steps',
        )
        ax.legend()

    ax.set_ylabel('Game lengths \n (%)', fontsize='large')


def plot_selfplay_games_winrate(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.black_won_rate * 100,
            color='black',
            linewidth=FLAGS.line_width,
            label='Black won',
        )
        ax.plot(
            df.training_steps,
            df.white_won_rate * 100,
            color='gray',
            linewidth=FLAGS.line_width,
            label='White won',
        )
        ax.legend()

    ax.set_ylabel('Win rate (%)', fontsize='large')


def plot_selfplay_num_games(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.total_games,
            color='steelblue',
            linewidth=FLAGS.line_width,
            label='Total',
        )

    ax.set_ylabel('Number of \n games', fontsize='large')
    ax.yaxis.set_major_formatter(FuncFormatter(shorten))


def plot_selfplay_game_length(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.avg_steps_per_game,
            color='steelblue',
            linewidth=FLAGS.line_width,
        )

    ax.set_ylabel('Avg steps \n per game', fontsize='large')


def plot_training_time(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.hours,
            color='steelblue',
            linewidth=FLAGS.line_width,
        )

    ax.set_ylabel('Training time (h)', fontsize='large')


def plot_training_samples(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.total_samples,
            color='steelblue',
            linewidth=FLAGS.line_width,
            label='Total samples',
        )

    ax.set_ylabel('Training samples \n (total)', fontsize='large')
    ax.yaxis.set_major_formatter(FuncFormatter(shorten))


def plot_training_lr(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.learning_rate,
            color='steelblue',
            linewidth=FLAGS.line_width,
        )

    ax.set_ylabel('Learning rate', fontsize='large')


def plot_training_value_loss(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.value_loss,
            color='steelblue',
            linewidth=FLAGS.line_width,
        )

    ax.set_ylabel('MSE loss', fontsize='large')


def plot_training_policy_loss(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.policy_loss,
            color='steelblue',
            linewidth=FLAGS.line_width,
        )

    ax.set_ylabel('Cross-entropy loss', fontsize='large')


def plot_eval_elo_rating(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.black_elo_rating,
            color='steelblue',
            linewidth=FLAGS.line_width,
            label='Elo rating',
        )

    ax.set_ylabel('Elo ratings', fontsize='large')


def plot_eval_game_length(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.game_length,
            color='steelblue',
            linewidth=FLAGS.line_width,
        )

    ax.set_ylabel('Evaluation \n game steps', fontsize='large')


if __name__ == '__main__':
    app.run(main)
