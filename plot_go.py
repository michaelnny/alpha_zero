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
flags.DEFINE_string('logs_dir', './logs/go/9x9', '')
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
    grouped_df['resign_false_positive_rate'] = grouped_df['marked_resign_could_won_games'] / grouped_df['marked_resign_games']

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

    axs_selfplay = subfigs[0].subplots(5, 1, sharex=True)
    axs_train = subfigs[1].subplots(5, 1, sharex=True)
    axs_eval = subfigs[2].subplots(4, 1, sharex=True)

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
                plot_selfplay_resign_fp_ratio(selfplay_df, ax)
        elif i == 3:
            if selfplay_df is not None:
                plot_selfplay_games_winrate(selfplay_df, ax)
        elif i == 4:
            if selfplay_df is not None:
                plot_selfplay_games_precentage(selfplay_df, ax)
            ax.set_xlabel('Training steps', fontsize='large')

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

    for i, ax in enumerate(axs_eval):
        ax.xaxis.set_major_formatter(FuncFormatter(shorten))
        if i == 0:
            if eval_df is not None and 'policy_top_1_accuracy' in eval_df.columns:
                plot_eval_policy_accuracy(eval_df, ax)
        elif i == 1:
            if eval_df is not None and 'value_mse_error' in eval_df.columns:
                plot_eval_mse_error(eval_df, ax)
        elif i == 2:
            if eval_df is not None:
                if 'policy_entropy' in eval_df.columns:
                    plot_eval_policy_entropy(eval_df, ax)
                else:
                    plot_eval_game_length(eval_df, ax)
        elif i == 3:
            if eval_df is not None:
                plot_eval_elo_rating(eval_df, ax)
            ax.set_xlabel('Training steps', fontsize='large')

    plt.show()


def plot_selfplay_games_precentage(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.steps_lt_60_rate * 100,
            color='steelblue',
            linewidth=FLAGS.line_width,
            label='< 60 steps',
        )
        ax.plot(
            df.training_steps,
            df.steps_60_to_100_rate * 100,
            color='purple',
            linewidth=FLAGS.line_width,
            label='60 - 100 steps',
        )
        ax.plot(
            df.training_steps,
            df.steps_gt_100_rate * 100,
            color='orange',
            linewidth=FLAGS.line_width,
            label='> 100 steps',
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

        ax.plot(
            df.training_steps,
            df.black_resign_rate * 100,
            '--',
            color='black',
            linewidth=FLAGS.line_width,
            label='Black resigned',
        )
        ax.plot(
            df.training_steps,
            df.white_resign_rate * 100,
            '--',
            color='gray',
            linewidth=FLAGS.line_width,
            label='White resigned',
        )
        ax.legend()

    ax.set_ylabel('Win rate (%)', fontsize='large')


def plot_selfplay_resign_fp_ratio(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.resign_false_positive_rate * 100,
            color='red',
            linewidth=FLAGS.line_width,
            label='False positive (estimated)',
        )
        ax.hlines(
            y=0.05 * 100,
            xmin=0,
            xmax=max(df.training_steps),
            color='black',
            linewidth=FLAGS.line_width,
            linestyle='--',
        )

    ax.set_ylabel('Resignation \n false positive (%)', fontsize='large')


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
            label='Game length',
        )
        ax.plot(
            df.training_steps,
            df.avg_passes_per_game,
            color='orange',
            linewidth=FLAGS.line_width,
            label='Number of passes',
        )
        ax.legend()

    ax.set_ylabel('Avg steps \n per game', fontsize='large')


def plot_training_time(df, ax):
    if df is not None:
        ax.plot(df.training_steps, df.hours, linewidth=FLAGS.line_width, color='steelblue')

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
            linewidth=FLAGS.line_width,
            color='steelblue',
        )

    ax.set_ylabel('Learning rate', fontsize='large')


def plot_training_value_loss(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.value_loss,
            linewidth=FLAGS.line_width,
            color='steelblue',
        )

    ax.set_ylabel('MSE loss', fontsize='large')


def plot_training_policy_loss(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.policy_loss,
            linewidth=FLAGS.line_width,
            color='steelblue',
        )

    ax.set_ylabel('Cross-entropy loss', fontsize='large')


def plot_eval_policy_entropy(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.policy_entropy,
            linewidth=FLAGS.line_width,
            color='steelblue',
        )

    ax.set_ylabel('Policy entropy', fontsize='large')


def plot_eval_mse_error(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.value_mse_error,
            linewidth=FLAGS.line_width,
            color='steelblue',
        )
        # scale by 1/4 to the range of 0-1
        # ax.plot(df.training_steps, df.value_mse_error * 0.25, linewidth=FLAGS.line_width, color='steelblue')

    ax.set_ylabel('MSE of professional \n game outcomes', fontsize='large')


def plot_eval_policy_accuracy(df, ax):
    if df is not None:
        ax.plot(
            df.training_steps,
            df.policy_top_1_accuracy * 100,
            color='steelblue',
            linewidth=FLAGS.line_width,
            label='Top 1 accuracy',
        )
        ax.plot(
            df.training_steps,
            df.policy_top_3_accuracy * 100,
            color='orange',
            linewidth=FLAGS.line_width,
            label='Top 3 accuracy',
        )
        ax.plot(
            df.training_steps,
            df.policy_top_5_accuracy * 100,
            color='green',
            linewidth=FLAGS.line_width,
            label='Top 5 accuracy',
        )
        ax.legend()

    ax.set_ylabel('Prediction accuracy \n on professional moves (%)', fontsize='large')


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
            label='Game length',
        )
        ax.plot(
            df.training_steps,
            df.num_passes,
            '--',
            color='orange',
            linewidth=FLAGS.line_width,
            label='Number of passes',
        )
        ax.legend()

    ax.set_ylabel('Evaluation \n game steps', fontsize='large')


if __name__ == '__main__':
    app.run(main)
