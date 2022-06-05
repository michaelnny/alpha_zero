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
"""Runs AlphaZero self-play training pipeline on free-style Gomoku game.

From the paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
https://arxiv.org/abs//1712.01815

In AlphaZero, we don't run evaluation to select 'best player' from new checkpoint as done in AlphaGo Zero.
Instead, we use a single network and the self-play actors always use the latest network weights to generate samples.

"""
from absl import app
from absl import flags
from absl import logging
import os
import multiprocessing
import threading
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

from alpha_zero.games.gomoku import GomokuEnv
from alpha_zero.network import AlphaZeroNet
from alpha_zero.replay import UniformReplay
from alpha_zero.pipeline_v2 import (
    run_self_play,
    run_training,
    run_evaluation,
    run_data_collector,
    load_checkpoint,
    load_from_file,
)


FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 11, 'Board size for Gomoku.')
flags.DEFINE_integer('stack_history', 4, 'Stack previous states.')

flags.DEFINE_integer('num_res_blocks', 6, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer('num_planes', 64, 'Number of planes for the conv2d layer in the neural network.')
flags.DEFINE_integer(
    'num_fc_units', 64, 'Number of hidden units for the fully connected layer in value head of the neural network.'
)

flags.DEFINE_integer('replay_capacity', 1000 * 50, 'Maximum replay size, approximately most recent 1000 games.')
flags.DEFINE_integer('min_replay_size', 5000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 128, 'Sample batch size when do learning.')

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('learning_rate_decay', 0.1, 'Adam learning rate decay rate.')
flags.DEFINE_multi_integer(
    'lr_milestones', [200000, 400000, 600000], 'The number of steps at which the learning rate will decay.'
)
flags.DEFINE_float('l2_decay', 0.0001, 'Adam L2 regularization.')

flags.DEFINE_integer('num_train_steps', 1000000, 'Number of training steps (measured in network updates).')

flags.DEFINE_integer('num_actors', 8, 'Number of self-play actor processes.')
flags.DEFINE_integer('num_simulations', 400, 'Number of simulations per MCTS search, per agent environment time step.')  # 800

flags.DEFINE_float('c_puct', 5.0, 'Puct constant of the UCB score.')

flags.DEFINE_float(
    'temp_begin_value', 1.0, 'Begin value of the temperature exploration rate after MCTS search to generate play policy.'
)
flags.DEFINE_float(
    'temp_end_value',
    0.1,
    'End (decayed) value of the temperature exploration rate after MCTS search to generate play policy.',
)
flags.DEFINE_integer(
    'temp_decay_steps', 30, 'Number of environment steps to decay the temperture from begin_value to end_value.'
)

flags.DEFINE_float(
    'train_delay',
    0.25,
    'Delay (in seconds) before training on next batch samples, if training on GPU, using large value (like 0.75, 1.0, 1.5).',
)
flags.DEFINE_float(
    'initial_elo', -2000.0, 'Initial elo rating, in case resume training, this should be the elo form last checkpoint.'
)

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.DEFINE_integer('checkpoint_frequency', 1000, 'The frequency (in training step) to create new checkpoint.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/gomoku_v2', 'Path for checkpoint file.')
flags.DEFINE_string(
    'load_checkpoint_file',
    '',
    'Load the checkpoint from file.',
)

flags.DEFINE_integer(
    'samples_save_frequency',
    10000,
    'The frequency (measured in number added in replay) to save self-play samples in replay.',
)
flags.DEFINE_string('samples_save_dir', 'samples/gomoku_v2', 'Path for save self-play samples in replay to file.')
flags.DEFINE_string('load_samples_file', '', 'Load the replay samples from file.')
flags.DEFINE_string('train_csv_file', 'logs/train_gomoku_v2.csv', 'A csv file contains training statistics.')
flags.DEFINE_string('eval_csv_file', 'logs/eval_gomoku_v2.csv', 'A csv file contains training statistics.')


def main(argv):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    runtime_device = torch.device(device)

    self_play_envs = [
        GomokuEnv(board_size=FLAGS.board_size, stack_history=FLAGS.stack_history) for i in range(FLAGS.num_actors)
    ]

    evaluation_env = GomokuEnv(board_size=FLAGS.board_size, stack_history=FLAGS.stack_history)

    input_shape = self_play_envs[0].observation_space.shape
    num_actions = self_play_envs[0].action_space.n

    network = AlphaZeroNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.num_fc_units)
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=FLAGS.lr_milestones, gamma=FLAGS.learning_rate_decay)

    actor_network = AlphaZeroNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.num_fc_units)
    actor_network.share_memory()

    old_checkpoint_network = AlphaZeroNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.num_fc_units)
    new_checkpoint_network = AlphaZeroNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.num_fc_units)

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    replay = UniformReplay(FLAGS.replay_capacity, random_state)

    train_steps = None
    # Load states from checkpoint to resume training.
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        optimizer.load_state_dict(loaded_state['optimizer'])
        lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
        train_steps = loaded_state['train_steps']

        actor_network.load_state_dict(loaded_state['network'])
        old_checkpoint_network.load_state_dict(loaded_state['network'])

        logging.info(f'Loaded state from checkpoint {FLAGS.load_checkpoint_file}')
        logging.info(f'Current state: train steps {train_steps}, learing rate {lr_scheduler.get_last_lr()}')

    # Load replay samples
    if FLAGS.load_samples_file is not None and os.path.isfile(FLAGS.load_samples_file):
        try:
            replay.reset()
            replay_state = load_from_file(FLAGS.load_samples_file)
            # replay.set_state(replay_state)
            for item in replay_state['storage']:
                if item is None:
                    break
                replay.add(item)

                if replay.num_added >= replay.capacity:
                    break
            logging.info(f"Loaded replay samples from file '{FLAGS.load_samples_file}'")
        except Exception:
            pass

    # Use the stop_event to signaling actors to stop running.
    stop_event = multiprocessing.Event()
    # Transfer samples from self-play process to training process.
    data_queue = multiprocessing.SimpleQueue()
    # A shared list to store most recent new checkpoint file paths.
    manager = multiprocessing.Manager()
    checkpoint_files = manager.list()

    # Start to collect samples from self-play on a new thread.
    data_collector = threading.Thread(
        target=run_data_collector,
        args=(data_queue, replay, FLAGS.samples_save_frequency, FLAGS.samples_save_dir),
    )
    data_collector.start()

    # Start the main training loop on a new thread.
    learner = threading.Thread(
        target=run_training,
        args=(
            network,
            optimizer,
            lr_scheduler,
            runtime_device,
            actor_network,
            replay,
            data_queue,
            FLAGS.min_replay_size,
            FLAGS.batch_size,
            FLAGS.num_train_steps,
            FLAGS.checkpoint_frequency,
            FLAGS.checkpoint_dir,
            checkpoint_files,
            FLAGS.train_csv_file,
            stop_event,
            FLAGS.train_delay,
            False,
            train_steps,
        ),
    )
    learner.start()

    # Start evaluation loop on a seperate process.
    evaluator = multiprocessing.Process(
        target=run_evaluation,
        args=(
            old_checkpoint_network,
            new_checkpoint_network,
            runtime_device,
            evaluation_env,
            FLAGS.c_puct,
            FLAGS.temp_end_value,
            FLAGS.num_simulations,
            checkpoint_files,
            FLAGS.eval_csv_file,
            stop_event,
            FLAGS.initial_elo,
        ),
    )
    evaluator.start()

    # Start self-play processes.
    actors = []
    for i in range(FLAGS.num_actors):
        actor = multiprocessing.Process(
            target=run_self_play,
            args=(
                i,
                actor_network,
                'cpu',
                self_play_envs[i],
                data_queue,
                FLAGS.c_puct,
                FLAGS.temp_begin_value,
                FLAGS.temp_end_value,
                FLAGS.temp_decay_steps,
                FLAGS.num_simulations,
                stop_event,
            ),
        )
        actor.start()
        actors.append(actor)

    for actor in actors:
        actor.join()
        actor.close()

    learner.join()
    data_collector.join()
    evaluator.join()


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
