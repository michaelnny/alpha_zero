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
"""Runs AlphaZero self-play training pipeline on Tic-Tac-Toe game.

From the paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
https://arxiv.org/abs//1712.01815

In AlphaZero, we don't run evaluation to select 'best player' from new checkpoint as done in AlphaGo Zero.
Instead, we use a single network and the self-play actors always use the latest network weights to generate samples.

"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from absl import app
from absl import flags
from absl import logging

import multiprocessing
import threading
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

from alpha_zero.games.tictactoe import TicTacToeEnv
from alpha_zero.network import AlphaZeroNet
from alpha_zero.replay import UniformReplay
from alpha_zero.pipeline_v2 import run_self_play, run_training, run_evaluation, run_data_collector
from alpha_zero.log import extract_args_from_flags_dict


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'num_filters',
    16,
    'Number of filters for the conv2d layers in the neural network.',
)
flags.DEFINE_integer(
    'num_fc_units',
    16,
    'Number of hidden units in the linear layer of the neural network.',
)

flags.DEFINE_integer('capacity', 10000 * 7, 'Replay buffer stores number of most recent self-play games.')
flags.DEFINE_integer('batch_size', 128, 'Sample batch size when do learning.')

flags.DEFINE_float('learning_rate', 0.02, 'Learning rate.')
flags.DEFINE_float('lr_decay', 0.1, 'Learning rate decay rate.')
flags.DEFINE_multi_integer('lr_decay_milestones', [20000], 'The number of steps at which the learning rate will decay.')
flags.DEFINE_float('sgd_momentum', 0.9, 'The momentum parameter for SGD optimizer.')
flags.DEFINE_float('l2_regularization', 1e-4, 'The L2 regularization parameter applied to weights.')

flags.DEFINE_integer('num_train_steps', 50000, 'Number of training steps (measured in network updates).')

flags.DEFINE_integer('num_actors', 16, 'Number of self-play actor processes.')
flags.DEFINE_integer(
    'num_simulations', 20, 'Number of iterations per MCTS search, this applies to both self-play and evaluation processes.'
)
flags.DEFINE_integer(
    'num_parallel',
    4,
    'Number of leaves to collect before using the neural network to evaluate the positions during MCTS search, 1 means no parallel search.',
)

flags.DEFINE_float('c_puct_base', 19652, 'Exploration constants balancing priors vs. search values.')
flags.DEFINE_float('c_puct_init', 1.25, 'Exploration constants balancing priors vs. search values.')

flags.DEFINE_integer(
    'warm_up_steps',
    5,
    'Number of opening environment steps to sample action according search policy instead of choosing most visited child.',
)

flags.DEFINE_float('train_delay', 0.1, 'Delay (in seconds) before training on next batch samples.')
flags.DEFINE_float(
    'initial_elo', 0.0, 'Initial elo rating, when resume training, this should be the elo from the loaded checkpoint.'
)

flags.DEFINE_integer('checkpoint_frequency', 1000, 'The frequency (in training step) to create new checkpoint.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/tictactoe_v2', 'Path for checkpoint file.')

flags.DEFINE_string('train_csv_file', 'logs/train_tictactoe_v2.csv', 'A csv file contains training statistics.')
flags.DEFINE_string('eval_csv_file', 'logs/eval_tictactoe_v2.csv', 'A csv file contains training statistics.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')


def main(argv):
    torch.manual_seed(FLAGS.seed)
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Running on {runtime_device}, with the following arguments:')
    args_dict = extract_args_from_flags_dict(FLAGS.flag_values_dict())
    logging.info(args_dict)

    def environment_builder():
        return TicTacToeEnv()

    evaluation_env = environment_builder()

    input_shape = evaluation_env.observation_space.shape
    num_actions = evaluation_env.action_space.n

    def network_builder():
        return AlphaZeroNet(input_shape, num_actions, 3, FLAGS.num_filters, FLAGS.num_fc_units)

    network = network_builder()
    optimizer = torch.optim.SGD(
        network.parameters(), lr=FLAGS.learning_rate, momentum=FLAGS.sgd_momentum, weight_decay=FLAGS.l2_regularization
    )
    # optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2_regularization)
    lr_scheduler = MultiStepLR(optimizer, milestones=FLAGS.lr_decay_milestones, gamma=FLAGS.lr_decay)

    actor_network = network_builder()
    actor_network.share_memory()

    old_ckpt_network = network_builder()
    new_ckpt_network = network_builder()

    replay = UniformReplay(FLAGS.capacity, random_state)

    # Use the stop_event to signaling actors to stop running.
    stop_event = multiprocessing.Event()
    # Transfer samples from self-play process to training process.
    data_queue = multiprocessing.SimpleQueue()
    # A shared list to store most recent new checkpoint file paths.
    manager = multiprocessing.Manager()
    checkpoint_files = manager.list()

    # Start to collect samples generated by self-play actors
    data_collector = threading.Thread(
        target=run_data_collector,
        args=(data_queue, replay),
    )
    data_collector.start()

    # Start learner
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
            FLAGS.batch_size,
            FLAGS.num_train_steps,
            FLAGS.checkpoint_frequency,
            FLAGS.checkpoint_dir,
            checkpoint_files,
            FLAGS.train_csv_file,
            stop_event,
            FLAGS.train_delay,
        ),
    )
    learner.start()

    # Start evaluator
    evaluator = multiprocessing.Process(
        target=run_evaluation,
        args=(
            old_ckpt_network,
            new_ckpt_network,
            runtime_device,
            evaluation_env,
            FLAGS.c_puct_base,
            FLAGS.c_puct_init,
            0.001,
            FLAGS.num_simulations,
            FLAGS.num_parallel,
            checkpoint_files,
            FLAGS.eval_csv_file,
            stop_event,
            FLAGS.initial_elo,
        ),
    )
    evaluator.start()

    # Start self-play actors
    actors = []
    for i in range(FLAGS.num_actors):
        actor = multiprocessing.Process(
            target=run_self_play,
            args=(
                i,
                actor_network,
                'cpu',
                environment_builder(),
                data_queue,
                FLAGS.c_puct_base,
                FLAGS.c_puct_init,
                FLAGS.warm_up_steps,
                FLAGS.num_simulations,
                FLAGS.num_parallel,
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
