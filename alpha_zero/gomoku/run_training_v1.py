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
"""Runs AlphaGo Zero self-play training pipeline on free-style Gomoku game with a small sized board.

From the paper "Mastering the game of Go without human knowledge"
https://www.nature.com/articles/nature24270/

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
from alpha_zero.pipeline_v1 import (
    run_self_play,
    run_training,
    run_evaluation,
    run_data_collector,
    load_checkpoint,
    load_from_file,
)


FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 15, 'Board size for Gomoku.')
flags.DEFINE_integer('stack_history', 4, 'Stack previous states, the state is an image of N x 2 + 1 binary planes.')
flags.DEFINE_integer('num_res_blocks', 10, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer(
    'num_planes',
    128,
    'Number of filters for the conv2d layers, this is also the number of hidden units in the linear layer of the neural network.',
)

flags.DEFINE_integer('replay_capacity', 200000, 'Maximum replay size, use most recent N positions for training.')
flags.DEFINE_integer('min_replay_size', 50000, 'Minimum replay size before learning starts.')
flags.DEFINE_integer('batch_size', 256, 'Sample batch size when do learning.')

flags.DEFINE_float('learning_rate', 0.002, 'Learning rate.')
flags.DEFINE_float('lr_decay', 0.1, 'Adam learning rate decay rate.')
flags.DEFINE_multi_integer(
    'lr_decay_milestones', [200000, 500000], 'The number of steps at which the learning rate will decay.'
)
flags.DEFINE_float('l2_decay', 0.0001, 'Adam L2 regularization.')

flags.DEFINE_integer('num_train_steps', 1000000, 'Number of training steps (measured in network updates).')
flags.DEFINE_integer('num_eval_games', 10, 'Number of games to play during evaluation to select best player.')

flags.DEFINE_integer('num_actors', 3, 'Number of self-play actor processes.')
flags.DEFINE_integer(
    'num_simulations', 600, 'Number of simulations per MCTS search, this applies to both self-play and evaluation processes.'
)
flags.DEFINE_integer('parallel_leaves', 8, 'Number of parallel leaves for MCTS search, 1 means do not use parallel search.')

flags.DEFINE_float('c_puct_base', 19652, 'Exploration constants balancing priors vs. value net output.')
flags.DEFINE_float('c_puct_init', 1.25, 'Exploration constants balancing priors vs. value net output.')

flags.DEFINE_float(
    'temp_begin_value', 1.0, 'Begin value of the temperature exploration rate after MCTS search to generate play policy.'
)
flags.DEFINE_float(
    'temp_end_value',
    0.1,
    'End (decayed) value of the temperature exploration rate after MCTS search to generate play policy.',
)
flags.DEFINE_integer(
    'temp_decay_steps', 30, 'Number of environment steps to decay the temperature from begin_value to end_value.'
)
flags.DEFINE_float('train_delay', 0.45, 'Delay (in seconds) before training on next batch samples.')
flags.DEFINE_float(
    'initial_elo', 0.0, 'Initial elo rating, when resume training, this should be the elo from the loaded checkpoint.'
)

flags.DEFINE_integer('checkpoint_frequency', 2000, 'The frequency (in training step) to create new checkpoint.')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/gomoku_v1', 'Path for checkpoint file.')
flags.DEFINE_string('load_checkpoint_file', '', 'Load the checkpoint from file to resume training.')

flags.DEFINE_integer(
    'samples_save_frequency',
    50000,
    'The frequency (measured in number added in replay) to save self-play samples to file.',
)
flags.DEFINE_string('samples_save_dir', 'samples/gomoku_v1', 'Path for save self-play samples file.')
flags.DEFINE_string('load_samples_file', '', 'Load the replay samples from file to resume training.')

flags.DEFINE_string('train_csv_file', 'logs/train_gomoku_v1.csv', 'A csv file contains training statistics.')
flags.DEFINE_string('eval_csv_file', 'logs/eval_gomoku_v1.csv', 'A csv file contains evaluation statistics.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')


def main(argv):
    torch.manual_seed(FLAGS.seed)
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def environment_builder():
        return GomokuEnv(board_size=FLAGS.board_size, stack_history=FLAGS.stack_history)

    evaluation_env = environment_builder()

    input_shape = evaluation_env.observation_space.shape
    num_actions = evaluation_env.action_space.n

    def network_builder():
        return AlphaZeroNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.num_planes)

    network = network_builder()
    optimizer = torch.optim.Adam(network.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=FLAGS.lr_decay_milestones, gamma=FLAGS.lr_decay)

    actor_network = network_builder()
    actor_network.share_memory()

    new_checkpoint_network = network_builder()

    replay = UniformReplay(FLAGS.replay_capacity, random_state)

    train_steps = None
    # Load states from checkpoint to resume training.
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        network.to(device=runtime_device)
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, runtime_device)
        network.load_state_dict(loaded_state['network'])
        optimizer.load_state_dict(loaded_state['optimizer'])
        lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
        train_steps = loaded_state['train_steps']

        actor_network.load_state_dict(loaded_state['network'])
        new_checkpoint_network.load_state_dict(loaded_state['network'])

        logging.info(f'Loaded state from checkpoint {FLAGS.load_checkpoint_file}')
        logging.info(f'Current state: train steps {train_steps}, learning rate {lr_scheduler.get_last_lr()}')

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
            train_steps,
        ),
    )
    learner.start()

    # Start evaluation loop on a separate process.
    evaluator = multiprocessing.Process(
        target=run_evaluation,
        args=(
            actor_network,
            new_checkpoint_network,
            runtime_device,
            evaluation_env,
            FLAGS.num_eval_games,
            FLAGS.c_puct_base,
            FLAGS.c_puct_init,
            FLAGS.temp_end_value,
            FLAGS.num_simulations,
            FLAGS.parallel_leaves,
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
                runtime_device,
                environment_builder(),
                data_queue,
                FLAGS.c_puct_base,
                FLAGS.c_puct_init,
                FLAGS.temp_begin_value,
                FLAGS.temp_end_value,
                FLAGS.temp_decay_steps,
                FLAGS.num_simulations,
                FLAGS.parallel_leaves,
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
