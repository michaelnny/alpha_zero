# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""
Trains the AlphaZero agent on a single machine for the game of Go.

IMPORTANT NOTE:

You should make sure you have the computation resource before running this module,
as this uses some of the hyper-parameters used by the original AlphaZero agent.
"""

import os

# This forces OpenMP to use 1 single thread, which is needed to
# prevent contention between multiple process.
os.environ['OMP_NUM_THREADS'] = '1'
# Tell numpy to only use one core. If we don't do this, each process may
# try to use all of the cores and the resulting contention may result
# in no speedup over the serial version. Note that if numpy is using
# OpenBLAS, then you need to set OPENBLAS_NUM_THREADS=1, and you
# probably need to do it from the command line (so it happens before
# numpy is imported).
os.environ['MKL_NUM_THREADS'] = '1'


import multiprocessing as mp
import sys
from absl import flags

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 19, 'Board size for Go.')
flags.DEFINE_float('komi', 7.5, 'Komi rule for Go.')
flags.DEFINE_integer(
    'num_stack',
    8,
    'Stack N previous states, the state is an image of N x 2 + 1 binary planes.',
)
flags.DEFINE_integer('num_res_blocks', 19, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer('num_filters', 256, 'Number of filters for the conv2d layers in the neural network.')
flags.DEFINE_integer(
    'num_fc_units',
    256,
    'Number of hidden units in the linear layer of the neural network.',
)

flags.DEFINE_integer('min_games', 50000, 'Collect number of self-play games before learning starts.')
flags.DEFINE_integer(
    'games_per_ckpt',
    25000,
    'Collect minimum number of self-play games using the last checkpoint before creating the next checkpoint.',
)
flags.DEFINE_integer(
    'replay_capacity',
    500000 * 100,
    'Replay buffer capacity is number of game * average game length. ',
)
flags.DEFINE_integer('batch_size', 2048, '')

flags.DEFINE_bool(
    'argument_data',
    True,
    'Apply random rotation and mirroring to the training data, default on.',
)
flags.DEFINE_bool('compress_data', False, 'Compress state when saving in replay buffer, default off.')

flags.DEFINE_float('init_lr', 0.2, 'Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, 'Learning rate decay rate.')
flags.DEFINE_multi_integer(
    'lr_milestones',
    [200000, 400000, 600000],
    'The number of training steps at which the learning rate will be decayed.',
)
flags.DEFINE_float('l2_regularization', 1e-4, 'The L2 regularization parameter applied to weights.')
flags.DEFINE_float('sgd_momentum', 0.9, '')

flags.DEFINE_integer(
    'max_training_steps',
    int(7e5),
    'Number of training steps (measured in network parameter update, one batch is one training step).',
)
flags.DEFINE_integer('num_actors', 2000, 'Number of self-play actor processes.')
flags.DEFINE_integer(
    'num_simulations',
    800,
    'Number of simulations per MCTS search, this applies to both self-play and evaluation processes.',
)
flags.DEFINE_integer(
    'num_parallel',
    8,
    'Number of leaves to collect before using the neural network to evaluate the positions during MCTS search,'
    '1 means no parallel search.',
)
flags.DEFINE_float(
    'c_puct_base',
    19652,
    'Exploration constants balancing priors vs. search values. Original paper use 19652',
)
flags.DEFINE_float(
    'c_puct_init',
    1.25,
    'Exploration constants balancing priors vs. search values. Original paper use 1.25',
)

flags.DEFINE_integer(
    'warm_up_steps',
    30,
    'Number of steps at the beginning of a self-play game where the search temperature is set to 1.',
)
flags.DEFINE_float(
    'init_resign_threshold',
    -0.88,
    'The self-play game is resigned if MCTS search values are lesser than this threshold.'
    'This value is also dynamically adjusted (decreased) during training to keep the false positive below the target level.'
    '-1 means no resign and it disables all the features related to resignations during self-play.',
)
flags.DEFINE_integer(
    'check_resign_after_steps',
    80,
    'Number steps into the self-play game before checking for resign.',
)
flags.DEFINE_float(
    'target_fp_rate',
    0.05,
    'Target resignation false positives rate, the resignation threshold is dynamically adjusted to keep the false positives rate below this value.',
)
flags.DEFINE_float(
    'disable_resign_ratio',
    0.1,
    'Disable resign for proportion of self-play games so we can measure resignation false positives.',
)
flags.DEFINE_integer(
    'reset_fp_interval',
    500000,
    'The frequency (measured in number of self-play games) to reset resignation threshold,'
    'so statistics from old games do not influence current play.',
)
flags.DEFINE_integer(
    'no_resign_games',
    200000,
    'Initial games played with resignation disable. '
    'This makes sense as when starting out, the prediction from the neural network is not accurate.',
)

flags.DEFINE_float(
    'default_rating',
    0,
    'Default elo rating, change to the rating (for black) from last checkpoint when resume training.',
)
flags.DEFINE_integer('ckpt_interval', 1000, 'The frequency (in training step) to create new checkpoint.')
flags.DEFINE_integer('log_interval', 200, 'The frequency (in training step) to log training statistics.')
flags.DEFINE_string('ckpt_dir', './checkpoints/go/19x19', 'Path for checkpoint file.')
flags.DEFINE_string(
    'logs_dir',
    './logs/go/19x19',
    'Path to save statistics for self-play, training, and evaluation.',
)
flags.DEFINE_string(
    'eval_games_dir',
    './games/pro_games/go/19x19',
    'Path contains evaluation games in sgf format.',
)
flags.DEFINE_string(
    'save_sgf_dir',
    './games/selfplay_games/go/19x19',
    'Path to selfplay and evaluation games in sgf format.',
)
flags.DEFINE_integer('save_sgf_interval', 500, 'How often to save self-play games.')

flags.DEFINE_integer(
    'save_replay_interval',
    0,
    'The frequency (in number of self-play games) to save the replay buffer state.'
    'So we can resume training without staring from zero. 0 means do not save replay state.'
    'If you set this to a non-zero value, you should make sure the path specified by "FLAGS.ckpt_dir" have at least 300GB of free space.',
)
flags.DEFINE_string('load_ckpt', '', 'Resume training by starting from last checkpoint.')
flags.DEFINE_string('load_replay', '', 'Resume training by loading saved replay buffer state.')

flags.DEFINE_string('log_level', 'INFO', '')
flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.register_validator('num_simulations', lambda x: x > 1)
flags.register_validator('log_level', lambda x: x in ['INFO', 'DEBUG'])
flags.register_multi_flags_validator(
    ['num_parallel', 'c_puct_base'],
    lambda flags: flags['c_puct_base'] >= 19652 * (flags['num_parallel'] / 800),
    '',
)


# Initialize flags
FLAGS(sys.argv)

os.environ['BOARD_SIZE'] = str(FLAGS.board_size)

from alpha_zero.envs.go import GoEnv
from alpha_zero.core.pipeline import (
    run_learner_loop,
    run_evaluator_loop,
    run_selfplay_actor_loop,
    set_seed,
    maybe_create_dir,
)
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.core.replay import UniformReplay
from alpha_zero.utils.util import extract_args_from_flags_dict, create_logger


def main():
    set_seed(FLAGS.seed)

    maybe_create_dir(FLAGS.ckpt_dir)
    maybe_create_dir(FLAGS.logs_dir)
    maybe_create_dir(FLAGS.save_sgf_dir)

    logger = create_logger(FLAGS.log_level)

    logger.info(extract_args_from_flags_dict(FLAGS.flag_values_dict()))

    # Default run all instances on CPU
    actor_devices = [torch.device('cpu')] * FLAGS.num_actors
    learner_device = eval_device = torch.device('cpu')

    # Check if GPU is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # Learner and evaluator on last two GPUs, so we might be able to run few more actors on the remaining GPUS
            learner_device = torch.device(f'cuda:{num_gpus-1}')
            eval_device = torch.device(f'cuda:{num_gpus-2}')
        else:
            learner_device = eval_device = torch.device('cuda')

        actor_devices = [torch.device(f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]

    def env_builder():
        return GoEnv(komi=FLAGS.komi, num_stack=FLAGS.num_stack)

    eval_env = env_builder()

    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    def network_builder():
        return AlphaZeroNet(
            input_shape,
            num_actions,
            FLAGS.num_res_blocks,
            FLAGS.num_filters,
            FLAGS.num_fc_units,
        )

    network = network_builder()
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=FLAGS.init_lr,
        momentum=FLAGS.sgd_momentum,
        weight_decay=FLAGS.l2_regularization,
    )
    lr_scheduler = MultiStepLR(optimizer, milestones=FLAGS.lr_milestones, gamma=FLAGS.lr_decay)

    # Use the events to synchronize work between learner and actors.
    stop_event = mp.Event()
    ckpt_event = mp.Event()
    # Transfer samples from self-play process to training process.
    data_queue = mp.Queue(maxsize=FLAGS.num_actors)

    with mp.Manager() as manager:
        var_ckpt = manager.Value('s', b'')
        var_resign_threshold = manager.Value('d', FLAGS.init_resign_threshold)

        replay = UniformReplay(
            capacity=FLAGS.replay_capacity,
            random_state=np.random.RandomState(),
            compress_data=FLAGS.compress_data,
        )

        # Start evaluator
        evaluator = mp.Process(
            target=run_evaluator_loop,
            args=(
                FLAGS.seed,
                network_builder(),
                eval_device,
                eval_env,
                FLAGS.eval_games_dir,
                FLAGS.num_simulations,
                FLAGS.num_parallel,
                FLAGS.c_puct_base,
                FLAGS.c_puct_init,
                FLAGS.default_rating,
                FLAGS.logs_dir,
                FLAGS.save_sgf_dir,
                FLAGS.load_ckpt,
                FLAGS.log_level,
                var_ckpt,
                stop_event,
            ),
        )

        evaluator.start()

        # Start self-play actors
        actors = []
        for i in range(FLAGS.num_actors):
            actor = mp.Process(
                target=run_selfplay_actor_loop,
                args=(
                    FLAGS.seed,
                    i,
                    network_builder(),
                    actor_devices[i],
                    data_queue,
                    env_builder(),
                    FLAGS.num_simulations,
                    FLAGS.num_parallel,
                    FLAGS.c_puct_base,
                    FLAGS.c_puct_init,
                    FLAGS.warm_up_steps,
                    FLAGS.check_resign_after_steps,
                    FLAGS.disable_resign_ratio,
                    FLAGS.save_sgf_dir,
                    FLAGS.save_sgf_interval,
                    FLAGS.logs_dir,
                    FLAGS.load_ckpt,
                    FLAGS.log_level,
                    var_ckpt,
                    var_resign_threshold,
                    ckpt_event,
                    stop_event,
                ),
            )
            actor.start()
            actors.append(actor)

        # Run learner loop on the main process
        run_learner_loop(
            seed=FLAGS.seed,
            network=network,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=learner_device,
            replay=replay,
            logger=logger,
            argument_data=FLAGS.argument_data,
            batch_size=FLAGS.batch_size,
            init_resign_threshold=FLAGS.init_resign_threshold,
            disable_resign_ratio=FLAGS.disable_resign_ratio,
            target_fp_rate=FLAGS.target_fp_rate,
            reset_fp_interval=FLAGS.reset_fp_interval,
            no_resign_games=FLAGS.no_resign_games,
            min_games=FLAGS.min_games,
            games_per_ckpt=FLAGS.games_per_ckpt,
            num_actors=FLAGS.num_actors,
            ckpt_interval=FLAGS.ckpt_interval,
            log_interval=FLAGS.log_interval,
            save_replay_interval=FLAGS.save_replay_interval,
            max_training_steps=FLAGS.max_training_steps,
            ckpt_dir=FLAGS.ckpt_dir,
            logs_dir=FLAGS.logs_dir,
            load_ckpt=FLAGS.load_ckpt,
            load_replay=FLAGS.load_replay,
            data_queue=data_queue,
            var_ckpt=var_ckpt,
            var_resign_threshold=var_resign_threshold,
            ckpt_event=ckpt_event,
            stop_event=stop_event,
        )

        # Wait for all actors to finish
        for actor in actors:
            actor.join()
            actor.close()

        evaluator.join()


if __name__ == '__main__':
    # Set multiprocessing start mode
    mp.set_start_method('spawn')
    main()
