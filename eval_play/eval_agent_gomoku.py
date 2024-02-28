# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Evaluate the AlphaZero agent on freestyle Gomoku game."""
from absl import flags
import os
import sys
import torch

FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 13, 'Board size for freestyle Gomoku.')
flags.DEFINE_integer(
    'num_stack',
    8,
    'Stack N previous states, the state is an image of N x 2 + 1 binary planes.',
)
flags.DEFINE_integer('num_res_blocks', 10, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer('num_filters', 40, 'Number of filters for the conv2d layers in the neural network.')
flags.DEFINE_integer(
    'num_fc_units',
    80,
    'Number of hidden units in the linear layer of the neural network.',
)

flags.DEFINE_string(
    'black_ckpt',
    './checkpoints/gomoku/13x13/training_steps_170000.ckpt',
    'Load the checkpoint file for black player.',
)
flags.DEFINE_string(
    'white_ckpt',
    './checkpoints/gomoku/13x13/training_steps_200000.ckpt',
    'Load the checkpoint file for white player.',
)

flags.DEFINE_integer('num_simulations', 400, 'Number of iterations per MCTS search.')
flags.DEFINE_integer(
    'num_parallel',
    8,
    'Number of leaves to collect before using the neural network to evaluate the positions during MCTS search, 1 means no parallel search.',
)

flags.DEFINE_float('c_puct_base', 19652, 'Exploration constants balancing priors vs. search values.')
flags.DEFINE_float('c_puct_init', 1.25, 'Exploration constants balancing priors vs. search values.')

flags.DEFINE_bool('human_vs_ai', True, 'Black player is human, default on.')
flags.DEFINE_bool('show_steps', False, 'Show step number on stones, default off.')

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

# Initialize flags
FLAGS(sys.argv)

from alpha_zero.envs.gomoku import GomokuEnv
from alpha_zero.envs.gui import BoardGameGui
from alpha_zero.core.network import AlphaZeroNet
from alpha_zero.core.pipeline import create_mcts_player, set_seed, disable_auto_grad
from alpha_zero.utils.util import create_logger


def main():
    set_seed(FLAGS.seed)
    logger = create_logger()

    runtime_device = 'cpu'
    if torch.cuda.is_available():
        runtime_device = 'cuda'
    elif torch.backends.mps.is_available():
        runtime_device = 'mps'

    eval_env = GomokuEnv(board_size=FLAGS.board_size, num_stack=FLAGS.num_stack)

    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    def network_builder():
        return AlphaZeroNet(
            input_shape,
            num_actions,
            FLAGS.num_res_blocks,
            FLAGS.num_filters,
            FLAGS.num_fc_units,
            True,
        )

    def load_checkpoint_for_net(network, ckpt_file, device):
        if ckpt_file and os.path.isfile(ckpt_file):
            loaded_state = torch.load(ckpt_file, map_location=torch.device(device))
            network.load_state_dict(loaded_state['network'])
        else:
            logger.warning(f'Invalid checkpoint file "{ckpt_file}"')

    def mcts_player_builder(ckpt_file, device):
        network = network_builder().to(device)
        disable_auto_grad(network)
        load_checkpoint_for_net(network, ckpt_file, device)
        network.eval()

        return create_mcts_player(
            network=network,
            device=device,
            num_simulations=FLAGS.num_simulations,
            num_parallel=FLAGS.num_parallel,
            root_noise=False,
            deterministic=False,
        )

    # Wrap MCTS player for the GUI program
    def wrap_player(mcts_player) -> int:
        def act(env):
            action, *_ = mcts_player(env, None, FLAGS.c_puct_base, FLAGS.c_puct_init)
            return action

        return act

    white_player = mcts_player_builder(FLAGS.white_ckpt, runtime_device)
    white_player = wrap_player(white_player)

    if FLAGS.human_vs_ai:
        black_player = 'human'
    else:
        black_player = mcts_player_builder(FLAGS.black_ckpt, runtime_device)
        black_player = wrap_player(black_player)

    game_gui = BoardGameGui(
        eval_env,
        black_player=black_player,
        white_player=white_player,
        show_steps=FLAGS.show_steps,
    )

    game_gui.start()


if __name__ == '__main__':
    main()
