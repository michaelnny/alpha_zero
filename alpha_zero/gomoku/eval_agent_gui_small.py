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
"""Evaluate the AlphaZero on free-style Gomoku game with a simple GUI program."""
from absl import app
from absl import flags
import os
import torch

from alpha_zero.games.gomoku import GomokuEnv
from alpha_zero.games.gui import BoardGameGui
from alpha_zero.network import AlphaZeroNet
from alpha_zero.pipeline_v1 import load_checkpoint
from alpha_zero.mcts_player import create_mcts_player


FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 9, 'Board size for Gomoku.')
flags.DEFINE_integer('stack_history', 4, 'Stack previous states, the state is an image of N x 2 + 1 binary planes.')
flags.DEFINE_integer('num_res_blocks', 5, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer(
    'num_planes',
    64,
    'Number of filters for the conv2d layers, this is also the number of hidden units in the linear layer of the neural network.',
)

flags.DEFINE_bool(
    'human_vs_ai',
    False,
    'Default plays in the Human vs. AlphaZero mode, if False, will play in AlphaZero vs. AlphaZero mode.',
)

flags.DEFINE_bool(
    'show_step',
    True,
    'Show step number on stones, default off.',
)

flags.DEFINE_string(
    'black_ckpt_file',
    'checkpoints/gomoku_small_v2/train_steps_148000',
    'Load the checkpoint file for black player, will only load if human_vs_ai is False.',
)
flags.DEFINE_string(
    'white_ckpt_file', 'checkpoints/gomoku_small_v2/train_steps_148000', 'Load the checkpoint file for white player.'
)

flags.DEFINE_integer('num_simulations', 200, 'Number of simulations per MCTS search.')
flags.DEFINE_integer('parallel_leaves', 8, 'Number of parallel leaves for MCTS search, 1 means do not use parallel search.')

flags.DEFINE_float('c_puct_base', 19652, 'Exploration constants balancing priors vs. value net output.')
flags.DEFINE_float('c_puct_init', 1.25, 'Exploration constants balancing priors vs. value net output.')

flags.DEFINE_float(
    'temperature',
    0.01,
    'Value of the temperature exploration rate after MCTS search to generate play policy.',
)

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')


def main(argv):
    torch.manual_seed(FLAGS.seed)
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_env = GomokuEnv(board_size=FLAGS.board_size, stack_history=FLAGS.stack_history)

    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    def network_builder():
        return AlphaZeroNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.num_planes)

    def load_checkpoint_for_net(network, ckpt_file):
        if ckpt_file and os.path.isfile(ckpt_file):
            loaded_state = load_checkpoint(ckpt_file, runtime_device)
            network.load_state_dict(loaded_state['network'])

    def mcts_player_builder(network):
        return create_mcts_player(
            network=network,
            device=runtime_device,
            num_simulations=FLAGS.num_simulations,
            parallel_leaves=FLAGS.parallel_leaves,
            root_noise=False,
            deterministic=True,
        )

    # White player is alway gonna be AI
    white_network = network_builder().to(device=runtime_device)
    load_checkpoint_for_net(white_network, FLAGS.white_ckpt_file)
    white_network.eval()
    white_mcts_player = mcts_player_builder(white_network)

    # Wrap MCTS player for the GUI program
    def white_player(env: GomokuEnv) -> int:
        action, _, _ = white_mcts_player(env, None, FLAGS.c_puct_base, FLAGS.c_puct_init, FLAGS.temperature)
        return action

    # Black player could be either human or AI
    if FLAGS.human_vs_ai:
        black_player = 'human'
    else:
        # Only create and load network if not in human vs. AI mode.
        black_network = network_builder().to(device=runtime_device)
        load_checkpoint_for_net(black_network, FLAGS.black_ckpt_file)
        black_network.eval()

        black_mcts_player = mcts_player_builder(black_network)

        # Wrap MCTS player for the GUI program
        def black_player(env: GomokuEnv) -> int:
            action, _, _ = black_mcts_player(env, None, FLAGS.c_puct_base, FLAGS.c_puct_init, FLAGS.temperature)
            return action

    game_gui = BoardGameGui(eval_env, black_player=black_player, white_player=white_player, show_step=FLAGS.show_step)
    game_gui.start()


if __name__ == '__main__':
    app.run(main)
