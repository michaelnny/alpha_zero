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
"""Evaluate the AlphaZero on Tic-Tac-Toe."""
from absl import app
from absl import flags
import os
import time
import torch

from alpha_zero.games.tictactoe import TicTacToeEnv
from alpha_zero.network import AlphaZeroNet
from alpha_zero.pipeline_v1 import load_checkpoint
from alpha_zero.mcts_player import create_mcts_player

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_res_blocks', 3, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer(
    'num_planes',
    16,
    'Number of filters for the conv2d layers, this is also the number of hidden units in the linear layer of the neural network.',
)

flags.DEFINE_string(
    'black_ckpt_file', 'checkpoints/tictactoe_v2/train_steps_19000', 'Load the checkpoint file for black player.'
)
flags.DEFINE_string(
    'white_ckpt_file', 'checkpoints/tictactoe_v2/train_steps_19000', 'Load the checkpoint file for white player.'
)

flags.DEFINE_integer('num_simulations', 24, 'Number of simulations per MCTS search.')
flags.DEFINE_integer('parallel_leaves', 4, 'Number of parallel leaves for MCTS search, 1 means do not use parallel search.')
flags.DEFINE_float('c_puct_base', 19652, 'Exploration constants balancing priors vs. value net output.')
flags.DEFINE_float('c_puct_init', 1.25, 'Exploration constants balancing priors vs. value net output.')
flags.DEFINE_float('temperature', 0.01, 'Value of the temperature exploration rate after MCTS search to generate play policy.')
flags.DEFINE_integer('seed', 1, 'Seed the runtime.')


def main(argv):
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_env = TicTacToeEnv()

    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    def network_builder():
        return AlphaZeroNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.num_planes)

    black_network = network_builder().to(device=runtime_device)
    white_network = network_builder().to(device=runtime_device)

    def load_checkpoint_for_net(network, ckpt_file):
        if ckpt_file and os.path.isfile(ckpt_file):
            loaded_state = load_checkpoint(ckpt_file, runtime_device)
            network.load_state_dict(loaded_state['network'])

    load_checkpoint_for_net(black_network, FLAGS.black_ckpt_file)
    load_checkpoint_for_net(white_network, FLAGS.white_ckpt_file)
    black_network.eval()
    white_network.eval()

    def mcts_player_builder(network):
        return create_mcts_player(
            network=network,
            device=runtime_device,
            num_simulations=FLAGS.num_simulations,
            parallel_leaves=FLAGS.parallel_leaves,
            root_noise=False,
            deterministic=True,
        )

    black_player = mcts_player_builder(black_network)
    white_player = mcts_player_builder(white_network)

    # Start to play game
    steps = 0
    _ = eval_env.reset()
    while True:
        if eval_env.current_player_name == 'black':
            player = black_player
        else:
            player = white_player

        action, _, _ = player(eval_env, None, FLAGS.c_puct_base, FLAGS.c_puct_init, FLAGS.temperature)

        _, reward, done, _ = eval_env.step(action)
        eval_env.render('human')

        steps += 1
        time.sleep(0.5)

        if done:
            break

    eval_env.close()


if __name__ == '__main__':
    app.run(main)
