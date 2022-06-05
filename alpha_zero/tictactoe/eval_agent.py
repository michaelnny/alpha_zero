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
"""AlphaZero vs. AlphaZero on free-style Gomoku game with a small sized board."""
from absl import app
from absl import flags
from absl import logging
import os
import time
import torch

from alpha_zero.games.tictactoe import TicTacToeEnv
from alpha_zero.network import AlphaZeroNet
from alpha_zero.pipeline_v1 import create_mcts_player, load_checkpoint


FLAGS = flags.FLAGS
flags.DEFINE_integer('num_res_blocks', 2, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer('num_planes', 16, 'Number of planes for the conv2d layer in the neural network.')
flags.DEFINE_integer(
    'num_fc_units', 16, 'Number of hidden units for the fully connected layer in value head of the neural network.'
)

flags.DEFINE_string(
    'load_black_checkpoint_file',
    'checkpoints/tictactoe_v2/train_steps_xx40000',
    'Load the last checkpoint from file.',
)
flags.DEFINE_string(
    'load_white_checkpoint_file',
    'checkpoints/tictactoe_v2/train_steps_30000',
    'Load the last checkpoint from file.',
)

flags.DEFINE_integer('num_simulations', 25, 'Number of simulations per MCTS search, per agent environment time step.')

flags.DEFINE_float('c_puct', 5.0, 'Puct constant of the UCB score.')

flags.DEFINE_float(
    'temp_value',
    0.1,
    'Value of the temperature exploration rate after MCTS search to generate play policy.',
)

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')


def main(argv):
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_env = TicTacToeEnv()

    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    black_network = AlphaZeroNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.num_fc_units).to(
        device=runtime_device
    )

    white_network = AlphaZeroNet(input_shape, num_actions, FLAGS.num_res_blocks, FLAGS.num_planes, FLAGS.num_fc_units).to(
        device=runtime_device
    )

    if FLAGS.load_black_checkpoint_file and os.path.isfile(FLAGS.load_black_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_black_checkpoint_file, runtime_device)
        black_network.load_state_dict(loaded_state['network'])

    if FLAGS.load_white_checkpoint_file and os.path.isfile(FLAGS.load_white_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_white_checkpoint_file, runtime_device)
        white_network.load_state_dict(loaded_state['network'])

    black_network.eval()
    white_network.eval()

    def alpha_zero_black_player(env: TicTacToeEnv) -> int:
        player = create_mcts_player(black_network, runtime_device)
        action, _, _ = player(env, None, 5.0, FLAGS.temp_value, FLAGS.num_simulations, False, True)
        return action

    def alpha_zero_white_player(env: TicTacToeEnv) -> int:
        player = create_mcts_player(white_network, runtime_device)
        action, _, _ = player(env, None, 5.0, FLAGS.temp_value, FLAGS.num_simulations, False, True)
        return action

    steps = 0
    returns = 0.0

    obs = eval_env.reset()
    while True:
        if eval_env.current_player_name == 'black':
            action = alpha_zero_black_player(eval_env)
        else:
            action = alpha_zero_white_player(eval_env)

        _, reward, done, _ = eval_env.step(action)
        eval_env.render('human')

        steps += 1
        returns += reward

        time.sleep(0.5)

        if done:
            break

    eval_env.close()
    logging.info(f'Episode returns: {returns}, steps: {steps}')


if __name__ == '__main__':
    app.run(main)
