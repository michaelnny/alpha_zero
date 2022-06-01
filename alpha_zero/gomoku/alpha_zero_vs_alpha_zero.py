"""AlphaZero vs. AlphaZero on free-style Gomoku game with a small sized board."""
from absl import app
from absl import flags
import os
import torch

from alpha_zero.games.gomoku import GomokuEnv
from alpha_zero.games.gui import BoardGameGui
from alpha_zero.network import AlphaZeroNet
from alpha_zero.pipeline_v1 import create_mcts_player, load_checkpoint


FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 11, 'Board size for Gomoku.')
flags.DEFINE_integer('stack_history', 4, 'Stack previous states.')

flags.DEFINE_integer('num_res_blocks', 6, 'Number of residual blocks in the neural network.')
flags.DEFINE_integer('num_planes', 64, 'Number of planes for the conv2d layer in the neural network.')
flags.DEFINE_integer(
    'num_fc_units', 64, 'Number of hidden units for the fully connected layer in value head of the neural network.'
)

flags.DEFINE_string(
    'load_black_checkpoint_file',
    'saved_checkpoints/gomoku_train_steps_502000',
    'Load the last checkpoint from file.',
)
flags.DEFINE_string(
    'load_white_checkpoint_file',
    'saved_checkpoints/gomoku_train_steps_502000',
    'Load the last checkpoint from file.',
)

flags.DEFINE_integer('num_simulations', 400, 'Number of simulations per MCTS search, per agent environment time step.')

flags.DEFINE_float('c_puct', 5.0, 'Puct constant of the UCB score.')

flags.DEFINE_float(
    'temp_value',
    0.1,
    'Value of the temperature exploration rate after MCTS search to generate play policy.',
)

flags.DEFINE_integer('seed', 1, 'Seed the runtime.')


def main(argv):
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_env = GomokuEnv(board_size=FLAGS.board_size, stack_history=FLAGS.stack_history)

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

    def alpha_zero_black_player(env: GomokuEnv) -> int:
        player = create_mcts_player(black_network, runtime_device)
        action, _, _ = player(env, None, 5.0, FLAGS.temp_value, FLAGS.num_simulations, False, True)
        return action

    def alpha_zero_white_player(env: GomokuEnv) -> int:
        player = create_mcts_player(white_network, runtime_device)
        action, _, _ = player(env, None, 5.0, FLAGS.temp_value, FLAGS.num_simulations, False, True)
        return action

    game_gui = BoardGameGui(eval_env, black_player=alpha_zero_black_player, white_player=alpha_zero_white_player)
    game_gui.start()


if __name__ == '__main__':
    app.run(main)
