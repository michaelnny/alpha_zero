# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


import math

from alpha_zero.core.network import AlphaZeroNet


def compute_nn_params(
    board_size,
    num_stack,
    resign,
    num_res_blocks,
    num_filters,
    num_fc_units,
    pad_3=False,
):
    input_shape = (num_stack * 2 + 1, board_size, board_size)
    num_actions = board_size * board_size + 1 if resign else board_size * board_size

    print(f'Board size: {board_size}')
    print(f'Number of stack: {num_stack}')
    print(f'Number of res-blocks: {num_res_blocks}')
    print(f'Number of conv filters: {num_filters}')
    print(f'Number of fc units: {num_fc_units}')
    print(f'Input shape: {input_shape}')

    net = AlphaZeroNet(input_shape, num_actions, num_res_blocks, num_filters, num_fc_units, pad_3)

    num_params = sum(p.data.nelement() for p in net.parameters())
    print(f'Number of parameters: {num_params} \n')


def compute_puct(n, c_puct_base, c_puct_init):
    return c_puct_init + math.log((n + c_puct_base + 1.0) / c_puct_base)


def print_title(title):
    print('-' * 60)
    print(f'{title}')
    print('-' * 60)


if __name__ == '__main__':
    # ========================================================================
    # Neural Network parameters
    # ========================================================================
    print_title('AZ - Neural Network Parameters')

    print_title('AlphaZero - 20 blocks')
    compute_nn_params(19, 8, True, 19, 256, 256)

    print_title('AlphaZero - 40 blocks')
    compute_nn_params(19, 8, True, 39, 256, 256)

    print_title('Gomoku')
    compute_nn_params(13, 8, False, 13, 32, 64, True)
    compute_nn_params(13, 8, False, 10, 40, 80, True)

    print_title('Go')
    compute_nn_params(9, 8, True, 19, 64, 64)
    compute_nn_params(9, 8, True, 11, 64, 64)
    compute_nn_params(9, 8, True, 10, 128, 128)

    # ========================================================================
    # Training sample ratio
    # ========================================================================

    print_title('AZ - Training sample ratio')
    window_size = 500000
    avg_length = 80

    batch_size = 2048
    ckpt_interval = 1000

    capacity = window_size * avg_length
    samples_per_ckpt = batch_size * ckpt_interval

    rate = samples_per_ckpt / capacity

    print(f'AZ training sample ratio per checkpoint: {rate:.2f}')
    print('\n\n')

    # ========================================================================
    # MCTS exploration
    # ========================================================================

    print_title('AZ - MCTS UCT exploration')

    print('AZ original')
    c_puct_base = 19652
    c_puct_init = 1.25

    for n in range(0, 800, 20):
        uct = compute_puct(n, c_puct_base, c_puct_init)
        print(f'N={n}, UCT={uct:.2f}')

    print('\n\n')

    print('AZ - scaled down')
    c_puct_base = 19652 / 2
    c_puct_init = 1.25

    for n in range(0, 400, 20):
        uct = compute_puct(n, c_puct_base, c_puct_init)
        print(f'N={n}, UCT={uct:.2f}')
