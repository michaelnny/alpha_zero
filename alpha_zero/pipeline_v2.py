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
"""Functions for AlphaZero agent self-play training pipeline.

From the paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
https://arxiv.org/abs//1712.01815

"""
from absl import logging
from typing import List
from pathlib import Path
import time
import timeit
import multiprocessing
import torch

from alpha_zero.games.env import BoardGameEnv
from alpha_zero.replay import UniformReplay
from alpha_zero.log import CsvWriter, write_to_csv
from alpha_zero.rating import compute_elo_rating
from alpha_zero.mcts_player import create_mcts_player
from alpha_zero.pipeline_v1 import (
    calc_loss,
    run_self_play,
    run_data_collector,
    init_absl_logging,
    get_time_stamp,
    create_checkpoint,
    load_checkpoint,
    disable_auto_grad,
    handle_exit_signal,
    load_from_file,
)


def run_training(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    device: torch.device,
    actor_network: torch.nn.Module,
    replay: UniformReplay,
    data_queue: multiprocessing.SimpleQueue,
    min_replay_size: int,
    batch_size: int,
    num_train_steps: int,
    checkpoint_frequency: int,
    checkpoint_dir: str,
    checkpoint_files: List,
    csv_file: str,
    stop_event: multiprocessing.Event,
    delay: float = 0.0,
    train_steps: int = 0,
    argument_data: bool = False,
):
    """Run the main training loop for N iterations, each iteration contains M updates.
    This controls the 'pace' of the pipeline, including when should the other parties to stop.

    Args:
        network: the neural network we want to optimize.
        optimizer: neural network optimizer.
        lr_scheduler: learning rate annealing scheduler.
        device: torch runtime device.
        actor_network: the neural network actors running self-play, for the case AlphaZero pipeline without evaluation.
        replay: a simple uniform experience replay.
        data_queue: a multiprocessing.SimpleQueue instance, only used to signal data collector to stop.
        min_replay_size: minimum replay size before start training.
        batch_size: sample batch size during training.
        num_train_steps: total number of training steps to run.
        checkpoint_frequency: the frequency to create new checkpoint.
        checkpoint_dir: create new checkpoint save directory.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint files.
        csv_file: a csv file contains the training statistics.
        stop_event: a multiprocessing.Event signaling other parties to stop running pipeline.
        delay: wait time (in seconds) before start training on next batch samples, default 0.
        train_steps: already trained steps, used when resume training, default 0.
        argument_data: if true, apply random rotation and reflection during training, default off.

    Raises:
        ValueError:
            if `min_replay_size` less than `batch_size`.
            if `checkpoint_dir` is invalid.
    """

    if min_replay_size < batch_size:
        raise ValueError(f'Expect min_replay_size > batch_size, got {min_replay_size}, and {batch_size}')
    if not isinstance(checkpoint_dir, str) or checkpoint_dir == '':
        raise ValueError(f'Expect checkpoint_dir to be valid path, got {checkpoint_dir}')

    writer = CsvWriter(csv_file)
    logging.info('Start training thread')
    start = None
    last_train_step = train_steps  # Store train step from last session incase resume training
    disable_auto_grad(actor_network)
    
    network = network.to(device=device)
    network.train()
    actor_network.eval()

    ckpt_dir = Path(checkpoint_dir)
    if checkpoint_dir is not None and checkpoint_dir != '' and not ckpt_dir.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_state_to_save():
        return {
            'network': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_steps': train_steps,
        }

    while True:
        if replay.size < min_replay_size:
            time.sleep(30)
            continue

        if start is None:
            start = timeit.default_timer()

        # Signaling other parties to stop running pipeline.
        if train_steps >= num_train_steps:
            break

        transitions = replay.sample(batch_size)
        optimizer.zero_grad()
        loss = calc_loss(network, device, transitions, argument_data)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_steps += 1

        if train_steps > 1 and train_steps % checkpoint_frequency == 0:
            train_rate = ((train_steps - last_train_step) * batch_size) / (timeit.default_timer() - start)
            logging.info(f'Train step {train_steps}, train sample rate {train_rate:.2f}')

            state_to_save = get_state_to_save()
            ckpt_file = ckpt_dir / f'train_steps_{train_steps}'
            create_checkpoint(state_to_save, ckpt_file)
            checkpoint_files.append(ckpt_file)

            actor_network.load_state_dict(network.state_dict())
            actor_network.eval()

            log_output = [
                ('timestamp', get_time_stamp(), '%1s'),
                ('train_steps', train_steps, '%3d'),
                ('checkpoint', ckpt_file, '%3s'),
                ('loss', loss.detach().item(), '%.4f'),
                ('learning_rate', lr_scheduler.get_last_lr()[0], '%.2f'),
                ('train_sample_rate', train_rate, '%.2f'),
            ]
            write_to_csv(writer, log_output)

        # Wait for sometime before start training on next batch.
        if delay is not None and delay > 0 and train_steps > 1:
            time.sleep(delay)

    stop_event.set()
    time.sleep(60)
    data_queue.put('STOP')


def run_evaluation(
    old_network: torch.nn.Module,
    new_network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    c_puct_base: float,
    c_puct_init: float,
    temperature: float,
    num_simulations: int,
    parallel_leaves: int,
    checkpoint_files: List,
    csv_file: str,
    stop_event: multiprocessing.Event,
    initial_elo: int = 0,
) -> None:
    """Monitoring training progress by play a single game with new checkpoint against last checkpoint.

    Args:
        old_network: the last checkpoint network.
        new_network: new checkpoint network we want to evaluate.
        device: torch runtime device.
        env: a BoardGameEnv type environment.
        c_puct: a constant controls the level of exploration during MCTS search.
        temperature: the temperature exploration rate after MCTS search
            to generate play policy.
        num_simulations: number of simulations for each MCTS search.
        parallel_leaves: Number of parallel leaves for MCTS search.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint.
        csv_file: a csv file contains the statistics for the best checkpoint.
        stop_event: a multiprocessing.Event signaling to stop running pipeline.
        initial_elo: initial elo ratings for the players, default 0.

     Raises:
        ValueError:
            if `env` is not a valid BoardGameEnv instance.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f'Expect env to be a valid BoardGameEnv instance, got {env}')

    init_absl_logging()
    handle_exit_signal()
    writer = CsvWriter(csv_file)
    logging.info('Start evaluator')

    disable_auto_grad(old_network)
    disable_auto_grad(new_network)

    old_network = old_network.to(device=device)
    new_network = new_network.to(device=device)

    # Set initial elo ratings
    black_elo = initial_elo
    white_elo = initial_elo

    while True:
        if stop_event.is_set() and len(checkpoint_files) == 0:
            break
        if len(checkpoint_files) == 0:
            continue

        # Remove the checkpoint file path from the shared list.
        ckpt_file = checkpoint_files.pop(0)
        loaded_state = load_checkpoint(ckpt_file, device)
        new_network.load_state_dict(loaded_state['network'])
        train_steps = loaded_state['train_steps']

        new_network.eval()
        old_network.eval()

        # Black is the new checkpoint, white is last checkpoint.
        black_player = create_mcts_player(
            network=new_network,
            device=device,
            num_simulations=num_simulations,
            parallel_leaves=parallel_leaves,
            root_noise=False,
            deterministic=True,
        )
        white_player = create_mcts_player(
            network=old_network,
            device=device,
            num_simulations=num_simulations,
            parallel_leaves=parallel_leaves,
            root_noise=False,
            deterministic=True,
        )

        env.reset()
        done = False
        root_node = None
        steps = 0

        while not done:
            if env.current_player == env.black_player:
                action, _, root_node = black_player(env, root_node, c_puct_base, c_puct_init, temperature)
            else:
                action, _, root_node = white_player(env, root_node, c_puct_base, c_puct_init, temperature)
            _, _, done, _ = env.step(action)
            steps += 1

        if env.winner == env.black_player:
            black_elo, _ = compute_elo_rating(0, black_elo, white_elo)
        elif env.winner == env.white_player:
            black_elo, _ = compute_elo_rating(1, black_elo, white_elo)
        white_elo = black_elo

        log_output = [
            ('timestamp', get_time_stamp(), '%1s'),
            ('train_steps', train_steps, '%3d'),
            ('checkpoint', ckpt_file, '%3s'),
            ('elo_rating', black_elo, '%1d'),
            ('episode_steps', steps, '%1d'),
        ]
        write_to_csv(writer, log_output)

        # Unlike in AlphaGo Zero, here we always use the latest checkpoint for next evaluation.
        old_network.load_state_dict(new_network.state_dict())
