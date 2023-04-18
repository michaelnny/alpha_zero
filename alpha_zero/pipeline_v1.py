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
"""Functions for AlphaGo Zero agent self-play training pipeline.

From the paper "Mastering the game of Go without human knowledge"
https://www.nature.com/articles/nature24270/

"""
from absl import logging
from typing import List, Tuple, Mapping, Union, Text, Any
from pathlib import Path
import os
import sys
import signal
import pickle
import collections
import time
import timeit
import queue
import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F

from alpha_zero.games.env import BoardGameEnv
from alpha_zero.replay import Transition, UniformReplay
from alpha_zero.log import CsvWriter, write_to_csv
from alpha_zero.data_processing import random_rotation_and_reflection
from alpha_zero.rating import compute_elo_rating
from alpha_zero.mcts_player import create_mcts_player


def run_self_play(
    rank: int,
    network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    data_queue: multiprocessing.Queue,
    c_puct_base: float,
    c_puct_init: float,
    warm_up_steps: int,
    num_simulations: int,
    num_parallel: int,
    stop_event: multiprocessing.Event,
) -> None:
    """Run self-play loop to generate training samples.
    Only stop the loop if `stop_event` is set to True.

    Args:
        rank: the rank of the self-play process.
        network: neural network to evaluate position,
            this is the current best network.
        device: torch runtime device.
        env: a BoardGameEnv type environment.
        data_queue: a multiprocessing.Queue to send samples to training process.
        c_puct: a constant controls the level of exploration during MCTS search.
        warm_up_steps: number of opening environment steps to
            sample action according search policy instead of choosing most visited child.
        num_simulations: number of simulations for each MCTS search.
        num_parallel: Number of parallel leaves for MCTS search.
        stop_event: a multiprocessing.Event that will signal the end of training.

    Raises:
        ValueError:
            if `env` is not a valid BoardGameEnv instance.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f'Expect env to be a valid BoardGameEnv instance, got {env}')

    init_absl_logging()
    handle_exit_signal()
    logging.info(f'Start self-play actor {rank}')
    disable_auto_grad(network)
    network = network.to(device=device)
    network.eval()

    played_games = 0
    mcts_player = create_mcts_player(
        network=network,
        device=device,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        root_noise=True,
        deterministic=False,
    )

    while not stop_event.is_set():
        # For each new game.
        obs = env.reset()
        done = False

        episode_states = []
        episode_search_pis = []
        episode_values = []
        player_ids = []

        root_node = None

        # Play and record transitions.
        while not done:
            temperature = 0.01 if env.steps >= warm_up_steps else 1.0
            move, search_pi, root_node = mcts_player(env, root_node, c_puct_base, c_puct_init, temperature)

            episode_states.append(obs)
            episode_search_pis.append(search_pi)
            episode_values.append(0.0)

            player_ids.append(env.current_player)

            obs, reward, done, _ = env.step(move)

        if reward != 0:
            for i, play_id in enumerate(player_ids):
                if play_id == env.last_player:
                    episode_values[i] = reward
                else:
                    episode_values[i] = -reward

        data_queue.put(
            [Transition(state=x, pi_prob=pi, value=v) for x, pi, v in zip(episode_states, episode_search_pis, episode_values)]
        )

        played_games += 1

    logging.info(f'Stop self-play actor {rank}')


def run_training(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    device: torch.device,
    replay: UniformReplay,
    data_queue: multiprocessing.SimpleQueue,
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
    log_interval: int = 1000,
):
    """Run the main training loop for N iterations, each iteration contains M updates.

    Args:
        network: the neural network we want to optimize.
        optimizer: neural network optimizer.
        lr_scheduler: learning rate annealing scheduler.
        device: torch runtime device.
        replay: a simple uniform experience replay.
        data_queue: a multiprocessing.SimpleQueue instance, only used to signal data collector to stop.
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
        log_interval: how often to log training statistics, default 1000.

    Raises:
        ValueError:
            if `min_replay_size` less than `batch_size`.
            if `checkpoint_dir` is invalid.
    """

    if not isinstance(checkpoint_dir, str) or checkpoint_dir == '':
        raise ValueError(f'Expect checkpoint_dir to be valid path, got {checkpoint_dir}')

    logging.info('Start training thread')
    start = None
    writer = CsvWriter(csv_file)

    last_train_step = train_steps  # Store train step from last session incase resume training

    network = network.to(device=device)
    network.train()

    if train_steps is None:
        train_steps = -1

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
        if replay.num_games_added < 200:
            time.sleep(30)
            continue

        if start is None:
            start = timeit.default_timer()

        # Signaling other parties to stop running pipeline.
        if train_steps >= num_train_steps:
            break

        transitions = replay.sample(batch_size)

        if transitions is None:
            continue

        optimizer.zero_grad()
        policy_loss, value_loss = calc_loss(network, device, transitions, argument_data)
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_steps += 1

        if train_steps % checkpoint_frequency == 0:
            state_to_save = get_state_to_save()
            ckpt_file = ckpt_dir / f'train_steps_{train_steps}'
            create_checkpoint(state_to_save, ckpt_file)
            checkpoint_files.append(ckpt_file)

            train_rate = ((train_steps - last_train_step) * batch_size) / (timeit.default_timer() - start)
            logging.info(f'Train step {train_steps}, train sample rate {train_rate:.2f}')

        if train_steps % log_interval == 0:
            log_output = [
                ('timestamp', get_time_stamp(), '%1s'),
                ('train_steps', train_steps, '%3d'),
                ('policy_loss', policy_loss.detach().item(), '%.4f'),
                ('value_loss', value_loss.detach().item(), '%.4f'),
                ('learning_rate', lr_scheduler.get_last_lr()[0], '%.2f'),
            ]
            write_to_csv(writer, log_output)

        # Wait for sometime before start training on next batch.
        if delay > 0:
            time.sleep(delay)

    stop_event.set()
    time.sleep(60)
    data_queue.put('STOP')


def run_evaluation(
    best_network: torch.nn.Module,
    new_network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    num_games: int,
    c_puct_base: float,
    c_puct_init: float,
    temperature: float,
    num_simulations: int,
    num_parallel: int,
    checkpoint_files: List,
    csv_file: str,
    stop_event: multiprocessing.Event,
    initial_elo: int = 0,
    win_ratio: float = 0.55,
) -> None:
    """Run the evaluation loop to select best player from new checkpoint,
    the 'new' best player is then used by the 'self-play' processes to generate training samples.
    Only stop the loop if `stop_event` is set to True.

    Args:
        best_network: the current best player.
        new_network: new checkpoint network we want to evaluate.
        device: torch runtime device.
        env: a BoardGameEnv type environment.
        num_games: number of games to play during each evaluation process.
        c_puct: a constant controls the level of exploration during MCTS search.
        temperature: the temperature exploration rate after MCTS search
            to generate play policy.
        num_simulations: number of simulations for each MCTS search.
        num_parallel: Number of parallel leaves for MCTS search.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint.
        csv_file: a csv file contains the statistics for the best checkpoint.
        stop_event: a multiprocessing.Event signaling to stop running pipeline.
        initial_elo: initial elo rating, default 0,
        win_ratio: the win ratio for new checkpoint to become new best player, default 0.55.

     Raises:
        ValueError:
            if `env` is not a valid BoardGameEnv instance.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f'Expect env to be a valid BoardGameEnv instance, got {env}')

    init_absl_logging()
    handle_exit_signal()

    logging.info('Start evaluator')
    writer = CsvWriter(csv_file)

    disable_auto_grad(best_network)
    disable_auto_grad(new_network)

    best_network = best_network.to(device=device)
    new_network = new_network.to(device=device)

    # Black is the new checkpoint, white is current best player.
    black_player = create_mcts_player(
        network=new_network,
        device=device,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        root_noise=False,
        deterministic=True,
    )

    white_player = create_mcts_player(
        network=best_network,
        device=device,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        root_noise=False,
        deterministic=True,
    )

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
        best_network.eval()

        won_games, loss_games, draw_games = 0, 0, 0
        is_new_best_player = False
        should_delete_ckpt_file = False

        steps = 0

        for _ in range(num_games):
            env.reset()
            done = False

            while not done:
                if env.current_player == env.black_player:
                    active_player = black_player
                else:
                    active_player = white_player

                action, _, _ = active_player(env, None, c_puct_base, c_puct_init, temperature)
                _, _, done, _ = env.step(action)
                steps += 1

            if env.winner is None:
                draw_games += 1
            elif env.winner == env.black_player:
                won_games += 1
            else:
                loss_games += 1

            # Check for early stop.
            # Case 1 - Won enough games
            if won_games / num_games >= win_ratio:
                is_new_best_player = True
                break
            # Case 2 - Loss enough games
            if loss_games / num_games > (1 - win_ratio):
                should_delete_ckpt_file = True
                break

        # Do a final check, this time exclude draw games.
        if won_games + loss_games > 0 and won_games / (won_games + loss_games) > win_ratio:
            is_new_best_player = True

        # We treat one evaluation as a 'single' game, otherwise the elo ratings might become too large.
        if is_new_best_player:
            best_network.load_state_dict(loaded_state['network'])
            logging.info(f"New best player loaded from checkpoint '{ckpt_file}'")

            black_elo, _ = compute_elo_rating(0, black_elo, white_elo)
            white_elo = black_elo
        elif should_delete_ckpt_file:
            os.remove(ckpt_file)

            black_elo, white_elo = compute_elo_rating(1, black_elo, white_elo)

        log_output = [
            ('timestamp', get_time_stamp(), '%1s'),
            ('train_steps', train_steps, '%3d'),
            ('elo_rating', black_elo, '%1d'),
            ('episode_steps', steps / num_games, '%1d'),  # mean episode steps
            ('won_games', won_games, '%1d'),
            ('loss_games', loss_games, '%1d'),
            ('draw_games', draw_games, '%1d'),
        ]
        write_to_csv(writer, log_output)


def run_data_collector(
    data_queue: multiprocessing.SimpleQueue,
    replay: UniformReplay,
    log_interval: int = 1000,  # every 1000 games
) -> None:
    """Collect samples from self-play,
    this runs on the same process as the training loop,
    but with a separate thread.

    Args:
        data_queue: a multiprocessing.SimpleQueue to receive samples from self-play processes.
        replay: a simple uniform random experience replay.
        log_interval: how often to log the statistic, measured in number of games received.

    """
    logging.info('Start data collector thread')
    start = timeit.default_timer()

    game_steps = collections.deque(maxlen=1000)

    while True:
        try:
            item = data_queue.get()
            if item == 'STOP':
                break

            replay.add_game(item)
            game_steps.append(len(item))

            if replay.num_games_added % log_interval == 0:
                sample_gen_rate = replay.num_samples_added / (timeit.default_timer() - start)
                logging.info(
                    f'Collected {replay.num_games_added} self-play games, sample generation rate {sample_gen_rate:.2f}'
                )

        except queue.Empty:
            pass
        except EOFError:
            pass


def process_episode_trajectory(winner: int, episode_trajectory: List[Transition]) -> None:
    """Update the reward for the transitions, note this operation is in-place."""
    for i in range(len(episode_trajectory)):
        transition = episode_trajectory[i]
        if transition.player_id == winner:
            value = 1.0
        else:
            value = -1.0
        episode_trajectory[i] = transition._replace(value=value)


def calc_loss(
    network: torch.nn.Module, device: torch.device, transitions: Transition, argument_data: bool = False
) -> torch.Tensor:
    """Compute the AlphaZero loss."""
    # [B, state_shape]
    state = torch.from_numpy(transitions.state).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, num_actions]
    pi_prob = torch.from_numpy(transitions.pi_prob).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, ]
    value = torch.from_numpy(transitions.value).to(device=device, dtype=torch.float32, non_blocking=True)

    if argument_data:
        # Argument data by apply random rotation and reflection.
        state, pi_prob = random_rotation_and_reflection(state, pi_prob)

    network_out = network(state)

    # value MSE loss
    value_loss = F.mse_loss(network_out.value.squeeze(1), value, reduction='mean')

    # policy cross-entropy loss
    policy_loss = F.cross_entropy(network_out.pi_logits, pi_prob, reduction='mean')

    return policy_loss, value_loss


def init_absl_logging():
    """Initialize absl.logging when run the process without app.run()"""
    logging._warn_preinit_stderr = 0  # pylint: disable=protected-access
    logging.set_verbosity(logging.INFO)
    logging.use_absl_handler()


def handle_exit_signal():
    """Listen to exit signal like ctrl-c or kill from os and try to exit the process forcefully."""

    def shutdown(signal_code, frame):
        del frame
        logging.info(
            f'Received signal {signal_code}: terminating process...',
        )
        sys.exit(128 + signal_code)

    # Listen to signals to exit process.
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


def get_time_stamp(as_file_name: bool = False) -> str:
    t = time.localtime()
    if as_file_name:
        timestamp = time.strftime('%Y%m%d_%H%M%S', t)
    else:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return timestamp


def create_checkpoint(state_to_save: Mapping[Text, Any], ckpt_file: str) -> None:
    torch.save(state_to_save, ckpt_file)


def load_checkpoint(ckpt_file: str, device: torch.device) -> Mapping[Text, Any]:
    return torch.load(ckpt_file, map_location=torch.device(device))


def save_to_file(obj: Any, file_name: str) -> None:
    """Save object to file."""
    pickle.dump(obj, open(file_name, 'wb'))


def load_from_file(file_name: str) -> Any:
    """Load object from file."""
    return pickle.load(open(file_name, 'rb'))


def disable_auto_grad(network: torch.nn.Module) -> None:
    for p in network.parameters():
        p.requires_grad = False
