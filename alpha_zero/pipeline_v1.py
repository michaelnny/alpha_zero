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
from typing import List, Tuple, Mapping, Text, Any
from pathlib import Path
import os
import sys
import signal
import pickle
import time
import queue
import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F

from alpha_zero.games.env import BoardGameEnv
from alpha_zero.replay import Transition, UniformReplay
from alpha_zero.log import CsvWriter, write_to_csv
from alpha_zero.mcts import Node, uct_search
from alpha_zero.data_processing import random_rotation_and_reflection
from alpha_zero.rating import compute_elo_rating


def run_self_play(
    rank: int,
    network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    data_queue: multiprocessing.Queue,
    c_puct: float,
    temp_begin_value: float,
    temp_end_value: float,
    temp_decay_steps: int,
    num_simulations: int,
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
        temp_begin_value: begin value of the temperature exploration rate
            after MCTS search to generate play policy.
        temp_end_value: end value of the temperature exploration rate
            after MCTS search to generate play policy.
        temp_decay_steps: number of environment steps to decay the temperture from begin_value to end_value
        num_simulations: number of simulations for each MCTS search.
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

    game = 0
    actor_player = create_mcts_player(network, device)

    while not stop_event.is_set():
        # For each new game.
        obs = env.reset()
        done = False
        reward = 0.0
        episode_trajectory: List[Transition] = []
        temp = temp_begin_value
        root_node = None

        # Play and record transitions.
        while not done:
            if env.steps >= temp_decay_steps:
                temp = temp_end_value
            action, pi_prob, root_node = actor_player(env, root_node, c_puct, temp, num_simulations, True)
            transition = Transition(
                state=obs,
                pi_prob=pi_prob,
                value=0.0,
                player_id=env.current_player,
            )
            episode_trajectory.append(transition)
            obs, reward, done, _ = env.step(action)

            # Add final observation, using uniform policy probabilities.
            if done:
                transition = Transition(
                    state=obs,
                    pi_prob=np.ones_like(pi_prob) / len(pi_prob),
                    value=reward,
                    player_id=env.current_player,
                )
                episode_trajectory.append(transition)

        game += 1
        if game % 1000 == 0:
            logging.info(f'Self-play actor {rank} played {game} games')

        # When game is over, the env stops updates the current player for timestep `t`.
        # So the current player for `t` is the same current player at `t-1` timestep who just won/loss the game.
        if reward != 0.0:
            process_episode_trajectory(env.current_player, reward, episode_trajectory)

        data_queue.put(episode_trajectory)

        del episode_trajectory[:]
    logging.info(f'Stop self-play actor {rank}')


def run_training(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    device: torch.device,
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
    train_steps: int = None,
):
    """Run the main training loop for N iterations, each iteration contains M updates.

    Args:
        network: the neural network we want to optimize.
        optimizer: neural network optimizer.
        lr_scheduler: learning rate annealing scheduler.
        device: torch runtime device.
        replay: a simple uniform experience replay.
        data_queue: a multiprocessing.SimpleQueue instance, only used to signal data collector to stop.
        min_replay_size: minimum replay size before start training.
        batch_size: sample batch size during training.
        num_train_steps: total number of traning steps to run.
        checkpoint_frequency: the frequency to create new checkpoint.
        checkpoint_dir: create new checkpoint save directory.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint files.
        csv_file: a csv file contains the training statistics.
        stop_event: a multiprocessing.Event signaling other parties to stop running pipeline.
        delay: wait time (in seconds) before start training on next batch samples, default 0.
        train_steps: already trained steps, used when resume training, default none.

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
        if replay.size < min_replay_size:
            continue

        # Signaling other parties to stop running pipeline.
        if train_steps >= num_train_steps:
            break

        transitions = replay.sample(batch_size)
        optimizer.zero_grad()
        loss = calc_loss(network, device, transitions, True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_steps += 1

        if train_steps > 1 and train_steps % checkpoint_frequency == 0:
            state_to_save = get_state_to_save()
            ckpt_file = ckpt_dir / f'train_steps_{train_steps}'
            create_checkpoint(state_to_save, ckpt_file)

            checkpoint_files.append(ckpt_file)

            log_output = [
                ('timestamp', get_time_stamp(), '%1s'),
                ('train_steps', train_steps, '%3d'),
                ('checkpoint', ckpt_file, '%3s'),
                ('loss', loss.detach().item(), '%.4f'),
                ('learning_rate', lr_scheduler.get_last_lr()[0], '%.2f'),
            ]
            write_to_csv(writer, log_output)

        # Wait for sometime before start training on next batch.
        if delay is not None and delay > 0 and train_steps > 1:
            time.sleep(delay)

    state_to_save = get_state_to_save()
    ckpt_file = ckpt_dir / f'train_steps_{train_steps}_final'
    create_checkpoint(state_to_save, ckpt_file)

    stop_event.set()
    time.sleep(60)
    data_queue.put('STOP')


def run_evaluation(
    best_network: torch.nn.Module,
    new_checkpoint_network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    num_games: int,
    c_puct: float,
    temperature: float,
    num_simulations: int,
    checkpoint_files: List,
    load_checkpoint_file: str,
    csv_file: str,
    stop_event: multiprocessing.Event,
    initial_elo: int = -2000,
    win_ratio: float = 0.55,
) -> None:
    """Run the evaluation loop to select best player from new checkpoint,
    the 'new' best player is then used by the 'self-play' processes to generate traning samples.
    Only stop the loop if `stop_event` is set to True.

    Args:
        best_network: the current best player.
        new_checkpoint_network: new checkpoint network we want to evaluate.
        device: torch runtime device.
        env: a BoardGameEnv type environment.
        num_games: number of games to play during each evaluation process.
        c_puct: a constant controls the level of exploration during MCTS search.
        temperature: the temperature exploration rate after MCTS search
            to generate play policy.
        num_simulations: number of simulations for each MCTS search.
        checkpoint_files: a shared list contains the full path for the most recent new checkpoint.
        load_checkpoint_file: resume training by load from checkpoint file.
        csv_file: a csv file contains the statistics for the best checkpoint.
        stop_event: a multiprocessing.Event signaling to stop running pipeline.
        initial_elo: initial elo rating, default -2000,
        win_ratio: the win ratio for new checkpoint to become new best player, default 0.55.

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

    disable_auto_grad(best_network)
    disable_auto_grad(new_checkpoint_network)

    # Load states from checkpoint to resume training.
    if load_checkpoint_file is not None and os.path.isfile(load_checkpoint_file):
        loaded_state = load_checkpoint(load_checkpoint_file, 'cpu')
        best_network.load_state_dict(loaded_state['network'])
        logging.info(f'Loaded best network from checkpoint {load_checkpoint_file}')

    best_network = best_network.to(device=device)
    new_checkpoint_network = new_checkpoint_network.to(device=device)

    # Black is the new checkpoint, white is current best player.
    black_player = create_mcts_player(new_checkpoint_network, device)
    white_player = create_mcts_player(best_network, device)

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
        new_checkpoint_network.load_state_dict(loaded_state['network'])
        train_steps = loaded_state['train_steps']

        new_checkpoint_network.eval()
        best_network.eval()

        won_games, loss_games, draw_games = 0, 0, 0
        is_new_best_player = False
        should_delete_ckpt_file = False

        for _ in range(num_games):
            env.reset()
            root_node = None
            done = False

            while not done:
                if env.current_player == env.black_player_id:
                    action, _, root_node = black_player(env, root_node, c_puct, temperature, num_simulations, False, True)
                else:
                    action, _, root_node = white_player(env, root_node, c_puct, temperature, num_simulations, False, True)
                _, _, done, _ = env.step(action)

            if env.winner is None:
                draw_games += 1
            elif env.winner == env.black_player_id:
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

        # We treat one evaluation as a 'single' game, otherwise the elo ratings will explode.
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
            ('checkpoint', ckpt_file, '%3s'),
            ('elo_rating', black_elo, '%1d'),
            ('won_games', won_games, '%1d'),
            ('loss_games', loss_games, '%1d'),
            ('draw_games', draw_games, '%1d'),
        ]
        write_to_csv(writer, log_output)


def run_data_collector(
    data_queue: multiprocessing.SimpleQueue,
    replay: UniformReplay,
    save_frequency: int,
    save_dir: str,
) -> None:
    """Collect samples from self-play,
    this runs on the same process as the training loop,
    but with a separate thread.

    Args:
        data_queue: a multiprocessing.SimpleQueue to receive samples from self-play processes.
        replay: a simple uniform random experience replay.
        save_frequency: the frequency to save replay state.
        save_dir: where to save replay state.

    """
    logging.info('Start data collector thread')

    save_samples_dir = Path(save_dir)
    if save_dir is not None and save_dir != '' and not save_samples_dir.exists():
        save_samples_dir.mkdir(parents=True, exist_ok=True)

    should_save = save_samples_dir.exists() and save_frequency > 0

    while True:
        try:
            item = data_queue.get()
            if item == 'STOP':
                break
            for sample in item:
                replay.add(sample)
                # Save replay samples preriodically to avoid restart from zero.
                if should_save and replay.num_added > 1 and replay.num_added % save_frequency == 0:
                    save_file = save_samples_dir / f'replay_{replay.size}_{get_time_stamp(True)}'
                    save_to_file(replay.get_state(), save_file)
                    logging.info(f"Replay samples saved to '{save_file}'")
        except queue.Empty:
            pass
        except EOFError:
            pass


def create_mcts_player(
    network: torch.nn.Module,
    device: torch.device,
):
    """Give a network and device, returns a 'act' function to act on the specific environment timestep."""

    @torch.no_grad()
    def evaluate_func(state_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """Give a game state tensor, returns the action probabilities
        and estimated winning probability from current player's perspective."""
        state = torch.from_numpy(state_tensor[None, ...]).to(device=device, dtype=torch.float32, non_blocking=True)
        output = network(state)
        pi_prob = F.softmax(output.pi_logits, dim=-1).cpu().numpy()
        value = torch.detach(output.value).cpu().numpy()

        # Remove batch dimensions
        pi_prob = np.squeeze(pi_prob, axis=0)
        value = np.squeeze(value, axis=0)

        # Convert value into float.
        value = value.item()

        return (pi_prob, value)

    def act(
        env: BoardGameEnv,
        root_node: Node,
        c_puct: float,
        temp: float,
        num_simulations: int,
        root_noise: bool = False,
        deterministic: bool = False,
    ):
        return uct_search(env, evaluate_func, root_node, c_puct, temp, num_simulations, root_noise, deterministic)

    return act


def process_episode_trajectory(final_player_id: int, final_reward: float, episode_trajectory: List[Transition]) -> None:
    """Update the final reward for the transitions, note this operation is in-place."""
    for i in range(len(episode_trajectory)):
        transition = episode_trajectory[i]
        if transition.player_id == final_player_id:
            value = final_reward
        else:
            value = -final_reward
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
        # Argument data by apply random roation and reflection.
        state, pi_prob = random_rotation_and_reflection(state, pi_prob)

    network_out = network(state)

    # value mse loss
    value_loss = F.mse_loss(network_out.value, value.unsqueeze(1), reduction='mean')

    # policy cross-entryopy loss
    policy_loss = F.cross_entropy(network_out.pi_logits, pi_prob, reduction='mean')

    return policy_loss + value_loss


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
