# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Implements the core functions of training the AlphaZero agent."""
import os
from typing import Any, Text, Callable, Mapping, Iterable, Tuple
import time
from pathlib import Path
from collections import OrderedDict, deque
import queue
import multiprocessing as mp
import threading
import pickle
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

import numpy as np
from copy import copy, deepcopy

# from alpha_zero.core.mcts_v1 import Node, parallel_uct_search, uct_search

from alpha_zero.core.mcts_v2 import Node, parallel_uct_search, uct_search

from alpha_zero.envs.base import BoardGameEnv
from alpha_zero.core.eval_dataset import build_eval_dataset
from alpha_zero.core.rating import EloRating
from alpha_zero.core.replay import UniformReplay, Transition
from alpha_zero.utils.csv_writer import CsvWriter
from alpha_zero.utils.transformation import apply_random_transformation
from alpha_zero.utils.util import Timer, create_logger, get_time_stamp


# =================================================================
# Helper functions
# =================================================================


def disable_auto_grad(network: torch.nn.Module) -> None:
    for p in network.parameters():
        p.requires_grad = False


def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def maybe_create_dir(dir) -> None:
    if dir is not None and dir != '' and not os.path.exists(dir):
        p = Path(dir)
        p.mkdir(parents=True, exist_ok=False)


def save_to_file(obj: Any, file_name: str) -> None:
    pickle.dump(obj, open(file_name, 'wb'))


def load_from_file(file_name: str) -> Any:
    return pickle.load(open(file_name, 'rb'))


def round_it(v, places=4) -> float:
    return round(v, places)


def _encode_bytes(in_str) -> Any:
    return str(in_str).encode('utf-8')


def _decode_bytes(b) -> str:
    return b.decode('utf-8')


def create_mcts_player(
    network: torch.nn.Module,
    device: torch.device,
    num_simulations: int,
    num_parallel: int,
    root_noise: bool = False,
    deterministic: bool = False,
) -> Callable[[BoardGameEnv, Node, float, float, bool], Tuple[int, np.ndarray, float, float, Node]]:
    @torch.no_grad()
    def eval_position(
        state: np.ndarray,
        batched: bool = False,
    ) -> Tuple[Iterable[np.ndarray], Iterable[float]]:
        """Give a game state tensor, returns the action probabilities
        and estimated state value from current player's perspective."""

        if not batched:
            state = state[None, ...]

        state = torch.from_numpy(state).to(dtype=torch.float32, device=device, non_blocking=True)
        pi_logits, v = network(state)

        pi_logits = torch.detach(pi_logits)
        v = torch.detach(v)

        pi = torch.softmax(pi_logits, dim=-1).cpu().numpy()
        v = v.cpu().numpy()

        B, *_ = state.shape

        v = np.squeeze(v, axis=1)
        v = v.tolist()  # To list

        # Unpack the batched array into a list of NumPy arrays
        pi = [pi[i] for i in range(B)]

        if not batched:
            pi = pi[0]
            v = v[0]

        return pi, v

    def act(
        env: BoardGameEnv,
        root_node: Node,
        c_puct_base: float,
        c_puct_init: float,
        warm_up: bool = False,
    ) -> Tuple[int, np.ndarray, float, float, Node]:
        if num_parallel > 1:
            return parallel_uct_search(
                env=env,
                eval_func=eval_position,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=root_noise,
                warm_up=warm_up,
                deterministic=deterministic,
            )
        else:
            return uct_search(
                env=env,
                eval_func=eval_position,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                num_simulations=num_simulations,
                root_noise=root_noise,
                warm_up=warm_up,
                deterministic=deterministic,
            )

    return act


# =================================================================
# Selfplay
# =================================================================


def run_selfplay_actor_loop(
    seed: int,
    rank: int,
    network: torch.nn.Module,
    device: torch.device,
    data_queue: mp.Queue,
    env: BoardGameEnv,
    num_simulations: int,
    num_parallel: int,
    c_puct_base: float,
    c_puct_init: float,
    warm_up_steps: int,
    check_resign_after_steps: int,
    disable_resign_ratio: float,
    save_sgf_dir: str,
    save_sgf_interval: int,
    logs_dir: str,
    load_ckpt: str,
    log_level: str,
    var_ckpt: mp.Value,
    var_resign_threshold: mp.Value,
    ckpt_event: mp.Event,
    stop_event: mp.Event,
) -> None:
    """Use the latest neural network to play against itself, and record the transitions for training."""
    assert num_simulations > 1

    set_seed(int(seed + rank))
    logger = create_logger(log_level)
    writer = CsvWriter(os.path.join(logs_dir, f'actor{rank}.csv'))
    timer = Timer()

    played_games = training_steps = 0
    last_ckpt = None

    should_save_sgf = False
    if save_sgf_dir is not None and os.path.isdir(save_sgf_dir) and os.path.exists(save_sgf_dir):
        should_save_sgf = True

    disable_auto_grad(network)
    network = network.to(device=device)

    if load_ckpt is not None and os.path.exists(load_ckpt):
        loaded_state = torch.load(load_ckpt, map_location=device)
        network.load_state_dict(loaded_state['network'])
        training_steps = loaded_state['training_steps']
        logger.debug(f'Actor{rank} loaded state from checkpoint "{load_ckpt}"')

    network.eval()

    # resign_threshold <= -1 means no resign
    resign_threshold = var_resign_threshold.value if env.has_resign_move else -1
    mcts_player = create_mcts_player(
        network=network,
        device=device,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        root_noise=True,
        deterministic=False,
    )

    while not stop_event.is_set():
        # Wait for learner to finish creating new checkpoint
        if ckpt_event.is_set():
            continue

        new_ckpt = _decode_bytes(var_ckpt.value)
        if new_ckpt != '' and new_ckpt != last_ckpt and os.path.exists(new_ckpt):
            loaded_state = torch.load(new_ckpt, map_location=torch.device(device))
            network.load_state_dict(loaded_state['network'])
            training_steps = loaded_state['training_steps']
            network.eval()
            last_ckpt = new_ckpt
            logger.debug(f'Actor{rank} switched to checkpoint "{new_ckpt}"')

        if env.has_resign_move:
            resign_threshold = var_resign_threshold.value

        resign_disabled = True
        if env.has_resign_move and resign_threshold > -1.0 and np.random.rand() > disable_resign_ratio:
            resign_disabled = False

        with timer:
            game_seq, stats = play_and_record_one_game(
                env=env,
                mcts_player=mcts_player,
                resign_disabled=resign_disabled,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                warm_up_steps=warm_up_steps,
                check_resign_after_steps=check_resign_after_steps,
                resign_threshold=resign_threshold,
                logger=logger,
            )

        played_games += 1

        # The second check is necessary, as the events could be set while the actor is in the middle of playing a game.
        if stop_event.is_set():
            break
        if ckpt_event.is_set():
            continue

        # Logging
        stats['time_per_game'] = round_it(timer.mean_time())
        stats['training_steps'] = training_steps
        log_stats = {'datetime': get_time_stamp(), **stats}
        writer.write(OrderedDict((n, v) for n, v in log_stats.items()))

        # For monitoring
        if should_save_sgf and played_games % save_sgf_interval == 0:
            sgf_content = env.to_sgf()
            sgf_file = os.path.join(save_sgf_dir, f'actor{rank}_{get_time_stamp(True)}.sgf')
            with open(sgf_file, 'w') as f:
                f.write(sgf_content)
                f.close()

        data_queue.put((game_seq, stats))

    logger.debug(f'Actor{rank} received stop signal.')
    writer.close()


def play_and_record_one_game(
    env: BoardGameEnv,
    mcts_player: Any,
    resign_disabled: bool,
    c_puct_base: float,
    c_puct_init: float,
    warm_up_steps: int,
    check_resign_after_steps: int,
    resign_threshold: float,
    logger: Any,
) -> Tuple[Iterable[Transition], Mapping[Text, Any]]:
    obs = env.reset()
    done = False

    episode_states = []
    episode_search_pis = []
    episode_values = []
    to_plays = []

    root_node = None
    marked_resign_player = None
    is_marked_for_resign = False
    is_could_won = False
    num_passes = 0

    while not done:  # For each step
        (move, search_pi, root_Q, best_child_Q, root_node) = mcts_player(
            env=env,
            root_node=root_node,
            c_puct_base=c_puct_base,
            c_puct_init=c_puct_init,
            warm_up=False if env.steps > warm_up_steps else True,
        )

        episode_states.append(obs)
        episode_search_pis.append(search_pi)
        episode_values.append(0.0)
        to_plays.append(env.to_play)

        if (
            env.has_resign_move
            and env.steps > check_resign_after_steps
            and root_Q < resign_threshold
            and best_child_Q < resign_threshold
        ):
            # Mark resigned player so we can compute false positive
            if marked_resign_player is None:
                marked_resign_player = copy(env.to_play)

            logger.debug(f'Search root value: {root_Q}, best child value: {best_child_Q}')
            # Only take the resign move for game where resignation is enabled
            if not resign_disabled:
                move = env.resign_move

        obs, reward, done, _ = env.step(move)

        if env.has_pass_move and move == env.pass_move:
            num_passes += 1

    # Do nothing if the game finished with draw
    if reward != 0.0:
        for i, play_id in enumerate(to_plays):
            if play_id == env.last_player:
                episode_values[i] = reward
            else:
                episode_values[i] = -reward

    game_seq = [
        Transition(state=x, pi_prob=pi, value=v) for x, pi, v in zip(episode_states, episode_search_pis, episode_values)
    ]

    # Use samples from those 10% games where resign is disabled to compute resignation false positive
    if env.has_resign_move and resign_disabled and marked_resign_player is not None:
        is_marked_for_resign = True
        # Despite marked for resign (not taking it as the move is disabled), but the game ended up won by the marked resign player
        if env.winner == marked_resign_player:
            is_could_won = True

    stats = {
        'game_length': len(game_seq),
        'game_result': env.get_result_string(),
    }

    if env.has_pass_move:
        stats['num_passes'] = num_passes

    if env.has_resign_move:
        stats['is_resign_disabled'] = resign_disabled
        stats['is_marked_for_resign'] = is_marked_for_resign
        stats['is_could_won'] = is_could_won
        stats['marked_resign_player'] = env.get_player_name_by_id(marked_resign_player)
        stats['resign_threshold'] = resign_threshold

    return game_seq, stats


# =================================================================
# Learner
# =================================================================


def run_learner_loop(  # noqa: C901
    seed: int,
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.MultiStepLR,
    device: torch.device,
    replay: UniformReplay,
    logger: Any,
    argument_data: bool,
    batch_size: int,
    disable_resign_ratio: float,
    init_resign_threshold: float,
    target_fp_rate: float,
    reset_fp_interval: int,
    no_resign_games: int,
    min_games: int,
    games_per_ckpt: int,
    num_actors: int,
    ckpt_interval: int,
    log_interval: int,
    save_replay_interval: int,
    max_training_steps: int,
    ckpt_dir: str,
    logs_dir: str,
    load_ckpt: str,
    load_replay: str,
    data_queue: mp.SimpleQueue,
    var_ckpt: mp.Value,
    var_resign_threshold: mp.Value,
    ckpt_event: mp.Event,
    stop_event: mp.Event,
    lock=threading.Lock(),
) -> None:
    """Update the neural network, dynamically adjust resignation threshold if required."""
    assert min_games >= 100
    assert init_resign_threshold < -0.5
    assert target_fp_rate <= 0.05
    assert games_per_ckpt >= 100
    assert ckpt_interval >= 100
    assert log_interval >= 100
    assert save_replay_interval >= 0
    assert max_training_steps > 0
    assert ckpt_dir is not None and os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir)

    set_seed(int(seed))
    writer = CsvWriter(os.path.join(logs_dir, 'training.csv'), buffer_size=1)
    game_time_que = deque(maxlen=2000)
    game_length_que = deque(maxlen=2000)
    training_steps = last_ckpt_games = last_ckpt_samples = 0
    resign_count = last_resign_count = could_won_count = 0
    training_sample_ratio = (batch_size * ckpt_interval) / replay.capacity

    logger.info(f'Training sample ratio (over one checkpoint) is {training_sample_ratio:.2f}')
    if training_sample_ratio > 0.25:
        logger.warning(f'Training sample ratio {training_sample_ratio:.2f} might be too high')

    if save_replay_interval > 0:
        logger.warning(f'Saving replay state has been enabled, ensure you have at least 100GB of free space at "{ckpt_dir}"')

    if init_resign_threshold <= -1:
        with lock:
            var_resign_threshold.value = -1
        logger.info('Resignation is permanently disabled for self-play games')
    elif no_resign_games > 0:
        with lock:
            var_resign_threshold.value = -1
        logger.info(f'Resignation is disabled for first {no_resign_games} self-play games')
    else:
        with lock:
            var_resign_threshold.value = init_resign_threshold
        logger.info(f'Resignation threshold is set to {init_resign_threshold}')

    network = network.to(device=device)

    with lock:
        var_ckpt.value = _encode_bytes('')
        ckpt_event.clear()

    if load_replay is not None and os.path.exists(load_replay):
        replay_state = load_from_file(load_replay)
        replay.set_state(replay_state)
        logger.info(f'Learner loaded replay state from "{load_replay}"')

    if load_ckpt is not None and os.path.exists(load_ckpt):
        loaded_state = torch.load(load_ckpt, map_location=device)
        network.load_state_dict(loaded_state['network'])
        optimizer.load_state_dict(loaded_state['optimizer'])
        lr_scheduler.load_state_dict(loaded_state['lr_scheduler'])
        training_steps = loaded_state['training_steps']
        logger.info(f'Learner loaded state from checkpoint "{load_ckpt}", last training step {training_steps}')

    network.train()

    while True:
        try:
            item = data_queue.get()
            if not isinstance(item, Tuple):
                continue

            game_seq, stats = item

            # Additional check to ensure that we collect equal amount of games from each checkpoint
            if stats['training_steps'] != training_steps:
                continue

            last_ckpt_games += 1
            last_ckpt_samples += stats['game_length']
            replay.add_game(game_seq)
            game_time_que.append(stats['time_per_game'])
            game_length_que.append(stats['game_length'])

            # Logging
            if replay.num_games_added % 10000 == 0:
                avg_time_per_game = round_it(np.mean(game_time_que) / num_actors)
                avg_game_length = np.mean(game_length_que)
                logger.info(
                    f'Collected total of {replay.num_games_added} self-play games, '
                    f'{replay.num_samples_added} samples. '
                    f'Average game length is {avg_game_length}. '
                    f'Average time per game (over {num_actors} actors) is {avg_time_per_game}'
                )

            # Save replay buffer state periodically to avoid starting from zero.
            if save_replay_interval > 0 and replay.num_games_added % save_replay_interval == 0:
                replay_file = os.path.join(ckpt_dir, 'replay_state.ckpt')
                save_to_file(replay.get_state(), replay_file)
                logger.debug(f'Replay buffer state saved at "{replay_file}"')

            # Adjust resignation threshold
            if init_resign_threshold > -1.0 and replay.num_games_added >= no_resign_games:
                if (
                    'is_resign_disabled' in stats
                    and 'is_marked_for_resign' in stats
                    and stats['is_resign_disabled']
                    and stats['is_marked_for_resign']
                ):
                    resign_count += 1
                    if 'is_could_won' in stats and stats['is_could_won']:
                        could_won_count += 1

                # Doing a hard reset without checking current false positive rate,
                # so statistics from long time along does not affect current play
                if replay.num_games_added == no_resign_games or replay.num_games_added % reset_fp_interval == 0:
                    resign_count = last_resign_count = could_won_count = 0
                    logger.info(f'Reset resignation threshold to {init_resign_threshold}')
                    with lock:
                        var_resign_threshold.value = init_resign_threshold
                # For those resignation have been disabled games, the agent may not chose to resign depending on the search results
                elif (
                    resign_count > last_resign_count
                    and resign_count % int(games_per_ckpt * 0.5 * disable_resign_ratio * 0.5) == 0
                ):
                    last_resign_count = resign_count

                    current_fp_rate = 0 if resign_count == 0 else round_it(could_won_count / resign_count)
                    current_threshold = var_resign_threshold.value
                    new_threshold = maybe_adjust_resign_threshold(current_threshold, current_fp_rate, target_fp_rate)
                    if new_threshold != current_threshold:
                        logger.info(
                            f'Current resignation false positive is {current_fp_rate}, target {target_fp_rate}, '
                            f'changing resignation threshold from {current_threshold} to {new_threshold}'
                        )
                        with lock:
                            var_resign_threshold.value = new_threshold

            # Perform network parameters update
            if replay.num_games_added == min_games or (
                replay.num_games_added >= min_games and last_ckpt_games >= games_per_ckpt
            ):
                logger.debug(
                    f'Collected {last_ckpt_games} games, {last_ckpt_samples} samples from last checkpoint (training steps {training_steps})'
                )

                with lock:
                    ckpt_event.set()

                network.train()

                target_t = training_steps + ckpt_interval

                while training_steps < target_t:
                    transitions = replay.sample(batch_size)
                    if transitions is None:
                        continue

                    optimizer.zero_grad()
                    pi_loss, v_loss = compute_losses(network, device, transitions, argument_data)
                    loss = pi_loss + v_loss
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    training_steps += 1

                    # Logging statistics
                    if training_steps % log_interval == 0 or training_steps % ckpt_interval == 0:
                        stats = {
                            'datetime': get_time_stamp(),
                            'training_steps': training_steps,
                            'policy_loss': pi_loss.detach().item(),
                            'value_loss': v_loss.detach().item(),
                            'learning_rate': lr_scheduler.get_last_lr()[0],
                            'total_games': replay.num_games_added,
                            'total_samples': replay.num_samples_added,
                        }
                        writer.write(OrderedDict((n, v) for n, v in stats.items()))

                # Create checkpoint
                ckpt_file = os.path.join(ckpt_dir, f'training_steps_{training_steps}.ckpt')
                torch.save(
                    {
                        'network': network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'training_steps': training_steps,
                    },
                    ckpt_file,
                )

                with lock:
                    var_ckpt.value = _encode_bytes(ckpt_file)
                    ckpt_event.clear()

                logger.debug(f'New checkpoint for training steps {training_steps} is created at "{ckpt_file}"')

                last_ckpt_games = 0
                last_ckpt_samples = 0

            if training_steps >= max_training_steps:
                break

        except (queue.Empty, EOFError) as error:  # noqa: F841
            pass

    writer.close()
    time.sleep(30)
    stop_event.set()
    time.sleep(60)

    try:
        data_queue.close()
    except Exception:
        pass


def compute_losses(network, device, transitions, argumentation=False) -> Tuple[torch.Tensor, torch.Tensor]:
    # [B, C, N, N]
    state = torch.from_numpy(transitions.state).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, num_actions]
    target_pi = torch.from_numpy(transitions.pi_prob).to(device=device, dtype=torch.float32, non_blocking=True)
    # [B, ]
    target_v = torch.from_numpy(transitions.value).to(device=device, dtype=torch.float32, non_blocking=True)

    if argumentation:
        state, target_pi, target_v = apply_random_transformation(state, target_pi, target_v)

    pred_pi_logits, pred_v = network(state)

    # Policy cross-entropy loss
    policy_loss = F.cross_entropy(pred_pi_logits, target_pi, reduction='mean')

    # State value MSE loss
    value_loss = F.mse_loss(pred_v.squeeze(), target_v, reduction='mean')

    return policy_loss, value_loss


def maybe_adjust_resign_threshold(current_v, current_rate, target_rate, min_v=-0.9999, smoothing_factor=0.5) -> float:
    """
    A smaller smoothing factor will result in more rapid adjustments to the threshold,
    while a larger smoothing factor will result in smoother adjustments that are
    less sensitive to short-term fluctuations in the false positive rate.
    """

    rate_delta = current_rate - target_rate

    if rate_delta <= 0:
        return current_v

    new_v = current_v + current_v * rate_delta
    smoothed_v = smoothing_factor * new_v + (1 - smoothing_factor) * current_v
    return round_it(max(min_v, smoothed_v))


# =================================================================
# Evaluator
# =================================================================


@torch.no_grad()
def run_evaluator_loop(
    seed: int,
    network: torch.nn.Module,
    device: torch.device,
    env: BoardGameEnv,
    eval_games_dir: str,
    num_simulations: int,
    num_parallel: int,
    c_puct_base: float,
    c_puct_init: float,
    default_rating: float,
    logs_dir: str,
    save_sgf_dir: str,
    load_ckpt: str,
    log_level: str,
    var_ckpt: mp.Value,
    stop_event: mp.Event,
) -> None:
    """Evaluate the latest neural network by paying against network from last checkpoint.
    Also compute the prediction accuracy on human games if applicable.
    """
    assert num_simulations > 1

    set_seed(int(seed))
    logger = create_logger(log_level)  # noqa: F841

    network = network.to(device=device)
    disable_auto_grad(network)

    writer = CsvWriter(os.path.join(logs_dir, 'evaluation.csv'), buffer_size=1)

    last_ckpt = None
    last_ckpt_step = 0

    if load_ckpt is not None and os.path.exists(load_ckpt):
        loaded_state = torch.load(load_ckpt, map_location=device)
        network.load_state_dict(loaded_state['network'])
        last_ckpt_step = loaded_state['training_steps']
        last_ckpt = load_ckpt
        logger.info(f'Evaluator loaded state from checkpoint "{load_ckpt}"')

    prev_ckpt_network = deepcopy(network).to(device=device)
    disable_auto_grad(prev_ckpt_network)
    network.eval()
    prev_ckpt_network.eval()

    dataloader = None
    if eval_games_dir is not None and eval_games_dir != '' and os.path.exists(eval_games_dir):
        eval_dataset = build_eval_dataset(eval_games_dir, env.num_stack, logger)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=1024,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

    # Create MCTS players for both players, note black always uses the latest checkpoint,
    # and white always uses the previous checkpoint
    black_elo = EloRating(rating=default_rating)
    white_elo = EloRating(rating=default_rating)

    black_player = create_mcts_player(
        network=network,
        device=device,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        root_noise=False,
        deterministic=True,
    )

    white_player = create_mcts_player(
        network=prev_ckpt_network,
        device=device,
        num_simulations=num_simulations,
        num_parallel=num_parallel,
        root_noise=False,
        deterministic=True,
    )

    while not stop_event.is_set():
        ckpt_file = _decode_bytes(var_ckpt.value)
        if ckpt_file == '' or ckpt_file == last_ckpt or not os.path.exists(ckpt_file):
            time.sleep(30)
            continue

        # Load states from checkpoint file
        loaded_state = torch.load(ckpt_file, map_location=torch.device(device))
        training_steps = loaded_state['training_steps']
        network.load_state_dict(loaded_state['network'])
        network.eval()
        last_ckpt = ckpt_file

        selfplay_game_stats = eval_against_prev_ckpt(
            env,
            black_player,
            white_player,
            black_elo,
            white_elo,
            c_puct_base,
            c_puct_init,
        )

        pro_game_stats = eval_on_pro_games(network, device, dataloader)

        stats = {
            'datetime': get_time_stamp(),
            'training_steps': training_steps,
            **selfplay_game_stats,
            **pro_game_stats,
        }

        writer.write(OrderedDict((n, v) for n, v in stats.items()))

        # Save the game in sgf format
        if save_sgf_dir is not None and os.path.isdir(save_sgf_dir) and os.path.exists(save_sgf_dir):
            sgf_content = env.to_sgf()
            sgf_file = os.path.join(
                save_sgf_dir,
                f'eval_training_steps_{training_steps}_vs_{last_ckpt_step}.sgf',
            )
            with open(sgf_file, 'w') as f:
                f.write(sgf_content)
                f.close()

        # Switching to new model
        prev_ckpt_network.load_state_dict(loaded_state['network'])
        prev_ckpt_network.eval()
        # We assume the new model will be the same level as previous model, since they are pretty close
        white_elo = deepcopy(black_elo)
        last_ckpt_step = training_steps

    writer.close()


@torch.no_grad()
def eval_against_prev_ckpt(
    env,
    black_player,
    white_player,
    black_elo,
    white_elo,
    c_puct_base,
    c_puct_init,
) -> Mapping[Text, Any]:
    _ = env.reset()
    mcts_player = None
    done = False
    num_passes = 0

    while not done:
        if env.to_play == env.black_player:
            mcts_player = black_player
        else:
            mcts_player = white_player
        move, *_ = mcts_player(
            env=env,
            root_node=None,
            c_puct_base=c_puct_base,
            c_puct_init=c_puct_init,
            warm_up=False,
        )

        _, _, done, _ = env.step(move)

        if env.has_pass_move and move == env.pass_move:
            num_passes += 1

    stats = {
        'game_length': env.steps,
        'game_result': env.get_result_string(),
    }

    if env.has_pass_move:
        stats['num_passes'] = num_passes

    # Update elo rating for both players
    if env.winner is not None:
        if env.winner == env.black_player:
            winner, loser = black_elo, white_elo
        elif env.winner == env.white_player:
            winner, loser = white_elo, black_elo

        winner.update_rating(loser.rating, 1)
        loser.update_rating(winner.rating, 0)

    stats['black_elo_rating'] = black_elo.rating
    stats['white_elo_rating'] = white_elo.rating
    return stats


@torch.no_grad()
def eval_on_pro_games(
    network,
    device,
    dataloader,
    k_list=(1, 3, 5),
) -> Mapping[Text, Any]:
    assert min(k_list) >= 1

    if dataloader is None or not isinstance(dataloader, DataLoader):
        return {}

    total_correct = {k: 0 for k in k_list}
    total_entropy = 0.0
    total_mse_loss = 0
    total_examples = 0
    for states, target_pi, target_v in dataloader:
        states = states.to(device=device, non_blocking=True)
        target_pi = target_pi.to(device=device, non_blocking=True)
        target_v = target_v.to(device=device, non_blocking=True)
        # Forward pass to get model predictions
        policy_logits_pred, value_pred = network(states)

        policy_pred = torch.softmax(policy_logits_pred, dim=-1)
        value_pred = value_pred.squeeze(-1)

        assert value_pred.shape == target_v.shape
        assert policy_pred.shape == target_pi.shape

        # Compute the number of correct predictions for each value of k
        # Note here the target is a one-hot vector,
        # so we only check if the indices of top k prediction contains the index of the actual human move
        batch_size = states.size(0)
        _, pred = torch.topk(policy_pred, max(k_list), dim=1)
        target_indices = torch.argmax(target_pi, dim=1)  # Get the index of the actual human move

        # for i in range(batch_size):
        #     for k in k_list:
        #         if target_indices[i] in pred[i, :k]:
        #             total_correct[k] += 1

        # This does the above counting, but much faster
        expanded_target_indices = target_indices.unsqueeze(1).expand(batch_size, max(k_list))
        matches = pred.eq(expanded_target_indices)
        for i, k in enumerate(k_list):
            total_correct[k] += matches[:, :k].any(dim=1).sum().item()

        # Compute the entropy of the predicted probability distribution
        entropy = -(policy_pred * torch.log(policy_pred)).sum(dim=1)
        total_entropy += entropy.sum().item()

        # Compute the mean absolute error for this batch
        total_mse_loss += F.mse_loss(value_pred, target_v, reduction='sum').item()

        # Update the total number of examples
        total_examples += batch_size

    # Compute the top-k accuracies and entropy for the dataset
    policy_accuracies = {k: total_correct[k] / total_examples for k in k_list}
    policy_entropy = total_entropy / total_examples

    # Compute the mean absolute error for the entire dataset
    value_mse_loss = total_mse_loss / total_examples

    stats = {
        'value_mse_error': value_mse_loss,
        'policy_entropy': policy_entropy,
    }
    for k, v in policy_accuracies.items():
        stats[f'policy_top_{k}_accuracy'] = v

    return stats
