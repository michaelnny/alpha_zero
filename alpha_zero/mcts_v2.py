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
"""Optimized MCTS class which uses numpy to speed up node's internal computation."""

from __future__ import annotations
import collections
import copy
import math
from typing import Callable, List, Tuple, Mapping, Union
import numpy as np
from alpha_zero.games.env import BoardGameEnv


class Node:
    """Node in the MCTS search tree."""

    def __init__(self, to_play: int, num_actions: int, move: int = None, parent: Node = None) -> None:
        """
        Args:
            to_play: the id of the current player.
            num_actions: number of actions, including illegal actions.
            move: the action associated with the prior probability.
            parent: the parent node, could be `None` if this is the root node.
        """
        self.to_play = to_play
        self.move = move
        self.parent = parent
        self.is_expanded = False

        self.num_actions = num_actions
        self.child_priors = np.zeros([self.num_actions], dtype=np.float32)
        self.child_total_value = np.zeros([self.num_actions], dtype=np.float32)
        self.child_number_visits = np.zeros([self.num_actions], dtype=np.float32)

        self.children: Mapping[int, Node] = {}

        # number of virtual losses on this node, only used in 'parallel_uct_search'
        self.losses_applied = 0

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        """Returns a 1D numpy.array contains prior score for all child."""

        pb_c = math.log((1 + self.number_visits + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * self.child_priors * (math.sqrt(self.number_visits) / (self.child_number_visits + 1))

    def child_Q(self):
        """Returns a 1D numpy.array contains mean action value for all child."""
        return self.child_total_value / (1 + self.child_number_visits)

    @property
    def number_visits(self):
        """The number of visits for current node is stored at parent's level."""
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        """The total value for current node is stored at parent's level."""
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    @property
    def has_parent(self) -> bool:
        return isinstance(self.parent, Node)


class DummyNode(object):
    """A place holder to make computation possible for the root node."""

    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def best_child(
    node: Node,
    current_player: int,
    opponent_player: int,
    actions_mask: np.ndarray,
    c_puct_base: float,
    c_puct_init: float,
) -> Node:
    """Returns best child node with maximum action value Q plus an upper confidence bound U,
    also create child node corresponding to the best move if not already exists.

    Args:
        node: the current node in the search tree.
        current_player: select best child from current_player's perspective.
        opponent_player: the opponent player id, used to add new child to node.
        actions_mask: a 1D bool numpy.array mask for all actions,
            where `True` represents legal move and `False` represents illegal move.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.

    Returns:
        The best child node.

    Raises:
        ValueError:
            if the node instance itself is a leaf node.
            if input argument `actions_mask` is not a valid 1D bool numpy.array.
    """
    if not node.is_expanded:
        raise ValueError('Expand leaf node first.')
    if not isinstance(actions_mask, np.ndarray) or actions_mask.dtype != np.bool8 or len(actions_mask.shape) != 1:
        raise ValueError(f'Expect `actions_mask` to be a 1D bool numpy.array, got {actions_mask}')

    # The child Q value is evaluated from opponent's perspective.
    # If the parent node id matches current_player id, then we want to switch the sign of child Q values
    # This is because we want to select the move that minimize the opponent's score, for example blocking their winning moves
    if node.to_play == current_player:
        ucb_scores = -1.0 * node.child_Q() + node.child_U(c_puct_base, c_puct_init)
    else:
        ucb_scores = node.child_Q() + node.child_U(c_puct_base, c_puct_init)

    # Exclude illegal actions, note in some cases, the max ucb_scores may be zero.
    ucb_scores = np.where(actions_mask, ucb_scores, -1000)

    # Break ties if we have multiple 'maximum' values.
    move = np.random.choice(np.where(ucb_scores == ucb_scores.max())[0])
    # Create the child node, note child node is 'owned' by opponent player
    child = maybe_add_child(node, move, opponent_player)
    return child


def maybe_add_child(parent: Node, move: int, child_to_play: int) -> Node:
    """Return child node corresponding to the given move, also create the child node if not already exists.

    Args:
        parent: the parent node to add child to.
        move: the move corresponding to the child node.
        child_to_play: the player id corresponding to child to node.
    """

    if move not in parent.children:
        parent.children[move] = Node(to_play=child_to_play, num_actions=parent.num_actions, move=move, parent=parent)
    return parent.children[move]


def expand(node: Node, prior_prob: np.ndarray) -> None:
    """Expand a node.

    Args:
        node: current leaf node in the search tree.
        prior_prob: 1D numpy.array contains prior probabilities of the state for all actions.

    Raises:
        ValueError:
            if node instance already expanded.
            if input argument `prior` is not a valid 1D float numpy.array.
    """
    if node.is_expanded:
        raise RuntimeError('Node already expanded.')
    if (
        not isinstance(prior_prob, np.ndarray)
        or len(prior_prob.shape) != 1
        or prior_prob.dtype not in (np.float32, np.float64)
    ):
        raise ValueError(f'Expect `prior_prob` to be a 1D float numpy.array, got {prior_prob}')

    node.child_priors = prior_prob
    node.is_expanded = True


def backup(node: Node, value: float, for_player: int) -> None:
    """Update statistics of the this node and all traversed parent nodes.

    Args:
        node: current leaf node in the search tree.
        value: the evaluation value.
        for_player: value evaluated from given player's perspective.

    Raises:
        ValueError:
            if input argument `value` is not float data type.
    """

    if not isinstance(value, float):
        raise ValueError(f'Expect `value` to be a float type, got {type(value)}')

    while node.parent is not None:
        node.number_visits += 1
        node.total_value += value if node.to_play == for_player else -value
        node = node.parent


def add_dirichlet_noise(node: Node, actions_mask: np.ndarray, eps: float = 0.25, alpha: float = 0.03) -> None:
    """Add dirichlet noise to a given node.

    Args:
        node: the root node we want to add noise to.
        actions_mask: a 1D bool numpy.array mask for all actions,
            where `True` represents legal move and `False` represents illegal move.
        eps: epsilon constant to weight the priors vs. dirichlet noise.
        alpha: parameter of the dirichlet noise distribution.

    Raises:
        ValueError:
            if input argument `node` is not expanded.
            if input argument `actions_mask` is not a valid 1D bool numpy.array.
            if input argument `eps` or `alpha` is not float type
                or not in the range of [0.0, 1.0].
    """

    if not isinstance(node, Node) or not node.is_expanded:
        raise ValueError(f'Expect `node` to be expanded')
    if not isinstance(actions_mask, np.ndarray) or actions_mask.dtype != np.bool8 or len(actions_mask.shape) != 1:
        raise ValueError(f'Expect `actions_mask` to be a 1D bool numpy.array, got {actions_mask}')
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(f'Expect `eps` to be a float in the range [0.0, 1.0], got {eps}')
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f'Expect `alpha` to be a float in the range [0.0, 1.0], got {alpha}')

    alphas = np.ones_like(actions_mask) * alpha
    noise = np.random.dirichlet(alphas)

    # Set noise to zero for illegal actions
    noise = np.where(actions_mask, noise, 0)

    noised_priors = node.child_priors * (1 - eps) + noise * eps

    node.child_priors = noised_priors


def generate_play_policy(node: Node, actions_mask: np.ndarray, temperature: float) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentialted visit count.

    Args:
        node: the root node of the search tree.
        actions_mask: a 1D bool numpy.array mask for all actions,
            where `True` represents legal move and `False` represents illegal move.
        temperature: a parameter controls the level of exploration.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if node instance not expanded.
            if input argument `actions_mask` is not a valid 1D bool numpy.array.
            if input argument `temperature` is not float type or not in range [0.0, 1.0].
    """
    if not node.is_expanded:
        raise ValueError('Node not expanded.')
    if not isinstance(actions_mask, np.ndarray) or actions_mask.dtype != np.bool8 or len(actions_mask.shape) != 1:
        raise ValueError(f'Expect `actions_mask` to be a 1D bool numpy.array, got {actions_mask}')
    if not isinstance(temperature, float) or not 0 <= temperature <= 1.0:
        raise ValueError(f'Expect `temperature` to be float type in the range [0.0, 1.0], got {temperature}')

    # Mask out illegal actions
    child_visits = np.where(actions_mask, node.child_number_visits, 0).astype(np.int64)

    if temperature > 0.0:
        # To avoid overflow when doing power operation over large numbers,
        # we limit the exponent in the range of [1.0, 5.0],
        exp = max(1.0, min(5.0, 1.0 / temperature))

        pi_logits = np.power(child_visits, exp)
        pi_probs = pi_logits / np.sum(pi_logits)
    else:
        pi_probs = child_visits / np.sum(child_visits)

    return pi_probs


def uct_search(
    env: BoardGameEnv,
    eval_func: Callable[[np.ndarray, bool], Tuple[np.ndarray, Union[np.ndarray, float]]],
    root_node: Node,
    c_puct_base: float,
    c_puct_init: float,
    temperature: float,
    num_simulations: int = 800,
    root_noise: bool = False,
    deterministic: bool = False,
) -> Tuple[int, np.ndarray, Node]:
    """Single-threaded Upper Confidence Bound (UCB) for Trees (UCT) search without any rollout.

    It follows the following general UCT search algorithm, except here we don't do rollout.
    ```
    function UCTSEARCH(r,m)
      i←1
      for i ≤ m do
          n ← select(r)
          n ← expand(n)
          ∆ ← rollout(n)
          backup(n,∆)
      end for
      return end function
    ```

    Args:
        env: a gym like custom BoardGameEnv environment.
        eval_func: a evaluation function when called returns the
            action probabilities and winning probability from
            current player's perspective.
        root_node: root node of the search tree, if none, new root node will be created.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        temperature: a parameter controls the level of exploration
            when generate policy action probabilities after MCTS search.
        num_simulations: number of simulations to run, default 800.
        root_noise: whether add dirichlet noise to root node to encourage exploration,
            default off.
        deterministic: after the MCTS search, choose the child node with most visits number to play in the game,
            instead of sample through a probability distribution, default off.

    Returns:
        tuple contains:
            a integer indicate the sampled action to play in the environment.
            a 1D numpy.array search policy action probabilities from the MCTS search result.
            a Node instance represent subtree of this MCTS search, which can be used as next root node for MCTS search.

    Raises:
        ValueError:
            if input argument `env` is not valid BoardGameEnv instance.
            if input argument `num_simulations` is not a positive integer.
        RuntimeError:
            if the game is over.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f'Expect `env` to be a valid BoardGameEnv instance, got {env}')
    if not 1 <= num_simulations:
        raise ValueError(f'Expect `num_simulations` to a positive integer, got {num_simulations}')
    if env.is_game_over:
        raise RuntimeError('Game is over.')

    if root_node is None:
        # Create new root node if not reusing sub-tree
        root_node = Node(to_play=env.current_player, num_actions=env.num_actions, parent=DummyNode())
        prior_prob, value = eval_func(env.observation(), False)
        expand(root_node, prior_prob)
        backup(root_node, value, env.current_player)

    assert root_node.to_play == env.current_player

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise:
        add_dirichlet_noise(root_node, env.actions_mask)

    # for simulation in range(num_simulations):
    while root_node.number_visits < num_simulations:
        # Make sure do not touch the actual environment.
        sim_env = copy.deepcopy(env)
        obs = sim_env.observation()
        done = sim_env.is_game_over

        # Phase 1 - Select
        # Select best child node until one of the following is true:
        # - reach a leaf node.
        # - game is over.
        node = root_node
        while node.is_expanded:
            node = best_child(
                node, sim_env.current_player, sim_env.opponent_player, sim_env.actions_mask, c_puct_base, c_puct_init
            )

            # Make move on the simulation environment.
            obs, reward, done, _ = sim_env.step(node.move)
            if done:
                break

        # Special case - If game is over, using the actual reward from the game to update statistics.
        if done:
            # Note when the game is over, the 'current_player' from the env 
            # is the same 'current_player' who made the move at timestep 'T-1' and won/loss the game
            # and the reward is also computed for (timestep 'T-1') 'current_player' perspective
            assert node.parent.to_play == sim_env.current_player
            backup(node, reward, sim_env.current_player)
            continue

        assert node.to_play == sim_env.current_player

        # Phase 2 - Expand and evaluation
        prior_prob, value = eval_func(obs, False)
        # Chidden nodes are evaluated from opponent player's perspective.
        expand(node, prior_prob)

        # Phase 3 - Backup statistics
        backup(node, value, sim_env.current_player)

    # Play - generate action probability from the root node.
    pi_probs = generate_play_policy(root_node, env.actions_mask, temperature)

    if deterministic:
        # Choose the action with most visit count.
        action_index = np.argmax(pi_probs)
    else:
        # Sample an action.
        action_index = np.random.choice(np.arange(pi_probs.shape[0]), p=pi_probs)

    # Reuse sub-tree.
    next_root_node = root_node.children[action_index]
    next_root_node.parent = DummyNode()
    return (next_root_node.move, pi_probs, next_root_node)


def add_virtual_loss(node: Node, loss_player: int) -> None:
    """Propagate a virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.
        loss_player: the player id to add virtual loss to,
            for any node in the traversed path, if the id equals to the loss_player,
            then it's a loss, otherwise it's a win,
            then the parent node is just the opposite of the leaf node

    """

    while node.parent is not None:
        node.losses_applied += 1
        node.total_value += -1.0 if node.to_play == loss_player else 1.0
        node = node.parent


def revert_virtual_loss(node: Node, loss_player: int) -> None:
    """Undo virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.
        loss_player: the player id for revert virtual loss.
    """

    while node.parent is not None:
        if node.losses_applied > 0:
            node.losses_applied -= 1
            node.total_value += 1.0 if node.to_play == loss_player else -1.0
        node = node.parent


def parallel_uct_search(
    env: BoardGameEnv,
    eval_func: Callable[[np.ndarray, bool], Tuple[np.ndarray, Union[np.ndarray, float]]],
    root_node: Node,
    c_puct_base: float,
    c_puct_init: float,
    temperature: float,
    num_simulations: int = 800,
    parallel_leaves: int = 8,
    root_noise: bool = False,
    deterministic: bool = False,
) -> Tuple[int, np.ndarray, Node]:
    """Single-threaded Upper Confidence Bound (UCB) for Trees (UCT) search without any rollout.

    Supports leaf-parallel search and batched evaluation.

    It follows the following general UCT search algorithm, except here we don't do rollout.
    ```
    function UCTSEARCH(r,m)
      i←1
      for i ≤ m do
          n ← select(r)
          n ← expand(n)
          ∆ ← rollout(n)
          backup(n,∆)
      end for
      return end function
    ```

    Args:
        env: a gym like custom BoardGameEnv environment.
        eval_func: a evaluation function when called returns the
            action probabilities and winning probability from
            current player's perspective.
        root_node: root node of the search tree, if none, new root node will be created.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        temperature: a parameter controls the level of exploration
            when generate policy action probabilities after MCTS search.
        num_simulations: number of simulations to run, default 800.
        parallel_leaves: Number of parallel leaves for MCTS search. This is also the batch size for neural network evaluation.
        root_noise: whether add dirichlet noise to root node to encourage exploration,
            default off.
        deterministic: after the MCTS search, choose the child node with most visits number to play in the game,
            instead of sample through a probability distribution, default off.

    Returns:
        tuple contains:
            a integer indicate the sampled action to play in the environment.
            a 1D numpy.array search policy action probabilities from the MCTS search result.
            a Node instance represent subtree of this MCTS search, which can be used as next root node for MCTS search.

    Raises:
        ValueError:
            if input argument `env` is not valid BoardGameEnv instance.
            if input argument `num_simulations` is not a positive integer.
        RuntimeError:
            if the game is over.
    """
    if not isinstance(env, BoardGameEnv):
        raise ValueError(f'Expect `env` to be a valid BoardGameEnv instance, got {env}')
    if not 1 <= num_simulations:
        raise ValueError(f'Expect `num_simulations` to a positive integer, got {num_simulations}')
    if env.is_game_over:
        raise RuntimeError('Game is over.')

    if root_node is None:
        # Create new root node if not reusing sub-tree
        root_node = Node(to_play=env.current_player, num_actions=env.num_actions, parent=DummyNode())
        prior_prob, value = eval_func(env.observation(), False)
        expand(root_node, prior_prob)
        backup(root_node, value, env.current_player)

    assert root_node.to_play == env.current_player

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise:
        add_dirichlet_noise(root_node, env.actions_mask)

    # for simulation in range(num_simulations):
    while root_node.number_visits < num_simulations:

        leaves = []
        failsafe = 0

        while len(leaves) < parallel_leaves and failsafe < parallel_leaves * 2:
            failsafe += 1

            # Make sure do not touch the actual environment.
            sim_env = copy.deepcopy(env)
            obs = sim_env.observation()
            done = sim_env.is_game_over

            # Phase 1 - Select
            # Select best child node until one of the following is true:
            # - reach a leaf node.
            # - game is over.
            node = root_node
            while node.is_expanded:
                node = best_child(
                    node, sim_env.current_player, sim_env.opponent_player, sim_env.actions_mask, c_puct_base, c_puct_init
                )
                # Make move on the simulation environment.
                obs, reward, done, _ = sim_env.step(node.move)
                if done:
                    break

            # Special case - If game is over, using the actual reward from the game to update statistics.
            if done:
                # Note when the game is over, the 'current_player' from the env 
                # is the same 'current_player' who made the move at timestep 'T-1' and won/loss the game
                # and the reward is also computed for (timestep 'T-1') 'current_player' perspective
                assert node.parent.to_play == sim_env.current_player
                backup(node, reward, sim_env.current_player)
                continue
            else:
                assert node.to_play == sim_env.current_player
                add_virtual_loss(node, root_node.to_play)
                leaves.append((node, obs, sim_env.current_player, sim_env.opponent_player))

        if leaves:
            batched_nodes, batched_obs, batched_current_player, batched_opponent_player = map(list, zip(*leaves))
            batched_obs = np.stack(batched_obs, axis=0)
            prior_probs, values = eval_func(batched_obs, True)

            for leaf, prior_prob, value, current_player, opponent_player in zip(
                batched_nodes, prior_probs, values, batched_current_player, batched_opponent_player
            ):
                revert_virtual_loss(leaf, root_node.to_play)

                # If a node was picked multiple times (despite virtual losses), we shouldn't
                # expand it more than once.
                if leaf.is_expanded:
                    continue

                value = value.item()  # To float

                expand(leaf, prior_prob)
                backup(leaf, value, current_player)

    # Play - generate action probability from the root node.
    pi_probs = generate_play_policy(root_node, env.actions_mask, temperature)

    if deterministic:
        # Choose the action with most visit count.
        action_index = np.argmax(pi_probs)
    else:
        # Sample an action.
        action_index = np.random.choice(np.arange(pi_probs.shape[0]), p=pi_probs)

    # Reuse sub-tree.
    next_root_node = root_node.children[action_index]
    next_root_node.parent = DummyNode()
    return (next_root_node.move, pi_probs, next_root_node)
