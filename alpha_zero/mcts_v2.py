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
"""MCTS class.

The positions are evaluated from the (to move) current player's perspective.

        A           Black to move

    B       C       White to move

  D   E             Black to move

For example, in the above two-player, zero-sum games search tree. 'A' is the root node,
and when the game is in state corresponding to node 'A', it's black's turn to move.
However the children nodes of 'A' are evaluated from white player's perspective.
So if we select the best child for node 'A', without further consideration,
we'd be actually selecting the best child for white player, which is not what we want.

Let's look at an simplified example where we don't consider number of visits and total values,
just the raw evaluation scores, if the evaluated scores (from white's perspective)
for 'B' and 'C' are 0.8 and 0.3 respectively. Then according to these results,
the best child of 'A' max(0.8, 0.3) is 'B', however this is done from white player's perspective.
But node 'A' represents black's turn to move, so we need to select the best child from black player's perspective,
which should be 'C' - the worst move for white, thus a best move for black.

Solution 1:
One way to resolve this issue is to always switching the signs of the child node's values when we select the best child.
For example:
    ucb_scores = -node.child_Q() + node.child_U()

In this case, a max(-0.8, -0.3) will give us the correct results for black player when we select the best child for node 'A'.
But this solution may introduce some confusion as the negative sign is not very self-explanation, and it's not in the pseudocode.

Solution 2:
Another more straight forward way is to simply store the values of the children nodes NOT from the current 'to move' player's perspective,
but from the their parent node's 'to move' player's perspective. So that when we select the best child,
we're always selecting the best child for the desired player. For the same example shown above,
if the evaluated scores (from white's perspective) for 'B' and 'C' are 0.8 and 0.3 respectively,
we'd actually store these scores as -0.8, -0.3 for 'B' and 'C' respectively. Since for two-player, zero-sum game,
one player's win is another player's loss, so we can simply switching the sign of the values.
Similar logic also applies to the rest of the search tree.

In this MCTS implementation, we chose to use the first one to show how it's done in code.
"""

import copy
import math
from typing import Callable, Tuple, Mapping, Union, Any
import numpy as np
from alpha_zero.games.env import BoardGameEnv


class Node:
    """Node in the MCTS search tree."""

    def __init__(self, legal_actions: np.ndarray, move: int = None, parent: Any = None) -> None:
        """
        Args:
            legal_actions: a 1D bool numpy.array mask for all actions,
                where `True` represents legal move and `False` represents illegal move.
            prior: a prior probability of the node for a specific action, could be empty in case of root node.
            move: the action associated with the prior probability.
            parent: the parent node, could be `None` if this is the root node.
        """
        self.move = move
        self.parent = parent
        self.legal_actions = legal_actions
        self.is_expanded = False

        self.number_visits = 0  # number of visits
        self.total_value = 0.0  # total action values
        self.child_priors = np.zeros_like(self.legal_actions).astype(np.float32)

        self.children: Mapping[int, Node] = {}

        # number of virtual losses on this node, only used in 'parallel_uct_search'
        self.losses_applied = 0

    def child_U(self, c_puct_base: float, c_puct_init: float) -> np.ndarray:
        """Returns a 1D numpy.array contains prior score for all child."""
        pb_c = math.log((1.0 + self.number_visits + c_puct_base) / c_puct_base) + c_puct_init
        return pb_c * self.child_priors * (math.sqrt(self.number_visits) / (1 + self.child_number_visits))

    def child_Q(self) -> np.ndarray:
        """Returns a 1D numpy.array contains mean action value for all child."""
        child_q = np.zeros_like(self.legal_actions).astype(np.float32)
        for a, child in self.children.items():
            if child.number_visits > 0:
                child_q[a] = child.total_value / child.number_visits

        return child_q

    @property
    def child_number_visits(self) -> np.ndarray:
        """Returns a 1D numpy.array contains visits count for all child."""
        child_n = np.zeros_like(self.legal_actions).astype(np.float32)
        for a, child in self.children.items():
            if child.number_visits > 0:
                child_n[a] = child.number_visits
        return child_n

    @property
    def has_parent(self) -> bool:
        return isinstance(self.parent, Node)


def best_child(node: Node, c_puct_base: float, c_puct_init: float) -> Node:
    """Returns best child node with maximum action value Q plus an upper confidence bound U.
    And creates the selected best child node if not already exists.

    Args:
        node: the current node in the search tree.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.

    Returns:
        The best child node.

    Raises:
        ValueError:
            if the node instance itself is a leaf node.
            if input argument `legal_actions` is not a valid 1D bool numpy.array.
    """
    if not node.is_expanded:
        raise ValueError('Expand leaf node first.')

    ucb_scores = node.child_Q() + node.child_U(c_puct_base, c_puct_init)

    # Exclude illegal actions, note in some cases, the max ucb_scores may be zero.
    ucb_scores = np.where(node.legal_actions, ucb_scores, -1000)

    # Break ties if we have multiple 'maximum' values.
    move = np.random.choice(np.where(ucb_scores == ucb_scores.max())[0])

    assert node.legal_actions[move] == True

    if move not in node.children:
        child_legal_actions = node.legal_actions.copy()
        child_legal_actions[move] = False
        node.children[move] = Node(legal_actions=child_legal_actions, move=move, parent=node)

    return node.children[move]


def expand(node: Node, prior_prob: np.ndarray) -> None:
    """Expand all actions, including illegal actions.

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


def backup(node: Node, value: float) -> None:
    """Update statistics of the this node and all traversed parent nodes.

    Args:
        node: current leaf node in the search tree.
        value: the evaluation value (from last player's perspective).

    Raises:
        ValueError:
            if input argument `value` is not float data type.
    """

    if not isinstance(value, float):
        raise ValueError(f'Expect `value` to be a float type, got {type(value)}')

    while node is not None:
        node.number_visits += 1
        node.total_value += value
        node = node.parent
        value = -value


def add_dirichlet_noise(node: Node, eps: float = 0.25, alpha: float = 0.03) -> None:
    """Add dirichlet noise to a given node.

    Args:
        node: the root node we want to add noise to.
        eps: epsilon constant to weight the priors vs. dirichlet noise.
        alpha: parameter of the dirichlet noise distribution.

    Raises:
        ValueError:
            if input argument `node` is not expanded.
            if input argument `legal_actions` is not a valid 1D bool numpy.array.
            if input argument `eps` or `alpha` is not float type
                or not in the range of [0.0, 1.0].
    """

    if not isinstance(node, Node) or not node.is_expanded:
        raise ValueError('Expect `node` to be expanded')
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(f'Expect `eps` to be a float in the range [0.0, 1.0], got {eps}')
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f'Expect `alpha` to be a float in the range [0.0, 1.0], got {alpha}')

    alphas = np.ones_like(node.legal_actions) * alpha
    noise = np.random.dirichlet(alphas)

    # Set noise to zero for illegal actions
    noise = np.where(node.legal_actions, noise, 0)

    node.child_priors = node.child_priors * (1 - eps) + noise * eps


def generate_play_policy(node: Node, temperature: float) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentialted visit count.

    Args:
        node: the root node of the search tree.
        temperature: a parameter controls the level of exploration.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if node instance not expanded.
            if input argument `legal_actions` is not a valid 1D bool numpy.array.
            if input argument `temperature` is not float type or not in range [0.0, 1.0].
    """
    if not node.is_expanded:
        raise ValueError('Node not expanded.')
    if not isinstance(temperature, float) or not 0 <= temperature <= 1.0:
        raise ValueError(f'Expect `temperature` to be float type in the range [0.0, 1.0], got {temperature}')

    if temperature > 0.0:
        # exp = 1.0 / temperature
        # To avoid overflow when doing power operation over large numbers
        exp = max(1.0, min(10.0, 1.0 / temperature))
        pi_logits = np.power(node.child_number_visits, exp)
    else:
        pi_logits = node.child_number_visits / np.sum(node.child_number_visits)

    # Mask out illegal actions
    pi_logits = np.where(node.legal_actions == True, pi_logits, 0.0).astype(np.float32)
    pi_probs = pi_logits / np.sum(pi_logits)

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
        root_node: root node of the search tree, this comes from reuse sub-tree.
        c_puct_base: a float constant determining the level of exploration.
        c_puct_init: a float constant determining the level of exploration.
        temperature: a parameter controls the level of exploration
            when generate policy action probabilities after MCTS search.
        num_simulations: number of simulations to run, default 800.
        root_noise: whether add dirichlet noise to root node to encourage exploration, default off.
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

    # Create root node
    if root_node is None:
        root_node = Node(legal_actions=env.legal_actions, parent=None)
        prior_prob, value = eval_func(env.observation(), False)
        expand(root_node, prior_prob)
        # Switching the sign of the evaluated value so it's from the last player's perspective.
        backup(root_node, -value)

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise:
        add_dirichlet_noise(root_node)

    while root_node.number_visits < num_simulations:
        node = root_node

        # Make sure do not touch the actual environment.
        sim_env = copy.deepcopy(env)
        obs = sim_env.observation()
        done = sim_env.is_game_over

        # Phase 1 - Select
        # Select best child node until one of the following is true:
        # - reach a leaf node.
        # - game is over.
        while node.is_expanded:
            node = best_child(node, c_puct_base, c_puct_init)
            # Make move on the simulation environment.
            obs, reward, done, _ = sim_env.step(node.move)
            if done:
                break

        # Special case - If game is over, using the actual reward from the game to update statistics
        if done:
            # The reward is for the last player who made the move won/loss the game.
            backup(node, reward)
            continue

        # Phase 2 - Expand and evaluation
        prior_prob, value = eval_func(obs, False)
        expand(node, prior_prob)

        # Phase 3 - Backup statistics
        # Switching the sign of the evaluated value so it's from the last player's perspective.
        backup(node, -value)

    # Play - generate action probability from the root node.
    pi_probs = generate_play_policy(root_node, temperature)

    move = None
    if deterministic:
        # Choose the action with most visit count.
        move = np.argmax(pi_probs)
    else:
        # Sample an action.
        move = np.random.choice(np.arange(pi_probs.shape[0]), p=pi_probs)

    next_root_node = None
    if move in root_node.children:
        next_root_node = root_node.children[move]
        next_root_node.parent = None
    
    return (move, pi_probs, next_root_node)


def add_virtual_loss(node: Node) -> None:
    """Propagate a virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.

    """
    # This is a loss for both players in the traversed path,
    # since we want to avoid multiple threads to select the same path.
    vloss = -1
    while node.parent is not None:
        node.losses_applied += 1
        node.total_value += vloss
        node = node.parent


def revert_virtual_loss(node: Node) -> None:
    """Undo virtual loss to the traversed path.

    Args:
        node: current leaf node in the search tree.
    """

    vloss = +1
    while node.parent is not None:
        if node.losses_applied > 0:
            node.losses_applied -= 1
            node.total_value += vloss
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

    This implementation uses tree parallel search and batched evaluation.

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
        root_node: root node of the search tree, this comes from reuse sub-tree.
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

    # Create root node
    if root_node is None:
        root_node = Node(legal_actions=env.legal_actions, parent=None)
        prior_prob, value = eval_func(env.observation(), False)
        expand(root_node, prior_prob)
        # Switching the sign of the evaluated value so it's from the last player's perspective.
        backup(root_node, -value)

    # Add dirichlet noise to the prior probabilities to root node.
    if root_noise:
        add_dirichlet_noise(root_node)

    while root_node.number_visits < num_simulations:
        leaves = []
        failsafe = 0

        while len(leaves) < parallel_leaves and failsafe < parallel_leaves * 2:
            # This is necessary as when a game is over no leaf is added to leaves,
            # as we use the actual game results to update statistic
            failsafe += 1
            node = root_node

            # Make sure do not touch the actual environment.
            sim_env = copy.deepcopy(env)
            obs = sim_env.observation()
            done = sim_env.is_game_over

            # Phase 1 - Select
            # Select best child node until one of the following is true:
            # - reach a leaf node.
            # - game is over.
            while node.is_expanded:
                node = best_child(node, c_puct_base, c_puct_init)
                # Make move on the simulation environment.
                obs, reward, done, _ = sim_env.step(node.move)
                if done:
                    break

            # Special case - If game is over, using the actual reward from the game to update statistics.
            if done:
                # The reward is for the last player who made the move won/loss the game.
                backup(node, reward)
                continue
            else:
                add_virtual_loss(node)
                leaves.append((node, obs))

        if leaves:
            batched_nodes, batched_obs = map(list, zip(*leaves))
            prior_probs, values = eval_func(np.stack(batched_obs, axis=0), True)

            for leaf, prior_prob, value in zip(batched_nodes, prior_probs, values):
                revert_virtual_loss(leaf)

                # If a node was picked multiple times (despite virtual losses), we shouldn't
                # expand it more than once.
                if leaf.is_expanded:
                    continue

                expand(leaf, prior_prob)
                # Switching the sign of the evaluated value so it's from the last player's perspective.
                backup(leaf, -value.item())

    # Play - generate action probability from the root node.
    pi_probs = generate_play_policy(root_node, temperature)

    move = None
    if deterministic:
        # Choose the action with most visit count.
        move = np.argmax(pi_probs)
    else:
        # Sample an action.
        move = np.random.choice(np.arange(pi_probs.shape[0]), p=pi_probs)

    next_root_node = None
    if move in root_node.children:
        next_root_node = root_node.children[move]
        next_root_node.parent = None
    
    return (move, pi_probs, next_root_node)
