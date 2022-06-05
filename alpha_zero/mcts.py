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
"""MCTS class."""

from __future__ import annotations

import copy
import math
from typing import Callable, List, Tuple, Mapping
import numpy as np
from alpha_zero.games.env import BoardGameEnv


class Node:
    """Node in the MCTS search tree."""

    def __init__(self, player_id: int, prior: float = None, move: int = None, parent: Node = None) -> None:
        """
        Args:
            player_id: the id of the current player, used to check values when we update node statistics.
            prior: a prior probability of the node for a specific action, could be empty in case root node.
            move: the action associated to the prior probability.
            parent: the parent node, could be `None` if this is the root node.
        """
        self.player_id = player_id
        self.prior = prior
        self.move = move
        self.parent = parent
        self.is_expanded = False

        self.N = 0  # number of visits
        self.W = 0.0  # total action values

        self.children: List[Node] = []

    def expand(self, prior: np.ndarray, player_id: int) -> None:
        """Expand all actions, including illegal actions.

        Args:
            prior: 1D numpy.array contains prior probabilities of the state for all actions,
                whoever calls this should pre-processing illegal actions first.
            player_id: the current player id in the environment timestep.

        Raises:
            ValueError:
                if node instance already expanded.
                if input argument `prior` is not a valid 1D float numpy.array.
        """
        if self.is_expanded:
            raise RuntimeError('Node already expanded.')
        if not isinstance(prior, np.ndarray) or len(prior.shape) != 1 or prior.dtype not in (np.float32, np.float64):
            raise ValueError(f'Expect `prior` to be a 1D float numpy.array, got {prior}')

        for action in range(0, prior.shape[0]):
            child = Node(player_id=player_id, prior=prior[action], move=action, parent=self)
            self.children.append(child)

        self.is_expanded = True

    def backup(self, score: Mapping[int, float]) -> None:
        """Update statistics of the this node and all travesed parent nodes.

        Args:
            score: a dict contains the evaulation value
                for both players (with player id as keys).

        Raises:
            ValueError:
                if input argument `score` is not a dict.
        """
        if not isinstance(score, dict):
            raise ValueError(f'Expect `score` to be a dict, got {score}')

        current = self
        while current is not None:
            current.N += 1
            current.W += float(score[current.player_id])
            current = current.parent

    def best_child(self, actions_mask: np.ndarray, c_puct: float = 5.0) -> Node:
        """Returns best child node with maximum action value Q plus an upper confidence bound U.

        Args:
            actions_mask: a 1D bool numpy.array contains legal actions for current state.
            c_puct: a float constatnt determining the level of exploration, default 5.0.

        Returns:
            The best child node.

        Raises:
            ValueError:
                if the node instance itself is a leaf node.
                if input argument `actions_mask` is not a valid 1D bool numpy.array.
                if input argument `c_puct` is not float type or not in the range [0.0, 10.0].
        """
        if not self.is_expanded:
            raise ValueError('Expand leaf node first.')
        if not isinstance(actions_mask, np.ndarray) or actions_mask.dtype != np.bool8 or len(actions_mask.shape) != 1:
            raise ValueError(f'Expect `actions_mask` to be a 1D bool numpy.array, got {actions_mask}')
        if not isinstance(c_puct, float) or not 0.0 <= c_puct <= 10.0:
            raise ValueError(f'Expect `c_puct` to be float type in the range [0.0, 10.0], got {c_puct}')

        # Here we're playing in a competitive mode, we want to select the action that minimize opponent's score
        # The Child Q value is from opponent's perspective, so we always switch it's sign.
        ucb_results = -1.0 * self.child_Q + c_puct * self.child_U

        # Exclude illegal actions, note max ucb_results may be zero.
        ucb_results = np.where(actions_mask, ucb_results, -1000)

        # Break ties when have multiple 'max' value.
        deterministic = np.random.choice(np.where(ucb_results == ucb_results.max())[0])
        best_child = self.children[deterministic]

        return best_child

    @property
    def Q(self) -> float:
        """Returns the mean action value Q(s, a)."""
        if self.N == 0:
            return 0.0
        return self.W / self.N

    @property
    def child_N(self) -> np.ndarray:
        """Returns a 1D numpy.array contains visits count for all child."""
        return np.array([child.N for child in self.children], dtype=np.int32)

    @property
    def child_Q(self) -> np.ndarray:
        """Returns a 1D numpy.array contains mean action value for all child."""
        return np.array([child.Q for child in self.children], dtype=np.float32)

    @property
    def child_U(self) -> np.ndarray:
        """Returns a 1D numpy.array contains UCB score for all child."""
        return np.array([child.prior * (math.sqrt(self.N) / (1 + child.N)) for child in self.children], dtype=np.float32)

    @property
    def has_parent(self) -> bool:
        return isinstance(self.parent, Node)


def add_dirichlet_noise(prob: np.ndarray, eps: float = 0.25, alpha: float = 0.03) -> np.ndarray:
    """Add dirichlet noise to a given probabilities.

    Args:
        prob: a numpy.array contains action probabilities we want to add noise to.
        eps: epsilon constant to weight the priors vs. dirichlet noise.
        alpha: parameter of the dirichlet noise distribution.

    Returns:
        action probabilities with added dirichlet noise.

    Raises:
        ValueError:
            if input argument `prob` is not a valid float numpy.array.
            if input argument `eps` or `alpha` is not float type
                or not in the range of [0.0, 1.0].
    """

    if not isinstance(prob, np.ndarray) or prob.dtype not in (np.float32, np.float64):
        raise ValueError(f'Expect `prob` to be a numpy.array, got {prob}')
    if not isinstance(eps, float) or not 0.0 <= eps <= 1.0:
        raise ValueError(f'Expect `eps` to be a float in the range [0.0, 1.0], got {eps}')
    if not isinstance(alpha, float) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f'Expect `alpha` to be a float in the range [0.0, 1.0], got {alpha}')

    alphas = np.ones_like(prob) * alpha
    noise = np.random.dirichlet(alphas)
    noised_prob = (1 - eps) * prob + eps * noise
    return noised_prob


def generate_play_policy(visits_count: np.ndarray, temperature: float) -> np.ndarray:
    """Returns a policy action probabilities after MCTS search,
    proportional to its exponentialted visit count.

    Args:
        visits_count: a 1D numpy.array contains child node visits count.
        temperature: a parameter controls the level of exploration.

    Returns:
        a 1D numpy.array contains the action probabilities after MCTS search.

    Raises:
        ValueError:
            if input argument `visits_count` is not a valid 1D numpy.array.
            if input argument `temperature` is not float type or not in the range [0.1, 1.0].
    """
    if not isinstance(visits_count, np.ndarray) or len(visits_count.shape) != 1 or visits_count.shape == (0,):
        raise ValueError(f'Expect `visits_count` to be a 1D numpy.array, got {visits_count}')
    if not isinstance(temperature, float) or not 0.1 <= temperature <= 1.0:
        raise ValueError(f'Expect `temperature` to be float type in the range [0.1, 1.0], got {temperature}')

    visits_count = np.asarray(visits_count, dtype=np.int64)

    if temperature > 0.0:
        # Wite the following hack, we limit the exponent in the range of [1.0, 5.0]
        # to avoid overflow when doing power operation over large numbers
        exp = max(1.0, min(5.0, 1.0 / temperature))

        pi_logits = np.power(visits_count, exp)
        pi_prob = pi_logits / np.sum(pi_logits)
    else:
        pi_prob = visits_count / np.sum(visits_count)

    return pi_prob


def set_illegal_action_probs_to_zero(actions_mask: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Set probabilities to zero for illegal actions.
    Args:
        actions_mask: a 1D bool numpy.array with valid actions True, invalid actions False.
        prob: a 1D float numpy.array prior/action probabilities.

    Returns:
        a 1D float numpy.array prior/action probabilities with invalid actions masked out.
    """

    assert actions_mask.shape == prob.shape

    prob = np.where(actions_mask, prob, 0.0)
    sumed = np.sum(prob)
    if sumed > 0:
        prob /= sumed
    return prob


def uct_search(
    env: BoardGameEnv,
    eval_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    root_node: Node,
    c_puct: float,
    temperature: float,
    num_simulations: int = 800,
    root_prior_noise: bool = False,
    deterministic: bool = False,
) -> Tuple[int, np.ndarray, Node]:
    """Single-threaded Upper Confidence Bound (UCB) for Trees (UCT) search without any rollout.

    It follows the following general UCT search algorithm.
    ```
    function UCTSEARCH(r,m)
      i←1
      for i ≤ m do
          n ← select(r)
          n ← expand_node(n)
          ∆ ← playout(n)
          update_statistics(n,∆)
      end for
      return end function
    ```

    Args:
        env: a gym like custom BoardGameEnv environment.
        eval_func: a evaluation function when called returns the
            action probabilities and winning probability from
            current player's perspective.
        root_node: root node of the search tree, this comes from reuse sub-tree.
        c_puct: a float constatnt determining the level of exploration
            when select child node during MCTS search.
        temperature: a parameter controls the level of exploration
            when generate policy action probabilities after MCTS search.
        num_simulations: number of simulations to run, default 800.
        root_prior_noise: whether add dirichlet noise to root node action priorities to encourage exploration,
            default off.
        deterministic: after the MCTS search, choose the child node with most visits number to play in the real environment,
            instead of sample through a probability distribution, default off.

    Returns:
        tuple contains:
            a integer indicate the sampled action to play in the environment.
            a 1D numpy.array search policy action probabilities from the MCTS search result.
            a Node instance represent subtree of this MCTS search, which can be used as next root node for MCTS search.

    Raises:
        ValueError:
            if input argument `env` is not valid BoardGameEnv instance.
            if input argument `num_simulations` is not a positive interger.
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
        root_node = Node(player_id=env.current_player, parent=None)

    assert root_node.player_id == env.current_player

    # for simulation in range(num_simulations):
    while root_node.N < num_simulations:
        # Make sure do not touch the actual environment.
        simulation_env = copy.deepcopy(env)
        obs = simulation_env.observation()
        done = simulation_env.is_game_over

        # Phase 1 - Select
        # Select best child node until one of the following is true:
        # - reach a leaf node.
        # - game is over.
        node = root_node
        while node.is_expanded:
            node = node.best_child(simulation_env.actions_mask, c_puct)
            # Make move on the simulation environment.
            obs, reward, done, _ = simulation_env.step(node.move)
            if done:
                break

        # Special case - Game over
        # If game is over, using the actual reward from the game to update statistics.
        if done:
            # When game is over, the env stops updates the current player for timestep `t`.
            # So the current player for `t` is the same current player at `t-1` timestep who just won/loss the game.
            score = {simulation_env.current_player: reward, simulation_env.opponent_player: -reward}
            node.backup(score)
            continue

        # Phase 2 - Expand and evaluation
        prior_prob, value = eval_func(obs)

        # Add dirichlet noise to the prior probabilities to root node.
        if root_prior_noise and not node.has_parent:
            prior_prob = add_dirichlet_noise(prior_prob)

        # Set prior probabilities to zero for illegal actions.
        prior_prob = set_illegal_action_probs_to_zero(simulation_env.actions_mask, prior_prob)
        # Sub-nodes are 'owned' by opponent player.
        node.expand(prior=prior_prob, player_id=simulation_env.opponent_player)

        # Phase 3 - Backup on leaf node
        score = {simulation_env.current_player: value, simulation_env.opponent_player: -value}
        node.backup(score)

    # Play - generate action probability from the root node.
    # Maskout illegal actions.
    child_visits = np.where(env.actions_mask, root_node.child_N, 0)
    pi_prob = generate_play_policy(child_visits, temperature)

    if deterministic:
        # Choose the action with most visit number.
        action_index = np.argmax(child_visits)
    else:
        # Sample a action.
        action_index = np.random.choice(np.arange(pi_prob.shape[0]), p=pi_prob)

    # Reuse sub-tree.
    next_root_node = root_node.children[action_index]
    return (next_root_node.move, pi_prob, next_root_node)
