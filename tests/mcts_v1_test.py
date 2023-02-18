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
"""Tests for mcts_v1.py"""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from alpha_zero import mcts_v1 as mcts
from alpha_zero.games.gomoku import GomokuEnv


class NodeTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.bool8)
        self.root_node = mcts.Node(to_play=0)

    def test_node_basics(self):
        root_node = self.root_node

        self.assertEqual(root_node.number_visits, 0)
        self.assertEqual(root_node.total_value, 0)
        self.assertEqual(root_node.Q, 0)
        self.assertEqual(root_node.move, None)
        self.assertEqual(root_node.prior, None)
        self.assertFalse(root_node.is_expanded)
        self.assertFalse(root_node.has_parent)


class ExpandNodeTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.bool8)

    def test_expand_node(self):
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)

        self.assertEqual(len(root_node.children), self.prior.shape[0])
        self.assertTrue(root_node.is_expanded)

        for i in range(len(self.prior)):
            child_node = root_node.children[i]
            self.assertEqual(child_node.number_visits, 0)
            self.assertEqual(child_node.total_value, 0)
            self.assertEqual(child_node.Q, 0)
            self.assertEqual(child_node.prior, self.prior[i])
            self.assertFalse(child_node.is_expanded)
            self.assertEqual(child_node.parent, root_node)

    def test_node_expand_invalid_type(self):
        dummy = [0.1, 3.3, {}, []]
        root_node = mcts.Node(to_play=0)
        with self.assertRaisesRegex(ValueError, 'Expect'):
            mcts.expand(root_node, dummy, 1)

    def test_expand_node_already_expanded(self):
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)

        with self.assertRaisesRegex(RuntimeError, 'Node already expanded'):
            mcts.expand(root_node, self.prior, 1)


class BestChildTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.bool8)

    def test_node_best_child_on_leaf_node(self):
        root_node = mcts.Node(to_play=0)
        with self.assertRaisesRegex(ValueError, 'Expand leaf node first'):
            mcts.best_child(root_node, self.legal_actions, 19652, 1.25)

    def test_node_best_child_high_priorabilities(self):
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)
        mcts.backup(root_node, 0.8, 0)
        best_node = mcts.best_child(root_node, self.legal_actions, 19652, 1.25)
        self.assertEqual(best_node, root_node.children[1])

    def test_node_best_child_high_action_values(self):
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)
        mcts.backup(root_node, 0.8, root_node.to_play)

        for _ in range(5):
            mcts.backup(root_node.children[0], 0.1, 0)
            mcts.backup(root_node.children[1], 0.6, 0)
            mcts.backup(root_node.children[2], 0.5, 0)
            mcts.backup(root_node.children[3], 0.4, 0)

        best_node = mcts.best_child(root_node, self.legal_actions, 19652, 1.25)
        # Child node Q is the opposite.
        self.assertEqual(best_node, root_node.children[1])


class MCTSUpdateStatisticsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.bool8)

    @parameterized.named_parameters(('black_player', 1), ('white_player', -1))
    def test_update_statistics_root_node(self, to_play):
        root_node = mcts.Node(to_play=to_play)
        mcts.expand(root_node, self.prior, -to_play)
        mcts.backup(root_node, 0.5, root_node.to_play)

        self.assertEqual(root_node.number_visits, 1)
        self.assertEqual(root_node.total_value, 0.5)

    def test_update_statistics_level_1_mixed(self):
        # Black player to move
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)

        # Expand level 1 children nodes
        # White player to move
        for child in root_node.children.values():
            mcts.expand(child, self.prior, 1)

        child_0 = root_node.children[0]
        child_1 = root_node.children[1]
        child_2 = child_0.children[0]
        child_3 = child_0.children[1]

        black_values = [0.5, 0.6]
        # Black player score is positive.
        for v, child in zip(black_values, [child_0, child_1]):
            mcts.backup(child, v, 0)

        white_values = [1.0, 2.0]
        # White player score is positive.
        for v, child in zip(white_values, [child_2, child_3]):
            mcts.backup(child, v, 1)

        self.assertEqual(root_node.number_visits, 4)
        self.assertAlmostEqual(root_node.total_value, 0.5 + 0.6 - 1 - 2, places=8)

        self.assertEqual(child_2.number_visits, 1)
        self.assertAlmostEqual(child_2.total_value, 1, places=8)

        self.assertEqual(child_3.number_visits, 1)
        self.assertAlmostEqual(child_3.total_value, 2, places=8)

    def test_update_statistics_level_2(self):
        # Black player to move
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)

        # Expand level 1 children nodes
        # White player to move
        for child in root_node.children.values():
            mcts.expand(child, self.prior, 1)

        # Expand level 2 children nodes
        # Black player to move
        for child in root_node.children.values():
            for c in child.children.values():
                mcts.expand(c, self.prior, 0)

        child_0 = root_node.children[0]
        child_1 = child_0.children[0]
        child_2 = child_1.children[0]
        child_3 = child_1.children[1]

        # First black move
        mcts.backup(child_0, 0.6, 0)

        # First white move
        mcts.backup(child_1, -0.8, 0)

        # Second black moves
        mcts.backup(child_2, 1.0, 0)

        mcts.backup(child_3, 2.0, 0)

        self.assertEqual(root_node.number_visits, 4)
        self.assertAlmostEqual(root_node.total_value, 0.6 - 0.8 + 1 + 2, places=8)


class MCTSGeneratePlayPolicyTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.bool8)

    def test_play_policy_error_on_leaf_node(self):
        root_node = mcts.Node(to_play=0)
        with self.assertRaisesRegex(ValueError, 'Node not expanded'):
            mcts.generate_play_policy(root_node, self.legal_actions, 0.1)

    def test_play_policy_root_node_greedy_equal_prob(self):
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)

        # Make sure each child is visited once
        for node in root_node.children.values():
            mcts.backup(node, 0.02, 0)

        pi_prob = mcts.generate_play_policy(root_node, self.legal_actions, 0.1)
        visits = np.array([1, 1, 1, 1])
        expected_prob = visits / np.sum(visits)
        np.testing.assert_allclose(pi_prob, expected_prob)

    def test_play_policy_prob_sums_1(self):
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)

        # Make sure each child is visited once
        for node in root_node.children.values():
            mcts.backup(node, 0.02, 0)

        pi_prob = mcts.generate_play_policy(root_node, self.legal_actions, 0.1)
        np.testing.assert_allclose(np.sum(pi_prob), np.ones(1))

    def test_play_policy_root_node_greedy_no_equal_prob(self):
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)

        # Make sure each child is visited once
        for node in root_node.children.values():
            mcts.backup(node, 0.02, 0)

        child = root_node.children[1]
        mcts.backup(child, 0.02, 0)

        pi_prob = mcts.generate_play_policy(root_node, self.legal_actions, 0.1)
        visits = np.array([1, 2, 1, 1], dtype=np.float64)
        exp = 1 / 0.1
        visits = visits**exp
        expected_prob = visits / np.sum(visits)
        np.testing.assert_allclose(pi_prob, expected_prob, atol=1e-6)

    def test_play_policy_root_node_exploration_equal_prob(self):
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)

        # Make sure each child is visited once
        for node in root_node.children.values():
            mcts.backup(node, 0.02, 0)

        pi_prob = mcts.generate_play_policy(root_node, self.legal_actions, 1.0)
        visits = np.array([1, 1, 1, 1])
        expected_prob = visits / np.sum(visits)
        np.testing.assert_allclose(pi_prob, expected_prob)

    @parameterized.named_parameters(('temp_1', -0.1), ('temp_2', 1.1))
    def test_play_policy_invalid_temp(self, tmp):
        root_node = mcts.Node(to_play=0)
        mcts.expand(root_node, self.prior, 1)

        for i in range(100):
            for node in root_node.children.values():
                mcts.backup(node, 0.02, 0)

        with self.assertRaisesRegex(ValueError, 'Expect'):
            pi_prob = mcts.generate_play_policy(root_node, self.legal_actions, tmp)


def mock_eval_func(state_tensor, batched=False):
    # Mock network output
    num_actions = state_tensor.shape[-1] ** 2
    if not batched:
        prior_shape = (num_actions,)
        value_shape = (1,)
    else:
        batch_size = state_tensor.shape[0]
        prior_shape = (
            batch_size,
            num_actions,
        )
        value_shape = (
            batch_size,
            1,
        )

    prior_prob = np.random.uniform(size=prior_shape)
    v = np.random.uniform(-1, 1, size=value_shape)

    if not batched:
        v = v.item()

    return (prior_prob, v)


class UCTSearchTest(parameterized.TestCase):
    def test_run_uct_search(self):
        env = GomokuEnv(board_size=7)
        obs = env.reset()

        while env.steps < 10:
            action, pi_prob = mcts.uct_search(env, mock_eval_func, 19652, 1.25, 1.0, 100)
            obs, reward, done, info = env.step(action)
            if done:
                break


class ParallelUCTSearchTest(parameterized.TestCase):
    def test_run_parallel_uct_search(self):
        env = GomokuEnv(board_size=7)
        obs = env.reset()

        while env.steps < 10:
            action, pi_prob = mcts.parallel_uct_search(env, mock_eval_func, 19652, 1.25, 1.0, 100, 4)
            obs, reward, done, info = env.step(action)
            if done:
                break


if __name__ == '__main__':
    absltest.main()
