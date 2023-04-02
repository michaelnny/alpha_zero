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

from alpha_zero import mcts_v2 as mcts
from alpha_zero.games.gomoku import GomokuEnv


class NodeTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.int8)
        self.root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())

    def test_node_basics(self):
        root_node = self.root_node

        self.assertEqual(root_node.number_visits, 0)
        self.assertEqual(root_node.total_value, 0)
        # self.assertEqual(root_node.Q, 0)
        self.assertEqual(root_node.move, None)
        self.assertFalse(root_node.is_expanded)
        self.assertFalse(root_node.has_parent)


class ExpandNodeTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.int8)

    def test_expand_node(self):
        root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())
        mcts.expand(root_node, self.prior)

        self.assertEqual(len(root_node.children), 0)
        self.assertTrue(root_node.is_expanded)

    def test_node_expand_invalid_type(self):
        dummy = [0.1, 3.3, {}, []]
        root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())
        with self.assertRaisesRegex(ValueError, 'Expect'):
            mcts.expand(root_node, dummy)

    def test_expand_node_already_expanded(self):
        root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())
        mcts.expand(root_node, self.prior)

        with self.assertRaisesRegex(RuntimeError, 'Node already expanded'):
            mcts.expand(root_node, self.prior)


class BestChildTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.int8)

    def test_node_best_child_on_leaf_node(self):
        root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())
        with self.assertRaisesRegex(ValueError, 'Expand leaf node first'):
            mcts.best_child(root_node, self.legal_actions, 19652, 1.25, 1)

    def test_node_best_child_high_priorabilities(self):
        root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())
        mcts.expand(root_node, self.prior)
        mcts.backup(root_node, 0.8)
        best_node = mcts.best_child(root_node, self.legal_actions, 19652, 1.25, 1)
        self.assertEqual(best_node, root_node.children[1])


class MCTSUpdateStatisticsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.int8)

    def test_update_statistics_root_node(self):
        root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())
        mcts.expand(root_node, self.prior)
        mcts.backup(root_node, 0.5)

        self.assertEqual(root_node.number_visits, 1)
        self.assertEqual(root_node.total_value, 0.5)

    def test_update_statistics_on_child(self):
        # Black player to move
        root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())
        mcts.expand(root_node, self.prior)
        mcts.backup(root_node, 0.5)

        child_1 = mcts.best_child(root_node, self.legal_actions, 19652, 1.25, 1)
        mcts.expand(child_1, self.prior)
        mcts.backup(child_1, 0.5)

        child_2 = mcts.best_child(root_node, self.legal_actions, 19652, 1.25, 1)
        mcts.expand(child_2, self.prior)
        mcts.backup(child_2, 0.5)

        self.assertEqual(root_node.number_visits, 3)
        self.assertAlmostEqual(root_node.total_value, -0.5, places=8)

        self.assertEqual(child_1.number_visits, 1)
        self.assertAlmostEqual(child_1.total_value, 0.5, places=8)

        self.assertEqual(child_2.number_visits, 1)
        self.assertAlmostEqual(child_2.total_value, 0.5, places=8)


class MCTSGeneratePlayPolicyTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.4, 0.25, 0.15])
        self.legal_actions = np.ones_like(self.prior, dtype=np.int8)

    def test_play_policy_prob_sums_1(self):
        root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())
        mcts.expand(root_node, self.prior)

        child_1 = mcts.best_child(root_node, self.legal_actions, 19652, 1.25, 1)
        mcts.expand(child_1, self.prior)
        mcts.backup(child_1, -0.5)

        pi_prob = mcts.generate_search_policy(root_node.child_number_visits, 0.1)
        np.testing.assert_allclose(np.sum(pi_prob), np.ones(1))

    @parameterized.named_parameters(('temp_1', -0.1), ('temp_2', 1.1))
    def test_play_policy_invalid_temp(self, tmp):
        root_node = mcts.Node(to_play=0, num_actions=self.num_actions, parent=mcts.DummyNode())
        mcts.expand(root_node, self.prior)

        for i in range(100):
            for node in root_node.children.values():
                mcts.backup(node, 0.02)

        with self.assertRaisesRegex(ValueError, 'Expect'):
            pi_prob = mcts.generate_search_policy(root_node.child_number_visits, tmp)


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
        root_node = None

        while env.steps < 10:
            action, pi_prob, root_node = mcts.uct_search(env, mock_eval_func, root_node, 19652, 1.25, 1.0, 100)
            obs, reward, done, info = env.step(action)
            if done:
                break


class ParallelUCTSearchTest(parameterized.TestCase):
    def test_run_parallel_uct_search(self):
        env = GomokuEnv(board_size=7)
        obs = env.reset()
        root_node = None

        while env.steps < 10:
            action, pi_prob, root_node = mcts.parallel_uct_search(env, mock_eval_func, root_node, 19652, 1.25, 1.0, 100, 4)
            obs, reward, done, info = env.step(action)
            if done:
                break


if __name__ == '__main__':
    absltest.main()
