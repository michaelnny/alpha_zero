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
"""Tests for mcts.py"""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from alpha_zero import mcts
from alpha_zero.games.gomoku import GomokuEnv


class NodeTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.3, 0.25, 0.15])
        self.actions_mask = np.ones_like(self.prior, dtype=np.bool8)
        self.root_node = mcts.Node(player_id=0)

    def test_node_basics(self):
        root_node = self.root_node

        self.assertEqual(root_node.N, 0)
        self.assertEqual(root_node.W, 0)
        self.assertEqual(root_node.Q, 0)
        self.assertEqual(root_node.move, None)
        self.assertEqual(root_node.prior, None)
        self.assertFalse(root_node.is_expanded)
        self.assertFalse(root_node.has_parent)

    def test_expand_node(self):
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 0)

        self.assertEqual(len(root_node.children), self.prior.shape[0])
        self.assertTrue(root_node.is_expanded)

        for i in range(len(self.prior)):
            child_node = root_node.children[i]
            self.assertEqual(child_node.N, 0)
            self.assertEqual(child_node.W, 0)
            self.assertEqual(child_node.Q, 0)
            self.assertEqual(child_node.prior, self.prior[i])
            self.assertFalse(child_node.is_expanded)
            self.assertEqual(child_node.parent, root_node)

    def test_node_expand_invalid_type(self):
        dummy = [0.1, 3.3, {}, []]
        root_node = mcts.Node(player_id=0)
        with self.assertRaisesRegex(ValueError, 'Expect'):
            root_node.expand(dummy, 0)

    def test_expand_node_already_expanded(self):
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 0)

        with self.assertRaisesRegex(RuntimeError, 'Node already expanded'):
            root_node.expand(self.prior, 0)

    def test_node_best_child_on_leaf_node(self):
        root_node = mcts.Node(player_id=0)
        with self.assertRaisesRegex(ValueError, 'Expand leaf node first'):
            root_node.best_child(self.actions_mask, 0.0)

    @parameterized.named_parameters(('c_puct_0.1', 0.1), ('c_puct_0.5', 0.5), ('c_puct_1', 1.0))
    def test_node_best_child_high_priorabilities(self, c_puct):
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 1)
        root_node.backup({0: 0.8, 1: -0.8})
        best_node = root_node.best_child(self.actions_mask, c_puct)
        self.assertEqual(best_node, root_node.children[1])

    @parameterized.named_parameters(('c_puct_0.1', 0.1), ('c_puct_0.5', 0.5), ('c_puct_1', 1.0))
    def test_node_best_child_high_action_values(self, c_puct):
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 1)
        root_node.backup({0: 0.8, 1: -0.4})

        for _ in range(5):
            root_node.children[0].backup({0: 0.8, 1: -0.4})
            root_node.children[1].backup({0: 0.6, 1: -0.5})
            root_node.children[2].backup({0: 0.5, 1: -0.6})
            root_node.children[3].backup({0: 0.4, 1: -0.7})

        best_node = root_node.best_child(self.actions_mask, c_puct)
        # Child node Q is the opposite.
        self.assertEqual(best_node, root_node.children[3])


class MCTSUpdateStatisticsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.num_actions = 4
        self.prior = np.array([0.2, 0.3, 0.25, 0.15])

    @parameterized.named_parameters(('black_player', 0), ('white_player', 1))
    def test_update_statistics_root_node(self, player_id):
        root_node = mcts.Node(player_id=player_id)
        root_node.expand(self.prior, player_id)
        values = {0: 0.5, 1: -0.5}
        root_node.backup(values)

        self.assertEqual(root_node.N, 1)
        self.assertEqual(root_node.W, values[player_id])
        self.assertEqual(root_node.Q, values[player_id])

    def test_update_statistics_invalid_score_type(self):
        root_node = mcts.Node(player_id=0)
        scores = [12, 0.5, 'ss', [], None]
        for score in scores:
            with self.assertRaisesRegex(ValueError, 'Expect'):
                root_node.backup(score)

    def test_update_statistics_level_1_mixed(self):
        # Black player to move
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 0)

        # Expand level 1 children nodes
        # White player to move
        for child in root_node.children:
            child.expand(self.prior, 1)

        child_0 = root_node.children[0]
        child_1 = root_node.children[1]
        child_2 = child_0.children[0]
        child_3 = child_0.children[1]

        black_values = [0.5, 0.6]
        # Black player score is positive.
        for v, child in zip(black_values, [child_0, child_1]):
            v_dict = {0: v, 1: -v}
            child.backup(v_dict)

        white_values = [1, 2]
        # White player score is positive.
        for v, child in zip(white_values, [child_2, child_3]):
            v_dict = {0: -v, 1: v}
            child.backup(v_dict)

        self.assertEqual(root_node.N, 4)
        self.assertAlmostEqual(root_node.W, 0.5 + 0.6 - 1 - 2, places=8)
        self.assertAlmostEqual(root_node.Q, (0.5 + 0.6 - 1 - 2) / 4, places=8)

        self.assertEqual(child_2.N, 1)
        self.assertAlmostEqual(child_2.W, 1, places=8)
        self.assertAlmostEqual(child_2.Q, 1, places=8)

        self.assertEqual(child_3.N, 1)
        self.assertAlmostEqual(child_3.W, 2, places=8)
        self.assertAlmostEqual(child_3.Q, 2, places=8)

    def test_update_statistics_level_2(self):
        # Black player to move
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 0)

        # Expand level 1 children nodes
        # White player to move
        for child in root_node.children:
            child.expand(self.prior, 1)

        # Expand level 2 children nodes
        # Black player to move
        for child in root_node.children:
            for c in child.children:
                c.expand(self.prior, 0)

        child_0 = root_node.children[0]
        child_1 = child_0.children[0]
        child_2 = child_1.children[0]
        child_3 = child_1.children[1]

        # First black move
        v_dict = {0: 0.6, 1: -0.6}
        child_0.backup(v_dict)

        # First white move
        v_dict = {0: -0.8, 1: 0.8}
        child_1.backup(v_dict)

        # Second black moves
        v_dict = {0: 1, 1: -1}
        child_2.backup(v_dict)
        v_dict = {0: 2, 1: -2}
        child_3.backup(v_dict)

        self.assertEqual(root_node.N, 4)
        self.assertAlmostEqual(root_node.W, 0.6 - 0.8 + 1 + 2, places=8)
        self.assertAlmostEqual(root_node.Q, (0.6 - 0.8 + 1 + 2) / 4, places=8)


class MCTSGeneratePlayPolicyTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.prior = np.array([0.2, 0.3, 0.25, 0.15])
        self.actions_mask = np.ones_like(self.prior, dtype=np.bool8)

    def test_play_policy_error_on_leaf_node(self):
        root_node = mcts.Node(player_id=0)
        with self.assertRaisesRegex(ValueError, 'Expect'):
            mcts.generate_play_policy(root_node.child_N, 0.1)

    def test_play_policy_root_node_greedy_equal_prob(self):
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 0)

        # Make sure each child is visited once
        for node in root_node.children:
            node.backup({0: 0.02, 1: -0.02})

        pi_prob = mcts.generate_play_policy(root_node.child_N, 0.1)
        visits = np.array([1, 1, 1, 1])
        expected_prob = visits / np.sum(visits)
        np.testing.assert_allclose(pi_prob, expected_prob)

    def test_play_policy_prob_sums_1(self):
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 0)

        # Make sure each child is visited once
        for node in root_node.children:
            node.backup({0: 0.02, 1: -0.02})

        pi_prob = mcts.generate_play_policy(root_node.child_N, 0.1)
        np.testing.assert_allclose(np.sum(pi_prob), np.ones(1))

    def test_play_policy_root_node_greedy_no_equal_prob(self):
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 0)

        # Make sure each child is visited once
        for node in root_node.children:
            node.backup({0: 0.02, 1: -0.02})

        child = root_node.children[1]
        child.backup({0: 0.02, 1: -0.02})

        pi_prob = mcts.generate_play_policy(root_node.child_N, 0.1)
        visits = np.array([1, 2, 1, 1], dtype=np.float64)
        exp = 5  # limit max to 5
        visits = visits**exp
        expected_prob = visits / np.sum(visits)
        np.testing.assert_allclose(pi_prob, expected_prob, atol=1e-6)

    def test_play_policy_root_node_exploration_equal_prob(self):
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 0)

        # Make sure each child is visited once
        for node in root_node.children:
            node.backup({0: 0.02, 1: -0.02})

        pi_prob = mcts.generate_play_policy(root_node.child_N, 1.0)
        visits = np.array([1, 1, 1, 1])
        expected_prob = visits / np.sum(visits)
        np.testing.assert_allclose(pi_prob, expected_prob)

    @parameterized.named_parameters(('temp_1e-4', 1e-4), ('temp_1e-10', 1e-10))
    def test_play_policy_invalid_temp(self, tmp):
        root_node = mcts.Node(player_id=0)
        root_node.expand(self.prior, 0)

        for i in range(100):
            for node in root_node.children:
                node.backup({0: 0.02, 1: -0.02})

        with self.assertRaisesRegex(ValueError, 'Expect'):
            pi_prob = mcts.generate_play_policy(root_node.child_N, tmp)


def mock_eval_func(state_tensor):
    # Mock network output
    num_actions = state_tensor.shape[-1] ** 2 + 1
    prior_prob = np.random.random(size=(num_actions,))

    prior_prob /= np.sum(prior_prob)

    v = np.array(np.random.uniform(-1, 1))
    return (prior_prob, v)


class UCTSearchTest(parameterized.TestCase):
    def test_run_uct_search(self):
        env = GomokuEnv(board_size=7)
        obs = env.reset()
        root_node = None
        while env.steps < 10:
            action, pi_prob, root_node = mcts.uct_search(env, mock_eval_func, root_node, 5.0, 1.0, 100)
            obs, reward, done, info = env.step(action)
            if done:
                break


if __name__ == '__main__':
    absltest.main()
