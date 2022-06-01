"""Tests for data_processing.py"""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from alpha_zero.data_processing import rotate_state_and_prob, mirror_horizonral, mirror_vertical


class PipelineDataArgumentationTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()

        self.board_size = 4
        self.state = torch.tensor(
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
            ]
        )

        self.pi_prob_2d = torch.tensor(
            [
                [0.0, 0.0, 0.1, 0.0],
                [0.4, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.0],
                [0.0, 0.2, 0.0, 0.0],
            ]
        )

        self.pi_prob_last_action = torch.tensor([[0.0]])

        self.pi_prob = torch.concat(
            [self.pi_prob_2d.view(1, self.board_size * self.board_size), self.pi_prob_last_action], dim=1
        )

        self.z = torch.tensor([1.0])

    def test_run_rotate_90(self):
        rotated_state, rotated_pi_prob = rotate_state_and_prob(self.state[None, ...], self.pi_prob, 90)

        expected_state = np.rot90(self.state.numpy(), 1, (1, 2))
        expected_pi_prob = np.rot90(self.pi_prob_2d.numpy(), 1, (0, 1))
        expected_pi_prob = expected_pi_prob.reshape((-1, self.board_size * self.board_size))
        expected_pi_prob = np.concatenate([expected_pi_prob, self.pi_prob_last_action.numpy()], axis=1)

        np.testing.assert_equal(rotated_state.squeeze(0).numpy(), expected_state)
        np.testing.assert_equal(rotated_pi_prob.numpy(), expected_pi_prob)

    def test_run_rotate_180(self):
        rotated_state, rotated_pi_prob = rotate_state_and_prob(self.state[None, ...], self.pi_prob, 180)

        expected_state = np.rot90(self.state.numpy(), 2, (1, 2))
        expected_pi_prob = np.rot90(self.pi_prob_2d.numpy(), 2, (0, 1))
        expected_pi_prob = expected_pi_prob.reshape((-1, self.board_size * self.board_size))
        expected_pi_prob = np.concatenate([expected_pi_prob, self.pi_prob_last_action.numpy()], axis=1)

        np.testing.assert_equal(rotated_state.squeeze(0).numpy(), expected_state)
        np.testing.assert_equal(rotated_pi_prob.numpy(), expected_pi_prob)

    def test_run_rotate_270(self):
        rotated_state, rotated_pi_prob = rotate_state_and_prob(self.state[None, ...], self.pi_prob, 270)

        expected_state = np.rot90(self.state.numpy(), 3, (1, 2))
        expected_pi_prob = np.rot90(self.pi_prob_2d.numpy(), 3, (0, 1))
        expected_pi_prob = expected_pi_prob.reshape((-1, self.board_size * self.board_size))
        expected_pi_prob = np.concatenate([expected_pi_prob, self.pi_prob_last_action.numpy()], axis=1)

        np.testing.assert_equal(rotated_state.squeeze(0).numpy(), expected_state)
        np.testing.assert_equal(rotated_pi_prob.numpy(), expected_pi_prob)

    def test_run_rotate_360(self):
        rotated_state, rotated_pi_prob = rotate_state_and_prob(self.state[None, ...], self.pi_prob, 360)
        np.testing.assert_equal(rotated_state.squeeze(0).numpy(), self.state.numpy())
        np.testing.assert_equal(rotated_pi_prob.numpy(), self.pi_prob.numpy())

    def test_mirror_horizontal(self):
        flipped_state, flipped_pi_prob = mirror_horizonral(self.state[None, ...], self.pi_prob)

        expected_state = np.stack([np.flip(s, axis=1) for s in self.state.numpy()], axis=0)
        expected_pi_prob = np.flip(self.pi_prob_2d.numpy(), axis=1)
        expected_pi_prob = expected_pi_prob.reshape((-1, self.board_size * self.board_size))
        expected_pi_prob = np.concatenate([expected_pi_prob, self.pi_prob_last_action.numpy()], axis=1)

        np.testing.assert_equal(flipped_state.squeeze(0).numpy(), expected_state)
        np.testing.assert_equal(flipped_pi_prob.numpy(), expected_pi_prob)

    def test_mirror_vertical(self):
        flipped_state, flipped_pi_prob = mirror_vertical(self.state[None, ...], self.pi_prob)

        expected_state = np.stack([np.flip(s, axis=0) for s in self.state.numpy()], axis=0)
        expected_pi_prob = np.flip(self.pi_prob_2d.numpy(), axis=0)
        expected_pi_prob = expected_pi_prob.reshape((-1, self.board_size * self.board_size))
        expected_pi_prob = np.concatenate([expected_pi_prob, self.pi_prob_last_action.numpy()], axis=1)

        np.testing.assert_equal(flipped_state.squeeze(0).numpy(), expected_state)
        np.testing.assert_equal(flipped_pi_prob.numpy(), expected_pi_prob)


if __name__ == '__main__':
    absltest.main()
