# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from absl.testing import absltest
import torch
from alpha_zero.utils.transformation import (
    apply_horizontal_flip,
    apply_vertical_flip,
    apply_rotation,
    probs_to_3d,
    flatten_probs,
)


class TestUtilityFunctions(absltest.TestCase):
    def test_probs_to_3d(self):
        x = torch.randn(2, 19 * 19)
        board_size = 19
        expected_output = torch.reshape(x, (-1, 1, board_size, board_size))
        output = probs_to_3d(x, board_size)
        self.assertEqual(output.shape, expected_output.shape)
        self.assertTrue(torch.all(torch.eq(output, expected_output)))

    def test_flatten_probs(self):
        # Define test inputs
        board_size = 3
        x = torch.randn(2, board_size, board_size)

        # Execute function
        x_flat = flatten_probs(x, board_size)

        # Compute expected output
        expected_x_flat = torch.reshape(x, (2, -1))

        # Check output is correct
        self.assertTrue(torch.all(torch.eq(x_flat, expected_x_flat)))


class TestHorizontalFlip(absltest.TestCase):
    def test_apply_horizontal_flip_valid_input(self):
        # Test with valid inputs
        states = torch.randn(2, 3, 19, 19)
        pi_probs = torch.randn(2, 361)
        states_out, pi_probs_out = apply_horizontal_flip(states, pi_probs)
        self.assertTrue(states_out.shape == (2, 3, 19, 19))
        self.assertTrue(pi_probs_out.shape == (2, 361))

    def test_apply_horizontal_flip_invalid_input(self):
        # Test with invalid input
        states = torch.randn(2, 3, 19)
        pi_probs = torch.randn(2, 19, 19)
        with self.assertRaisesRegex(ValueError, 'Expect'):
            states_out, pi_probs_out = apply_horizontal_flip(states, pi_probs)

    def test_apply_horizontal_flip_no_pass_move(self):
        # Test without pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size)
        states_out, pi_probs_out = apply_horizontal_flip(states, pi_probs)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.flip(expected_state, dims=[2])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs = torch.reshape(expected_pi_probs, (1, board_size, board_size))
            expected_pi_probs = torch.flip(expected_pi_probs, dims=[2])
            expected_pi_probs = torch.reshape(expected_pi_probs, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))

    def test_apply_horizontal_flip_pass_move(self):
        # Test with pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size + 1)

        states_out, pi_probs_out = apply_horizontal_flip(states, pi_probs)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.flip(expected_state, dims=[2])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs_no_pass = torch.reshape(expected_pi_probs[:-1], (1, board_size, board_size))
            expected_pi_probs_no_pass = torch.flip(expected_pi_probs_no_pass, dims=[2])
            expected_pi_probs[:-1] = torch.reshape(expected_pi_probs_no_pass, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))


class TestVerticalFlip(absltest.TestCase):
    def test_apply_vertical_flip_valid_input(self):
        # Test with valid inputs
        states = torch.randn(2, 3, 19, 19)
        pi_probs = torch.randn(2, 361)
        states_out, pi_probs_out = apply_vertical_flip(states, pi_probs)
        self.assertTrue(states_out.shape == (2, 3, 19, 19))
        self.assertTrue(pi_probs_out.shape == (2, 361))

    def test_apply_vertical_flip_invalid_input(self):
        # Test with invalid input
        states = torch.randn(2, 3, 19)
        pi_probs = torch.randn(2, 19, 19)
        with self.assertRaisesRegex(ValueError, 'Expect'):
            states_out, pi_probs_out = apply_vertical_flip(states, pi_probs)

    def test_apply_vertical_flip_no_pass_move(self):
        # Test without pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size)
        states_out, pi_probs_out = apply_vertical_flip(states, pi_probs)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.flip(expected_state, dims=[1])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs = torch.reshape(expected_pi_probs, (1, board_size, board_size))
            expected_pi_probs = torch.flip(expected_pi_probs, dims=[1])
            expected_pi_probs = torch.reshape(expected_pi_probs, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))

    def test_apply_vertical_flip_pass_move(self):
        # Test with pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size + 1)

        states_out, pi_probs_out = apply_vertical_flip(states, pi_probs)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.flip(expected_state, dims=[1])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs_no_pass = torch.reshape(expected_pi_probs[:-1], (1, board_size, board_size))
            expected_pi_probs_no_pass = torch.flip(expected_pi_probs_no_pass, dims=[1])
            expected_pi_probs[:-1] = torch.reshape(expected_pi_probs_no_pass, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))


class TestRotation(absltest.TestCase):
    def test_apply_rotation_valid_input(self):
        # Test with valid inputs
        states = torch.randn(2, 3, 19, 19)
        pi_probs = torch.randn(2, 361)
        states_out, pi_probs_out = apply_rotation(states, pi_probs, 90)
        self.assertTrue(states_out.shape == (2, 3, 19, 19))
        self.assertTrue(pi_probs_out.shape == (2, 361))

    def test_apply_rotation_invalid_input(self):
        # Test with invalid input
        states = torch.randn(2, 3, 19)
        pi_probs = torch.randn(2, 19, 19)
        with self.assertRaisesRegex(ValueError, 'Expect'):
            states_out, pi_probs_out = apply_rotation(states, pi_probs, 90)

        for angle in [30, 46, 220]:
            with self.assertRaisesRegex(ValueError, 'Expect'):
                states_out, pi_probs_out = apply_rotation(states, pi_probs, angle)

    def test_apply_rotation_90_no_pass_move(self):
        # Test without pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size)
        states_out, pi_probs_out = apply_rotation(states, pi_probs, 90)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.rot90(expected_state, k=1, dims=[1, 2])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs = torch.reshape(expected_pi_probs, (1, board_size, board_size))
            expected_pi_probs = torch.rot90(expected_pi_probs, k=1, dims=[1, 2])
            expected_pi_probs = torch.reshape(expected_pi_probs, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))

    def test_apply_rotation_180_no_pass_move(self):
        # Test without pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size)
        states_out, pi_probs_out = apply_rotation(states, pi_probs, 180)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.rot90(expected_state, k=2, dims=[1, 2])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs = torch.reshape(expected_pi_probs, (1, board_size, board_size))
            expected_pi_probs = torch.rot90(expected_pi_probs, k=2, dims=[1, 2])
            expected_pi_probs = torch.reshape(expected_pi_probs, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))

    def test_apply_rotation_270_no_pass_move(self):
        # Test without pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size)
        states_out, pi_probs_out = apply_rotation(states, pi_probs, 270)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.rot90(expected_state, k=3, dims=[1, 2])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs = torch.reshape(expected_pi_probs, (1, board_size, board_size))
            expected_pi_probs = torch.rot90(expected_pi_probs, k=3, dims=[1, 2])
            expected_pi_probs = torch.reshape(expected_pi_probs, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))

    def test_apply_rotation_90_with_pass_move_small_case(self):
        # Test with pass move
        states = torch.tensor(
            [
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                ],
                [
                    [
                        [3, 6, 9],
                        [2, 5, 8],
                        [1, 4, 7],
                    ]
                ],
            ]
        )
        expected_states = torch.tensor(
            [
                [
                    [
                        [3, 6, 9],
                        [2, 5, 8],
                        [1, 4, 7],
                    ]
                ],
                [
                    [
                        [9, 8, 7],
                        [6, 5, 4],
                        [3, 2, 1],
                    ]
                ],
            ]
        )
        pi_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.001]])
        expected_pi_probs = torch.tensor([[0.3, 0.6, 0.9, 0.2, 0.5, 0.8, 0.1, 0.4, 0.7, 0.001]])
        states_out, pi_probs_out = apply_rotation(states, pi_probs, 90)

        self.assertTrue(torch.all(torch.eq(states_out, expected_states)))
        self.assertTrue(torch.all(torch.eq(pi_probs_out, expected_pi_probs)))

    def test_apply_rotation_90_with_pass_move(self):
        # Test with pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size + 1)

        states_out, pi_probs_out = apply_rotation(states, pi_probs, 90)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.rot90(expected_state, k=1, dims=[1, 2])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs_no_pass = torch.reshape(expected_pi_probs[:-1], (1, board_size, board_size))
            expected_pi_probs_no_pass = torch.rot90(expected_pi_probs_no_pass, k=1, dims=[1, 2])
            expected_pi_probs[:-1] = torch.reshape(expected_pi_probs_no_pass, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))

    def test_apply_rotation_180_with_pass_move(self):
        # Test with pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size + 1)

        states_out, pi_probs_out = apply_rotation(states, pi_probs, 180)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.rot90(expected_state, k=2, dims=[1, 2])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs_no_pass = torch.reshape(expected_pi_probs[:-1], (1, board_size, board_size))
            expected_pi_probs_no_pass = torch.rot90(expected_pi_probs_no_pass, k=2, dims=[1, 2])
            expected_pi_probs[:-1] = torch.reshape(expected_pi_probs_no_pass, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))

    def test_apply_rotation_270_with_pass_move(self):
        # Test with pass move
        board_size = 19
        states = torch.randn(2, 3, board_size, board_size)
        pi_probs = torch.randn(2, board_size * board_size + 1)

        states_out, pi_probs_out = apply_rotation(states, pi_probs, 270)

        # Check all elements of the output tensors match the expected values
        for i in range(states.shape[0]):
            expected_state = states[i, ...]
            expected_state = torch.rot90(expected_state, k=3, dims=[1, 2])
            self.assertTrue(torch.all(torch.eq(states_out[i, ...], expected_state)))

            expected_pi_probs = pi_probs[i, ...]
            expected_pi_probs_no_pass = torch.reshape(expected_pi_probs[:-1], (1, board_size, board_size))
            expected_pi_probs_no_pass = torch.rot90(expected_pi_probs_no_pass, k=3, dims=[1, 2])
            expected_pi_probs[:-1] = torch.reshape(expected_pi_probs_no_pass, (-1,))
            self.assertTrue(torch.all(torch.eq(pi_probs_out[i, ...], expected_pi_probs)))


if __name__ == '__main__':
    absltest.main()
