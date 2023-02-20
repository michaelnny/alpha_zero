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
from typing import Tuple
import numpy as np
import torch
from torchvision.transforms.functional import rotate, hflip, vflip


def flat_probs_to_3d_grid(pi_prob, board_size):
    return torch.reshape(pi_prob, (-1, 1, board_size, board_size))


def flatten_3d_grid_probs(pi_prob, board_size):
    return pi_prob.view(-1, board_size * board_size)


def rotate_state_and_prob(state: torch.Tensor, pi_prob: torch.Tensor, angle: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns rotated state and action probabilities."""

    if not isinstance(state, torch.Tensor) or len(state.shape) != 4:
        raise ValueError(f'Expect state to be a 4D torch.Tensor, got {state}')
    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 2:
        raise ValueError(f'Expect pi_prob to be a 2D torch.Tensor, got {pi_prob}')
    if not isinstance(angle, int) or angle % 90 != 0:
        raise ValueError(f'Expect angle to be a integer and divisible by 90, got {angle}')

    # Rotate the state tensor
    rotated_state = rotate(state, angle)

    # Rotate the action probabilities, using the same angle.
    board_size = state.shape[-1]

    pi_prob = flat_probs_to_3d_grid(pi_prob, board_size)
    rotated_pi_prob = rotate(pi_prob, angle)

    rotated_pi_prob = flatten_3d_grid_probs(rotated_pi_prob, board_size)
    return rotated_state, rotated_pi_prob


def mirror_horizontal(state: torch.Tensor, pi_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns horizontal flipped 'mirror' of state and action probabilities."""

    if not isinstance(state, torch.Tensor) or len(state.shape) != 4:
        raise ValueError(f'Expect state to be a 4D torch.Tensor, got {state}')
    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 2:
        raise ValueError(f'Expect pi_prob to be a 2D torch.Tensor, got {pi_prob}')

    fliped_state = hflip(state)

    board_size = state.shape[-1]
    pi_prob = flat_probs_to_3d_grid(pi_prob, board_size)
    fliped_pi_prob = hflip(pi_prob)

    fliped_pi_prob = flatten_3d_grid_probs(fliped_pi_prob, board_size)

    return fliped_state, fliped_pi_prob


def mirror_vertical(state: torch.Tensor, pi_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns vertical flipped 'mirror' of state and action probabilities."""

    if not isinstance(state, torch.Tensor) or len(state.shape) != 4:
        raise ValueError(f'Expect state to be a 4D torch.Tensor, got {state}')
    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 2:
        raise ValueError(f'Expect pi_prob to be a 2D torch.Tensor, got {pi_prob}')

    fliped_state = vflip(state)

    board_size = state.shape[-1]
    pi_prob = flat_probs_to_3d_grid(pi_prob, board_size)
    fliped_pi_prob = vflip(pi_prob)

    fliped_pi_prob = flatten_3d_grid_probs(fliped_pi_prob, board_size)

    return fliped_state, fliped_pi_prob


def random_rotation_and_reflection(state: torch.Tensor, pi_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly rotate and mirror batch of samples."""

    rnd = np.random.rand()

    if 0.4 < rnd and rnd <= 0.6:
        # Apply random (counter clockwise) rotation.
        angle = int(np.random.choice([90, 180, 270]))
        state, pi_prob = rotate_state_and_prob(state, pi_prob, angle)
    elif 0.6 < rnd and rnd <= 0.8:
        # Mirroring horizontally
        state, pi_prob = mirror_horizontal(state, pi_prob)
    elif 0.8 < rnd:
        # Mirroring vertically
        state, pi_prob = mirror_vertical(state, pi_prob)

    return state, pi_prob
