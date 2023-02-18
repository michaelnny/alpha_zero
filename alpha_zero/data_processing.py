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
    pi_prob = torch.reshape(pi_prob, (-1, 1, board_size, board_size))
    rotated_pi_prob = rotate(pi_prob, angle)

    rotated_pi_prob = rotated_pi_prob.view(-1, board_size * board_size)

    return rotated_state, rotated_pi_prob


def mirror_horizontal(state: torch.Tensor, pi_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns horizontal flipped 'mirror' of state and action probabilities."""

    if not isinstance(state, torch.Tensor) or len(state.shape) != 4:
        raise ValueError(f'Expect state to be a 4D torch.Tensor, got {state}')
    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 2:
        raise ValueError(f'Expect pi_prob to be a 2D torch.Tensor, got {pi_prob}')

    fliped_state = hflip(state)

    board_size = state.shape[-1]
    pi_prob = torch.reshape(pi_prob, (-1, 1, board_size, board_size))
    fliped_pi_prob = hflip(pi_prob)

    fliped_pi_prob = fliped_pi_prob.view(-1, board_size * board_size)

    return fliped_state, fliped_pi_prob


def mirror_vertical(state: torch.Tensor, pi_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns vertical flipped 'mirror' of state and action probabilities."""

    if not isinstance(state, torch.Tensor) or len(state.shape) != 4:
        raise ValueError(f'Expect state to be a 4D torch.Tensor, got {state}')
    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 2:
        raise ValueError(f'Expect pi_prob to be a 2D torch.Tensor, got {pi_prob}')

    fliped_state = vflip(state)

    board_size = state.shape[-1]
    pi_prob = torch.reshape(pi_prob, (-1, 1, board_size, board_size))
    fliped_pi_prob = vflip(pi_prob)

    fliped_pi_prob = fliped_pi_prob.view(-1, board_size * board_size)

    return fliped_state, fliped_pi_prob


def random_rotation_and_reflection(state: torch.Tensor, pi_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns randomly rotated, mirrored, and switched player perspective samples."""

    # Apply random (counter clockwise) rotation.
    angles = [90, 180, 270]
    if np.random.rand() > 0.5:
        angle = int(np.random.choice(angles))
        state, pi_prob = rotate_state_and_prob(state, pi_prob, angle)

    # Mirroring horizontally
    if np.random.rand() > 0.5:
        state, pi_prob = mirror_horizontal(state, pi_prob)

    # Mirroring vertically
    if np.random.rand() > 0.5:
        state, pi_prob = mirror_vertical(state, pi_prob)

    return state, pi_prob
