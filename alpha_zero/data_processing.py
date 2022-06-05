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


def reshape_action_probs(pi_prob: torch.Tensor, board_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns reshaped action probabilities in the shape of (batch_size, 1, board_size, board_dize)
    and the extracted last action (resign action).
    """
    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 2:
        raise ValueError(f'Expect pi_prob to be a 2D torch.Tensor, got {pi_prob}')
    if not isinstance(board_size, int) or board_size < 1:
        raise ValueError(f'Expect board_size to be a positive integer, got {board_size}')

    # First, extract the 'resign' which is the last action.
    last_actions = torch.unsqueeze(pi_prob[:, -1], dim=1)

    # Next, convert flat action probabilities into 'image' like shape,
    # so we can rotate or flip operations on it.
    pi_prob = torch.reshape(pi_prob[:, 0:-1], (-1, 1, board_size, board_size))

    return pi_prob, last_actions


def flat_and_conct_action_probs(pi_prob: torch.Tensor, last_actions: torch.Tensor, board_size: int) -> torch.Tensor:
    """Returns flattend and concated action probabilities with extracted last actions."""

    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 4:
        raise ValueError(f'Expect pi_prob to be a 4D torch.Tensor, got {pi_prob}')
    if not isinstance(last_actions, torch.Tensor) or len(last_actions.shape) != 2:
        raise ValueError(f'Expect last_actions to be a 2D torch.Tensor, got {last_actions}')
    if not isinstance(board_size, int) or board_size < 1:
        raise ValueError(f'Expect board_size to be a positive integer, got {board_size}')

    return torch.concat([pi_prob.view(-1, board_size * board_size), last_actions], dim=1)


def rotate_state_and_prob(state: torch.Tensor, pi_prob: torch.Tensor, angle: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns rotated state and action probabilities."""

    if not isinstance(state, torch.Tensor) or len(state.shape) != 4:
        raise ValueError(f'Expect state to be a 4D torch.Tensor, got {state}')
    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 2:
        raise ValueError(f'Expect pi_prob to be a 2D torch.Tensor, got {pi_prob}')
    if not isinstance(angle, int) or angle % 90 != 0:
        raise ValueError(f'Expect angle to be a integer and divisible by 90, got {angle}')

    board_size = state.shape[-1]

    # Rotate the state tensor is easy.
    rotated_state = rotate(state, angle)

    # To rotate the action probabilities is more complex.
    # Since we have board_size * board_size + 1 (resign) actions.
    # And the original pi_prob is a 1D vector (2D if batched).

    # Extract the 'resign' action, and reshape into 'image' like shape.
    pi_prob, last_actions = reshape_action_probs(pi_prob, board_size)

    # Now we can rotate the action probabilities, using the same angle.
    rotated_pi_prob = rotate(pi_prob, angle)

    # Finally, we flatten the action probabilities and add the 'resign' action back.
    rotated_pi_prob = flat_and_conct_action_probs(rotated_pi_prob, last_actions, board_size)

    return rotated_state, rotated_pi_prob


def mirror_horizonral(state: torch.Tensor, pi_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns horizontal flipped 'mirror' of state and action probabilities."""

    if not isinstance(state, torch.Tensor) or len(state.shape) != 4:
        raise ValueError(f'Expect state to be a 4D torch.Tensor, got {state}')
    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 2:
        raise ValueError(f'Expect pi_prob to be a 2D torch.Tensor, got {pi_prob}')

    board_size = state.shape[-1]
    fliped_state = hflip(state)

    pi_prob, last_actions = reshape_action_probs(pi_prob, board_size)
    fliped_pi_prob = hflip(pi_prob)
    fliped_pi_prob = flat_and_conct_action_probs(fliped_pi_prob, last_actions, board_size)
    return fliped_state, fliped_pi_prob


def mirror_vertical(state: torch.Tensor, pi_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns vertical flipped 'mirror' of state and action probabilities."""

    if not isinstance(state, torch.Tensor) or len(state.shape) != 4:
        raise ValueError(f'Expect state to be a 4D torch.Tensor, got {state}')
    if not isinstance(pi_prob, torch.Tensor) or len(pi_prob.shape) != 2:
        raise ValueError(f'Expect pi_prob to be a 2D torch.Tensor, got {pi_prob}')

    board_size = state.shape[-1]
    fliped_state = vflip(state)

    pi_prob, last_actions = reshape_action_probs(pi_prob, board_size)
    fliped_pi_prob = vflip(pi_prob)
    fliped_pi_prob = flat_and_conct_action_probs(fliped_pi_prob, last_actions, board_size)
    return fliped_state, fliped_pi_prob


def random_rotation_and_reflection(state: torch.Tensor, pi_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns randomly rotated, mirrored, and switched player perspective samples."""

    # Apply random (counter clockwise) rotation.
    if np.random.rand() > 0.5:
        angles = [90, 180, 270]
        angle = int(np.random.choice(angles))
        state, pi_prob = rotate_state_and_prob(state, pi_prob, angle)

    # Mirroring horizontaly
    if np.random.rand() > 0.5:
        state, pi_prob = mirror_horizonral(state, pi_prob)

    # Mirroring vertically
    if np.random.rand() > 0.5:
        state, pi_prob = mirror_vertical(state, pi_prob)

    return state, pi_prob
