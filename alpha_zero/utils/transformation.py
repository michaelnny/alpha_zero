# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import Tuple
import random
import torch
from torchvision.transforms.functional import rotate, hflip, vflip


def probs_to_3d(x, board_size):
    return torch.reshape(x, (-1, 1, board_size, board_size))


def flatten_probs(x, board_size):
    return torch.reshape(x, (-1, board_size * board_size))


def extract_pi_probs(pi_probs, board_size):
    has_pass_move = pi_probs.shape[-1] == (board_size**2) + 1
    pass_move_prob = None

    if has_pass_move:
        pass_move_prob = pi_probs[..., -1:]  # Extract pass move probability
        pi_probs_3d = torch.reshape(pi_probs[..., :-1], (-1, 1, board_size, board_size))
    else:
        pi_probs_3d = torch.reshape(pi_probs, (-1, 1, board_size, board_size))

    return pi_probs_3d, pass_move_prob


def apply_horizontal_flip(states: torch.Tensor, pi_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns horizontal flipped 'mirror' of state and action probabilities."""

    if not isinstance(states, torch.Tensor) or len(states.shape) != 4:
        raise ValueError(f'Expect states to be a 4D torch.Tensor, got {states}')
    if not isinstance(pi_probs, torch.Tensor) or len(pi_probs.shape) != 2:
        raise ValueError(f'Expect pi_probs to be a 2D torch.Tensor, got {pi_probs}')

    states = torch.clone(states)  # [B, C, H, W]
    pi_probs = torch.clone(pi_probs)  # [B, H*W] or [B, H*W+1]
    board_size = states.shape[-1]

    states = hflip(states)

    pi_probs_3d, pass_move_prob = extract_pi_probs(pi_probs, board_size)

    pi_probs_3d = hflip(pi_probs_3d)
    pi_probs = flatten_probs(pi_probs_3d, board_size)

    if pass_move_prob is not None:
        pi_probs = torch.cat((pi_probs, pass_move_prob), dim=-1)

    return states, pi_probs


def apply_vertical_flip(states: torch.Tensor, pi_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns vertical flipped 'mirror' of state and action probabilities."""

    if not isinstance(states, torch.Tensor) or len(states.shape) != 4:
        raise ValueError(f'Expect states to be a 4D torch.Tensor, got {states}')
    if not isinstance(pi_probs, torch.Tensor) or len(pi_probs.shape) != 2:
        raise ValueError(f'Expect pi_probs to be a 2D torch.Tensor, got {pi_probs}')

    states = torch.clone(states)  # [B, C, H, W]
    pi_probs = torch.clone(pi_probs)  # [B, H*W] or [B, H*W+1]
    board_size = states.shape[-1]

    states = vflip(states)

    pi_probs_3d, pass_move_prob = extract_pi_probs(pi_probs, board_size)

    pi_probs_3d = vflip(pi_probs_3d)
    pi_probs = flatten_probs(pi_probs_3d, board_size)

    if pass_move_prob is not None:
        pi_probs = torch.cat((pi_probs, pass_move_prob), dim=-1)

    return states, pi_probs


def apply_rotation(states: torch.Tensor, pi_probs: torch.Tensor, angle: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns rotated state and action probabilities."""

    if not isinstance(states, torch.Tensor) or len(states.shape) != 4:
        raise ValueError(f'Expect states to be a 4D torch.Tensor, got {states}')
    if not isinstance(pi_probs, torch.Tensor) or len(pi_probs.shape) != 2:
        raise ValueError(f'Expect pi_probs to be a 2D torch.Tensor, got {pi_probs}')
    if angle not in [90, 180, 270]:
        raise ValueError(f'Expect angle to be one of [90, 180, 270], got {angle}')

    states = torch.clone(states)  # [B, C, H, W]
    pi_probs = torch.clone(pi_probs)  # [B, H*W] or [B, H*W+1]
    board_size = states.shape[-1]

    # Rotate the state tensor
    states = rotate(states, angle)

    # Rotate the action probabilities, using the same angle.
    pi_probs_3d, pass_move_prob = extract_pi_probs(pi_probs, board_size)

    pi_probs_3d = rotate(pi_probs_3d, angle)
    pi_probs = flatten_probs(pi_probs_3d, board_size)

    if pass_move_prob is not None:
        pi_probs = torch.cat((pi_probs, pass_move_prob), dim=-1)

    return states, pi_probs


def rotate_90(
    states: torch.Tensor, pi_probs: torch.Tensor, values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, pi_probs = apply_rotation(states, pi_probs, 90)
    return states, pi_probs, values


def rotate_180(
    states: torch.Tensor, pi_probs: torch.Tensor, values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, pi_probs = apply_rotation(states, pi_probs, 180)
    return states, pi_probs, values


def rotate_270(
    states: torch.Tensor, pi_probs: torch.Tensor, values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, pi_probs = apply_rotation(states, pi_probs, 270)
    return states, pi_probs, values


def v_flip(
    states: torch.Tensor, pi_probs: torch.Tensor, values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, pi_probs = apply_vertical_flip(states, pi_probs)
    return states, pi_probs, values


def h_flip(
    states: torch.Tensor, pi_probs: torch.Tensor, values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, pi_probs = apply_horizontal_flip(states, pi_probs)
    return states, pi_probs, values


SUPPORTED_TRANSFORMATIONS = {
    # 'identity': lambda x, y, z: (x, y, z),
    'h_flip': h_flip,
    'v_flip': v_flip,
    'rotate90': rotate_90,
    'rotate180': rotate_180,
    'rotate270': rotate_270,
}

TRANSFORMATIONS = list(SUPPORTED_TRANSFORMATIONS.keys())


def apply_random_transformation(
    states: torch.Tensor, pi_probs: torch.Tensor, values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if random.random() > 0.5:
        transformation = random.choice(TRANSFORMATIONS)
        states, pi_probs, values = SUPPORTED_TRANSFORMATIONS[transformation](states, pi_probs, values)

    return states, pi_probs, values
