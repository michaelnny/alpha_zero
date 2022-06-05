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
"""Elo ratings."""


def estimate_win_probability(ra, rb, c_elo: float = 1 / 400) -> float:
    """Returns the estimated probability of winning from 'player A' perspective.

    Args:
        ra: elo rating for player A.
        rb: elo rating for player B.
        c_elo: the constant for elo, default 1 / 400.

    Returns:
        the win probability of player A.
    """

    return 1.0 / (1 + 10 ** ((rb - ra) * c_elo))


def compute_elo_rating(winner: int, ra=0, rb=0, k=32) -> float:
    """Returns the Elo rating from 'player A' perspective.

    Args:
        winner: who won the game, `0` for player A, `1` for player B.
        ra: current elo rating for player A, default 2500.
        rb: current elo rating for player B, default 2500.
        k: the factor, default 32.

    Returns:
        A tuple contains new estimated Elo ratings for player A and player B.
        format (elo_player_A, elo_player_B)
    """
    if winner is None:
        return (ra, rb)
    if not isinstance(winner, int) or winner not in [0, 1]:
        raise ValueError(f'Expect input argument `winner` to be [0, 1], got {winner}')

    c_elo = 1.0 / 400.0

    # Compute the winning probability of player A
    prob_a = estimate_win_probability(ra, rb, c_elo)

    # Compute the winning probability of player B
    prob_b = estimate_win_probability(rb, ra, c_elo)

    # Updating the Elo Ratings
    # Case -1 When player A wins
    if winner == 0:
        new_ra = ra + k * (1 - prob_a)
        new_rb = rb + k * (0 - prob_b)
    # Case -1 When player B wins
    else:
        new_ra = ra + k * (0 - prob_a)
        new_rb = rb + k * (1 - prob_b)

    return (new_ra, new_rb)
