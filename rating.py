# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Implements code for the Elo rating system."""
from typing import Iterable
import math


def get_k_factor(player_ratings: Iterable[float]) -> int:
    """
    We follow the formal USCF rule on setting the k-factor, which is much simpler:

    K-factor Used for players with ratings:
        K = 32  below 2100
        K = 24  between 2100 and 2400
        K = 16  above 2400
    """
    k = 32  # default 32

    if all(r < 2100 for r in player_ratings):
        k = 32
    elif all(r < 2400 for r in player_ratings) and any(r >= 2100 for r in player_ratings):
        k = 24
    elif all(r >= 2400 for r in player_ratings):
        k = 16

    return k


class EloRating:
    """
    A very simple implementation of the elo rating system.

    Usage example:
    ```
    player1 = EloRating()
    player2 = EloRating()

    for i in range(10):
        player1_won = True if i % 2 == 0 else False  # player1 wins

        if player1_won:
            winner, loser = player1, player2
        else:
            winner, loser = player2, player1
        winner.update_rating(loser.rating, 1)
        loser.update_rating(winner.rating, 0)

        print(f'Player 1 rating: {player1.rating}')
        print(f'Player 2 rating: {player2.rating}')
    ```

    """

    def __init__(self, rating=0):
        self.rating = rating

    def expected_score(self, opponent_rating: float) -> float:
        return 1 / (1 + math.pow(10, (opponent_rating - self.rating) / 400))

    def update_rating(self, opponent_rating: float, actual_score: float) -> None:
        expected_score = self.expected_score(opponent_rating)

        k_factor = get_k_factor((self.rating, opponent_rating))
        self.rating += k_factor * (actual_score - expected_score)
