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
"""Replay components for training agents."""

from typing import Any, NamedTuple, Optional, Sequence
import numpy as np
import collections


class Transition(NamedTuple):
    state: Optional[np.ndarray]
    pi_prob: Optional[np.ndarray]
    value: Optional[float]


TransitionStructure = Transition(state=None, pi_prob=None, value=None)


class UniformReplay:
    """Uniform replay, with window size to limit maximum number of games stored in buffer,
    and periodically adjusting the buffer capacity based on the mean game length."""

    def __init__(
        self,
        window_size: int,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        min_game_step: int = 10,  # capacity = window_size x min_game_step, a small number to flush out initial random games quickly
    ):
        if window_size <= 0:
            raise ValueError(f'Expect window_size to be a positive integer, got {window_size}')
        self.structure = TransitionStructure
        self.window_size = window_size
        self.min_game_step = min_game_step
        self.avg_game_steps = min_game_step
        self.random_state = random_state
        self.storage = []

        self.game_steps = collections.deque(maxlen=1000)
        self.num_games_added = 0
        self.num_samples_added = 0

    def add_game(self, game: Sequence[Any]) -> None:
        """Adds entire game to replay."""

        self.game_steps.append(len(game))

        for item in game:
            self.add(item)

        self.num_games_added += 1

    def add(self, item: Any) -> None:
        """Adds single item to replay."""

        if self.size > self.capacity:
            self.storage.pop(0)

        self.storage.append(item)
        self.num_samples_added += 1

    def get(self, indices: Sequence[int]) -> Sequence[Any]:
        """Retrieves items by indices."""
        return [self.storage[i] for i in indices]

    def sample(self, batch_size: int) -> Any:
        """Samples batch of items from replay uniformly, with replacement."""
        if self.size < batch_size:
            return

        indices = self.random_state.randint(self.size, size=batch_size)
        samples = self.get(indices)

        transposed = zip(*samples)
        stacked = [np.stack(xs, axis=0) for xs in transposed]  # Stack on batch dimension (0)
        return type(self.structure)(*stacked)

    def adjust_capacity(self, new_avg_game_steps: int) -> int:
        assert new_avg_game_steps > 0

        self.avg_game_steps = new_avg_game_steps

        deltas = self.capacity - self.size

        if deltas < 0:
            for _ in range(abs(deltas)):
                self.storage.pop(0)

        return self.capacity

    def reset(self) -> None:
        """Reset the state of replay."""
        self.num_games_added = 0
        self.num_samples_added = 0
        self.avg_game_steps = self.min_game_step
        del self.storage[:]

    @property
    def size(self) -> int:
        """Number of items currently contained in replay."""
        return len(self.storage)

    @property
    def capacity(self) -> int:
        return self.window_size * self.avg_game_steps
