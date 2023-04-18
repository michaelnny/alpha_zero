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
    """Uniform replay."""

    def __init__(
        self,
        capacity: int,
        random_state: np.random.RandomState,  # pylint: disable=no-member
    ):
        if capacity <= 0:
            raise ValueError(f'Expect capacity to be a positive integer, got {capacity}')
        self.structure = TransitionStructure
        self.capacity = capacity
        self.random_state = random_state
        self.storage = []

        self.num_games_added = 0
        self.num_samples_added = 0

    def add_game(self, game: Sequence[Any]) -> None:
        """Adds entire game to replay."""

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
