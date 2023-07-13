# Copyright (c) 2023 Michael Hu
# All rights reserved.
"""Replay components for training agents."""

from typing import Mapping, Text, Any, NamedTuple, Optional, Sequence
import numpy as np
import snappy


class Transition(NamedTuple):
    state: Optional[np.ndarray]
    pi_prob: Optional[np.ndarray]
    value: Optional[float]


TransitionStructure = Transition(state=None, pi_prob=None, value=None)


def compress_array(array):
    """Compresses a numpy array with snappy."""
    return snappy.compress(array), array.shape, array.dtype


def uncompress_array(compressed):
    """Uncompresses a numpy array with snappy given its shape and dtype."""
    compressed_array, shape, dtype = compressed
    byte_string = snappy.uncompress(compressed_array)
    return np.frombuffer(byte_string, dtype=dtype).reshape(shape)


class UniformReplay:
    """Uniform replay, with circular buffer storage for flat named tuples."""

    def __init__(
        self,
        capacity: int,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        compress_data: bool = True,
    ):
        if capacity <= 0:
            raise ValueError(f'Expect capacity to be a positive integer, got {capacity}')
        self.structure = TransitionStructure
        self.capacity = capacity
        self.random_state = random_state
        self.compress_data = compress_data
        self.storage = [None] * capacity

        self.num_games_added = 0
        self.num_samples_added = 0

    def add_game(self, game_seq: Sequence[Transition]) -> None:
        """Add an entire game to replay."""

        for transition in game_seq:
            self.add(transition)

        self.num_games_added += 1

    def add(self, transition: Any) -> None:
        """Adds single transition to replay."""
        index = self.num_samples_added % self.capacity
        self.storage[index] = self.encoder(transition)
        self.num_samples_added += 1

    def get(self, indices: Sequence[int]) -> Sequence[Transition]:
        """Retrieves items by indices."""
        return [self.decoder(self.storage[i]) for i in indices]

    def sample(self, batch_size: int) -> Transition:
        """Samples batch of items from replay uniformly, with replacement."""
        if self.size < batch_size:
            return

        indices = self.random_state.randint(low=0, high=self.size, size=batch_size)
        samples = self.get(indices)

        transposed = zip(*samples)
        stacked = [np.stack(xs, axis=0) for xs in transposed]  # Stack on batch dimension (0)
        return type(self.structure)(*stacked)

    def encoder(self, transition: Transition) -> Transition:
        if self.compress_data:
            return transition._replace(
                state=compress_array(transition.state),
            )
        return transition

    def decoder(self, transition: Transition) -> Transition:
        if self.compress_data:
            return transition._replace(
                state=uncompress_array(transition.state),
            )
        return transition

    def get_state(self) -> Mapping[Text, Any]:
        """Retrieves replay state as a dictionary (e.g. for serialization)."""
        return {'num_games_added': self.num_games_added, 'num_samples_added': self.num_samples_added, 'storage': self.storage}

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets replay state from a (potentially de-serialized) dictionary."""
        self.num_games_added = state['num_games_added']
        self.num_samples_added = state['num_samples_added']
        self.storage = state['storage']

    @property
    def size(self) -> int:
        """Number of items currently contained in replay."""
        return min(self.num_samples_added, self.capacity)
