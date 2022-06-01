"""Replay components for training agents."""

from typing import Any, Text, Mapping, NamedTuple, Generic, List, Optional, Sequence, Tuple, TypeVar
import numpy as np

CompressedArray = Tuple[bytes, Tuple, np.dtype]

# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar('ReplayStructure', bound=Tuple[Any, ...])


class Transition(NamedTuple):
    state: Optional[np.ndarray]
    pi_prob: Optional[np.ndarray]
    value: Optional[float]
    player_id: Optional[int]


TransitionStructure = Transition(state=None, pi_prob=None, value=None, player_id=None)


class UniformReplay(Generic[ReplayStructure]):
    """Uniform replay, with circular buffer storage for flat named tuples."""

    def __init__(
        self,
        capacity: int,
        random_state: np.random.RandomState,  # pylint: disable=no-member
        structure: ReplayStructure = TransitionStructure,
    ):
        if capacity <= 0:
            raise ValueError(f'Expect capacity to be a positive integer, got {capacity}')
        self.structure = structure
        self._capacity = capacity
        self._random_state = random_state
        self._storage = [None] * capacity
        self._num_added = 0

    def add(self, item: ReplayStructure) -> None:
        """Adds single item to replay."""
        self._storage[self._num_added % self._capacity] = item
        self._num_added += 1

    def get(self, indices: Sequence[int]) -> List[ReplayStructure]:
        """Retrieves items by indices."""
        return [self._storage[i] for i in indices]

    def sample(self, batch_size: int) -> ReplayStructure:
        """Samples batch of items from replay uniformly, with replacement."""
        if self.size < batch_size:
            raise RuntimeError(f'Replay only have {self.size} samples, got sample batch size {batch_size}')

        indices = self._random_state.randint(self.size, size=batch_size)
        samples = self.get(indices)

        transposed = zip(*samples)
        stacked = [np.stack(xs, axis=0) for xs in transposed]  # Stack on batch dimension (0)
        return type(self.structure)(*stacked)

    @property
    def num_added(self) -> int:
        """Number of items added into replay."""
        return self._num_added

    @property
    def size(self) -> int:
        """Number of items currently contained in replay."""
        return min(self._num_added, self._capacity)

    @property
    def capacity(self) -> int:
        """Total capacity of replay (max number of items stored at any one time)."""
        return self._capacity

    def reset(self) -> None:
        """Reset the state of replay."""
        self._num_added = 0

    def get_state(self) -> Mapping[Text, Any]:
        """Retrieves replay state as a dictionary (e.g. for serialization)."""
        return {'num_added': self._num_added, 'storage': self._storage}

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets replay state from a (potentially de-serialized) dictionary."""
        self._num_added = state['num_added']
        self._storage = state['storage']
