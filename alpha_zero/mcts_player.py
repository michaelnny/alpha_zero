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
"""MCTS player."""

from typing import Tuple, Union
import numpy as np
import torch
from alpha_zero.games.env import BoardGameEnv

# from alpha_zero.mcts_v1 import Node, uct_search, parallel_uct_search

from alpha_zero.mcts_v2 import Node, uct_search, parallel_uct_search

# from alpha_zero.mcts_v3 import Node, uct_search, parallel_uct_search


def create_mcts_player(
    network: torch.nn.Module,
    device: torch.device,
    num_simulations: int,
    num_parallel: int,
    root_noise: bool = False,
    deterministic: bool = False,
):
    """Give a network and device, returns a 'act' function to act on the specific environment."""

    @torch.no_grad()
    def eval_func(state_tensor: np.ndarray, batched: bool = False) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Give a game state tensor, returns the action probabilities
        and estimated winning probability from current player's perspective."""

        if not batched:
            state_tensor = state_tensor[None, ...]

        state = torch.from_numpy(state_tensor).to(device=device, dtype=torch.float32, non_blocking=True)
        output = network(state)
        pi_prob = torch.softmax(output.pi_logits, dim=-1).cpu().numpy()
        value = torch.detach(output.value).cpu().numpy()

        if not batched:
            # Remove batch dimensions
            pi_prob = np.squeeze(pi_prob, axis=0)
            value = np.squeeze(value, axis=0)

            # Convert value into float.
            value = value.item()

        return (pi_prob, value)

    def act(
        env: BoardGameEnv,
        root_node: Node,
        c_puct_base: float,
        c_puct_init: float,
        temperature: float,
    ):
        if num_parallel > 1:
            return parallel_uct_search(
                env=env,
                eval_func=eval_func,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                temperature=temperature,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=root_noise,
                deterministic=deterministic,
            )
        else:
            return uct_search(
                env=env,
                eval_func=eval_func,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                temperature=temperature,
                num_simulations=num_simulations,
                root_noise=root_noise,
                deterministic=deterministic,
            )

    return act
