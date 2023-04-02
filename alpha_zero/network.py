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
"""AlphaZero Neural Network component."""
import math
from typing import NamedTuple, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class NetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor
    value: torch.Tensor


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class ResNetBlock(nn.Module):
    """Basic redisual block."""

    def __init__(
        self,
        num_filters: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """Policy network for AlphaZero agent."""

    def __init__(
        self,
        input_shape: Tuple,
        num_actions: int,
        num_res_block: int = 19,
        num_filters: int = 256,
        num_fc_units: int = 256,
    ) -> None:
        super().__init__()
        c, h, w = input_shape
        conv_out_hw = calc_conv2d_output((h, w), 3, 1, 1)
        # FIX BUG, Python 3.7 has no math.prod()
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        # First convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, num_fc_units),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> NetworkOutputs:
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""

        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)

        # Predict raw logits distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict evaluated value from current player's perspective.
        value = self.value_head(features)

        return NetworkOutputs(pi_logits=pi_logits, value=value)
