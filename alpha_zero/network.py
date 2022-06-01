"""Neural Network component."""
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
        num_planes: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_planes,
                out_channels=num_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_planes),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_planes,
                out_channels=num_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_planes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """Policy network with additional value head for Alpha Zero agent."""

    def __init__(
        self,
        input_shape: Tuple,
        num_actions: int,
        num_res_block: int = 10,
        num_planes: int = 256,
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
                out_channels=num_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_planes),
            nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_block = ResNetBlock(num_planes)
            res_blocks.append(res_block)
        self.res_blocks = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_planes,
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
                in_channels=num_planes,
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
        """Given raw state x, predict the action probability distribution
        and the winning probability for current player's perspective."""

        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)

        # Predict action distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict winning probability for current player's perspective.
        value = self.value_head(features)

        return NetworkOutputs(pi_logits=pi_logits, value=value)
