# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

# from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import (
    SkipMode,
    look_up_option,
)
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import UNet

__all__ = ["NoSkipUNet", "NoSkipUnet"]


class NoSkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::

        --+--submodule--o--
          |_____________|

    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(
        self,
        submodule,
    ) -> None:
        """

        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
        """
        super().__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.submodule(x)
        return y


class NoSkipUNet(UNet):
    """
    Enhanced version of UNet with no skip connections.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:

        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            channels,
            strides,
            kernel_size,
            up_kernel_size,
            num_res_units,
            act,
            norm,
            dropout,
            bias,
            adn_ordering,
        )

        def _create_block(
            inc: int,
            outc: int,
            channels: Sequence[int],
            strides: Sequence[int],
            is_top: bool,
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(
                    c, c, channels[1:], strides[1:], False
                )  # continue recursion down
                upc = c
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = channels[1]

            down = self._get_down_layer(
                inc, c, s, is_top
            )  # create layer in downsampling path
            up = self._get_up_layer(
                upc, outc, s, is_top
            )  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, True
        )

    def _get_connection_block(
        self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module
    ) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, NoSkipConnection(subblock), up_path)


NoSkipUnet = NoSkipUNet
