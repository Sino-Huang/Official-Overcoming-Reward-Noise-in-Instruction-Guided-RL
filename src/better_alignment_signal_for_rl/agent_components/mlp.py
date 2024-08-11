from typing import Dict

import torch as th
import torch.nn as nn

from .torch_util import FanInInitReLULayer
from icecream import ic 


class MLP(nn.Module):
    def __init__(
        self,
        insize: int,
        nhidlayer: int,
        outsize: int,
        hidsize: int,
        dense_init_norm_kwargs: Dict = {},
        layer_type: str = "linear",
    ):
        super().__init__()

        # Layers
        insizes = [insize] + nhidlayer * [hidsize]
        outsizes = nhidlayer * [hidsize] + [outsize]
        self.layers = nn.ModuleList()

        for i, (insize, outsize) in enumerate(zip(insizes, outsizes)):
            use_activation = i < nhidlayer
            init_scale = 1.4 if use_activation else 1.0
            init_norm_kwargs = dense_init_norm_kwargs if use_activation else {}
            layer = FanInInitReLULayer(
                insize,
                outsize,
                layer_type=layer_type,
                use_activation=use_activation,
                init_scale=init_scale,
                **init_norm_kwargs,
            )
            self.layers.append(layer)

    def forward(self, x: th.Tensor) -> th.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class GatedResidualMLP(nn.Module):
    def __init__(
        self,
        insize: int,
        nhidlayer: int,
        outsize: int,
        hidsize: int,
        dense_init_norm_kwargs: Dict = {},
        layer_type: str = "linear",
    ):
        super().__init__()

        # Layers
        insizes = [insize] + nhidlayer * [hidsize]
        outsizes = nhidlayer * [hidsize] + [outsize]
        self.layers = nn.ModuleList()
        self.gates = nn.ParameterList()
        self.projections = nn.ModuleList()

        for i, (insize, outsize) in enumerate(zip(insizes, outsizes)):
            use_activation = i < nhidlayer
            init_scale = 1.4 if use_activation else 1.0
            init_norm_kwargs = dense_init_norm_kwargs if use_activation else {}
            layer = FanInInitReLULayer(
                insize,
                outsize,
                layer_type=layer_type,
                use_activation=use_activation,
                init_scale=init_scale,
                **init_norm_kwargs,
            )
            self.layers.append(layer)
            # adding a gate for each layer 
            gate = nn.Parameter(th.tensor([0.]))
            self.gates.append(gate)
            
            # check if a projection is needed 
            if insize != outsize:
                projection = nn.Linear(insize, outsize)
                self.projections.append(projection)
            else:
                self.projections.append(None)

    def forward(self, x: th.Tensor) -> th.Tensor:
        for layer, gate, projection in zip(self.layers, self.gates, self.projections):
            y = layer(x)
            if projection:
                x = projection(x)
            x = x + y * gate.tanh() 
        return x