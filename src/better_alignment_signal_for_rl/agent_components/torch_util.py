# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from typing import Dict, Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic 

class FanInInitReLULayer(nn.Module):
    """Fan-in initialization followed by default ReLU activation.
    If you use SwiGLU layer, then every layer will have a SwiGLU activation.
    """
    def __init__(
        self,
        inchan: int,
        outchan: int,
        layer_type: str = "conv",
        init_scale: float = 1.0,
        batch_norm: bool = False,
        batch_norm_kwargs: Dict = {},
        group_norm_groups: Optional[int] = None,
        layer_norm: bool = False,
        rms_norm: bool = False,
        use_activation: bool = True,
        **layer_kwargs,
    ):
        super().__init__()

        # Normalization
        self.norm = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan, **batch_norm_kwargs)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)
        elif rms_norm:
            self.norm = RMSNorm(inchan)

        # Layer
        layer = dict(conv=nn.Conv2d, conv3d=nn.Conv3d, linear=nn.Linear, swiglu=SwiGLU)[layer_type]
        self.layer = layer(inchan, outchan, bias=self.norm is None, **layer_kwargs)
        self.use_activation = use_activation and layer_type != "swiglu"

        # Initialization
        if not isinstance(self.layer, SwiGLU):
            self.layer.weight.data *= init_scale / self.layer.weight.norm(
                dim=tuple(range(1, self.layer.weight.data.ndim)), p=2, keepdim=True
            )
            if self.layer.bias is not None:
                self.layer.bias.data *= 0
        else:
            self.layer.w1.weight.data *= init_scale / self.layer.w1.weight.norm(
                dim=tuple(range(1, self.layer.w1.weight.data.ndim)), p=2, keepdim=True
            )
            if self.layer.w1.bias is not None:
                self.layer.w1.bias.data *= 0
            self.layer.w2.weight.data *= init_scale / self.layer.w2.weight.norm(
                dim=tuple(range(1, self.layer.w2.weight.data.ndim)), p=2, keepdim=True
            )
            if self.layer.w2.bias is not None:
                self.layer.w2.bias.data *= 0
            self.layer.w3.weight.data *= init_scale / self.layer.w3.weight.norm(
                dim=tuple(range(1, self.layer.w3.weight.data.ndim)), p=2, keepdim=True
            )
            if self.layer.w3.bias is not None:
                self.layer.w3.bias.data *= 0
            

    def forward(self, x: th.Tensor):
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x





class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(th.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(th.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = th.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
    
    
class SwiGLU(nn.Module):
    """Feed forward block for LLAMA model.

    Args:
        d_model (int): input dimension of model
        d_ff (int): hidden dimension of feed forward block
        ffn_dim_multiplier (int): custom multiplier for hidden dimension of feed forward block
        multiple_of (int): value to make hidden dimension of feed forward block multiple of
    """

    def __init__(self, insize: int, outsize:int, bias: bool, ):
        super().__init__()

        hidsize= insize * 4
        self.w1 = nn.Linear(insize, hidsize, bias=bias)
        self.w2 = nn.Linear(hidsize, outsize, bias=bias)
        self.w3 = nn.Linear(insize, hidsize, bias=bias)

    def forward(self, x):
        swish = F.silu(self.w1(x))  # (batch_size, seq_len, d_ff)
        x_v = self.w3(x)  # (batch_size, seq_len, d_ff)
        x = swish * x_v  # (batch_size, seq_len, d_ff)
        x = self.w2(x)  # (batch_size, seq_len, d_model)
        return x