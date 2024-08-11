from random import shuffle, seed, randint
import math
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import einsum
from einops import reduce, repeat, rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding

class SimpleSwiGLU(nn.Module):
    # note that this will divide the feature dim by 2
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def exists(val):
    return val is not None

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        *,
        eps = 1e-8,
        gated = False
    ):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.weight = nn.Parameter(torch.ones(dim)) if gated else None

    def forward(self, x):
        norm = x.norm(keepdim = True, dim = -1) * self.scale
        out = (x / norm.clamp(min = self.eps)) * self.gamma

        if not exists(self.weight):
            return out

        return out * (x * self.weight).sigmoid()

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class NormMLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, act_layer=SimpleSwiGLU, factor = 4, dropout_rate = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.factor = factor
        self.middle_dim = int(self.factor * self.input_dim)
        self.block = nn.Sequential(
            nn.Linear(self.input_dim, self.middle_dim * 2, bias=False),
            act_layer(),
            nn.Dropout(dropout_rate),
            RMSNorm(self.middle_dim),
            nn.Linear(self.middle_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Attention(nn.Module):
    def __init__(self, q_dim, k_dim, v_and_output_dim, num_heads, dropout_p) -> None:
        """_summary_

        query (Tensor) Query tensor; shape (N,...,L,E)
        key (Tensor) Key tensor; shape (N,...,S,E)
        value (Tensor) Value tensor; shape (N,...,S,Ev)
        Attention output; shape (N,...,L,Ev)
        
        N:Batch size...:Any number of other batch dimensions (optional)
        S:Source sequence lengthS:Source sequence length
        L:Target sequence lengthL:Target sequence length
        E:Embedding dimension of the query and keyE:Embedding dimension of the query and key
        Ev:Embedding dimension of the valueEv:Embedding dimension of the value

        """
        super().__init__()
        self.q_linear = nn.Linear(q_dim, v_and_output_dim)
        self.k_linear = nn.Linear(k_dim, v_and_output_dim)
        self.num_head = num_heads
        self.dropout_p = dropout_p
        self.rearrange_layer = Rearrange('b n (h d) -> b h n d', h = self.num_head)
        
    def forward(self, q, k, v, mask=None, pos_emb:RotaryEmbedding= None):
        # mask is the transformer mask scheme 
        # amplify q and k 
        q = self.q_linear(q)
        k = self.k_linear(k)
        # split head 
        # q = rearrange(q, 'b n (h d) -> b h n d', h = self.num_head)
        # k = rearrange(k, 'b n (h d) -> b h n d', h = self.num_head)
        # v = rearrange(v, 'b n (h d) -> b h n d', h = self.num_head)
        
        q = self.rearrange_layer(q)
        k = self.rearrange_layer(k)
        v = self.rearrange_layer(v)
        
        if pos_emb is not None:
            q = pos_emb.rotate_queries_or_keys(q)
            k = pos_emb.rotate_queries_or_keys(k)
        
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            output = F.scaled_dot_product_attention(q,k,v, attn_mask=mask, dropout_p=self.dropout_p) # shape (N,H...,L,Ev)
        output = rearrange(output, 'b h l v -> b l (h v)')
        return output 

class Gated_XAtten_Dense_Block(nn.Module):
    def __init__(self,
                 q_dim,
                 kv_dim,
                 v_dim=None,
                 num_heads=8,
                 drop_ratio=0.2,
                 attn_drop_ratio=0.2,
                 drop_path_ratio=0.2,  # set dropout rate to be 0.2 as default
                 act_layer=SimpleSwiGLU,
                 norm_layer=LayerNorm):
        r'''
        :param q_dim:  q feature dim, it should be the target encoding feature size, please change lang feature size to target size in advance
        :param kv_dim: orignial k and v feature dim, should be the same size, (action and image)
        :IMPORTANT: v_dim should be LARGE
        :param num_heads: num of heads for multi head attention
        :param qkv_bias: whether need bias param
        :param qk_scale:
        :param drop_ratio: dropout rate
        :param attn_drop_ratio: droprate in the attention
        :param drop_path_ratio: droppath rate between layers
        :param act_layer:
        :param norm_layer:
        
        output shape will be the same as [batch, q_len, v_dim] 
        '''

        super(Gated_XAtten_Dense_Block, self).__init__()
        if v_dim is None:
            v_dim = kv_dim
        self.norm1_q = norm_layer(q_dim)
        self.norm1_k = norm_layer(kv_dim)
        self.norm1_v = norm_layer(v_dim)
        self.norm1_post = norm_layer(v_dim)
        self.crossattn = Attention(q_dim=q_dim,k_dim=kv_dim, v_and_output_dim=v_dim, num_heads=num_heads, dropout_p=attn_drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.selfattn_option_freeze_pretrain = Attention(q_dim=v_dim,k_dim=v_dim, v_and_output_dim=v_dim, num_heads=num_heads, dropout_p=attn_drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(v_dim)
        self.norm3 = norm_layer(v_dim)
        self.norm3_post = norm_layer(v_dim)
        self.norm4 = norm_layer(v_dim)
        self.mlp1 = NormMLPBlock(input_dim=v_dim, output_dim=v_dim, act_layer=act_layer, dropout_rate=drop_ratio)
        self.mlp2_option_freeze_pretrain = NormMLPBlock(input_dim=v_dim, output_dim=v_dim, act_layer=act_layer, dropout_rate=drop_ratio)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.mlp_gate = nn.Parameter(torch.tensor([0.]))
        self.q_mapper = nn.Linear(q_dim, v_dim)
        
        
        # rotatory pos embd follow v_dim 
        self.rotary_emb = RotaryEmbedding(dim = v_dim//2//num_heads)

    def forward(self, q, k, v, mask=None):
        # mask = 1 means we will let that token take part in 
        q = self.q_mapper(q) + self.drop_path(self.norm1_post(
            self.crossattn(
                self.norm1_q(q), self.norm1_k(k), self.norm1_v(v), mask, self.rotary_emb)
            )
                               ) * self.attn_gate.tanh()
        q = q + self.drop_path(self.mlp1(self.norm2(q))) * self.mlp_gate.tanh()
        q = self.norm3(q)
        q = q + self.drop_path(self.norm3_post(self.selfattn_option_freeze_pretrain(q, q, q, mask, self.rotary_emb)))
        q = q + self.drop_path(self.mlp2_option_freeze_pretrain(self.norm4(q)))
        return q
    
if __name__ == "__main__":
    model = Gated_XAtten_Dense_Block(
        12,128
    )
    q = torch.rand((2,10,12))
    k = v = torch.rand((2,100,128))
    
    output = model(q, k, v)
    print(output.shape)
    # torch.Size([2, 10, 128])