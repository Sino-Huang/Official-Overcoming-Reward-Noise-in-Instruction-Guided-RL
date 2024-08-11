import abc
from typing import Dict, Sequence

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from better_alignment_signal_for_rl.agent_components.mlp import MLP
from better_alignment_signal_for_rl.agent_components.torch_util import (
    FanInInitReLULayer,
)
from better_alignment_signal_for_rl.pipelines.train_lang_rew_model.pretrained_vision_language_model import (
    get_pretrained_vision_language_model,
    postprocess_clip_outputs,
    postprocess_xclip_outputs,
)
from torch.nn import GRU, LSTM, RNN
import sys


class ExtractRNN(nn.Module):
    def forward(self, x):
        # Output shape (batch, seq_len, hidden_size) and (h_n, c_n)
        tensor, _ = x
        # we only need the tensor
        return tensor


class TransitionDifference(nn.Module):
    def __init__(self, env_name):
        super().__init__()
        self.env_name = env_name
        if env_name == "crafter":
            self.forward = self.forward_follow_camera
        elif env_name in ['minigrid', 'montezuma']:
            self.forward = self.forward_fixed_camera
            
    def forward_fixed_camera(self, x):
        # Input shape (batch, seq_len, dim)
        # Output shape (batch, seq_len - 1, dim)
        x = x[:, 1:, :] - 0.5*(x[:, :-1, :] )
        return x
    
    def forward_follow_camera(self, x):
        # Input shape (batch, seq_len, dim)
        # Output shape (batch, seq_len - 1, dim)
        x = x[:, 1:, :] - x[:, :-1, :]
        return x


class BaseModel(nn.Module, abc.ABC):
    """
    The model should have three layers, the pretrained model layer, the post clip layers and the dense model layer, the dense model layer will be implemented in the child class
    """

    def __init__(
        self,
        pretrained_model_cls,
        pretrained_model_output_size: int,
        is_pretrained_module_freezed,
        is_markovian: bool,
        rnn_model_cls,
        rnn_kwargs,
        dense_kwargs,
        has_extra_data_manipulation: bool,
        env_name :str,
        traj_length: int,
    ):
        super().__init__()

        # save this attributes
        self.pretrained_model_cls = pretrained_model_cls
        self.pretrained_model_output_size = pretrained_model_output_size
        self.is_pretrained_module_freezed = is_pretrained_module_freezed
        assert is_pretrained_module_freezed  # ! only support freezed pretrained module
        self.is_markovian = is_markovian
        self.rnn_model_cls = rnn_model_cls
        self.rnn_kwargs = rnn_kwargs
        self.dense_kwargs = dense_kwargs
        self.has_extra_data_manipulation = has_extra_data_manipulation
        self.env_name = env_name
        self.traj_length = traj_length

        # load the pretrained model
        self.pretrained_model_layer, self.tokenizer = (
            get_pretrained_vision_language_model(pretrained_model_cls)
        )  # * tokenizer will not load to cuda by default as it has no nn.Parameter

        # freeze the pretrained model
        if is_pretrained_module_freezed:
            for param in self.pretrained_model_layer.parameters():
                param.requires_grad = False

        # intermediate mlp to convert the vision and text embeddings to the hidden size

        layer_lst = []
        if (
            "xclip" not in pretrained_model_cls
        ):  # clip, may need to handle multiple inputs
            layer_lst.append(
                TransitionDifference(self.env_name)
            )  # output shape (batch, seq_len-1, hidden_size)
            layer_lst.append(
                FanInInitReLULayer(
                    pretrained_model_output_size,
                    rnn_kwargs["input_size"],
                    layer_type="linear",
                    use_activation=False,
                    rms_norm=True,
                )
            ),
            if not is_markovian:  # means we will use the rnn model
                layer_lst.extend(
                    [
                        getattr(sys.modules[__name__], rnn_model_cls)(**rnn_kwargs),
                        ExtractRNN(),  # (batch, seq_len-1, hidden_size)
                    ]
                )
        else:  # xclip, 
            layer_lst.append(nn.Identity())

        self.post_clip_vision_layer = nn.Sequential(*layer_lst)  # ! it is post clip layers

    def clip_format_prepare(
        self, vision_input: th.Tensor, text_input: Sequence[str]
    ):
        """Make sure you run this function before the forward pass, so that you can control the device of the input data
        """
        
        if "xclip" in self.pretrained_model_cls:
            assert vision_input.shape[1] == 8, "xclip require 8 frames"
        else: 
            # need to flatten it 
            vision_input = rearrange(vision_input, "B L C H W -> (B L) C H W")
            
        text_dict = self.tokenizer(text=text_input, padding=True, return_tensors="pt")
        inputs = dict(
            pixel_values=vision_input,
            **text_dict,
        )
        return inputs
        
    def clip_encode(self, inputs):
        """
        The method will encode the clip inputs, run it inside the forward method
        Args:
            vision_input (th.Tensor): the transformed and normalized vision input, shape ( (batch, seq_len) , C, H, W)
            text_input (Sequence[str]): the text input
        Return:
            vision_embeds (th.Tensor): the vision embeddings, shape (batch, seq_len, hidden_size) if clip, else (batch, hidden_size)
            text_embeds (th.Tensor): the text embeddings, shape (batch, hidden_size)
        """
        B = inputs['input_ids'].shape[0]
        
        # start encoding 
        with th.no_grad():
            outputs = self.pretrained_model_layer(**inputs)
            # return the vision and text embeddings
            if "xclip" in self.pretrained_model_cls:
                vision_embeds, text_embeds = postprocess_xclip_outputs(outputs)
            else:
                vision_embeds, text_embeds = postprocess_clip_outputs(outputs)
                # reshape the vision_embeds
                vision_embeds = rearrange(vision_embeds, "(B L) C -> B L C", L=self.traj_length)
        return vision_embeds, text_embeds
            

    @abc.abstractmethod
    def generate_extra_hard_negatives(
        self, vision_input: th.Tensor, text_input: Sequence[str],
        target_cls: th.Tensor = None
    ):
        # ! this is used to generate extra hard negatives if has_extra_data_manipulation is True
        pass

    @abc.abstractmethod
    def forward(self, inputs):
        """the inputs needs to be prepared by the clip_format_prepare method
        the correct order of the process is 
        1. vision transformation and augmentation  (done by dataloader already)
        2. hard negative data generation 
        3. clip_format_prepare
        4. inputs to device 
        5. forward 
        """
        pass

    def compute_cosine_similarity(
        self, vision_embeds: th.Tensor, text_embeds: th.Tensor
    ):
        # unsqueeze the vision_embeds
        vision_embeds = vision_embeds.unsqueeze(1) # shape (batch, 1, hidden_size)
        text_embeds = text_embeds.unsqueeze(0) # shape (1, num_texts, hidden_size)
        cosine_similarity = F.cosine_similarity(vision_embeds, text_embeds, dim=-1)
        # shape (batch, num_texts) 
        
        # deprecated: add normalization to the cosine similarity from [-1, 1] to [0, 1]
        # cosine_similarity = (cosine_similarity + 1) / 2
        # # all_above_zero = (cosine_similarity >= 0).all()
        # # all_below_one = (cosine_similarity <= 1).all()
        # # assert all_above_zero
        # # assert all_below_one
        
        return cosine_similarity

    @abc.abstractmethod
    def compute_losses(self, **kwargs) -> Dict[str, th.Tensor]:
        # we may want to consider has_extra_data_manipulation
        pass
