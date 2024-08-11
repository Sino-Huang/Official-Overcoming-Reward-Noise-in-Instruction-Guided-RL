from copy import deepcopy
from typing import Sequence
from einops import rearrange
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from focal_loss.focal_loss import FocalLoss
from torchvision.transforms.functional import resize as resize_image

from better_alignment_signal_for_rl.agent_components.attention import Gated_XAtten_Dense_Block
from better_alignment_signal_for_rl.agent_components.impala_cnn import ImpalaCNN
from better_alignment_signal_for_rl.agent_components.mlp import GatedResidualMLP
from .base import BaseModel
from icecream import ic 

class TrajRecogLangRewModel(BaseModel):
    def __init__(
        self,
        pretrained_model_cls,
        pretrained_model_output_size,
        is_pretrained_module_freezed,
        is_markovian: bool,
        rnn_model_cls,
        rnn_kwargs,
        dense_kwargs,
        has_extra_data_manipulation: bool,
        cls_weight,
        minigrid_no_pretrain : bool, 
        env_name,
        traj_length,
        
        
    ):
        super().__init__(
            pretrained_model_cls,
            pretrained_model_output_size,
            is_pretrained_module_freezed,
            is_markovian,
            rnn_model_cls,
            rnn_kwargs,
            dense_kwargs,
            has_extra_data_manipulation,
            env_name,
            traj_length,
        )
        self.output_size = 3 
        self.minigrid_no_pretrain = minigrid_no_pretrain
        
        self.focal_loss =  FocalLoss(gamma=2.0, weights=th.FloatTensor(cls_weight))
        # gamma =2 follows the default value of focal loss

        assert not is_markovian, "Trajectory recognition model does not support markovian"

        assert has_extra_data_manipulation , "Traj Recog model need extra data manipulation to get more complex data, otherwise the training will be overfitting"

        assert "xclip" not in pretrained_model_cls, "Trajectory recognition model does not support xclip as it require a sequence of trajectory data"

        # clip require a attention based layer
        # ! the main layer will require two inputs: the video sequence embedding (B, L, Ev) and the text sentence  embedding (B, 1, El)
        self.attention_layer = Gated_XAtten_Dense_Block(
            q_dim=pretrained_model_output_size,  # be the video feature size
            kv_dim=pretrained_model_output_size,  # be the text feature size
        )  # output shape (batch, seq_len, kv_dim),

        self.prediction_layer = GatedResidualMLP(
            insize=dense_kwargs["insize"],
            nhidlayer=dense_kwargs["nhidlayer"],
            outsize=self.output_size,
            hidsize=int(dense_kwargs["insize"] // 2),
            dense_init_norm_kwargs=dense_kwargs["dense_init_norm_kwargs"],
            layer_type=dense_kwargs["dense_model_cls"],
        )  # output shape (batch, seq_len, output_size=3)
        
        # ! ---- extra debugging setting ----
        if self.env_name == "minigrid" and self.minigrid_no_pretrain:
            assert "xclip" not in pretrained_model_cls # we only support clip for minigrid
            dense_init_norm_kwargs = {"layer_norm": True}
            impala_kwargs = {
                "chans": [64, 128, 128],
                "outsize": dense_kwargs['insize'],  # match clip output size
                "nblock": 2,
                "post_pool_groups": 1,
                "init_norm_kwargs": {"batch_norm": True, "group_norm_groups": 1},
            }
            self.minigrid_ipalacnn = ImpalaCNN(
                (3, 64, 64),  # match clip input size
                dense_init_norm_kwargs=dense_init_norm_kwargs,  # this will decide whether to add norm layer
                **impala_kwargs,
            )
            
            self.preprocess_input = self.minigrid_no_vision_pretrain_preprocess
        else:
            self.preprocess_input = self.default_preprocess
        # ! ---- end extra debugging setting ----
        
    # ! ---- extra debugging setting ----
    def forward_minigrid_no_vision_pretrain(self, inputs):
        B = inputs['input_ids'].shape[0]
        pixel_values = inputs['pixel_values'] # shape ((B L), C, H, W)
        pixel_values = resize_image(pixel_values, (64, 64), antialias=True) # resize the image
        vision_embeds = self.minigrid_ipalacnn(pixel_values) # shape ((B L), 512)
        vision_embeds = rearrange(vision_embeds, "(B L) C -> B L C", L=self.traj_length)
        return vision_embeds
    
    def minigrid_no_vision_pretrain_preprocess(self, inputs):
        text_embeds = self.clip_encode(inputs)[1]
        vision_embeds = self.forward_minigrid_no_vision_pretrain(inputs)
        return vision_embeds, text_embeds
    
    def default_preprocess(self, inputs):
        vision_embeds, text_embeds = self.clip_encode(inputs)
        return vision_embeds, text_embeds
    
    # ! ---- end extra debugging setting ----

    def forward(self, inputs):
        vision_embeds, text_embeds = self.preprocess_input(inputs)
        # post_clip_vision_layer
        vision_embeds = self.post_clip_vision_layer(vision_embeds)
        # shape (b, l-1, Ev) and (b, El)
        # we need to unsqueeze the text_embeds
        text_embeds = text_embeds.unsqueeze(-2)

        # main layer
        x = self.attention_layer(vision_embeds, text_embeds, text_embeds)
        x = self.prediction_layer(x)

        # output shape (batch, seq_len, 3)
        return x 

    def compute_losses(
        self,
        outputs,
        target_cls,
        **kwargs,
    ) -> th.Dict[str, th.Tensor]:
        # output shape (batch, seq_len, 3)
        # target_cls shape (batch, seq_len)
        # classification
        # 0 = achieved
        # 1 = intend to achieve
        # 2 = irrelevant
        # classification loss
        # flatten the output and target_cls
        outputs = outputs.view(-1, 3)
        target_cls = target_cls.view(-1)

        loss = self.focal_loss(F.softmax(outputs, dim=-1), target_cls)
        
        # calculate accuracy 
        preds = th.argmax(outputs, dim=-1) # shape (batch * seq_len)
        correct = th.sum(preds == target_cls)
        accuracy = correct.float() / target_cls.shape[0]
        accuracy = accuracy.item()

        return {"loss": loss, "accuracy": accuracy}

    def get_raw_target_cls(self, vision_input):
        # vision_input shape (batch, seq_len, *), * is the vision feature size
        # our target_cls output shape should be (batch, seq_len)
        target_cls = th.full((vision_input.shape[0], vision_input.shape[1]), 2, dtype=th.long,)

        # Create a tensor with some increasing trend
        if not hasattr(self, "target_cls_rand_p_weight"):
            self.register_buffer("target_cls_rand_p_weight", th.arange(0, vision_input.shape[1] - 2, dtype=th.float, ) + 10, persistent=False)
            # 10 will smooth the distribution so every frame has a chance to be selected

        rand_decided_indices = th.multinomial(self.target_cls_rand_p_weight, vision_input.shape[0], replacement=True) # shape (batch_size)

        # Efficient way to set indices without using a loop
        if not hasattr(self, "indices_range") or self.indices_range.shape[0] != vision_input.shape[0]:
            self.register_buffer("indices_range", th.arange(vision_input.shape[1],).expand(vision_input.shape[0], -1), persistent=False)

        mask = self.indices_range >= rand_decided_indices.unsqueeze(1)

        # Apply mask to target_cls where relevant, changing from irrelevant (2) to intend to achieve (1)
        target_cls[mask] = 1

        # the last frame is the achieved frame (0), and the 2nd last frame is the intend to achieve frame (1)
        target_cls[:, -1] = 0

        return target_cls


    def generate_extra_hard_negatives(
        self, vision_input: th.Tensor, text_input: Sequence[str], target_cls: th.Tensor = None
    ):
        assert target_cls is not None, "Traj Recog model require target_cls to generate hard negatives"
        # ! this is used to generate extra hard negatives if has_extra_data_manipulation is True
        assert vision_input.shape[1] > 2 # the sequence length should be greater than 2

        manipulated_vision_input = th.zeros_like(vision_input)
        # shape [batch, seq, vision_feature*]
        manipulated_text_seq = [] # shape [batch, seq]

        manipulated_target_cls = th.full_like(target_cls, 2) # shape [batch, seq]

        # ! types of manipulation
        # 1. add irrelevant frames to the sequence (front and end)
        # 2. reverse the sequence and delete the first frame, set all being irrelevant
        # 3. change the text instruction to other instruction and set all being irrelevant

        text_ind_dict = dict()
        for i, text in enumerate(text_input):
            if text not in text_ind_dict:
                text_ind_dict[text] = [i]
            else:
                text_ind_dict[text].append(i)

        for i in range(len(text_input)):
            same_text_inds = text_ind_dict[text_input[i]]
            other_i_pool = [x for x in range(len(text_input)) if x not in same_text_inds]
            other_i = np.random.choice(other_i_pool)

            manipulation = np.random.choice([0, 1, 2], p=[0.5, 0.25, 0.25])
            if manipulation == 0: # add irrelevant frames to the sequence

                manipulated_text_seq.append(text_input[i])

                is_irrelevant_front = np.random.choice([0, 1])

                if not hasattr(self, "normal_dist_mean"):
                    self.normal_dist_mean = (vision_input.shape[1])/2
                    self.normal_dist_std_dev= (vision_input.shape[1])/6

                cut_index = int(np.random.normal(loc=self.normal_dist_mean, scale=self.normal_dist_std_dev)) 

                if is_irrelevant_front: # cut the front frame of the original sequence and add other irrelevant frames
                    # Ensure cut_index is within the valid range
                    cut_index = max(1, min(cut_index - 2, vision_input.shape[1]-2))

                    # relevant
                    manipulated_target_cls[i, cut_index:].copy_(target_cls[i, cut_index:]) # set the relevant frames to the end
                    manipulated_vision_input[i, cut_index:].copy_(vision_input[i, cut_index:]) # add the relevant frames to the end

                    # irrelevant
                    manipulated_vision_input[i, :cut_index].copy_(vision_input[other_i, -cut_index:]) # add irrelevant frames to the front

                else: # move the relevant frames to the front and add other irrelevant frames to the end
                    cut_index = max(2, min(cut_index+2, vision_input.shape[1]-1))

                    # relevant
                    manipulated_target_cls[i, :cut_index].copy_(target_cls[i, -cut_index:]) # set the relevant frames to the front
                    manipulated_vision_input[i, :cut_index].copy_(vision_input[i, -cut_index:]) # add the relevant frames to the front

                    # irrelevant
                    manipulated_vision_input[i, cut_index:].copy_(vision_input[other_i, cut_index:]) # add irrelevant frames to the end

            elif manipulation == 1: # reverse the sequence and delete the first frame, set all being irrelevant
                manipulated_text_seq.append(text_input[i])
                # manipulated_target_cls[i] = 2 # all irrelevant
                manipulated_vision_input[i].copy_(vision_input[i].flip(0)) 
                # change the first frame to the second frame
                # this will remove the achieved frame
                manipulated_vision_input[i][0].copy_(manipulated_vision_input[i][1])

            else: # change the text instruction to other instruction and set all being irrelevant
                manipulated_text_seq.append(text_input[other_i])
                # manipulated_target_cls[i] = 2 # all irrelevant
                manipulated_vision_input[i].copy_(vision_input[i]) # copy the original vision input

        # mix two batch
        mixed_vision_input = th.cat([vision_input, manipulated_vision_input], dim=0)
        mixed_text_input = text_input + manipulated_text_seq
        mixed_target_cls = th.cat([target_cls, manipulated_target_cls], dim=0)
        # shuffle the two batch
        indices = th.randperm(mixed_vision_input.shape[0])
        mixed_vision_input = mixed_vision_input[indices]
        mixed_text_input = [mixed_text_input[i] for i in indices]
        mixed_target_cls = mixed_target_cls[indices]
        # split the batch
        vision_input_1 = mixed_vision_input[:len(text_input)]
        text_input_1 = mixed_text_input[:len(text_input)]
        target_cls_1 = mixed_target_cls[:len(text_input)]
        vision_input_2 = mixed_vision_input[len(text_input):]
        text_input_2 = mixed_text_input[len(text_input):]
        target_cls_2 = mixed_target_cls[len(text_input):]
        assert len(text_input_1) == len(text_input_2) == len(text_input)
        return (vision_input_1, text_input_1, target_cls_1), (vision_input_2, text_input_2, target_cls_2)    
