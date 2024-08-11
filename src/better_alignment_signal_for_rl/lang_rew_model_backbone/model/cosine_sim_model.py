from copy import deepcopy
import random
import time
from einops import rearrange
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Sequence
from icecream import ic

from better_alignment_signal_for_rl.agent_components.impala_cnn import ImpalaCNN
from better_alignment_signal_for_rl.agent_components.mlp import GatedResidualMLP
from torchvision.transforms.functional import resize as resize_image
from .base import BaseModel

class ExtractLast(nn.Module):
    def forward(self, x):
        return x[:, -1, :]

class SqueezeLength(nn.Module):
    # shape (batch, 1, hidden_size) -> (batch, hidden_size)
    def forward(self, x):
        return x.squeeze(-2)

class CustomImageCosineSimLoss(nn.Module):
    def __init__(self, temperature=0.07): # temperature follows CLIP default
        super(CustomImageCosineSimLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features, instr_d):
        # Detach text features so gradients are not computed for them
        text_features = text_features.detach()

        # Compute img-img similarity
        # image_similarity = th.einsum('i d, j d -> i j', image_features, image_features) 
        
        # Compute text-text similarity
        text_similarity = th.einsum('i d, j d -> i j', text_features, text_features)
        # Min-max normalization of text similarities
        min_val = th.min(text_similarity, dim=1, keepdim=True).values
        max_val = th.max(text_similarity, dim=1, keepdim=True).values
        text_weights = (text_similarity - min_val) / (max_val - min_val + 1e-6) # shape (batch, batch) range will be [0, 1]
        
        # ! --- slower but more accurate way of calculating the loss
        loss = 0 
        for i in range(image_features.shape[0]):
            # Compute image-text similarity
            for j in range(text_features.shape[0]):
                # cosine similarity loss 
                aligned_cond = i == j
                aligned_cond = aligned_cond or instr_d[i] == instr_d[j]
                if aligned_cond: 
                    y = th.tensor(1, dtype=th.long).to(image_features.device)
                    margin = 0
                else:
                    y = th.tensor(-1, dtype=th.long).to(image_features.device)
                    margin = text_weights[i, j].item()
                loss += F.cosine_embedding_loss(image_features[i], text_features[j], y, margin=margin) # if not aligned, but the text is similar, the margin will cause the loss to be smaller
                    
        # mean the loss 
        loss /= (image_features.shape[0] * text_features.shape[0]) 
        
        
        return loss
        # ! --- end slower but more accurate way of calculating the loss ---

        # ! --- faster way of calulating the loss, but it seems not working 
        # p50_vals = th.quantile(text_weights, 0.5, dim=-1, keepdim=True) # shape (batch, 1)
        # under_p50_inds = (text_weights < p50_vals).nonzero(as_tuple=True) 
        # # positive pairs
        # emb_one = image_features
        # emb_two = text_features
        # y = th.ones(emb_one.shape[0], dtype=th.long, device=emb_one.device)
        # # negative pairs
        # emb_one_neg = emb_one[under_p50_inds[0]]
        # emb_two_neg = emb_two[under_p50_inds[1]]
        # y_neg = th.full((emb_one_neg.shape[0],), -1, dtype=th.long, device=emb_one.device)
        # # concatenate the positive and negative pairs
        # emb_one = th.cat([emb_one, emb_one_neg], dim=0)
        # emb_two = th.cat([emb_two, emb_two_neg], dim=0)
        # y = th.cat([y, y_neg], dim=0)
        # loss = F.cosine_embedding_loss(emb_one, emb_two, y)
        # loss /= image_features.shape[0]
        # return loss
        # ! --- end faster way of calculating the loss ---

class CosineSimLangRewModel(BaseModel):
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
        traj_length,
        
        # extra arguments
        has_hard_signal: bool, 
        alpha: int, # for the quantile-based thresholding
        env_name :str , 
        minigrid_no_pretrain: bool, 
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
        self.has_hard_signal = has_hard_signal
        self.alpha = alpha # ! it is a list 
        self._cp_thred_dict = None 

        self.output_size = pretrained_model_output_size
        self.image_clip_loss = CustomImageCosineSimLoss()
        self.minigrid_no_pretrain = minigrid_no_pretrain

        # ! the main layer will only convert the video embedding
        # if xclip, the output should be (batch, hidden_size)
        layer_lst = []
        if "xclip" not in pretrained_model_cls:

            if not is_markovian: # we have a sequence of video embeddings but we only want the last one
                layer_lst.append(ExtractLast()) # output shape (batch, hidden_size)
            else:
                layer_lst.append(SqueezeLength()) # output shape (batch, hidden_size)

        layer_lst.append(GatedResidualMLP(
                insize = dense_kwargs['insize'],
                nhidlayer = dense_kwargs['nhidlayer'],
                outsize = self.output_size,
                hidsize = self.output_size,
                dense_init_norm_kwargs = dense_kwargs['dense_init_norm_kwargs'],
                layer_type = dense_kwargs['dense_model_cls'],
            ) # output shape (batch, output_size), after this we can do cosine similarity
        )
        self.main_vision_layer = nn.Sequential(*layer_lst)

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

    # getter and setter of self.cp_thred_dict
    @property
    def cp_thred_dict(self):
        return self._cp_thred_dict

    @cp_thred_dict.setter
    def cp_thred_dict(self, value):
        self._cp_thred_dict = value

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
        # main_vision_layer
        vision_embeds = self.main_vision_layer(vision_embeds)

        # output shape (batch, hidden_size), (batch, hidden_size), we can do cosine similarity
        return vision_embeds, text_embeds

    def calculate_threshold(self, cal_smx, alpha):
        # Quantile-based Thresholding for Conformal Prediction
        # score vector shape [N, M], where the N is the batch number, and the M is actually the number of classes, however, in our case, we are doing the cosine similarity, so the M is the number of text batch, which is the same as the vision batch
        assert isinstance(cal_smx, np.ndarray)
        # ! 1: get conformal scores
        cal_labels = np.arange(cal_smx.shape[1]) # shape [M] just the index of the pair

        n = cal_smx.shape[0] # number of batch
        cal_scores = 1 - cal_smx[range(n), cal_labels] # get the score of the correct label, the score is 1 - cosine similarity

        # ! 2: get adjusted quantile
        output_thres = dict() 
        if isinstance(alpha, list):
            pass
        else:
            alpha = [alpha]
        for a in alpha:
            q_level = np.ceil((n+1) * (1-a))/n
            qhat = np.quantile(cal_scores, q_level, method="higher")
            threshold = 1 - qhat
            output_thres[a] = threshold

        return output_thres

    def calculate_metrics(self, cosine_sim_vector, threshold):
        assert isinstance(cosine_sim_vector, np.ndarray)
        # shape (i, j) where i is the vision batch and j is the text batch
        predictions = (cosine_sim_vector >= threshold).astype(int)
        # true_labels are just the diagonal matrix
        mshape = cosine_sim_vector.shape[0]
        true_labels = np.eye(mshape).astype(int)
        # flatten both
        predictions = predictions.flatten()
        true_labels = true_labels.flatten()
        # calculate the metrics
        TP = np.sum((predictions == 1) & (true_labels == 1))
        FP = np.sum((predictions == 1) & (true_labels == 0))
        TN = np.sum((predictions == 0) & (true_labels == 0))
        FN = np.sum((predictions == 0) & (true_labels == 1))

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        return precision, recall, f1, accuracy

    def compute_losses(
        self, vision_embeds: th.Tensor, text_embeds: th.Tensor, instr_d :Sequence[str] = None,  **kwargs
    ) -> Dict[str, th.Tensor]:
        # first, calculate the cosine similarity across the batch

        loss = self.image_clip_loss(vision_embeds, text_embeds, instr_d)

        output_dict = {"loss": loss}

        if self.has_hard_signal:  # measure CP quantile for the alignment
            if instr_d is not None:
                # ! if not None, it means we are measuring the performance during evaluation, we need to set incoming batch unique
                # get unique instr_d lst
                instr_d_unique = list(set(instr_d))
                # get the index
                instr_d_idx = []
                # Loop through each unique item
                for item in instr_d_unique:
                    # Find all indices of the current item in instr_d
                    indices = [i for i, x in enumerate(instr_d) if x == item]
                    # Randomly select one index from the list of indices
                    random_index = random.choice(indices)
                    instr_d_idx.append(random_index)
                # get the unique vision_embeds and text_embeds
                cosine_sim_vector = self.compute_cosine_similarity(
                    vision_embeds=vision_embeds[instr_d_idx],
                    text_embeds=text_embeds[instr_d_idx],
                )

            else:
                cosine_sim_vector = self.compute_cosine_similarity(
                    vision_embeds=vision_embeds, 
                    text_embeds=text_embeds,
                ) # shape (i, j) where i is the vision batch and j is the text batch
            cosine_sim_vector_np = cosine_sim_vector.cpu().detach().numpy()

            if cosine_sim_vector_np.shape[0] > 8 and not self.training:
                # two conditions: 1. we have more than one batch, 2. we are not training
                thresholds = self.calculate_threshold(cosine_sim_vector_np, self.alpha)
                output_dict['cp_threshold'] = thresholds # structure dict(alpha: threshold_val)

            # we can also calculate precision, recall, f1 and accuracy
            if self.cp_thred_dict is not None:
                output_dict['precision'] = dict()
                output_dict['recall']  = dict()
                output_dict['f1'] = dict()
                output_dict['accuracy'] = dict()
                for alpha_k in self.cp_thred_dict:
                    thres_v = self.cp_thred_dict[alpha_k]
                    precision, recall, f1, accuracy = self.calculate_metrics(cosine_sim_vector_np, thres_v)
                    output_dict['precision'][alpha_k] = precision
                    output_dict['recall'][alpha_k] = recall
                    output_dict['f1'][alpha_k] = f1
                    output_dict['accuracy'][alpha_k] = accuracy

        return output_dict

    def generate_extra_hard_negatives(self, vision_input: th.Tensor, text_input: Sequence[str], target_cls: th.Tensor = None):
        """
        Generates extra hard negatives for the model.  
        Types of hard negatives:
        1. Negate the instruction and flip the sequence
        2. Concatenate two sequences and the instruction (half half)

        Args:
            vision_input (th.Tensor): The input tensor representing the visual input.
            text_input (Sequence[str]): The input sequence of text.

        Returns:
            Tuple[Tuple[th.Tensor, Sequence[str]], Tuple[th.Tensor, Sequence[str]]]: A tuple containing two tuples.
            The first tuple contains the mixed vision input and text input for the first batch.
            The second tuple contains the mixed vision input and text input for the second batch.
        """

        assert not self.is_markovian # ! we do not support markovian
        # aim: it will generate a same batch size hard negatives, after that we shuffle the two batches and get two mixed set of batch
        # output : ((vision_mixed_input_1, text_mixed_input_1), (vision_mixed_input_2, text_mixed_input_2))
        manipulated_vision_input = th.zeros_like(vision_input)
        # shape [batch, seq, vision_feature*]
        manipulated_text_seq = [] # shape [batch, seq]
        if self.is_markovian:
            assert vision_input.shape[1] == 2 # we want the sequence to be 2 when markovian
        else:
            assert vision_input.shape[1] > 2 
        if "xclip" in self.pretrained_model_cls:
            assert vision_input.shape[1] == 8 # we want the sequence to be 8 when xclip

        # ! types of manipulation
        # 1. negate the instruction and flip the sequence
        # 2. concatenate two sequence and the instruction (half half)
        for i in range(len(text_input)):
            # randomly choose the manipulation
            manipulation = np.random.choice([0, 1])
            if manipulation == 0:
                manipulated_vision_input[i].copy_(vision_input[i].flip(0))
                if self.env_name == "minigrid":
                    rand_type = np.random.choice([0,1])
                    if rand_type == 0:
                        manipulated_text_seq.append(text_input[i].replace("go to", "go away from"))
                    else:
                        manipulated_text_seq.append("do not " + text_input[i])
                else:
                    manipulated_text_seq.append("go away from " + text_input[i])
            else: 
                # concatenate the sequence
                other_i = (i + 1) % len(text_input)
                # concat the last half of the sequence
                who_first = np.random.choice([0, 1])
                first_length = vision_input.shape[1]//2
                second_length = vision_input.shape[1] - first_length
                if who_first == 0:
                    first_input = vision_input[i]
                    second_input = vision_input[other_i]
                    first_input_idx = i
                    second_input_idx = other_i

                else:
                    first_input = vision_input[other_i]
                    second_input = vision_input[i]
                    first_input_idx = other_i
                    second_input_idx = i

                manipulated_vision_input[i][:first_length] = first_input[-first_length:]
                manipulated_vision_input[i][first_length:] = second_input[-second_length:]

                # concat the text
                choice_word = np.random.choice([0, 1]) # either after or , and then 
                if choice_word == 0:
                    manipulated_text_seq.append(text_input[first_input_idx] + ", and then " + text_input[second_input_idx])
                else:
                    manipulated_text_seq.append(text_input[second_input_idx] + " after " + text_input[first_input_idx]) # since we use after, we need to switch the order

        # mix two batch
        mixed_vision_input = th.cat([vision_input, manipulated_vision_input], dim=0)
        mixed_text_input = text_input + manipulated_text_seq
        # shuffle the two batch
        indices = th.randperm(mixed_vision_input.shape[0])
        mixed_vision_input = mixed_vision_input[indices]
        mixed_text_input = [mixed_text_input[i] for i in indices]
        # split the batch
        vision_input_1 = mixed_vision_input[:len(text_input)]
        text_input_1 = mixed_text_input[:len(text_input)]
        vision_input_2 = mixed_vision_input[len(text_input):]
        text_input_2 = mixed_text_input[len(text_input):]
        assert len(text_input_1) == len(text_input_2) == len(text_input)
        return (vision_input_1, text_input_1, None), (vision_input_2, text_input_2, None)      # the last None is for target_cls 
