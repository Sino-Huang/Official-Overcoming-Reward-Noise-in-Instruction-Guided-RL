import os 
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomApply, RandomAffine, RandomHorizontalFlip, Normalize, ToTensor, Compose

from torchvision.transforms.functional import to_tensor, normalize 
from einops import rearrange
from einops.layers.torch import Rearrange
import imageio 

import numpy as np 
from pathlib import Path


class StoreVideoFrames(nn.Module):
    def __init__(self, store_dir, env_name, helper_instance, *args, **kwargs) -> None:
        super().__init__()
        self.store_dir = store_dir
        self.env_name = env_name
        self.helper_instance = helper_instance
        # mkdir 
        Path(os.path.join(os.environ['PWD'], store_dir, env_name)).mkdir(parents=True, exist_ok=True)
        self.file_name = "transformed_frames"
        
    def forward(self, x):
        # x shape ( (b l), 3, clip_size, clip_size)
        unsqueeze_x = rearrange(x, "(b l) c h w -> b l h w c", b=self.helper_instance.B)
        for i in range(10): # only store 10 videos
            output_path = os.path.join(os.environ['PWD'], self.store_dir, self.env_name, f"{self.file_name}_{i}.mp4")
            # check if exist, if exist, skip
            if os.path.exists(output_path):
                continue
            video_frame_lst = []
            for j in range(unsqueeze_x.shape[1]):
                numpy_frame = unsqueeze_x[i, j].numpy().astype(np.uint8)
                video_frame_lst.append(numpy_frame)
                
            imageio.mimsave(output_path, video_frame_lst)
        return x

class FixTrajLength(nn.Module):
    def __init__(self, traj_length) -> None:
        super().__init__()
        self.traj_length = traj_length
        
    def forward(self, x):
        # x shape (chunks, mem_length, 3, clip_size, clip_size)
        return x[:, -self.traj_length :]
    
    
class NumpyToTensor(nn.Module):
    def forward(self, x):
        # rearrange 
        x = th.from_numpy(x)
        x = rearrange(x, "b l h w c -> b l c h w")
        return x

class CustomRearrange(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.B = None
        
    def forward(self, x):
        # x shape [b l c h w]
        # rearrange 
        self.B = x.shape[0]
        x = rearrange(x, "b l c h w -> (b l) c h w")
        return x

class CustomSplit(nn.Module):
    def __init__(self, pattern, instance_helper) -> None:
        super().__init__()
        self.instance_helper = instance_helper
        self.pattern = pattern
        
    def forward(self, x):
        # x shape [(b l) c h w]
        x = rearrange(x, self.pattern, b=self.instance_helper.B)
        return x
        

class CustomToTensorNormalize(nn.Module):
    def __init__(self, mean_lst, std_lst) -> None:
        super().__init__()
        self.mean = mean_lst
        self.std = std_lst
        
    def forward(self, x: th.Tensor):
        # x shape [(b l) c h w]
        # ensure float32 
        x = x.to(th.float32)
        # devide 255
        x = x / 255.0
        x = normalize(x, mean=self.mean, std=self.std)
        return x

class CustomRandomApply(RandomApply):
    """Apply randomly a list of transformations with a given probability. edited to support shape ((b l) c h w)
    """

    def __init__(self, transforms, p, instance_helper):
        super().__init__(transforms, p)
        self.instance_helper = instance_helper

    def forward(self, img):
        # Calculate the batch size B and the number of sequences per batch L from the total length of the batch dimension
        B, L = self.instance_helper.B, img.shape[0] // self.instance_helper.B

        # Process each sequence in each batch separately
        for b in range(B):
            idx_start = b * L
            idx_end = (b + 1) * L
            img[idx_start:idx_end] = super().forward(img[idx_start:idx_end])
            
        return img


def get_traj_transform(lang_rew_model_cfg, general_cfg):
    # chunk_data shape (chunks, mem_length, clip_size, clip_size, 3)
    
    # need to 
    # 1. shorten the traj length
    # 1.5. flatten 
    # 2. video image manipulation (same video should have same manipulation)
    # - horizontal flip 
    # - RandomAffine 
    # - wrap by RandomApply
    # - record as video 
    # 3. to tensor and normalize 
    # 3.5. rearrange back the shape  
    
    # ! the output should be a function that can take in chunk_data and output the transformed chunk_data
    has_data_augmentation = lang_rew_model_cfg['has_data_augmentation']
    if not has_data_augmentation:
        return None 
    
    traj_length = lang_rew_model_cfg['traj_length']
    clip_mean = general_cfg['constant']['clip_mean']   
    clip_std = general_cfg['constant']['clip_std']
    env_name = general_cfg['env_name']
    if env_name == "crafter" or env_name == "montezuma": # flip the image will change the semantics
        flip_p = 0.0
    else:
        flip_p = 0.5
    custom_rearrrange = CustomRearrange()
    compose = Compose([
        NumpyToTensor(), # shape [b l c h w]
        FixTrajLength(traj_length),
        custom_rearrrange, # shape [(b l) c h w]
        CustomRandomApply(nn.ModuleList([
            RandomHorizontalFlip(p=flip_p), # If the image is torch Tensor, it is expected  to have [..., H, W] shape,
            RandomAffine(degrees=10, translate=(0.13, 0.13), scale=(0.8, 1.1), shear=10),
        ]), p=0.5, instance_helper=custom_rearrrange),
        CustomToTensorNormalize(mean_lst=clip_mean, std_lst=clip_std),
        CustomSplit(pattern="(b l) c h w -> b l c h w", instance_helper=custom_rearrrange)
    ])
    
    return compose

