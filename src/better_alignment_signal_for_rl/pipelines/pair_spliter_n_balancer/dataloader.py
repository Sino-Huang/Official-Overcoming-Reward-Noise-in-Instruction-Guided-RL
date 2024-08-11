from collections import deque
from pathlib import Path
import imageio
import torch as th
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms 
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms.functional import normalize
import os 
import sys 
import time 
import numpy as np
from icecream import ic 
from tqdm.auto import tqdm 
from torchvision.transforms import InterpolationMode
from einops import rearrange
from functools import partial
import threading
import pandas as pd 

# df cols [data_id,instruction,trajectory_chunk_file,trajectory_local_idx]
# ! the input traj shape should be (chunks, mem_length, clip_size, clip_size, 3) image numpy array

def default_transform(data: np.ndarray, traj_length: int, transform_mean, transform_std):
    # shape of data (chunks, mem_length, clip_size, clip_size, 3)
    images_tensor = th.from_numpy(data).float()
    images_tensor = images_tensor[:, -traj_length:]
    c = images_tensor.shape[0]
    images_tensor = rearrange(images_tensor, 'c m h w d -> (c m) d h w')
    # normalize 
    images_tensor /= 255.0
    images_tensor = normalize(images_tensor, mean=transform_mean, std=transform_std)
    # split (c m)
    images_tensor = rearrange(images_tensor, '(c m) d h w -> c m d h w', c=c)
    return images_tensor
    
class ChunkSampler(Sampler):
    """Sampler that fetches chunks of indices, shuffling chunks and within each chunk each epoch."""
    def __init__(self, num_samples, chunk_size):
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.indices = None 
        self.cur_index = None 
        self.has_init = False
    
    def reset(self):
        if self.has_init:
            if self.cur_index != self.indices[-1]:
                # ! it means that we have not finished the last chunk, thus no need to reset
                return 
        self.has_init = True
        # Generate a new random order of chunks each epoch
        chunk_indices = th.randperm(self.num_samples // self.chunk_size)
        indices = []
        for chunk_idx in chunk_indices:
            # Shuffling within the chunk
            internal_indices = th.randperm(self.chunk_size) + chunk_idx * self.chunk_size
            indices.extend(internal_indices.tolist())
        self.indices = indices
        
    def __iter__(self):
        for i in self.indices:
            self.cur_index = i
            yield i

    def __len__(self):
        return self.num_samples

class VisionLanguageDataset(Dataset):
    """By default it will minibatch shuffle the data each epoch."""
    def __init__(
        self,
        annotation_df,
        traj_data_partition_dict,
        traj_length,
        chunksize,
        transform_mean,
        transform_std,
        env_name,
        batch_size,
        traj_transform=None, # ! handled by training process 
        is_dataloader_tested = True,
    ):
        self.annotation_df = annotation_df
        self.traj_data_partition_dict = traj_data_partition_dict
        self.traj_length = traj_length
        self.chunksize = chunksize
        self.env_name = env_name
        self.batch_size = batch_size # just for info 
        self.cache_traj_chunk_key_deque = deque(maxlen=2)
        self.cache_traj_chunk_data_deque = deque(maxlen=2)
        self.is_dataloader_tested = is_dataloader_tested
        if not is_dataloader_tested:
            self.video_eval_count = 0
            self.video_count_limit = 10
        

        if traj_transform is None:
            # default transform ref: https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L81
            # totensor, normalize
            self.traj_transform = partial(default_transform, traj_length=traj_length, transform_mean=transform_mean, transform_std=transform_std)
        else:
            self.traj_transform = traj_transform
            
        # sort the df by trajectory_chunk_file and then by trajectory_local_idx
        self.annotation_df = self.annotation_df.sort_values(by=["trajectory_chunk_file", "trajectory_local_idx"])
        
        # threading 
        self.prefetch_thread = None
        self.prefetch_lock = threading.Lock()
        self.determined_looping_indices = None 
        self.cur_index = None 
        self.cur_i_index = None
        self.thread_running = False 
        
    def reset_and_balance(self):
        self.balanced_annotation_df = self.annotation_df.copy()
        if self.env_name == "crafter" and self.batch_size > 1:
            # ! balance the data
            for start_i in tqdm(range(0, len(self.balanced_annotation_df), self.chunksize), desc="Balancing data"):
                end_i = start_i + self.chunksize
                group = self.balanced_annotation_df.iloc[start_i:end_i]
                # check if it is balanced 
                value_counts = group['instruction'].value_counts()
                # ic("Before balancing")
                # ic(value_counts)
                # calculate how many samples to upsample
                mean = int(value_counts.mean())
                upsample_dict = {}
                downsample_dict = {}
                for instr, count in value_counts.items():
                    if count < mean:
                        upsample_dict[instr] = mean - count
                    elif count - 1 > mean:
                        downsample_dict[instr] = count - 1 - mean
                
                exact_total_manipulation_num = sum(downsample_dict.values())
                
                upsample_sum = sum(upsample_dict.values())
                
                while upsample_sum != exact_total_manipulation_num:
                    diff_num = np.abs(exact_total_manipulation_num - upsample_sum)
                    if upsample_sum < exact_total_manipulation_num:
                        if upsample_sum == 0:
                            part = {k: v for k, v in value_counts.items() if v <= mean}
                            part_sum = sum(part.values())
                            instrs = np.random.choice(list(part.keys()), diff_num, p=[v/part_sum for v in part.values()], replace=True)
                        else:
                            instrs = np.random.choice(list(upsample_dict.keys()), diff_num, p=[v/upsample_sum for v in upsample_dict.values()], replace=True)
                        for instr in instrs:
                            if instr not in upsample_dict:
                                upsample_dict[instr] = 1
                            else:
                                upsample_dict[instr] += 1
                            upsample_sum += 1
                    else:
                        instrs = np.random.choice(list(upsample_dict.keys()), diff_num, replace=False)
                        for instr in instrs:
                            upsample_dict[instr] -= 1
                            upsample_sum -= 1
                            
                # remove key if val == 0
                upsample_dict = {k: v for k, v in upsample_dict.items() if v != 0}
                    
                # group by instruction
                group_by_instr = group.groupby('instruction')
                # get downsample iloc candidates 
                downsample_locs_candidates = []
                for instr, prob in downsample_dict.items():
                    avail_downsample_locs = group_by_instr.get_group(instr).index
                    
                    num = downsample_dict[instr]
                    
                    downsample_locs = np.random.choice(avail_downsample_locs, num, replace=False)
                    downsample_locs_candidates.extend(downsample_locs)
                
                # upsample
                for instr, count in upsample_dict.items():
                    upsample_locs_cand = group_by_instr.get_group(instr).index
                    # pick random iloc to upsample
                    upsampled_locs = np.random.choice(upsample_locs_cand, count, replace=True)
                    upsampled_rows = group.loc[upsampled_locs]
                    
                    # these will occupy some of the downsampled rows
                    downsample_locs = downsample_locs_candidates[:count]
                    group.loc[downsample_locs] = upsampled_rows.values
                    # remove the downsampled ilocs from the candidates
                    downsample_locs_candidates = downsample_locs_candidates[count:]
                    
                # check if it is balanced 
                value_counts = self.balanced_annotation_df.iloc[start_i:end_i]['instruction'].value_counts()
                # ic("After balancing")
                # ic(value_counts)
            
            # ! --- end of balance the data ---
        # get instruction series
        self.instruction_series = self.balanced_annotation_df["instruction"]
        # get trajectory_chunk_file series, remove ".pkl" extension
        self.traj_chunk_file_series = self.balanced_annotation_df["trajectory_chunk_file"].str.replace(".pkl", "").str.split("/", expand=True).loc[:,1]
        # get local index series
        self.traj_local_idx_series = self.balanced_annotation_df["trajectory_local_idx"]

    def init_prefetch(self):
        # start prefetching and first chunks 
        if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
            self.prefetch_thread = threading.Thread(target=self._prefetch_data)
            self.thread_running = True
            self.prefetch_thread.start()
            
    def _prefetch_data(self):
        while self.thread_running:
            next_chunk_key = self._determine_next_chunk_key()
            if next_chunk_key is not None and next_chunk_key not in self.cache_traj_chunk_key_deque:
                # ic("going to prefetch", next_chunk_key)
                # load the chunk data (IO bound)
                chunk_data = self.traj_data_partition_dict[next_chunk_key]()
                # * transform the chunk data
                chunk_data_trans = self.traj_transform(chunk_data) # chunk_data shape (chunks, mem_length, clip_size, clip_size, 3)
                del chunk_data
                chunk_data = chunk_data_trans
                
                # load to the cache
                with self.prefetch_lock:
                    # ic("finished prefetching", next_chunk_key)
                    self.cache_traj_chunk_key_deque.append(next_chunk_key)
                    self.cache_traj_chunk_data_deque.append(chunk_data)
            time.sleep(0.5) 
        # clear the cache
        with self.prefetch_lock:
            self.cache_traj_chunk_key_deque.clear()
            self.cache_traj_chunk_data_deque.clear()
            
    def stop(self):
        self.thread_running = False
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()
            
    def _determine_next_chunk_key(self):      
        if self.determined_looping_indices is None:
            raise RuntimeError("determined_looping_indices is not set, need to set it before each epoch, obtaining from the Sampler")
        # check the size of the cache_traj_chunk_key_deque
        if len(self.cache_traj_chunk_key_deque) == 0:
            # get the first chunk key
            index = self.determined_looping_indices[0]
            # ic("first chunk key associated index", index)
            trajectory_chunk_file = self.traj_chunk_file_series.iloc[index]
            return trajectory_chunk_file
        else:
            if self.cur_index is not None: 
                # get the next chunk key 
                i_index = self.determined_looping_indices.index(self.cur_index)
                # the next chunk index will associated with the index = (i_index + chunksize) % len(determined_looping_indices)
                next_i_index = i_index + self.chunksize
                if next_i_index >= len(self.determined_looping_indices):
                    return None
                next_index = self.determined_looping_indices[next_i_index]
                trajectory_chunk_file = self.traj_chunk_file_series.iloc[next_index]
                return trajectory_chunk_file
            else:
                return None 
    def _read_captions(
        self, index,
    ):
        return self.instruction_series.iloc[index]

    def __len__(self):
        return len(self.annotation_df)
    
    def __getitem__(self, index):
        self.cur_index = index
        assert self.determined_looping_indices[self.cur_i_index + 1] == index
        self.cur_i_index += 1
        
        
        instruction = self._read_captions(index)
        trajectory_chunk_file = self.traj_chunk_file_series.iloc[index]
            
        while trajectory_chunk_file not in self.cache_traj_chunk_key_deque:
            # ic(index, trajectory_chunk_file, self.traj_chunk_file_series.iloc[index-5:index+5])
            time.sleep(1) # default 1, somehow decrease the sleep time will freeze the prefetching thread

        with self.prefetch_lock:
            cache_ind = self.cache_traj_chunk_key_deque.index(trajectory_chunk_file)
            chunk_data = self.cache_traj_chunk_data_deque[cache_ind]
                
        
        local_ind = self.traj_local_idx_series.iloc[index]
        
        traj_data = chunk_data[local_ind] # shape (mem_length, 3, clip_size, clip_size)
        # if not self.is_dataloader_tested:
        #     save_dir = os.path.join(os.environ['PWD'], "data/02_intermediate/dataloader_alignment_test", self.env_name)
        #     Path(save_dir).mkdir(parents=True, exist_ok=True)
        #     video_name = f"{instruction}.mp4"
        #     video_lst = []
        #     mean = th.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        #     std = th.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        #     for t in traj_data:
        #         # denormalize
        #         t = t * std + mean
        #         # 255
        #         t = t * 255.0
        #         # clamp 
        #         t = th.clamp(t, 0, 255)
        #         t = t.permute(1, 2, 0).numpy().astype(np.uint8)
        #         video_lst.append(t)
                
        #     imageio.mimsave(os.path.join(save_dir, video_name), video_lst)
        #     if self.video_eval_count < self.video_count_limit:
        #         self.video_eval_count += 1
        #     else:
        #         raise RuntimeError("Video evaluation limit reached, set is_dataloader_tested to True to disable this feature")
            
        return traj_data, instruction
     
class VisionLanguageDataloader:
    def __init__(self, dataloader : DataLoader):
        self.dataloader = dataloader
        self.sampler = dataloader._index_sampler
        self.dataset : VisionLanguageDataset= dataloader.dataset
        
    def before_epoch(self):
        # assign the looping indices to the dataset
        with self.dataset.prefetch_lock:
            self.dataset.cache_traj_chunk_key_deque.clear()
            self.dataset.cache_traj_chunk_data_deque.clear()
        self.dataset.cur_i_index = -1
        self.dataset.cur_index = None
        self.dataset.reset_and_balance()
        self.sampler.sampler.reset()
        self.dataset.determined_looping_indices = self.sampler.sampler.indices
        self.dataset.init_prefetch()
        # ic("finish init prefetch", self.dataset.determined_looping_indices[:20])
        
    def after_epoch(self):
        # stop the prefetching threading 
        self.dataset.stop()
    
    def __iter__(self):
        self.before_epoch()
        try:
            for data in self.dataloader:
                yield data
        finally:
            self.after_epoch()

    def __len__(self):
        return len(self.dataloader)
           
        
def create_dataloader(
    annotation_df,
    traj_data_partition_dict,
    traj_length,
    chunksize,
    transform_mean,
    transform_std,
    batch_size,
    num_workers,
    env_name,
    traj_transform=None,
    is_dataloader_tested=True, 
):
    
    dataset = VisionLanguageDataset(
        annotation_df = annotation_df,
        traj_data_partition_dict = traj_data_partition_dict,
        traj_length = traj_length,
        chunksize = chunksize,
        transform_mean = transform_mean,
        transform_std = transform_std,
        env_name=env_name,
        batch_size= batch_size,
        traj_transform =  traj_transform,
        is_dataloader_tested= is_dataloader_tested,
    )
    
    sampler = ChunkSampler(len(dataset), chunksize)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True, 
    )
    
    custom_dataloader = VisionLanguageDataloader(dataloader)
    
    return custom_dataloader

# Assuming you have a batch of data 'data' and 'target' to be transferred to GPU
# data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
