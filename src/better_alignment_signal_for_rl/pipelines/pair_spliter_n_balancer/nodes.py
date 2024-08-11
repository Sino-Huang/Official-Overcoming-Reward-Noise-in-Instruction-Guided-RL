"""
This is a boilerplate pipeline 'pair_spliter_n_balancer'
generated using Kedro 0.19.3
"""

import time
from icecream import ic
import numpy as np
from pandas import DataFrame
import random
import pandas as pd

from better_alignment_signal_for_rl.pipelines.train_lang_rew_model.vision_transform import get_traj_transform 
from .dataloader import create_dataloader
from tqdm.auto import tqdm 
import torch as th

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=6)

def check_in_keys(row, keys):
    for key in keys:
        if key+".pkl" in row['trajectory_chunk_file'] :
            return True
    return False


def iterating_dataloader_for_debug(train_dataloader, val_dataloader, test_dataloader):
    
    def iter_helper(dloader, desc):
        for _ in tqdm(range(debug_epoch), desc='Epoch'):
            showing_flag = True 
            instr_count_dict = dict()
            for data in tqdm(dloader, desc=desc, leave=False):
                traj_d, instr_d = data
                if showing_flag:
                    showing_flag = False 
                    # display shapes 
                    ic(traj_d.shape)
                    ic(instr_d)
                # count instr
                for instr in instr_d:
                    if instr not in instr_count_dict:
                        instr_count_dict[instr] = 0
                    instr_count_dict[instr] += 1
            ic(instr_count_dict)
    
    
    ic(train_dataloader)
    ic(val_dataloader)
    ic(test_dataloader)
    debug_epoch =  2
    
    iter_helper(train_dataloader, 'Train')
    iter_helper(val_dataloader, 'Val')
    iter_helper(test_dataloader, 'Test')
    
                


# ! NODE
def train_validate_test_split(
    expert_traj_data_partitions,
    expert_instr_data_df: DataFrame,
    pair_spliter_n_balancer_cfg,
):
    train_validate_test_ratio = pair_spliter_n_balancer_cfg["train_validate_test_ratio"] # default [46, 2, 2]

    # shuffle the data
    traj_data_keys = list(expert_traj_data_partitions.keys())
    random.shuffle(traj_data_keys)
    expert_traj_data_partitions = {k: expert_traj_data_partitions[k] for k in traj_data_keys}
    traj_data_keys = list(expert_traj_data_partitions.keys())

    # split the data
    train_keys = traj_data_keys[:train_validate_test_ratio[0]]
    val_keys = traj_data_keys[train_validate_test_ratio[0]:train_validate_test_ratio[0] + train_validate_test_ratio[1]]
    test_keys = traj_data_keys[train_validate_test_ratio[0] + train_validate_test_ratio[1]:]
    # handle montezuma case 
    if len(val_keys) == 0:
        val_keys = [train_keys[-1]]
    if len(test_keys) == 0:
        test_keys = [val_keys[-1]]
    # split df based on keys
    expert_instr_data_df["trajectory_chunk_file"] = expert_instr_data_df["trajectory_chunk_file"].astype(str)

    train_cond = expert_instr_data_df.parallel_apply(check_in_keys, args=(train_keys,), axis=1)
    train_df = expert_instr_data_df[train_cond]
    val_cond = expert_instr_data_df.parallel_apply(check_in_keys, args=(val_keys,), axis=1)
    val_df = expert_instr_data_df[val_cond]
    test_cond = expert_instr_data_df.parallel_apply(check_in_keys, args=(test_keys,), axis=1)
    test_df = expert_instr_data_df[test_cond]
    
    if len(train_keys) > 1: # do not assert for montezuma case
        assert ic(len(train_df) + len(val_df) + len(test_df)) == ic(len(expert_instr_data_df))
        assert len(val_df) == len(test_df)
        assert len(train_df) > len(val_df)


    expert_traj_data_partitions_train = {k: expert_traj_data_partitions[k] for k in train_keys}
    expert_traj_data_partitions_val = {k: expert_traj_data_partitions[k] for k in val_keys}
    expert_traj_data_partitions_test = {k: expert_traj_data_partitions[k] for k in test_keys}

    return [
        expert_traj_data_partitions_train,
        expert_traj_data_partitions_val,
        expert_traj_data_partitions_test,
    ], [train_df, val_df, test_df]


# ! NODE
def setup_balanced_dataloader(
    traj_partition_group, df_group, pair_spliter_n_balancer_cfg, lang_rew_model_cfg, traj_instr_pairs_cfg, general_cfg,
):
    env_name = general_cfg['env_name']
    batch_size = pair_spliter_n_balancer_cfg["batch_size"]
    is_dataloader_tested = pair_spliter_n_balancer_cfg["is_dataloader_tested"]
    num_workers = pair_spliter_n_balancer_cfg['num_workers']
    data_partition_train, data_partition_val, data_partition_test = traj_partition_group
    df_train, df_val, df_test = df_group

    traj_length = lang_rew_model_cfg['traj_length']
    chunksize = traj_instr_pairs_cfg['chunksize']
    transform_mean = general_cfg['constant']['clip_mean']
    transform_std = general_cfg['constant']['clip_std']
    
    # ! we may want to fill the df to make sure the length is at least a chunksize
    while len(df_train) < chunksize:
        lack_num = chunksize - len(df_train)
        if lack_num > len(df_train):
            df_train = pd.concat([df_train]*2, ignore_index=True)
        else:
            df_train = pd.concat([df_train, df_train[:lack_num]], ignore_index=True)
    
    while len(df_val) < chunksize:
        lack_num = chunksize - len(df_val)
        if lack_num > len(df_val):
            df_val = pd.concat([df_val]*2, ignore_index=True)
        else:
            df_val = pd.concat([df_val, df_val[:lack_num]], ignore_index=True)
            
    while len(df_test) < chunksize:
        lack_num = chunksize - len(df_test)
        if lack_num > len(df_test):
            df_test = pd.concat([df_test]*2, ignore_index=True)
        else:
            df_test = pd.concat([df_test, df_test[:lack_num]], ignore_index=True)
        
    seed = general_cfg['seed']
    # init seed 
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    

    common_kwargs = dict(
        env_name= env_name,
        traj_length=traj_length,
        chunksize=chunksize,
        transform_mean=transform_mean,
        transform_std=transform_std,
        num_workers=num_workers,
        is_dataloader_tested=is_dataloader_tested,
    )

    train_dataloader = create_dataloader(
        annotation_df=df_train,
        traj_data_partition_dict=data_partition_train,
        batch_size=batch_size,
        traj_transform=get_traj_transform(lang_rew_model_cfg, general_cfg),  # add transform when we setup train lang rew model
        **common_kwargs,
        
    ) 

    validate_dataloader = create_dataloader(
        annotation_df=df_val,
        traj_data_partition_dict=data_partition_val,
        batch_size=batch_size,
        traj_transform=get_traj_transform(lang_rew_model_cfg, general_cfg),
        **common_kwargs,
    )
    
    test_dataloader = create_dataloader(
        annotation_df=df_test,
        traj_data_partition_dict=data_partition_test,
        batch_size=8, # ! set it 8
        traj_transform=None, # test dataloader do not need data augmentation as we are testing the model
        **common_kwargs,
    )
    
    if not is_dataloader_tested:
        # test dataloader by iterating it 
        ic.enable()
        iterating_dataloader_for_debug(train_dataloader, validate_dataloader, test_dataloader)
        # ic| traj_d.shape: torch.Size([32, 10, 3, 224, 224]) shape [batch_size, traj_length, H, W, 3]
        # ic| instr_d: ['make_stone_sword','place_plant','collect_drink','make_stone_sword','make_stone_pickaxe','collect_drink','wake_up','place_stone','collect_sapling','place_furnace','make_stone_sword','make_wood_sword','collect_stone','collect_stone','collect_drink','collect_sapling','place_furnace','collect_wood','defeat_zombie','place_plant','defeat_zombie','place_plant','place_furnace','place_plant','collect_wood','collect_wood','defeat_zombie','wake_up','collect_stone','place_table','place_furnace','defeat_skeleton']
        
        # train 1 epoch about 13 mins
        ic.disable()
    
    return train_dataloader, validate_dataloader, test_dataloader
    
