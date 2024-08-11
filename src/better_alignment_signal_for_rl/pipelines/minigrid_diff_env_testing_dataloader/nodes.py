"""
This is a boilerplate pipeline 'minigrid_diff_env_testing_dataloader'
generated using Kedro 0.19.3
"""
from icecream import ic
import numpy as np
from pandas import DataFrame
import random 
from better_alignment_signal_for_rl.pipelines.pair_spliter_n_balancer.dataloader import create_dataloader
from tqdm.auto import tqdm 
import torch as th

from pandarallel import pandarallel

from better_alignment_signal_for_rl.pipelines.train_lang_rew_model.vision_transform import get_traj_transform
pandarallel.initialize(progress_bar=False, nb_workers=6)


def iterating_minigrid_test_dataloader_for_debug(smaller_env_dataloader, larger_env_dataloader):
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
    
    
    ic(smaller_env_dataloader)
    ic(larger_env_dataloader)
    debug_epoch =  2
    
    iter_helper(smaller_env_dataloader, 'Smaller Env Test')
    iter_helper(larger_env_dataloader, 'Larger Env Test')
    

# ! NODE 
def setup_minigrid_generalization_testing_dataloader(
    smaller_env_expert_traj_data_partitions,
    smaller_env_expert_instr_data_df: DataFrame,
    larger_env_expert_traj_data_partitions,
    larger_env_expert_instr_data_df: DataFrame, minigrid_diff_env_testing_dataloader_cfg, lang_rew_model_cfg, traj_instr_pairs_cfg, general_cfg,
):
    env_name = general_cfg['env_name']
    env_name = "minigrid"
    assert env_name == 'minigrid' # only minigrid supporting diff env size
    is_dataloader_tested = minigrid_diff_env_testing_dataloader_cfg["is_dataloader_tested"]
    num_workers = minigrid_diff_env_testing_dataloader_cfg['num_workers']

    traj_length = lang_rew_model_cfg['traj_length']
    chunksize = traj_instr_pairs_cfg['chunksize']
    transform_mean = general_cfg['constant']['clip_mean']
    transform_std = general_cfg['constant']['clip_std']
    
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
        traj_transform=get_traj_transform(lang_rew_model_cfg, general_cfg), # transform when we setup train lang rew model
    )

    smaller_env_test_dataloader = create_dataloader(
        annotation_df=smaller_env_expert_instr_data_df,
        traj_data_partition_dict=smaller_env_expert_traj_data_partitions,
        batch_size=1,
        **common_kwargs,
    )
    larger_env_test_dataloader = create_dataloader(
        annotation_df=larger_env_expert_instr_data_df,
        traj_data_partition_dict=larger_env_expert_traj_data_partitions,
        batch_size=1,
        **common_kwargs,
    )
    
    if not is_dataloader_tested:
        # test dataloader by iterating it 
        ic.enable()
        iterating_minigrid_test_dataloader_for_debug(smaller_env_test_dataloader, larger_env_test_dataloader)
        # ic| traj_d.shape: torch.Size([1, 10, 3, 224, 224]) shape [batch_size, traj_length, 3, H, W]
        # ic| instr_d: ['go to the blue box']
        ic.disable()
        
    
    return smaller_env_test_dataloader, larger_env_test_dataloader