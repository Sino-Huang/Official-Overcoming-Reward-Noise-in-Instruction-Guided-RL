"""
This is a boilerplate pipeline 'eval_lrm'
generated using Kedro 0.19.3
"""
from copy import deepcopy
import os
import random

from better_alignment_signal_for_rl.lang_rew_model_backbone.model.base import BaseModel
from better_alignment_signal_for_rl.pipelines.eval_lrm.exist_impact_eval_enum import Stage_1_Offline_Evaluation_Type
from better_alignment_signal_for_rl.pipelines.train_lang_rew_model.nodes import generate_tag_lst_and_str, setup_lang_rew_model
from natsort import natsorted
from pathlib import Path
import torch as th
import pandas as pd 
import numpy as np 
import pickle 
import time 
import json 
import wandb 
from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from einops import repeat
from .rephrase_constant import MINIGRID_TASKS_REPHRASED, CRAFTER_TASKS_REPHRASED, MONTEZUMA_TASKS_REPHRASED

def manipulation_eval_helper(dataloader, lang_rew_model, device, eval_type, traj_length, env_name):
    with th.no_grad():
        vision_embeds_lst = []
        text_embeds_lst = []
        for data in (testpbar:=tqdm(dataloader, desc=f"Test on {eval_type.name}")):
            traj_d, instr_d = data
            if traj_length == 2:
                traj_d = traj_d[:, -2:, :]
                
            # manipulate the data 
            if eval_type == Stage_1_Offline_Evaluation_Type.H1_1_Compo_Reserved_Traj:
                # reserve the trajectory
                traj_d = traj_d.flip(dims=[1]) # reverse the trajectory
            elif eval_type == Stage_1_Offline_Evaluation_Type.H1_2_State_Not_Do: 
                for i in range(len(instr_d)):
                    instr_d[i] = "Do not " + instr_d[i]
            elif eval_type == Stage_1_Offline_Evaluation_Type.H1_2_State_Rephrase:
                if env_name == "minigrid":
                    rephrase_dict = MINIGRID_TASKS_REPHRASED
                elif env_name == "crafter":
                    rephrase_dict = CRAFTER_TASKS_REPHRASED
                elif env_name == "montezuma":
                    rephrase_dict = MONTEZUMA_TASKS_REPHRASED
                for i in range(len(instr_d)):
                    instr_d[i] = rephrase_dict[instr_d[i]]
                 
            elif eval_type in [Stage_1_Offline_Evaluation_Type.H1_1_Compo_Concat_Two_Instr_Swaped_Traj, Stage_1_Offline_Evaluation_Type.H1_1_Compo_Concat_Two_Instr_First_Only, Stage_1_Offline_Evaluation_Type.H1_1_Compo_Concat_Two_Instr_Second_Only]:   
            
                # have a two two combination 
                inds = list(range(len(instr_d)))
                random.shuffle(inds)
                paired_inds = [(inds[i], inds[i+1]) for i in range(0, len(inds), 2)]
                
                # concat the two instructions 
                new_instr_d = []
                new_traj_d = []
                for first_id, second_id in paired_inds: 
                    if random.random() > 0.5:
                        new_instr_d.append(instr_d[first_id] + ", and then " + instr_d[second_id])
                    else:
                        new_instr_d.append(instr_d[second_id] + " after we " + instr_d[first_id])
                        
                    if eval_type == Stage_1_Offline_Evaluation_Type.H1_1_Compo_Concat_Two_Instr_Swaped_Traj:
                        local_traj_d = [] 
                        local_traj_d.append(traj_d[first_id, -int(traj_length/2):, :])
                        local_traj_d.append(traj_d[second_id, -int(traj_length/2):, :])
                        local_traj_d = th.cat(local_traj_d, dim=0)
                        new_traj_d.append(local_traj_d)
                    elif eval_type == Stage_1_Offline_Evaluation_Type.H1_1_Compo_Concat_Two_Instr_First_Only:
                        new_traj_d.append(traj_d[first_id, :, :])
                        
                    elif eval_type == Stage_1_Offline_Evaluation_Type.H1_1_Compo_Concat_Two_Instr_Second_Only:
                        new_traj_d.append(traj_d[second_id, :, :])    
                        
                new_traj_d = th.stack(new_traj_d, dim=0)
                
                instr_d = new_instr_d
                traj_d = new_traj_d
                
            inputs = lang_rew_model.clip_format_prepare(traj_d, instr_d)
            # to device
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            outputs = lang_rew_model(inputs)
            vision_embeds, text_embeds = outputs
            vision_embeds_lst.append(vision_embeds)
            text_embeds_lst.append(text_embeds)
        vision_embeds = th.cat(vision_embeds_lst, dim=0)
        text_embeds = th.cat(text_embeds_lst, dim=0)
        
        # measure the cosine sim
        cosine_sim_lst_for_manipulation = th.nn.functional.cosine_similarity(vision_embeds, text_embeds)
        cosine_sim_lst_for_manipulation = cosine_sim_lst_for_manipulation.detach().cpu().tolist()
        assert isinstance(cosine_sim_lst_for_manipulation[0], float)
        return cosine_sim_lst_for_manipulation
        
def normal_eval_helper(dataloader, lang_rew_model, device, eval_type, traj_length):
    with th.no_grad():
        vision_embeds_lst = []
        text_embeds_lst = []
        instr_d_lst = []
        for data in (testpbar:=tqdm(dataloader, desc=f"Test on {eval_type.name}")):
            traj_d, instr_d = data
            instr_d_lst.extend(instr_d)
            if traj_length == 2:
                traj_d = traj_d[:, -2:, :] # only take the last two frames
            inputs = lang_rew_model.clip_format_prepare(traj_d, instr_d)
            # to device
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            outputs = lang_rew_model(inputs)
            vision_embeds, text_embeds = outputs
            vision_embeds_lst.append(vision_embeds)
            text_embeds_lst.append(text_embeds)
        vision_embeds = th.cat(vision_embeds_lst, dim=0)
        text_embeds = th.cat(text_embeds_lst, dim=0)
        # matched are in index i,i 
        # not matched are in index i, j where instr_d[i] != instr_d[j]
        cosine_sim_lst_for_matched = th.nn.functional.cosine_similarity(vision_embeds, text_embeds)
        cosine_sim_lst_for_matched = cosine_sim_lst_for_matched.detach().cpu().tolist()
        cosine_sim_lst_for_not_matched = []
        for i in tqdm(range(len(vision_embeds)), desc="Compute Cosine Sim for Not Matched"):
            want_ids = [] 
            for j in range(len(instr_d_lst)):
                if instr_d_lst[i] != instr_d_lst[j]:
                    want_ids.append(j)
            
            expand_vision = repeat(vision_embeds[i], 'C -> N C', N=len(want_ids))
            target_text_embeds = []
            for j in want_ids:
                target_text_embeds.append(text_embeds[j])
            target_text_embeds = th.stack(target_text_embeds, dim=0)
            
            cosine_sim = th.nn.functional.cosine_similarity(expand_vision, target_text_embeds)
            cosine_sim = cosine_sim.detach().cpu().tolist()
            cosine_sim_lst_for_not_matched.extend(cosine_sim)
            
        # check shape are correct
        assert isinstance(cosine_sim_lst_for_matched[0], float)
        assert isinstance(cosine_sim_lst_for_not_matched[0], float)
        return cosine_sim_lst_for_matched, cosine_sim_lst_for_not_matched
        
def form_dataframe_and_draw_single_box_plot(cosine_sim_lst_for_matched, cosine_sim_lst_for_not_matched, lang_rew_model_cfg, env_name, eval_type, cosine_sim_df):
    # form dataframe
    new_rows = pd.DataFrame(columns=['Cosine Sim Score', 'Env', 'Pair Type', "Model"])
    new_rows['Cosine Sim Score'] = cosine_sim_lst_for_matched + cosine_sim_lst_for_not_matched
    new_rows['Env'] = env_name
    new_rows['Pair Type'] = ["Matched"] * len(cosine_sim_lst_for_matched) + ["Not Matched"] * len(cosine_sim_lst_for_not_matched)
    if lang_rew_model_cfg['traj_length'] == 2:
        new_rows['Model'] = "LARVA Goyal et. al."
    elif lang_rew_model_cfg['traj_length'] == 10:
        new_rows['Model'] = "ELLM- Du et. al."
        
    cosine_sim_df = pd.concat([cosine_sim_df, new_rows], ignore_index=True)
    return cosine_sim_df
    

    
    
def update_dataframe_to_score_eval_result(cosine_sim_lst_for_manipulation, lang_rew_model_cfg, env_name, eval_type, cosine_sim_df):
    if eval_type == Stage_1_Offline_Evaluation_Type.H1_1_Compo_Reserved_Traj:
        pair_type_str = "Reserved Traj"
    elif eval_type == Stage_1_Offline_Evaluation_Type.H1_2_State_Not_Do:
        pair_type_str = "Do Not Do `Instr`"
    elif eval_type == Stage_1_Offline_Evaluation_Type.H1_2_State_Rephrase:
        pair_type_str = "Rephrased Instr"
    elif eval_type == Stage_1_Offline_Evaluation_Type.H1_1_Compo_Concat_Two_Instr_Swaped_Traj:
        pair_type_str = "Concatenated Two with Swapped Traj"
    elif eval_type == Stage_1_Offline_Evaluation_Type.H1_1_Compo_Concat_Two_Instr_First_Only:
        pair_type_str = "Concatenated Two with First Traj Only"
    elif eval_type == Stage_1_Offline_Evaluation_Type.H1_1_Compo_Concat_Two_Instr_Second_Only:
        pair_type_str = "Concatenated Two with Second Traj Only"
        
    traj_length = lang_rew_model_cfg['traj_length']
    if traj_length == 2:
        model_str = "LARVA Goyal et. al."
    elif traj_length == 10:
        model_str = "ELLM- Du et. al."
    
    new_rows = pd.DataFrame({
        'Cosine Sim Score': cosine_sim_lst_for_manipulation,
        'Env': [env_name] * len(cosine_sim_lst_for_manipulation),
        'Pair Type': [pair_type_str] * len(cosine_sim_lst_for_manipulation),
        'Model': [model_str] * len(cosine_sim_lst_for_manipulation)
    })
    
    cosine_sim_df = pd.concat([cosine_sim_df, new_rows], ignore_index=True)
    return cosine_sim_df

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def obtain_lang_rew_model_for_eval(updated_lang_rew_model_cfg, general_cfg):
    lang_model = setup_lang_rew_model(updated_lang_rew_model_cfg, general_cfg) 
    return lang_model

# ! NODE
def eval_lang_rew_model(
    test_dataloader,
    smaller_env_test_dataloader,
    larger_env_test_dataloader,
    lang_rew_model_cfg_original,
    general_cfg,
    eval_lrm_cfg,
):
    # get the lang rew model 
    # -- need to loop over the diff settings
    is_offline_eval_completed = eval_lrm_cfg["is_offline_eval_completed"]
    env_name = general_cfg["env_name"]
    
    if not is_offline_eval_completed:
        diff_setting_dict_for_lrm = {
            
            "markovian_hard_signal": {
                "is_markovian": True,
                "traj_length": 2,
                "has_data_augmentation": True,
                "has_extra_data_manipulation": False,
            },
        
            "traj_length_10_hard_signal": {
                "is_markovian": False,
                "traj_length": 10,
                "has_data_augmentation": True,
                "has_extra_data_manipulation": True,
                "has_hard_signal": True,
            },
        
        }
        
        if env_name == "minigrid":
            # add cnn option
            for key, value in diff_setting_dict_for_lrm.items():
                value['model_kwargs'] = dict()
                value['model_kwargs']['minigrid_no_pretrain'] = True
    
            
        stage_1_offline_evaluation_type_list = list(Stage_1_Offline_Evaluation_Type)
        # init dataframe 
        cosine_sim_df = pd.DataFrame(columns=['Cosine Sim Score', 'Env', 'Pair Type', "Model"])
        
        # loop over  
        for key, value in diff_setting_dict_for_lrm.items():
            lang_rew_model_cfg = deepcopy(lang_rew_model_cfg_original)
            lang_rew_model_cfg = recursive_update(lang_rew_model_cfg, value)
            
            lang_rew_model: BaseModel = obtain_lang_rew_model_for_eval(lang_rew_model_cfg, general_cfg)
            
            tag_lst = [f"is_mark_{lang_rew_model_cfg['is_markovian']}", f"type_{lang_rew_model_cfg['lang_rew_type']}", f"extra_mnpl_{lang_rew_model_cfg['has_extra_data_manipulation']}", f"traj_l_{lang_rew_model_cfg['traj_length']}"]
        
            has_hard_signal = lang_rew_model_cfg['cosine_sim_based_params']['has_hard_signal']
            cls_weight = lang_rew_model_cfg['recognition_based_params']['cls_weight']
            env_name = general_cfg['env_name']
            
            if lang_rew_model_cfg['lang_rew_type'] == "trajectory_recognition":
                tag_lst.append(f"cls_w_{cls_weight}")
                
            minigrid_no_pretrain = lang_rew_model_cfg['model_kwargs']['minigrid_no_pretrain']
            if env_name == "minigrid":
                if minigrid_no_pretrain:
                    tag_lst.append("no_pretrain")
                
            tag_lst = natsorted(tag_lst)
            tag_lst_str = "-".join(tag_lst)
            save_dir = os.path.join(os.environ['PWD'], "data/04_lang_rew_model_checkpoint", "lang_rew_models",f"{env_name}",tag_lst_str)
            load_path = os.path.join(save_dir, "best_for_f1.pth")
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Model checkpoint not found at {load_path}")
            lang_rew_model.load_state_dict(th.load(load_path))
            lang_rew_model.eval()
            # to device 
            lang_rew_model = lang_rew_model.to(general_cfg["device"])
            print(f"Eval for Model with Setting {key}, Env: {env_name}")
            time.sleep(1)
            for eval_type in stage_1_offline_evaluation_type_list:
                if eval_type == Stage_1_Offline_Evaluation_Type.H0_1_Normal_Eval_On_Cosine_Sim:
                    cosine_sim_lst_for_matched, cosine_sim_lst_for_not_matched = normal_eval_helper(test_dataloader, lang_rew_model, general_cfg["device"], eval_type, lang_rew_model_cfg['traj_length'])
                    cosine_sim_df = form_dataframe_and_draw_single_box_plot(
                        cosine_sim_lst_for_matched = cosine_sim_lst_for_matched, 
                        cosine_sim_lst_for_not_matched = cosine_sim_lst_for_not_matched, 
                        lang_rew_model_cfg = lang_rew_model_cfg, 
                        env_name = env_name, 
                        eval_type = eval_type, 
                        cosine_sim_df = cosine_sim_df
                    )
                else:
                    cosine_sim_lst_for_manipulation = manipulation_eval_helper(test_dataloader, lang_rew_model, general_cfg["device"], eval_type, lang_rew_model_cfg['traj_length'], env_name) 
                    cosine_sim_df = update_dataframe_to_score_eval_result(
                        cosine_sim_lst_for_manipulation = cosine_sim_lst_for_manipulation, 
                        lang_rew_model_cfg = lang_rew_model_cfg, 
                        env_name = env_name, 
                        eval_type = eval_type, 
                        cosine_sim_df = cosine_sim_df
                    )
                    
            # ! finish 
            del lang_rew_model
            th.cuda.empty_cache()
    
    else:
        # directly load pickle 
        loadpath = os.path.join(os.environ['PWD'], "data/08_reporting/offline_eval/dataframes",  f"dataframe_cosine_sim_score_{env_name}.pkl")
        cosine_sim_df = pd.read_pickle(loadpath)
        
    # save dataframe 
    save_dir = os.path.join(os.environ['PWD'], "data/08_reporting/offline_eval/dataframes")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cosine_sim_df.to_pickle(os.path.join(save_dir, f"dataframe_cosine_sim_score_{env_name}.pkl"))
    
    
    # change the pair type to shorter
    change_dict = {
        "Concatenated Two with Swapped Traj": "Concat But Swap Traj",
        "Concatenated Two with First Traj Only": "Concat But Half Traj",
        "Concatenated Two with Second Traj Only": "Concat But Half Traj",
    }
    
    cosine_sim_df['Pair Type'] = cosine_sim_df['Pair Type'].apply(lambda x: change_dict.get(x, x))
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    # ! seaborn box plot
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.boxplot(data=cosine_sim_df, ax=ax, x='Pair Type', hue='Model', y='Cosine Sim Score')
    plt.title(f"Boxplot of Cosine Similarity Score in `{env_name}` Environment")
    # savefig
    save_dir = os.path.join(os.environ['PWD'], "data/08_reporting/offline_eval", env_name, "diagrams")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, f"cosine_sim_score_offline_eval_{env_name}.png")
    plt.xticks(rotation=30, ha='right')
    plt.subplots_adjust(bottom=0.2)  # Adjust this value as needed
    plt.savefig(save_path, dpi=500)
    return None
        
