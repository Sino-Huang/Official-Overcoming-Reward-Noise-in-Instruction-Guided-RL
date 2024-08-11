"""
This is a boilerplate pipeline 'generate_traj_instr_pairs'
generated using Kedro 0.19.3
"""

from pathlib import Path
from icecream import ic
import imageio
import numpy as np
import pandas as pd
import torch as th
import pickle
import os
import sys
import math
from tqdm import tqdm
from collections import deque
from better_alignment_signal_for_rl.pipelines.expert_policy_setup.baby_ai_bot import BabyAIBot
from better_alignment_signal_for_rl.pipelines.expert_policy_setup.nodes import tensor_to_img_numpy
from better_alignment_signal_for_rl.pipelines.env_setup.env_wrapper import CRAFTER_TASKS
import torch.nn.functional as F
from better_alignment_signal_for_rl.pipelines.expert_policy_setup.nodes import get_minigrid_eval_env_mission
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling


# obs shape (batch, 3, H, W)
# done shape (batch, 1)


def gen_traj_instr_pairs_helper(
    df,
    chunk_id,
    expert_model,
    expert_model_eval_env,
    traj_memory_length,
    chunksize,
    general_cfg,
    progress_qbar,
    is_generation_evaluated: bool, 
):
    hidsize = general_cfg["hidsize"]
    env_name = general_cfg["env_name"]
    device = general_cfg["device"]
    
    constant_cfg = general_cfg["constant"]
    clip_mean = constant_cfg["clip_mean"]
    clip_std = constant_cfg["clip_std"]
    rescale_size = constant_cfg["clip_size"]
    
    trajectory_chunk_file = f"{env_name}/expert_traj_chunk_{chunk_id}.pkl"
    chunk_data_lst = []
    
    
    def _crafter_func():
        if not is_generation_evaluated:
            video_gen_eval_count = 0 
            video_gen_eval_limit = 10 
        return_sum = 0 
        env_reset_count = 0
        qbar = tqdm(total=chunksize, leave=False, desc=f"Construct crafter chunk idx:{chunk_id}, return_avg: 0.00, env_reset_count:{env_reset_count}")
        local_idx = 0 
        # reset env
        obs = expert_model_eval_env.reset()
        states = th.zeros(1, hidsize, device=device)
        prev_achievements = th.zeros((len(CRAFTER_TASKS),), device=device)
        memory_buffer = deque(maxlen=traj_memory_length) # will store image numpy array
        while local_idx < chunksize:
            obs = expert_model.obs_transform_forward(obs)
            outputs = expert_model.act(obs, states=states)
            latents = outputs['latents']
            actions = outputs['actions'] # shape (1, 1) (env_process_num, action_dim)
            obs, rewards, dones, infos = expert_model_eval_env.step(actions)
            
            achievements = infos['achievements'][0] # shape (task_num,)
            
            assert achievements.shape == prev_achievements.shape
            
            # check if done 
            if dones.any():
                env_reset_count += 1
                return_sum += infos['episode_rewards'][0].item()
                return_avg = np.round(return_sum / env_reset_count, 2)
                qbar.set_description(desc=f"Construct crafter chunk idx:{chunk_id}, return_avg:{return_avg}, env_reset_count:{env_reset_count}")
                # refresh states 
                states = th.zeros(1, hidsize, device=device)
                # clear memory buffer
                memory_buffer.clear()
                # refresh prev_achievements
                prev_achievements = th.zeros((len(CRAFTER_TASKS),), device=device)
                
            else:
                # convert obs to image numpy 
                img = tensor_to_img_numpy(
                    x = obs,
                    prev_mean = clip_mean,
                    prev_std = clip_std,
                    rescale_size = rescale_size,
                )
                memory_buffer.append(img)
                # if memory buffer is not full, try to prepend the first image 
                while len(memory_buffer) < memory_buffer.maxlen:
                    memory_buffer.appendleft(memory_buffer[0])
                    
                # ! check if any achievement increment 
                cond = achievements > prev_achievements
                if_record = False 
                if cond.any():
                    # get the last true index 
                    idx = th.nonzero(cond).max().detach().cpu().item()
                    prob = np.random.rand()
                    # only record if prob is smaller than 1/achievements[idx]
                    if_record = prob < (1 / (achievements[idx] + 0.01) + 0.01).detach().cpu().item()
                    
                # update prev_achievements
                prev_achievements = achievements
                
                if if_record: # ! record the trajectory
                    instruction = CRAFTER_TASKS[idx]
                    data_id = chunksize * chunk_id + local_idx 
                    trajectory_local_idx = local_idx
                    # cast the memory buffer to numpy 
                    numpy_memory_buffer = np.stack(memory_buffer, axis=0) # shape (traj_memory_length, H, W, 3)
                    if not is_generation_evaluated:
                        if video_gen_eval_count < video_gen_eval_limit:
                            evaluate_traj_instr_pairs([x for x in numpy_memory_buffer], instruction, env_name)
                            video_gen_eval_count += 1
                        elif video_gen_eval_count == video_gen_eval_limit:
                            raise RuntimeError("Video generation evaluation limit reached, you can set is_generation_evaluated to True to skip this")
                    chunk_data_lst.append(numpy_memory_buffer)
                    
                    # update df
                    df.loc[len(df)] = [data_id, instruction, trajectory_chunk_file, trajectory_local_idx]
                    
                    # increment local idx
                    local_idx += 1
                    qbar.update(1)
                    
                    
            # Update states
            if (rewards > 0.1).any():
                with th.no_grad():
                    obs_encode_ver = expert_model.obs_transform_forward(obs)
                    next_latents = expert_model.encode(obs_encode_ver)
                states = next_latents - latents
                states = F.normalize(states, dim=-1)
        
        # change the chunk data list to numpy array
        chunk_data = np.stack(chunk_data_lst, axis=0) # shape (chunksize, traj_memory_length, H, W, 3)
        
        qbar.close()
        progress_qbar.update(1)
        
        return chunk_data   

    def _minigrid_func():
        nonlocal expert_model
        is_full_observability = expert_model.is_full_observability
        env_reset_count = 0
        due_to_bad_count = 0
        qbar = tqdm(total=chunksize, leave=False, desc=f"Construct minigrid chunk idx:{chunk_id}, env_reset_count:{env_reset_count}, due_to_bad_count:{due_to_bad_count}")
        
        local_idx = 0
        # reset env
        _ = expert_model_eval_env.reset()
        mission = get_minigrid_eval_env_mission(expert_model_eval_env)
        expert_model = BabyAIBot(mission=mission, is_full_observability=is_full_observability)
        instruction = mission.surface(expert_model_eval_env.envs[0])
        
        memory_buffer = deque(maxlen=traj_memory_length) # will store image numpy array
        
        while local_idx < chunksize:
            action = expert_model.replan() # type: int
            # transform to torch tensor
            action = th.tensor([action], dtype=th.int64).view(1, 1)
            # step the env
            obs, _, dones, infos = expert_model_eval_env.step(action)
            
            # check instruction_bak a special case for minigrid
            refresh_env_due_to_bad = False 
            instruction_bak = infos["mission_texts"][0]
            if instruction != instruction_bak:
                # check if done
                if not dones.any(): 
                    refresh_env_due_to_bad = True
                    due_to_bad_count += 1 
                
            # check if done
            if dones.any() or refresh_env_due_to_bad:
                env_reset_count += 1
                
                
                if not refresh_env_due_to_bad: # ! it is a valid done 
                    # append the real final frame
                    real_final = infos['terminal_observation']
                    memory_buffer.append(tensor_to_img_numpy(real_final, clip_mean, clip_std, rescale_size))
                    if len(memory_buffer) == traj_memory_length:
                        data_id = chunksize * chunk_id + local_idx
                        trajectory_local_idx = local_idx
                        
                        # cast the memory buffer to numpy
                        numpy_memory_buffer = np.stack(memory_buffer, axis=0) # shape (traj_memory_length, H, W, 3)
                        chunk_data_lst.append(numpy_memory_buffer)
                        
                        # update df
                        df.loc[len(df)] = [data_id, instruction, trajectory_chunk_file, trajectory_local_idx]
                        
                        # increment local idx
                        local_idx += 1
                        qbar.update(1)
                
                # refresh 
                
                qbar.set_description(desc=f"Construct minigrid chunk idx:{chunk_id}, env_reset_count:{env_reset_count}, due_to_bad_count:{due_to_bad_count}")

                # reset the expert model
                mission = get_minigrid_eval_env_mission(expert_model_eval_env)
                expert_model = BabyAIBot(mission=mission, is_full_observability=is_full_observability)
                instruction = mission.surface(expert_model_eval_env.envs[0])
                # clear memory buffer
                memory_buffer.clear()
                
            else:
                # convert obs to image numpy
                img = tensor_to_img_numpy(
                    x = obs,
                    prev_mean = clip_mean,
                    prev_std = clip_std,
                    rescale_size = rescale_size,
                )
                memory_buffer.append(img)
                # if memory buffer is not full, try to prepend the first image
                while len(memory_buffer) < memory_buffer.maxlen:
                    memory_buffer.appendleft(memory_buffer[0])
                    
        # change the chunk data list to numpy array
        chunk_data = np.stack(chunk_data_lst, axis=0) # shape (chunksize, traj_memory_length, H, W, 3)
        
        qbar.close()
        progress_qbar.update(1)
        
        return chunk_data
        
            
    if env_name == "crafter":
        return _crafter_func
    elif env_name == "minigrid":
        return _minigrid_func
    else:
        raise ValueError(f"env_name {env_name} not supported")


def evaluate_traj_instr_pairs(img_lst, instr, env_name):
    save_path = os.path.join(os.environ["PWD"], f"data/02_intermediate/eval_traj_instr_pairs/{env_name}")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    save_name = f"{instr}.mp4"
    imageio.mimsave(os.path.join(save_path, save_name), img_lst)



# ! NODE
def generate_traj_instr_pairs(
    expert_model,
    expert_model_eval_env,
    general_cfg,
    traj_instr_pairs_cfg,
):
    env_name = general_cfg["env_name"]
    num_pairs = traj_instr_pairs_cfg["num_pairs"]
    chunksize = traj_instr_pairs_cfg["chunksize"]
    assert env_name in ["crafter", "minigrid"]
    if env_name == "crafter":
        traj_memory_length = traj_instr_pairs_cfg["crafter_env_params"][
            "traj_memory_length"
        ]
    elif env_name == "minigrid":
        traj_memory_length = traj_instr_pairs_cfg["minigrid_env_params"][
            "traj_memory_length"
        ]
    else:
        raise ValueError(f"env_name {env_name} not supported")

    chunk_num = math.ceil(num_pairs / chunksize)
    is_generation_evaluated = traj_instr_pairs_cfg["is_generation_evaluated"]
    
    # init pandas dataframe

    columns = [
        "data_id",
        "instruction",
        "trajectory_chunk_file",
        "trajectory_local_idx",
    ]

    df = pd.DataFrame(columns=columns)

    output_dict = dict()

    progress_qbar = tqdm(total=chunk_num, desc="generate trajectory instruction pairs")
    
    for chunk_id in range(chunk_num):
        traj_filename = f"expert_traj_chunk_{chunk_id}"
        # lazy save
        output_dict[traj_filename] = gen_traj_instr_pairs_helper(
            df,
            chunk_id,
            expert_model,
            expert_model_eval_env,
            traj_memory_length,
            chunksize,
            general_cfg,
            progress_qbar,
            is_generation_evaluated,
        )

    # save csv too using lazy save

    return output_dict, df
