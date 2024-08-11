from copy import deepcopy
import os
from pathlib import Path
import time
from typing import Dict

import numpy as np
from better_alignment_signal_for_rl.pipelines.env_setup.env_wrapper import CRAFTER_TASKS, MINIGRID_TASKS_ONEHOT_ENCODER, VecPyTorch
from better_alignment_signal_for_rl.pipelines.env_setup.nodes import MONTEZUMA_NSTEP_EPOCH_RATIO, RolloutStorage
from better_alignment_signal_for_rl.pipelines.reward_machine_setup.nodes import CRAFTER_MINE_DIAMOND, calculate_lang_rew_raw
from better_alignment_signal_for_rl.pipelines.rl_policy_setup.int_rew_utils import create_intrinsic_rewards_deir, create_intrinsic_rewards_rnd
from better_alignment_signal_for_rl.ppo_backbone.model.montezuma_model import RNDModel
from better_alignment_signal_for_rl.ppo_backbone.model.base import BaseModel
import torch as th
from better_alignment_signal_for_rl.agent_components.ir_deir_model import DiscriminatorModel
from icecream import ic 
from better_alignment_signal_for_rl.ppo_backbone.model.ppo_rnn import PPORNNModel
from torchvision.transforms.functional import resize, rgb_to_grayscale
from einops import rearrange
import pickle
from einops import repeat
from tqdm.auto import tqdm 


# handle sample rollout to bridge the Storage Update and Policy Update
# ! the training process involves running the env and collect rollout data, then optimize the model using the rollout data

# handle extra instr_str inputs when env_name == "minigrid"
# add beta parameter into consideration, .act is responsible for the beta, currently we do not modify .act in the PPOADModel but just in sample_rollouts function because they only use that to get pi_logits distribution and does not impact the action sampling

def postprocess_inputs(
    inputs: Dict[str, th.Tensor], env_name: str, beta_ceoff, nproc
) -> Dict[str, th.Tensor]:
    if env_name == "minigrid":
        goal_info = MINIGRID_TASKS_ONEHOT_ENCODER.transform(
            inputs["instr_str"]
        )  # shape [env_size, dim], dim = 55
        goal_info = th.from_numpy(goal_info).float().to(inputs["obs"].device)
        inputs["goal_info"] = goal_info  # shape [env_size, dim]
    else:
        goal_info = None
        inputs["goal_info"] = goal_info

    if beta_ceoff is not None:  # if beta is activated, then we need to pass beta to the policy model
        inputs["beta"] = th.full(
            (nproc, 1), beta_ceoff, device=inputs["obs"].device
        )  # shape [env_size, 1]
    else:
        inputs["beta"] = None
        
    return inputs, goal_info


def safe_stack(arrays, axis=0):
    if len(arrays) == 0: # handle empty arrays
        return np.array([])
    return np.stack(arrays, axis=axis)

def postprocess_inputs_montezuma(inputs):
    inputs['obs_single'] = resize(rgb_to_grayscale(inputs['obs']), (84, 84), antialias=False)
    rnn_obs = inputs['rnn_obs'] # shape [nproc, 3, 3, 64, 64]
    nproc = rnn_obs.shape[0]
    rnn_obs_flatten = rearrange(rnn_obs, "b r c h w -> (b r) c h w")
    rnn_obs_flatten = resize(rgb_to_grayscale(rnn_obs_flatten), (84, 84), antialias=False) # shape [nproc*3, 1, 84, 84]
    rnn_obs_flatten = rnn_obs_flatten.squeeze(1) # shape [nproc*3, 84, 84]
    rnn_obs = rearrange(rnn_obs_flatten, "(b c) h w -> b c h w", b=nproc) # shape [nproc, 3, 84, 84]
    inputs['obs'] = rnn_obs # we replace the obs with rnn_obs


def sample_rollouts(
    env_name: str, 
    venv: VecPyTorch,
    policy_model: BaseModel,
    int_rew_model : DiscriminatorModel, 
    lang_rew_model,
    has_exploration_reward,
    has_language_reward,
    storage: RolloutStorage,
    reward_machine_lst, 
    nproc,
    epoch,
    tag_lst_str,
    lang_rew_coef,
    lang_reward_function_type,
    traj_length,
    resize_size_for_lang_rew_model,
    has_hard_signal,
    seed,
    **kwargs,
) -> Dict[str, np.ndarray]:
    # at this point we shall compute the int and lang rew and save to storage

    if "beta_ceoff" in kwargs:
        beta_ceoff = kwargs["beta_ceoff"]
    else:
        beta_ceoff = None

    # Set model to eval model
    policy_model.eval()
    if has_exploration_reward:
        int_rew_model.eval()
    if has_language_reward and lang_rew_model is not None and reward_machine_lst is not None:
        lang_rew_model.eval()

    # Sample rollout
    episode_rewards = []
    episode_lengths = []
    achievements = []
    successes = []

    # Sample int rew
    if has_exploration_reward:
        if "obs_rms" in kwargs:
            last_model_mems = storage.init_int_rew_storage(obs_rms = kwargs["obs_rms"])
        else:
            last_model_mems = storage.init_int_rew_storage()
        if isinstance(int_rew_model, DiscriminatorModel):
            # check how to init them at line 214 in ppo_rollout.py in deir project
            if getattr(storage, "episodic_obs_emb_history", None) is not None:
                episodic_obs_emb_history = storage.episodic_obs_emb_history
                episodic_trj_emb_history = storage.episodic_trj_emb_history
            else:
                episodic_obs_emb_history = [None for _ in range(nproc)]
                episodic_trj_emb_history = [None for _ in range(nproc)]
        else:
            episodic_obs_emb_history = None
            episodic_trj_emb_history = None
            last_model_mems = None
    else:
        last_model_mems = None

    # we shall measure the mean episodic return
    episode_sum_return_ext = 0.0
    episode_cached_return_ext = np.zeros(nproc)
    episode_sum_return_int = 0.0
    episode_cached_return_int = np.zeros(nproc)
    episode_sum_return_lang = 0.0
    episode_cached_return_lang = np.zeros(nproc)
    episode_num = 0 
    episode_sum_step = 0

    # Init agent location info for montezuma and minigrid
    if env_name in ["montezuma", "minigrid"]:
        agent_location_storage_per_episode = []
        if getattr(storage, "agent_location_storage_cache", None) is not None:
            agent_location_storage_cache = storage.agent_location_storage_cache
        else:
            agent_location_storage_cache = [[] for _ in range(nproc)]

    # ! Init the per episode lang rew triggered flag for diff instructions
    # it should be a dictionary, the key is the instruction_with_room_id (montezuma), or instruction with seed (minigrid) or instruction (crafter), the value is the triggered times across the rollout. if the instruction is not triggered, it should not be in the dictionary.
    if lang_rew_model is not None:
        lang_rew_instr_trigger_count_per_rollout = dict()

        if getattr(storage, 'lang_rew_trigger_list_cache', None) is not None:
            # lang_rew_trigger_list_cache will store the triggered instruction list for each env in one episode
            lang_rew_trigger_list_cache = storage.lang_rew_trigger_list_cache
        else:
            lang_rew_trigger_list_cache = [set() for _ in range(nproc)]
    else:
        lang_rew_instr_trigger_count_per_rollout = None

    # Initialize policy memory for PPO RNN
    if isinstance(policy_model, PPORNNModel):
        last_policy_mems = storage.init_policy_memory(rnn_layer_num=policy_model.gru_layers, feature_dim = policy_model.gru_output_size, device=storage.device, policy_model=policy_model )
    else:
        last_policy_mems = None

    # ! Initialize Reward Heatmap for montezuma
    if env_name == "montezuma":
        reward_heatmap_per_episode = []
        if getattr(storage, "reward_heatmap_cache", None) is not None:
            reward_heatmap_cache = storage.reward_heatmap_cache
        else:
            reward_heatmap_cache = [np.zeros((400, 400, 20), dtype=float) for _ in range(nproc)]  # shape, x ,y , room_id

    # ! initialize the instr_text for language reward model
    if has_language_reward and lang_rew_model is not None and reward_machine_lst is not None:
        if getattr(storage, "cur_instr_text_lst", None) is not None:
            cur_instr_text_lst = storage.cur_instr_text_lst
        else:
            cur_instr_text_lst = []
            if env_name == "crafter":
                for idx in range(nproc):
                    cur_instr_text_lst.append(reward_machine_lst[idx].reset(epoch, "normal", 0))
            elif env_name == "montezuma":
                for idx in range(nproc):
                    cur_instr_text_lst.append(reward_machine_lst[idx].reset("normal", 0))
            elif env_name == "minigrid":
                goal_str_lst = venv.get_attr("goal_str_lst")
                for idx, goal_str in enumerate(goal_str_lst):
                    cur_instr_text_lst.append(reward_machine_lst[idx].reset(goal_str[0], goal_str[1], "normal", 0))

            # put them into storage reward_machine_instr_str step 0
            if not (env_name == "crafter" and epoch < reward_machine_lst[0].exploration_epoch_num):
                cur_instr_text_lst_np = np.array(cur_instr_text_lst, dtype="<U100").reshape(-1, 1)
                storage.reward_machine_instr_str[0] = cur_instr_text_lst_np
            
            if env_name == "montezuma":
                reward_machine_instr_additional_key = venv.get_attr("cur_room")
                storage.reward_machine_instr_additional_key[0] = np.array(reward_machine_instr_additional_key, dtype="<U100").reshape(-1, 1)
            elif env_name == "minigrid":
                reward_machine_instr_additional_key = venv.get_attr("seed_idx")
                storage.reward_machine_instr_additional_key[0] = np.array(reward_machine_instr_additional_key, dtype="<U100").reshape(-1, 1)
                
        if env_name == "crafter" and epoch == reward_machine_lst[0].exploration_epoch_num:
            # we reset to normal immediately after exploration
            cur_instr_text_lst = []
            for idx in range(nproc):
                cur_instr_text_lst.append(reward_machine_lst[idx].reset(epoch, "normal", 0))
                
    # ! initialize lang_rew_trajectory_history
    if has_language_reward and lang_rew_model is not None and reward_machine_lst is not None:
        if getattr(storage, "lang_rew_traj_history", None) is not None:
            lang_rew_traj_history = storage.lang_rew_traj_history
        else:
            # keep the lang_rew_traj_history to be small so we can save memory
            # shape [env_size, traj_length, 3, 64, 64] 
            lang_rew_traj_history = th.zeros((nproc, traj_length, *venv.observation_space.shape), dtype=th.float32, device=storage.device)
            # copy step 0 obs to lang_rew_traj_history
            obs_full = repeat(storage.obs[0], 'b c h w -> b r c h w', r=traj_length)
            lang_rew_traj_history[:] = obs_full
    # ! init episode step count 
    if getattr(storage, "nproc_episode_step_count", None) is not None:
        nproc_episode_step_count = storage.nproc_episode_step_count
    else:
        nproc_episode_step_count = [0 for _ in range(nproc)]
            
    # ! initialize Offset Histogram Analysis
    if has_language_reward and lang_rew_model is not None and reward_machine_lst is not None and not has_hard_signal:
        if getattr(storage, "offset_analysis_info", None) is not None:
            offset_analysis_info = storage.offset_analysis_info
        else:
            if env_name == "montezuma":
                offset_analysis_info = venv.get_attr('cur_goal_index')
                
            elif env_name == "minigrid":
                offset_analysis_info = venv.get_attr('bonus_flag')
                
            elif env_name == "crafter":
                # crafter is special as it have a separated achievement info 
                offset_analysis_info = [
                    [0 for task in CRAFTER_TASKS] for _ in range(nproc)
                ] # shape [env_size, len(CRAFTER_TASKS)]
                
        

    for step in tqdm(range(storage.nstep), desc="rollout...", leave=False): # nstep is the number of steps in each rollout

        # Pass through model
        # storage.get_inputs input requires goal_info for onehot encoding of instr_str
        inputs = storage.get_inputs(step) # keys(): ['obs', 'states'], shape of states: [8, 1024], shape of obs: [8,3,64,64], 8 is the env_size

        if env_name == "montezuma":
            postprocess_inputs_montezuma(inputs)

        inputs, goal_info = postprocess_inputs(inputs, env_name, beta_ceoff, nproc)

        outputs = policy_model.act(**inputs) # act will only use obs value from inputs, output is a dict containing ["latents", "pi_latents", "vf_latents", "pi_logits", "vpreds", "actions", "log_probs"], if montezuma, we may have vpreds_int
        actions = outputs["actions"] # shape: [8, 1] (env_size, action_dim)

        # Step environment
        obs, rewards, dones, infos = venv.step(actions)
        for i in range(nproc):
            nproc_episode_step_count[i] += 1
        # ! Offset analysis 
        if has_language_reward and lang_rew_model is not None and reward_machine_lst is not None and not has_hard_signal:
            if env_name == "montezuma":
                new_offset_analysis_info = venv.get_attr('cur_goal_index')
                
            elif env_name == "minigrid":
                new_offset_analysis_info = venv.get_attr('bonus_flag')
                
            elif env_name == "crafter":
                new_offset_analysis_info = infos['achievements']
                new_offset_analysis_info = new_offset_analysis_info.detach().cpu().numpy().tolist()
                
            if not (env_name == "crafter" and epoch < reward_machine_lst[0].exploration_epoch_num):
                for i in range(nproc):
                    reward_machine_lst[i].analyze_offset_actual_status(offset_analysis_info[i], new_offset_analysis_info[i], nproc_episode_step_count[i])
                
            # update offset_analysis_info
            offset_analysis_info = new_offset_analysis_info
            
            
        # save agent location info for montezuma and minigrid
        if env_name in ["montezuma", "minigrid"]:
            for env_id, lst in enumerate(agent_location_storage_cache):
                lst.append(infos["agent_locations"][env_id])

        if env_name == "montezuma":
            is_state_visited : np.ndarray = infos["is_state_visited"]
            # true become 1, false become 0
        outputs["obs"] = obs # this is actually the next obs
        outputs["rewards"] = rewards # shape [env_size, 1]
        outputs["masks"] = 1.0 - dones # if done, mask = 0
        outputs["successes"] = infos["successes"]
        if env_name in ["minigrid", "montezuma"]:
            instr_str = infos["cur_goals"] # check line 480 in env_wrapper.py shape [env_size, 1]
        else:
            instr_str = None

        # ! int_rew, lang_rew should be handled here rather than storage
        if has_exploration_reward:
            # obtain mem from storage, the previous obs_mem and tra_mem
            if isinstance(int_rew_model, DiscriminatorModel):
                int_rew, model_mems = create_intrinsic_rewards_deir(
                    last_obs = inputs["obs"], 
                    new_obs = outputs["obs"], # not resized
                    last_model_mems = last_model_mems,
                    episodic_obs_emb_history = episodic_obs_emb_history,
                    episodic_trj_emb_history = episodic_trj_emb_history,
                    int_rew_model = int_rew_model,
                    int_rew_stats_mean = storage.int_rew_stats.mean,
                    is_obs_queue_init = storage.is_obs_queue_init,
                ) # shape [env_size, 1]
            elif isinstance(int_rew_model, RNDModel):
                int_rew = create_intrinsic_rewards_rnd(
                    last_obs = inputs["obs_single"],
                    new_obs = outputs["obs"], # not resized
                    int_rew_model = int_rew_model,
                    is_state_visited = is_state_visited, 
                    mode = "rnd", # "rnd" | noveld # ! we found that noveld did not converge faster in the current configs
                    obs_rms = kwargs["obs_rms"],
                    
                ) # shape [env_size, 1], type 
                # ! apply NovelD is_state_visited, if visited, int_rew = 0
                model_mems = None # for RNDModel, we do not need to update model_mems

            if not storage.is_obs_queue_init:
                storage.is_obs_queue_init = True

            assert int_rew.shape == (nproc, 1)
            episode_cached_return_int += int_rew.reshape(-1)

        else:
            int_rew = None

        # ! calculate lang_rew
        if has_language_reward and lang_rew_model is not None and reward_machine_lst is not None:
            # update obs to lang_rew_traj_history, obs shape [env_size, 3, 64, 64]
            # push lang_rew_traj_history 1 step forward
            latest_obs = []
            for i, done in enumerate(dones):
                if done:
                    # get it from infos['terminal_observation']
                    latest_obs.append(infos['terminal_observation'][i]) 
                else:
                    latest_obs.append(obs[i])
            latest_obs = th.stack(latest_obs, dim=0) # shape [env_size, 3, 64, 64]
            lang_rew_traj_history = th.cat([lang_rew_traj_history[:, 1:], latest_obs.unsqueeze(1)], dim=1) # shape [env_size, traj_length, 3, 64, 64]
            if env_name == "crafter" and epoch < reward_machine_lst[0].exploration_epoch_num:
                lang_reward_function_type_temp = "normal"
            else:
                lang_reward_function_type_temp = lang_reward_function_type
            
            if lang_reward_function_type_temp in ['cmi_log', 'cmi_linear']:
                cur_instr_text_lst_temp = []
                for env_i, instr in enumerate(cur_instr_text_lst):
                    output_l = [instr]
                    # add other instrs
                    if env_name == "crafter":
                        other_instrs = deepcopy(reward_machine_lst[env_i].all_task_name_lst)
                        other_instrs.remove(instr)
                        output_l.extend(other_instrs)
                    elif env_name == "minigrid":
                        other_instrs = [reward_machine_lst[env_i].first_instruction, reward_machine_lst[env_i].second_instruction]
                        other_instrs.remove(instr)
                        output_l.extend(other_instrs)
                    elif env_name == "montezuma":
                        other_instrs = reward_machine_lst[env_i].room_task_group[reward_machine_lst[env_i].cur_goal_idx]
                        other_instrs = reward_machine_lst[env_i].walkthrough_df.iloc[other_instrs]['goal'].tolist()
                        other_instrs.remove(instr)
                        output_l.extend(other_instrs)
                        
                    cur_instr_text_lst_temp.append(output_l)
                    
            else:
                cur_instr_text_lst_temp = cur_instr_text_lst
                
            lang_rew_raw, score_output_for_measure_frequency = calculate_lang_rew_raw(cur_instr_text_lst_temp, lang_rew_traj_history, lang_rew_model, resize_size_for_lang_rew_model, storage.device, lang_reward_function_type_temp) # shape [env_size, 1]
            processed_lang_rew = []
            if env_name == "montezuma":
                room_ids = venv.get_attr("cur_room")
            for idx, env_rew in enumerate(lang_rew_raw):
                if env_name == "crafter":
                    update_input_dict = dict(
                        reward=env_rew, 
                        inventory=infos['inventories'][idx], 
                        lang_reward_function_type=lang_reward_function_type_temp,
                    )
                elif env_name == "minigrid":
                    update_input_dict = dict(
                        reward=env_rew, 
                        lang_reward_function_type=lang_reward_function_type_temp,
                    )
                elif env_name == "montezuma":
                    update_input_dict = dict(
                        reward=env_rew, 
                        cur_room_id=room_ids[idx],
                        lang_reward_function_type=lang_reward_function_type_temp,
                    )
                    
                if lang_reward_function_type_temp in ['cmi_log', 'cmi_linear']:
                    update_input_dict['score_output_for_measure_frequency'] = score_output_for_measure_frequency[idx]
                    update_input_dict['other_instrs'] = cur_instr_text_lst_temp[idx][1:]
                    update_input_dict['cur_instr'] = cur_instr_text_lst_temp[idx][0]
                    
                updated_rew, instr_text, next_flag = reward_machine_lst[idx].update(**update_input_dict)
                    
                # ! offset analysis after triggering lang_rew
                if updated_rew > 0.0 and not (env_name == "crafter" and epoch < reward_machine_lst[0].exploration_epoch_num): 
                    reward_machine_lst[idx].offset_analysis_after_triggering_lang_rew(env_rew, next_flag, nproc_episode_step_count[idx])
                    
                # ! update to lang_rew_trigger_list_cache
                if not (env_name == "crafter" and epoch < reward_machine_lst[0].exploration_epoch_num) and next_flag:
                    
                    completed_instr_str_with_id = f"{cur_instr_text_lst[idx]}_{str(inputs['env_instr_additional_key'][idx][0])}" # ! in storage.compute_language_rewards function, we should use the same method to get the completed_instr_str_with_id
                    lang_rew_trigger_list_cache[idx].add(completed_instr_str_with_id)
                cur_instr_text_lst[idx] = instr_text
                processed_lang_rew.append(updated_rew)

            lang_rew = th.as_tensor(processed_lang_rew, dtype=th.float32, device=storage.device).reshape(-1, 1)

            assert lang_rew.shape == (nproc, 1)
            episode_cached_return_lang += lang_rew.cpu().numpy().reshape(-1)
            if env_name == "montezuma":
                # ! update reward heatmap
                # get the agent location
                for env_id, loc in enumerate(infos["agent_locations"]):
                    # loc shape is [3,]
                    x, y, room_id = loc
                    reward_heatmap_cache[env_id][x, y, room_id] += lang_rew[env_id].cpu().numpy().item()

        else:
            lang_rew = None

        episode_cached_return_ext += rewards.cpu().numpy().reshape(-1)
        # convert to tensor
        if int_rew is not None:
            int_rew = th.as_tensor(int_rew).float().to(storage.device) # shape [env_size, 1]
        outputs['exploration_rewards'] = int_rew 
        outputs['language_rewards'] = lang_rew
        if lang_rew is not None:
            # compute reward_machine_instr_additional_key      
            if not (env_name == "crafter" and epoch < reward_machine_lst[0].exploration_epoch_num): # ! special case crafter exploration period 
                outputs['reward_machine_instr_str'] = np.array(cur_instr_text_lst,  dtype="<U100").reshape(-1, 1)
                
            if env_name == "montezuma":
                reward_machine_instr_additional_key = venv.get_attr("cur_room")
                outputs['reward_machine_instr_additional_key'] = np.array(reward_machine_instr_additional_key, dtype="<U100").reshape(-1, 1)
            elif env_name == "minigrid":
                reward_machine_instr_additional_key = venv.get_attr("seed_idx")
                outputs['reward_machine_instr_additional_key'] = np.array(reward_machine_instr_additional_key, dtype="<U100").reshape(-1, 1)
                    
            
        outputs['instr_str'] = instr_str
        outputs['goal_info'] = goal_info # only for minigrid env 
        outputs['last_model_mems'] = last_model_mems

        if isinstance(policy_model, PPORNNModel):
            assert "policy_mem" in list(outputs.keys())
            last_policy_mems = outputs["policy_mem"]

        # Update stats (clear_on_episode_end) # see clear_on_episode_end line 366 in ppo_rollout.py in deir project
        for i, done in enumerate(dones):
            if done:
                # Episode lengths
                episode_length = infos["episode_lengths"][i].cpu().numpy()
                episode_lengths.append(episode_length)

                # Episode rewards
                episode_reward = infos["episode_rewards"][i].cpu().numpy()
                episode_rewards.append(episode_reward)

                # Achievements
                achievement = infos["achievements"][i].cpu().numpy()
                achievements.append(achievement)

                # Successes
                success = infos["successes"][i].cpu().numpy()
                successes.append(success)

                # compute the mean episodic return

                episode_num += 1
                episode_sum_return_ext += episode_cached_return_ext[i]
                episode_cached_return_ext[i] = 0.0

                episode_sum_step += episode_length
                
                

                if has_exploration_reward:
                    episode_sum_return_int += episode_cached_return_int[i]
                    episode_cached_return_int[i] = 0.0

                    # update episodic_obs_emb_history and episodic_trj_emb_history and model_mems
                    if model_mems is not None:
                        model_mems[i] *= 0.0

                    if episodic_obs_emb_history is not None:
                        episodic_obs_emb_history[i] = None
                        episodic_trj_emb_history[i] = None

                # postprocess offset analysis 
                if has_language_reward and lang_rew_model is not None and reward_machine_lst is not None and not has_hard_signal:
                    reward_machine_lst[i].save_offset_value(i, tag_lst_str)

                if has_language_reward and lang_rew_model is not None and reward_machine_lst is not None:
                    episode_sum_return_lang += episode_cached_return_lang[i]
                    episode_cached_return_lang[i] = 0.0
                    
                    # update lang_rew_trigger_list_cache
                    for completed_instr_str_with_id in lang_rew_trigger_list_cache[i]:
                        if completed_instr_str_with_id in lang_rew_instr_trigger_count_per_rollout:
                            lang_rew_instr_trigger_count_per_rollout[completed_instr_str_with_id] += 1
                        else:
                            lang_rew_instr_trigger_count_per_rollout[completed_instr_str_with_id] = 1
                    lang_rew_trigger_list_cache[i] = set()
                    
                    
                    # update cur_instr_text_lst and reset reward machine
                    if env_name == "crafter":
                        cur_instr_text_lst[i] = reward_machine_lst[i].reset(epoch, lang_reward_function_type_temp, nproc_episode_step_count[i])
                    elif env_name == "montezuma":
                        cur_instr_text_lst[i] = reward_machine_lst[i].reset(lang_reward_function_type_temp, nproc_episode_step_count[i])
                    elif env_name == "minigrid":
                        goal_str = venv.get_attr("goal_str_lst")[i]
                        cur_instr_text_lst[i] = reward_machine_lst[i].reset(goal_str[0], goal_str[1], lang_reward_function_type_temp, nproc_episode_step_count[i])
                        
                    # update lang_rew_traj_history
                    # repeat the obs outputs["obs"] will contain the first obs of the next episode
                    new_obs_full = repeat(outputs["obs"][i], 'c h w -> r c h w', r=traj_length)
                    lang_rew_traj_history[i] = new_obs_full
                    
                    
                        

                if last_policy_mems is not None:
                    last_policy_mems[i] *= 0.0

                if env_name in ["montezuma", "minigrid"]: # we need to reset the agent location cache
                    agent_loc_traj = np.stack(agent_location_storage_cache[i], axis=0)
                    agent_location_storage_per_episode.append(agent_loc_traj)
                    agent_location_storage_cache[i] = []

                # save reward heatmap for montezuma
                if env_name == "montezuma":
                    reward_heatmap_per_episode.append(reward_heatmap_cache[i])
                    reward_heatmap_cache[i] = np.zeros((400, 400, 20), dtype=float)
                    
                # reset nproc_episode_step_count    
                nproc_episode_step_count[i] = 0

        # update last_policy_mems to outputs
        outputs['policy_mem'] = last_policy_mems       

        # Update storage
        storage.insert(**outputs, 
                        model = policy_model,
                        has_exploration_reward = has_exploration_reward,
                        has_language_reward = has_language_reward,
                       )

        if has_exploration_reward:
            if model_mems is not None: # for other tensors, we handle them in storage as well as create_intrinsic_rewards function in int_rew_utils.py
                last_model_mems = model_mems.detach().clone()

    # Pass through model
    inputs = storage.get_inputs(step=-1) # this will get the vpreds for the last step, rmb that obs from env is actually the next obs, thus the last step is the evaluation of the last obs
    if env_name == "montezuma":
        postprocess_inputs_montezuma(inputs)
    inputs, goal_info = postprocess_inputs(inputs, env_name, beta_ceoff, nproc)
    outputs = policy_model.act(**inputs)
    vpreds = outputs["vpreds"]
    # Update storage
    storage.vpreds[-1].copy_(vpreds)
    if env_name == "montezuma":
        storage.vpreds_int[-1].copy_(outputs["vpreds_int"])

    # postprocessing intrinsic reward
    if has_exploration_reward:
        if env_name == "montezuma":
            storage.compute_intrinsic_rewards_montezuma(reward_rms = kwargs['reward_rms'], discounted_reward = kwargs['discounted_reward'])
        else:
            storage.compute_intrinsic_rewards()
    # postprocessing language reward
    if has_language_reward and lang_rew_model is not None and reward_machine_lst is not None:
        
        storage.compute_language_rewards(lang_rew_coef)

    # Stack stats (again, redundant)
    episode_lengths = safe_stack(episode_lengths, axis=0).astype(np.int32) # shape [episode_num]
    episode_rewards = safe_stack(episode_rewards, axis=0).astype(np.float32)
    achievements = safe_stack(achievements, axis=0).astype(np.int32)
    successes = safe_stack(successes, axis=0).astype(np.int32)

    # Define rollout stats
    rollout_stats = {
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
        "achievements": achievements,
        "successes": successes,
        'episode_sum_return_ext': episode_sum_return_ext,
        'episode_sum_return_int': episode_sum_return_int,
        'episode_sum_return_lang': episode_sum_return_lang,
        'episode_num': episode_num,
        'episode_sum_step': episode_sum_step,
    }

    # save episodic_obs_emb_history and episodic_trj_emb_history to storage
    if has_exploration_reward:
        storage.episodic_obs_emb_history = episodic_obs_emb_history
        storage.episodic_trj_emb_history = episodic_trj_emb_history

    # save agent location info for montezuma and minigrid
    # TODO do not save 
    if False and env_name in ["montezuma", "minigrid"] and seed == 1234:
        if env_name == "montezuma" and epoch % (MONTEZUMA_NSTEP_EPOCH_RATIO * 20) != 0:
            pass
        else:
            storage.agent_location_storage_cache = agent_location_storage_cache
            save_dir = os.path.join(os.environ['PWD'], 'data/07_model_output/agent_location_track', env_name, tag_lst_str)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(save_dir, f'agent_location_epoch{epoch}_traj_list.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(agent_location_storage_per_episode, f)

    # save reward heatmap for montezuma
    # TODO do not save 
    if False and env_name == "montezuma" and seed == 1234:
        if epoch % (MONTEZUMA_NSTEP_EPOCH_RATIO * 10) != 0:
            pass
        else:
            storage.reward_heatmap_cache = reward_heatmap_cache
            save_dir = os.path.join(os.environ['PWD'], 'data/07_model_output/reward_heatmap_track', env_name, tag_lst_str)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(save_dir, f'reward_heatmap_epoch{epoch}_traj_list.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(reward_heatmap_per_episode, f)
            
    # save lang_rew_trigger_list_cache
    if lang_rew_model is not None:
        storage.lang_rew_trigger_list_cache = lang_rew_trigger_list_cache
        storage.cur_instr_text_lst = cur_instr_text_lst
        
        # also save lang_rew_traj_history
        storage.lang_rew_traj_history = lang_rew_traj_history
        
    # temp save nproc_episode_step_count
    storage.nproc_episode_step_count = nproc_episode_step_count

    return rollout_stats
