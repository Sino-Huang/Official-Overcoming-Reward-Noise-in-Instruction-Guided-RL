"""
This is a boilerplate pipeline 'reward_machine_setup'
generated using Kedro 0.19.3
"""

from typing import Sequence

import numpy as np
from better_alignment_signal_for_rl.lang_rew_model_backbone.model.cosine_sim_model import CosineSimLangRewModel
from better_alignment_signal_for_rl.pipelines.env_setup.env_wrapper import CRAFTER_TASKS
import torch as th 
from einops import rearrange, repeat
from torchvision.transforms.functional import resize as th_f_resize
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import pickle 
import os
from pathlib import Path
from glob import glob 

CRAFTER_EXPLORATION_EPOCH_NUM = 50 

CRAFTER_MINE_DIAMOND = ['collect_wood', 'collect_wood', 'place_table', 'make_wood_pickaxe', 'collect_stone', 'make_stone_pickaxe', 'collect_stone', 'place_furnace', 'collect_coal', 'collect_iron', 'make_iron_pickaxe', 'collect_diamond']

THIRSTY = ['collect_drink', 'collect_drink']
HUNGRY_AND_LOW_HEALTH = ['eat_cow', 'collect_sapling', 'place_plant', 'eat_plant']
LOW_ENERGY = ['place_stone', 'wake_up']

class Reward_Machine(object): # this is just for single environment 
    def __init__(self, reward_machine_type, env_name, reward_cap, has_hard_signal, seed, **kwargs):
        """Reward Machine will control the subgoal sequence and postprocess the reward signal from the language reward model

        kwargs will have cp_threshold, walkthrough_df, etc. depending on the environment
       
        """
        assert reward_machine_type in ['standard']
        self.reward_machine_type = reward_machine_type
        self.env_name = env_name
        self.reward_cap = reward_cap
        self.has_hard_signal = has_hard_signal
        self.seed = seed
        self.cp_threshold = kwargs['cp_threshold']
        
        if not has_hard_signal:
            self.scale_fac = 0.5 # from NovelD, we calculate the difference between the current reward and the highest reward, and then multiply by this scale factor to get the novelty bias score
        # dynamic 
        self.freeze_flag = False # freeze the reward machine 
        if not self.has_hard_signal:
            self.current_rew_sum = 0.0
            self.current_rew_highest_score = 0.0
            
        self.p_for_cmi = None # for CMI reward function
        self.instr_count_per_episode_dict = dict() # for CMI reward function
            
        if self.env_name == "montezuma":
            self.walkthrough_df = kwargs['walkthrough_df']
            # setup the room_task_group
            self.room_task_group = dict() # shape {goal_index: [group of goal_index]}
            room_temp = 1 
            task_group = [] 
            
            for i in range(len(self.walkthrough_df)):
                room = self.walkthrough_df.iloc[i]['room']
                if room == room_temp:
                    task_group.append(i)
                else:
                    # update the room_task_group
                    for index in task_group:
                        self.room_task_group[index] = task_group
                    # update the room_temp
                    room_temp = room
                    # reset the task_group
                    task_group = [i]
            # update the last task group
            for index in task_group:
                self.room_task_group[index] = task_group
            
            
            
            
        # Offset Histogram Analysis for Reward Magnitude and Goal Proximity
        self.offset_hist_dict_for_lang_rew = dict() # structure 
        # {k=instr_id, v={
        #     k=inverval[0.0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0],
        #     v=[venv_step when the reward is given, ...]
        # }}
        self.actual_reward_dict = dict() # structure 
        # {k=instr_id, v=actual completion venv_step, ...}
        self.offset_value_lst = dict() # structure 
        self.offset_value_lst[0.25] = [] 
        self.offset_value_lst[0.5] = []
        self.offset_value_lst[0.75] = []
        self.offset_value_lst[1.0] = []
        
        # {k=inverval[0.0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0], v=[offset value, ...]}
            
        # assign reset and update functions
        if self.env_name == "minigrid":
            self.reset = self.minigrid_reset_walkthrough_data
            self.update = self.minigrid_update
        elif self.env_name == "montezuma":
            self.reset = self.montezuma_reset_walkthrough_data
            self.update = self.montezuma_update
        elif self.env_name == "crafter":
            self.exploration_epoch_num = CRAFTER_EXPLORATION_EPOCH_NUM # ! changeable but notice that exploration is computationally expensive
            self.reset = self.crafter_reset_walkthrough_data
            self.update = self.crafter_update

        
        
        
    # if env_name == "montezuma", it is a dataframe
    # if env_name == "minigrid", it is a instruction list collected from the environment 
    # if env_name == "crafter", we use the manual defined in from ELLM paper, but dynamically changes according to the state, before 100 epochs we just do exploration and gives rewards as long as the language reward model detect that some achievements are made, after 100 epochs we apply the CRAFTER_MINE_DIAMOND task
    
    def reset_offset_histogram_analysis_vars(self, instr_ids): # reset once episode ends
        self.offset_hist_dict_for_lang_rew.clear()
        self.actual_reward_dict.clear()
        for instr_id in instr_ids:
            self.offset_hist_dict_for_lang_rew[instr_id] = dict()
            for val_interval in [0.25, 0.5, 0.75, 1.0]:
                self.offset_hist_dict_for_lang_rew[instr_id][val_interval] = []
                
            self.actual_reward_dict[instr_id] = None
                
        
    def offset_analysis_after_triggering_lang_rew(self, rew_val, next_flag, env_step):
        if getattr(self, 'cur_goal_idx', None) is None:
            return
        if next_flag:
            # it means lang rew model believe that the subgoal is achieved
            the_subgoal_id = self.cur_goal_idx -1 # the previous subgoal id
        else:
            # it means lang rew model believe that the subgoal is not achieved but we still give partial 
            the_subgoal_id = self.cur_goal_idx # the current subgoal id
            
        if self.env_name == "crafter":
            the_subgoal_id = the_subgoal_id % len(CRAFTER_MINE_DIAMOND)
        elif self.env_name == "minigrid":
            the_subgoal_id = the_subgoal_id % 2
        elif self.env_name == "montezuma":
            the_subgoal_id = the_subgoal_id % len(self.walkthrough_df)
            
        for val_interval in [0.25, 0.5, 0.75, 1.0]:
            if rew_val <= val_interval:
                interval_key = val_interval
                self.offset_hist_dict_for_lang_rew[the_subgoal_id][val_interval].append(env_step)
                break
        # ! check if the actual reward has been given or not
        if self.actual_reward_dict[the_subgoal_id] is None:
            # it means the actual reward has not been given
            pass 
        else:
            # it means the actual reward has been given
            # but we need to ensure that we do not consider the actual reward timestep before we even trigger the lang rew model
            actual_reward_given_step = self.actual_reward_dict[the_subgoal_id]
            for val_interval, venv_step_lst in self.offset_hist_dict_for_lang_rew[the_subgoal_id].items():
                if len(venv_step_lst) == 0:
                    continue
                # we check the first element of the venv_step_lst as it is the earliest venv_step that the reward is given by the lang rew model
                else:
                    earliest_lang_rew_step = venv_step_lst[0]
                    break
            if actual_reward_given_step < earliest_lang_rew_step:
                # it means the actual reward is given before the lang rew model gives the reward
                # it means it is not acceptable
                self.actual_reward_dict[the_subgoal_id] = None
                
        # ! add the negative offset 
        if self.actual_reward_dict[the_subgoal_id] is not None:
            offset_val = self.actual_reward_dict[the_subgoal_id] - env_step
            self.offset_value_lst[interval_key].append(offset_val)
                
    
    def analyze_offset_actual_status(self, previous_env_current_goal_id, new_previous_env_current_goal_id, env_step):
        """this method will record the actual reward venv_step for each instruction id
        """
        if getattr(self, 'cur_goal_idx', None) is None:
            return
        completed_goal_id = None 
        
        
        if self.env_name == "montezuma":
            if new_previous_env_current_goal_id > previous_env_current_goal_id:
                # it means the env has moved to the next goal
                self.actual_reward_dict[previous_env_current_goal_id] = env_step
                completed_goal_id = previous_env_current_goal_id
                
                
        elif self.env_name == "minigrid":
            if new_previous_env_current_goal_id > previous_env_current_goal_id:
                # it means the env has moved to the next goal
                self.actual_reward_dict[previous_env_current_goal_id] = env_step
                completed_goal_id = previous_env_current_goal_id
            elif new_previous_env_current_goal_id < previous_env_current_goal_id:
                # it means we indeed complete the goal but the venv reset 
                self.actual_reward_dict[previous_env_current_goal_id] = env_step
                completed_goal_id = previous_env_current_goal_id
                
        elif self.env_name == "crafter":
            # they are list 
            for i, success_count in enumerate(new_previous_env_current_goal_id):
                if success_count > previous_env_current_goal_id[i]:
                    # convert i to our local 
                    task_name = CRAFTER_TASKS[i]
                    if task_name in CRAFTER_MINE_DIAMOND:
                        local_goal_id = CRAFTER_MINE_DIAMOND.index(task_name)
                        self.actual_reward_dict[local_goal_id] = env_step
                        completed_goal_id = local_goal_id
           
        # ! add the positive offset             
        # check if we have lang rew scores for this instruction id
        if completed_goal_id is not None:
            for val_interval, venv_step_lst in self.offset_hist_dict_for_lang_rew[completed_goal_id].items():
                offset_lst_local = []
                if len(venv_step_lst) == 0:
                    continue
                else:
                    # we measure all the offset values
                    for lang_rew_step in venv_step_lst:
                        offset_val = env_step - lang_rew_step
                        offset_lst_local.append(offset_val)
                    # clear the list
                    self.offset_hist_dict_for_lang_rew[completed_goal_id][val_interval] = []
                self.offset_value_lst[val_interval].extend(offset_lst_local)
                
                
        
        
    def save_offset_value(self, env_id, tag_lst_str, force_save=False): # call this after the episode ends
        if self.seed != 1234:
            return        
        # temp save 
        for val_interval, offset_lst in self.offset_value_lst.items():
            
            if len(offset_lst) > 2000 or force_save:
                save_dir = os.path.join(os.environ['PWD'], 'data/07_model_output/reward_offset_vals', self.env_name, tag_lst_str)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                pattern_path = os.path.join(save_dir, f"env_id_{env_id}_val_interval_{val_interval}_idx_*.pkl") # * is the index count 
                existing_files = glob(pattern_path)
                if len(existing_files) > 0:
                    idx = len(existing_files)
                else:
                    idx = 0
                save_path = os.path.join(save_dir, f"env_id_{env_id}_val_interval_{val_interval}_idx_{idx}.pkln")
                with open(save_path, 'wb') as f:
                    pickle.dump(offset_lst, f)
                self.offset_value_lst[val_interval] = []
                
        
                
    def reset_cmi_info(self, lang_reward_function_type, episode_length):
        if lang_reward_function_type in ['cmi_log', 'cmi_linear']:
            p_for_cmi_dict = dict()
            for instr in self.instr_count_per_episode_dict:
                p_for_cmi_dict[instr] = self.instr_count_per_episode_dict[instr] / episode_length
            self.p_for_cmi = p_for_cmi_dict
            self.instr_count_per_episode_dict.clear()
    
    # reset function will give the first subgoal_text
    def crafter_reset_walkthrough_data(self, epoch, lang_reward_function_type, episode_length):
        self.freeze_flag = False
        if epoch < self.exploration_epoch_num:
            self.task = "exploration"
            if self.has_hard_signal:
                self.task_achieved_flag = [False for _ in CRAFTER_TASKS]
            else:
                self.task_achieved_flag = [0.0 for _ in CRAFTER_TASKS]
                
            self.all_task_name_lst = CRAFTER_TASKS
            return CRAFTER_TASKS
        else:
            self.task = "exploitation" 
            self.cur_goal_idx = 0 
            self.emergency_flag = False 
            self.emergency_goal_idx = 0
            self.emergency_task_title = None
            self.all_task_name_lst = CRAFTER_MINE_DIAMOND
            
            instr_ids = list(range(len(CRAFTER_MINE_DIAMOND)))
            self.reset_offset_histogram_analysis_vars(instr_ids)
            
            self.reset_cmi_info(lang_reward_function_type, episode_length)
            
            return CRAFTER_MINE_DIAMOND[self.cur_goal_idx]
            
    def minigrid_reset_walkthrough_data(self, first_instruction, second_instruction, lang_reward_function_type, episode_length):
        self.first_instruction = first_instruction
        self.second_instruction = second_instruction
        self.cur_goal_idx = 0
        self.reset_offset_histogram_analysis_vars([0,1])
        self.freeze_flag = False
        self.reset_cmi_info(lang_reward_function_type, episode_length)
        return self.first_instruction

        
    def montezuma_reset_walkthrough_data(self, lang_reward_function_type, episode_length):
        self.cur_goal_idx = 0 
        self.reset_offset_histogram_analysis_vars(list(range(len(self.walkthrough_df))))
        self.freeze_flag = False
        self.reset_cmi_info(lang_reward_function_type, episode_length)
        
        return self.walkthrough_df.iloc[self.cur_goal_idx]['goal']

        
    # update function will output the processed reward signal and the new subgoal_text    
    def soft_reward_calculate(self, reward):
        novelty_bias_score = np.max([reward - (self.scale_fac * self.current_rew_highest_score), 0.0])
        if reward > self.current_rew_highest_score:
            self.current_rew_highest_score = reward
        self.current_rew_sum += novelty_bias_score
        output_reward = novelty_bias_score
        return output_reward
    
    def update_cmi_info_data_for_next_episode(self, score_output_for_measure_frequency, other_instrs):
        
        for idx, reward in enumerate(score_output_for_measure_frequency):
             # we also support soft reward but it will use the same cp_threshold
            if reward > self.cp_threshold: 
                the_instr = other_instrs[idx]
                if the_instr not in self.instr_count_per_episode_dict:
                    self.instr_count_per_episode_dict[the_instr] = 1
                else:
                    self.instr_count_per_episode_dict[the_instr] += 1
                
            
    def update_reward_based_on_cmi(self, reward, lang_reward_function_type, cur_instr):
        assert lang_reward_function_type in ['cmi_log', 'cmi_linear']
        if self.p_for_cmi is None:
            return reward
        if cur_instr in self.p_for_cmi:
            p = self.p_for_cmi[cur_instr] * 0.95 # scale factor to make p less confident
            if not self.has_hard_signal: # for soft reward, we set p value to be proportional to the reward 
                pass
                # p = reward * p 
            if lang_reward_function_type == 'cmi_log':
                return np.max([0.0, np.log(reward / p)])
            elif lang_reward_function_type == 'cmi_linear':
                return np.max([0.0, reward - p])
        else:
            return reward # if the instruction is not in the p_for_cmi, we do not change the reward
            
                
        
    def crafter_update(self, reward, inventory, lang_reward_function_type, **kwargs):
        # infos['inventories'][0]
        # {'health': 9, 'food': 9, 'drink': 9, 'energy': 9, 'sapling': 0, 'wood': 0, 'stone': 0, 'coal': 0, 'iron': 0, 'diamond': 0, 'wood_pickaxe': 0, 'stone_pickaxe': 0, 'iron_pickaxe': 0, 'wood_sword': 0, 'stone_sword': 0, 'iron_sword': 0}
        
        if lang_reward_function_type in ['cmi_log', 'cmi_linear']:
            self.update_cmi_info_data_for_next_episode(kwargs['score_output_for_measure_frequency'], kwargs['other_instrs'])
        
        if self.task == "exploration":
            assert isinstance(reward, list) # the reward is a list of rewards for each task
            if self.has_hard_signal:
                accu_rew = 0.0
                for i, task in enumerate(CRAFTER_TASKS):
                    if reward[i] > self.cp_threshold:
                        if not self.task_achieved_flag[i]:
                            self.task_achieved_flag[i] = True
                            accu_rew += 1.0 # it means we only give reward once for each task
                return accu_rew, CRAFTER_TASKS, False 
            else:
                accu_rew = 0.0
                for i, task in enumerate(CRAFTER_TASKS):
                    if reward[i] > 0.1: # start from 0.1 otherwise very noisy
                        if self.task_achieved_flag[i] < self.reward_cap:
                            self.task_achieved_flag[i] += reward[i]
                            accu_rew += reward[i] # it means we do not give reward if the cap is reached
                return accu_rew, CRAFTER_TASKS, False 
        else:
            next_flag = False 
            if not self.emergency_flag:
                if self.has_hard_signal:
                    if reward > self.cp_threshold: 
                        self.cur_goal_idx += 1
                        next_flag = True
                        output_reward = 1.0
                    else:
                        output_reward = 0.0
                    
                else:
                    if reward > 0.1:
                        output_reward = self.soft_reward_calculate(reward)
                        if self.current_rew_sum > self.reward_cap:
                            self.current_rew_sum = 0.0
                            self.current_rew_highest_score = 0.0
                            self.cur_goal_idx += 1
                            next_flag = True
                    else:
                        output_reward = 0.0
                            
                # check if we continue the next task or we have emergency tasks
                # check health and food 
                if inventory['health'] < 3 or inventory['food'] < 3:
                    self.emergency_flag = True
                    self.emergency_task_title = "hungry_and_low_health"
                    instr_text = HUNGRY_AND_LOW_HEALTH[self.emergency_goal_idx]
                    self.all_task_name_lst = HUNGRY_AND_LOW_HEALTH
                elif inventory['energy'] < 3:
                    self.emergency_flag = True
                    self.emergency_task_title = "low_energy"
                    instr_text = LOW_ENERGY[self.emergency_goal_idx]
                    self.all_task_name_lst = LOW_ENERGY
                elif inventory['drink'] < 3:
                    self.emergency_flag = True
                    self.emergency_task_title = "thirsty"
                    instr_text = THIRSTY[self.emergency_goal_idx]
                    self.all_task_name_lst = THIRSTY
                    
                else:
                    instr_text = CRAFTER_MINE_DIAMOND[self.cur_goal_idx % len(CRAFTER_MINE_DIAMOND)]
                    
                if lang_reward_function_type in ['cmi_log', 'cmi_linear'] and output_reward > 0.1:
                    output_reward = self.update_reward_based_on_cmi(output_reward, lang_reward_function_type, kwargs['cur_instr'])
                    
                    
                return output_reward, instr_text, next_flag
            else: # we are in emergency mode
                if self.has_hard_signal:
                    if reward > self.cp_threshold: 
                        self.emergency_goal_idx += 1
                        output_reward = 1.0
                    else:
                        output_reward = 0.0
                else:
                    if reward > 0.1:
                        output_reward = self.soft_reward_calculate(reward)
                        if self.current_rew_sum > self.reward_cap:
                            self.current_rew_sum = 0.0
                            self.current_rew_highest_score = 0.0
                            self.emergency_goal_idx += 1
                    else:
                        output_reward = 0.0
                        
                is_emergency_done = False    
                if self.emergency_task_title == "hungry_and_low_health":
                    if inventory['health'] >= 3 and inventory['food'] >= 3:
                        is_emergency_done = True
                     
                    else:
                        instr_text = HUNGRY_AND_LOW_HEALTH[self.emergency_goal_idx%len(HUNGRY_AND_LOW_HEALTH)]
                elif self.emergency_task_title == "low_energy":
                    if inventory['energy'] >= 3:
                        is_emergency_done = True
                    
                    else:
                        instr_text = LOW_ENERGY[self.emergency_goal_idx%len(LOW_ENERGY)]
                elif self.emergency_task_title == "thirsty":
                    if inventory['drink'] >= 3:
                        is_emergency_done = True
                        
                    else:
                        instr_text = THIRSTY[self.emergency_goal_idx%len(THIRSTY)]
                        
                if is_emergency_done:
                    self.emergency_flag = False
                    self.emergency_task_title = None
                    self.emergency_goal_idx = 0
                    instr_text = CRAFTER_MINE_DIAMOND[self.cur_goal_idx % len(CRAFTER_MINE_DIAMOND)]
                    self.all_task_name_lst = CRAFTER_MINE_DIAMOND
                return output_reward, instr_text, False
                        
    
    def minigrid_update(self, reward, lang_reward_function_type, **kwargs):
        if lang_reward_function_type in ['cmi_log', 'cmi_linear']:
            self.update_cmi_info_data_for_next_episode(kwargs['score_output_for_measure_frequency'], kwargs['other_instrs'])
        
        if self.freeze_flag:
            return 0.0, self.freeze_time_instr_text, self.freeze_time_next_flag
        
        next_flag = False
        
        if self.has_hard_signal:
            if reward > self.cp_threshold: 
                self.cur_goal_idx += 1
                output_reward = 1.0
                next_flag = True
            else:
                output_reward = 0.0
            
        else:
            if reward > 0.1:
                output_reward = self.soft_reward_calculate(reward)
                if self.current_rew_sum > self.reward_cap:
                    self.current_rew_sum = 0.0
                    self.current_rew_highest_score = 0.0
                    self.cur_goal_idx += 1
                    next_flag = True
            else:
                output_reward = 0.0
        
        goal_idx = self.cur_goal_idx % 2
        if goal_idx == 0:
            instr_text = self.first_instruction
        else:
            instr_text = self.second_instruction
            
        # check if freeze 
        if self.cur_goal_idx == 2:
            self.freeze_flag = True
            self.freeze_time_instr_text = instr_text
            self.freeze_time_next_flag = False
            
        if lang_reward_function_type in ['cmi_log', 'cmi_linear'] and output_reward > 0.1:
            output_reward = self.update_reward_based_on_cmi(output_reward, lang_reward_function_type, kwargs['cur_instr'])
            
        return output_reward, instr_text, next_flag
            
    
    def montezuma_update(self, reward, cur_room_id, lang_reward_function_type, **kwargs):
        
        if lang_reward_function_type in ['cmi_log', 'cmi_linear']:
            self.update_cmi_info_data_for_next_episode(kwargs['score_output_for_measure_frequency'], kwargs['other_instrs'])
            
        if self.freeze_flag:
            if int(cur_room_id) == self.lang_predict_room_id:
                self.freeze_flag = False
            else:
                return 0.0, self.freeze_time_instr_text, self.freeze_time_next_flag
            
        next_flag = False
        if self.has_hard_signal:
            if reward > self.cp_threshold: 
                self.cur_goal_idx += 1
                next_flag = True
                output_reward = 1.0
            else:
                output_reward = 0.0
            
        else:
            if reward > 0.1:
                output_reward = self.soft_reward_calculate(reward)
                if self.current_rew_sum > self.reward_cap:
                    self.current_rew_sum = 0.0
                    self.current_rew_highest_score = 0.0
                    self.cur_goal_idx += 1
                    next_flag = True
            else:
                output_reward = 0.0
        # check if the updated cur_goal_idx lead to right room_id, 
        # if not, we freeze until the room_id is correct
        lang_predict_room_id = int(self.walkthrough_df.iloc[self.cur_goal_idx]['room'])
        cur_room_id = int(cur_room_id)
        assert lang_predict_room_id in [i for i in range(21)]
        assert cur_room_id in [i for i in range(21)]
        
        goal_idx = self.cur_goal_idx % len(self.walkthrough_df)
        self.cur_goal_idx = goal_idx
        instr_text = self.walkthrough_df.iloc[goal_idx]['goal']
        if lang_predict_room_id != cur_room_id:
            if cur_room_id == 0 and lang_predict_room_id == 1: # it means the agent has already moved to the next room
                next_flag = True
                for i in range(self.cur_goal_idx, len(self.walkthrough_df)):
                    if int(self.walkthrough_df.iloc[i]['room']) == cur_room_id:
                        self.cur_goal_idx = i
                        break
                instr_text = self.walkthrough_df.iloc[self.cur_goal_idx]['goal']
            else:
                self.freeze_flag = True
                self.lang_predict_room_id = lang_predict_room_id
                self.freeze_time_instr_text = instr_text
                self.freeze_time_next_flag = False
                
        if lang_reward_function_type in ['cmi_log', 'cmi_linear'] and output_reward > 0.1:
            output_reward = self.update_reward_based_on_cmi(output_reward, lang_reward_function_type, kwargs['cur_instr'])
        
        return output_reward, instr_text, next_flag
    
    
    
def calculate_lang_rew_raw(cur_instr_text_lst, lang_rew_traj_history, lang_rew_model:CosineSimLangRewModel, resize_size_for_lang_rew_model, device, lang_reward_function_type):
    assert lang_reward_function_type in ["normal", "cmi_log", "cmi_linear"]
    def local_forward_helper(traj_d, text_d:Sequence[str], len_lst):
        # ! traj_d batch size can be smaller than text_d size
        # follow line 163 from process_and_forward_helper in train_lang_rew_model pipeline
        inputs = lang_rew_model.clip_format_prepare(traj_d, text_d)
        
        # to device
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        
        # forward 
        outputs = lang_rew_model(inputs)
        vision_embeds, text_embeds = outputs # shape (large_batch, hidden_size)
        # now, given the len_lst, we repeat the vision_embeds
        expand_vision_embeds = []
        for i, l in enumerate(len_lst):
            temp_vision_embeds = [vision_embeds[i] for _ in range(l)]
            expand_vision_embeds.extend(temp_vision_embeds)
            
        expand_vision_embeds = th.stack(expand_vision_embeds, dim=0)
        
        cosine_similarity_flatten = F.cosine_similarity(expand_vision_embeds, text_embeds, dim=-1) # shape (large_batch, )
        
        # make it to list 
        cosine_similarity_flatten = cosine_similarity_flatten.tolist()
        
        # restore the original batch size
        output = [] 
        cur_indx = 0 
        for l in len_lst:
            if l > 1:
                output.append(cosine_similarity_flatten[cur_indx:cur_indx+l])
            else:
                output.append(cosine_similarity_flatten[cur_indx])
            cur_indx += l
        return output
    
    
    # obs is actually the next obs
    # lang_rew_traj_history shape (batch, seq_len, 3, H, W) where H W are small shape 
    lang_rew_model.eval()
    with th.no_grad():
        # we follow line 163 from process_and_forward_helper in train_lang_rew_model pipeline 
        B = lang_rew_traj_history.shape[0]
        lang_rew_traj_history = rearrange(lang_rew_traj_history, "B L C H W -> (B L) C H W")
        lang_rew_traj_history_resize = th_f_resize(lang_rew_traj_history, size=(resize_size_for_lang_rew_model, resize_size_for_lang_rew_model), interpolation=InterpolationMode.BICUBIC, antialias=True)
        # reshape back 
        lang_rew_traj_history_resize = rearrange(lang_rew_traj_history_resize, "(B L) C H W -> B L C H W", B=B)
        
        # check cur_instr_text_lst if the element is a list, if it is a list, we need to repeat the lang_rew_traj_history_resize
        
        flatten_cur_instr_text_lst = [] # shape (large_batch, )
        len_lst = []
        for env_id, cur_instr_text in enumerate(cur_instr_text_lst):
            if isinstance(cur_instr_text, list):
                flatten_cur_instr_text_lst.extend(cur_instr_text)
                len_lst.append(len(cur_instr_text))
                
            else:
                assert lang_reward_function_type not in ['cmi_log', 'cmi_linear']
                flatten_cur_instr_text_lst.append(cur_instr_text)
                len_lst.append(1)
                
                
        score_output = local_forward_helper(lang_rew_traj_history_resize, flatten_cur_instr_text_lst, len_lst)
 
        assert len(score_output) == len(cur_instr_text_lst)
        if lang_reward_function_type in ['cmi_log', 'cmi_linear']:
            # we strip out the actual reward value (which is at index 0) and let the other values to be a split list 
            score_output_actual = [i[0] for i in score_output]
            score_output_for_measure_frequency = [i[1:] for i in score_output]
            return score_output_actual, score_output_for_measure_frequency
        else:
            return score_output, None
        
                
        
        
        
    
    
    

    
    