import pickle
import time
from stable_baselines3.common.vec_env import VecEnvWrapper
import gymnasium as gym
import gym as old_gym
from gymnasium.core import Wrapper
from gymnasium import logger, spaces
from gymnasium.spaces import Discrete, Box
from enum import IntEnum
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
import numpy as np
import torch as th
from typing import Dict, Sequence
from minigrid.envs.babyai.core.verifier import AfterInstr, BeforeInstr
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.transforms import InterpolationMode
from icecream import ic 
import os 
from pathlib import Path 
import pickle
from sklearn.preprocessing import OneHotEncoder
from minigrid.core.constants import OBJECT_TO_IDX

CRAFTER_TASKS = [
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
]

CRAFTER_STANDARD_CASE_REWARDING_TASKS = [
    'place_furnace',
    "collect_coal",
    "collect_iron",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
]


# first, cap the montezuma max step for each room to 1200 steps

MINIGRID_TASKS = ['go to the yellow key', 'go to the yellow door', 'go to a door',
       'go to the blue door', 'go to a ball', 'go to the ball',
       'go to the green box', 'go to the red key', 'go to the grey box',
       'go to a purple door', 'go to the yellow box',
       'go to a green door', 'go to the green door', 'go to the red door',
       'go to the green ball', 'go to the green key', 'go to a grey door',
       'go to the purple key', 'go to the blue key',
       'go to the purple box', 'go to the grey door',
       'go to a yellow door', 'go to the box', 'go to a key',
       'go to the grey key', 'go to a red door', 'go to the key',
       'go to a blue ball', 'go to the blue ball', 'go to the grey ball',
       'go to the red ball', 'go to a blue door', 'go to the yellow ball',
       'go to the blue box', 'go to the purple ball', 'go to a box',
       'go to the red box', 'go to the purple door', 'go to a green key',
       'go to a purple ball', 'go to a purple key', 'go to a yellow ball',
       'go to a red ball', 'go to a purple box', 'go to a yellow box',
       'go to a green box', 'go to a grey box', 'go to a blue box',
       'go to a red box', 'go to a grey key', 'go to a yellow key',
       'go to a blue key', 'go to a green ball', 'go to a grey ball',
       'go to a red key', ""]

MINIGRID_TASKS_ONEHOT_PATH = os.path.join(os.environ['PWD'], 'data/05_model_input', "minigrid_tasks_onehot.pkl")

if not os.path.exists(MINIGRID_TASKS_ONEHOT_PATH):
    # Reshaping the task list to fit the encoder input shape
    tasks_array = np.array(MINIGRID_TASKS).reshape(-1, 1)

    # Creating the OneHotEncoder
    MINIGRID_TASKS_ONEHOT_ENCODER = OneHotEncoder(sparse_output=False)

    # Fitting the encoder and transforming the tasks list
    one_hot_encoded_tasks = MINIGRID_TASKS_ONEHOT_ENCODER.fit_transform(tasks_array)
    
    with open(MINIGRID_TASKS_ONEHOT_PATH, 'wb') as f:
        pickle.dump(MINIGRID_TASKS_ONEHOT_ENCODER, f)
else:
    with open(MINIGRID_TASKS_ONEHOT_PATH, 'rb') as f:
        MINIGRID_TASKS_ONEHOT_ENCODER : OneHotEncoder = pickle.load(f)

# ! --- Montezuma Part ---

class MR_ATARI_ACTION(IntEnum):
    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    RIGHTFIRE = 11
    LEFTFIRE = 12

class MR_ActionsWrapper(gym.ActionWrapper):
    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = Discrete(len(MR_ATARI_ACTION))
        self.action_choice_list = list(MR_ATARI_ACTION)

    def action(self, act):
        return self.action_choice_list[act].value
    
    
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            if i > 0 and action in [1, 11, 12]:
                act = 0 
            else:
                act = action
            obs, reward, done, trunc, info = self.env.step(act)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer[-1]
        return max_frame, total_reward, done, trunc, info


class MontezumaInfoWrapper(gym.Wrapper):
    """apply this before FourOutputWrapper
    goal_seq_df cols = ['goal', 'room', 'x', 'y'], it is a game walkthrough df 
    """
    def __init__(self, env, goal_seq_df=None, lang_rew_machine_type='standard', **kwargs):
        super().__init__(env)
        
        # static
        self.lang_rew_machine_type = lang_rew_machine_type
        if lang_rew_machine_type == 'false_negative_sim':
            self.goal_seq_df_original = goal_seq_df
            self.goal_seq_df = kwargs['false_negative_sim_goal_seq_df']
        else:
            self.goal_seq_df = goal_seq_df
            
        # false positive case 
        if lang_rew_machine_type == 'false_positive_sim':
            self.false_positive_reward_locations = kwargs['false_positive_reward_locations']
            self.is_testing_false_positive = True
        else:
            self.is_testing_false_positive = False
            
        if self.goal_seq_df is not None:
            self.total_goal_seq = self.goal_seq_df['goal'].values.tolist()
            self.total_goal_num = len(self.total_goal_seq)
            
            # dynamic
            self.cur_goal_index = 0
            self.cur_goal = self.total_goal_seq[self.cur_goal_index]
            self.completed_goals = []
            
        self.cur_lives = 6
        self.STEP_LIMIT_PER_ROOM = 1200 # ! 1200 steps per room MAX STEP
        self.step_count = 0
        self.cur_room = 0 
        
        # visited states 
        self.visited_states_mem = np.zeros((20, 400, 400), dtype=bool)
        
        # setup the room_task_group
        self.room_task_group = dict() # shape {goal_index: [group of goal_index]}
        room_temp = 1 
        task_group = [] 
        
        for i in range(self.total_goal_num):
            room = self.goal_seq_df.iloc[i]['room']
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
        
        if self.lang_rew_machine_type == "no_temporal_order_sim":
            self.no_temporal_order_sim_goal_tick_table = [0 for _ in range(self.total_goal_num)]
        
        
    def reset_goal_seq_info(self):
        self.cur_goal_index = 0
        self.cur_goal = self.total_goal_seq[self.cur_goal_index]
        self.completed_goals = []
        if self.lang_rew_machine_type == "no_temporal_order_sim":
            self.no_temporal_order_sim_goal_tick_table = [0 for _ in range(self.total_goal_num)]
            
                
        
    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        # remove info 
        self.cur_lives = 6
        # reset step count
        self.step_count = 0
        self.cur_room = 0 
        self.reset_visited_states()
        
        return obs

    def reset_visited_states(self):
        self.visited_states_mem = np.zeros((20, 400, 400), dtype=bool)
        
        
    def update_step_count(self, new_room):
        if new_room != self.cur_room:
            self.cur_room = new_room
            self.step_count = 0
        else:
            self.step_count += 1
        return self.step_count
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # add agent_pos_info
        x_pos, y_pos = self.agent_pos()
        
        info['agent_pos'] = [x_pos, y_pos]
        # add room info 
        new_room = self.room()
        info['agent_location'] = np.array([x_pos, y_pos, new_room])
        
        # update visited states
        is_visited_before = self.visited_states_mem[self.cur_room, x_pos, y_pos]
        if not is_visited_before:
            # try to let round +- 5 pixels to be visited
            self.visited_states_mem[self.cur_room, x_pos - 5: x_pos + 5, y_pos - 5: y_pos + 5] = True
            # alternative: just let the current pixel to be visited
            # self.visited_states_mem[self.cur_room, x_pos, y_pos] = True
            info['is_state_visited'] = False
        else:
            info['is_state_visited'] = True
            
        info['room'] = new_room
        # update step count and check if the step limit is reached
        step_count = self.update_step_count(new_room)
        if step_count >= self.STEP_LIMIT_PER_ROOM:
            done = True
        
        # add goal info 
        if self.goal_seq_df is not None:
            info['total_goal_seq'] = self.total_goal_seq
            info['total_goal_num'] = self.total_goal_num
            
            # handling no_temporal_order_sim case
            if self.lang_rew_machine_type == "no_temporal_order_sim":
                # for all goals within the same room, we give reward for each goal completion
                # get group of goal index
                task_group = self.room_task_group[self.cur_goal_index]
                for index in task_group:
                    goal_x = int(self.goal_seq_df.iloc[index]['x'])
                    goal_y = int(self.goal_seq_df.iloc[index]['y'])
                    goal_room = int(self.goal_seq_df.iloc[index]['room'])
                    if self.reached_pos(goal_x, goal_y, goal_room):
                        if self.no_temporal_order_sim_goal_tick_table[index] == 0:
                            reward = 1.0
                            self.no_temporal_order_sim_goal_tick_table[index] = 1   
            
            
            # ! special case for first goal
            if self.cur_goal_index == 0:
                t_range = range(self.cur_goal_index, self.cur_goal_index + 3)
            else:
                t_range = range(self.cur_goal_index, self.cur_goal_index + 1)
            for goal_index in t_range:
                goal_x = int(self.goal_seq_df.iloc[goal_index]['x'])
                goal_y = int(self.goal_seq_df.iloc[goal_index]['y'])
                goal_room = int(self.goal_seq_df.iloc[goal_index]['room'])
                
                if self.reached_pos(goal_x, goal_y, goal_room):
                    # oracle case  
                    if self.lang_rew_machine_type in ['oracle', 'false_negative_sim', 'false_positive_sim']:
                        reward = 1.0 # we give reward for each goal completion
                    
                    self.completed_goals.append(self.total_goal_seq[goal_index])
                    self.cur_goal_index = goal_index + 1
                    self.cur_goal = self.total_goal_seq[self.cur_goal_index]
                    break # only one goal can be completed in one step
        
            info['cur_goal'] =  self.cur_goal   
            info['cur_goal_index'] = self.cur_goal_index
            
            # process the false positive case 
            if self.is_testing_false_positive:
                x_pos, y_pos = self.agent_pos()
                if self.false_positive_reward_locations[x_pos, y_pos] and not is_visited_before:
                    reward += 0.15 # give a small reward for false positive case, 0.15 is from previous experiment showing that the cosine sim score for not-matched pairs is around 0.15
        
        # if lives being 0, reset the goal seq info
        if info['lives'] == 0:
            self.reset_goal_seq_info()
            
        if info['lives'] != self.cur_lives:
            self.cur_lives = info['lives']
            self.reset_visited_states() # reset the visited states when lives change
            # skip 15 frames
            for _ in range(15):
                obs, _, _, _, _ = self.env.step(0)
            
        if self.lang_rew_machine_type in ['oracle', 'false_negative_sim', 'false_positive_sim']:
            # normalize reward 
            if reward > 1.0:
                reward = 1.0
        return obs, reward, done, truncated, info
        
    def agent_pos(self):
        ale = self.env.get_wrapper_attr('ale')
        x, y = ale.getRAM()[42:44]
        return int(x), int(y)
    
    def room(self):
        ale = self.env.get_wrapper_attr('ale')
        return int(ale.getRAM()[3])
    
    def reached_pos(self, x_, y_, goal_room_):
        x, y = self.agent_pos()
        pos_cond = (x_ - 5 <= x <= x_ + 5) and (y_ - 5 <= y <= y_ + 5)
        room = self.room()
        room_cond = room == goal_room_
        return pos_cond and room_cond
    
    def save_state(self, filename):
        state = self.env.clone_full_state()
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print('File written : {}'.format(filename))
    
    def load_state(self, filename):
        with open(filename, 'rb') as f:
            state_bak = pickle.load(f)
        self.env.restore_full_state(state_bak)
        obs, *_ = self.step(0)
        return obs

# ! --- End Montezuma Part ---


class FourOutputWrapper(Wrapper):
    """The step output of the environment is a tuple of four elements: observation, reward, done, info. The truncated value will be integrated with the done value in the info dictionary."""

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        info["truncated"] = truncated
        done = done or truncated
        return obs, reward, done, info

class MinigridCustomWrapper(gym.Wrapper):
    "Put in advance of FourOutputWrapper but after MinigridRGBImgObsWrapper"

    def __init__(self, env: gym.Env, max_difficulty=3, lang_rew_machine_type='standard'):
        super().__init__(env)

        # define the observation space only keep the image
        self.observation_space = self._transform_observation_space()
        self.max_difficulty = max_difficulty
        self.goal_str_lst = []
        self.bonus_flag = 0 # number of goals done
        self.lang_rew_machine_type = lang_rew_machine_type
        self.get_second_reward_flag = False 
        self.mission = ""

    def step(self, action):
        if action > 2:
            action = 5 # https://minigrid.farama.org/environments/babyai/GoToSeq/, only 5 actions is useful here
        obs, reward, done, trunc, info = self.env.step(action)
        # Modify the observation
        new_obs = self._modify_observation(obs)
        # Augment the info dictionary
        new_info = self._augment_info(info, obs, action)     
        if new_info["first_instr_done"] and self.bonus_flag == 0: 
            self.bonus_flag += 1
            if self.lang_rew_machine_type == "oracle":
                reward = 1.0 # reward for completing the first instruction
        if self.lang_rew_machine_type == "no_temporal_order_sim":
            second_instr_done_raw = new_info["second_instr_done_raw"]
            if second_instr_done_raw:
                reward = 1.0
                
        elif self.lang_rew_machine_type == "false_negative_sim":
            pass # no need to change the reward, as it only have 2 instructions and in this case, we do not reward the first instruction
                
               
        return new_obs, reward, done, trunc, new_info
    
    def reset(self, *args, **kwargs):
        # reset the environment until the difficulty is less than the max_difficulty
        while True:
            obs, _ = super().reset(*args, **kwargs)
            # check difficulty
            instrs = self.env.get_wrapper_attr("instrs")
            num_navs_needed = self.env.get_wrapper_attr("num_navs_needed")(instrs)
            if num_navs_needed <= self.max_difficulty:
                    break
        mission = obs["mission"]
        self.mission = mission
        new_obs = self._modify_observation(obs)
        self.goal_str_lst[:] = []
        self.bonus_flag = 0
        self.get_second_reward_flag = False 
        
        
        if not isinstance(instrs, AfterInstr) and not isinstance(instrs, BeforeInstr):
            self.goal_str_lst.append(instrs.surface(self.env))
        else:
            if ", then" in mission:
                first_instr = instrs.instr_a
                second_instr = instrs.instr_b
                
            elif "after you" in mission:
                first_instr = instrs.instr_b
                second_instr = instrs.instr_a
            else:
                raise ValueError("The instruction is not supported")
            
            if len(self.goal_str_lst) == 0:
                first_instr_str = first_instr.surface(self.env)
                second_instr_str = second_instr.surface(self.env)
                self.goal_str_lst.append(first_instr_str)
                self.goal_str_lst.append(second_instr_str)
        
        
        return new_obs

    def _transform_observation_space(self):
        obs_space = self.env.observation_space
        img_obs_space = obs_space['image']
        new_obs_space = spaces.Box(
            low=img_obs_space.low, high=img_obs_space.high, dtype=img_obs_space.dtype
        )
        return new_obs_space

    def _modify_observation(self, obs):
        return obs["image"]

    def _augment_info(self, info, obs, action):
        info["direction"] = obs["direction"]
        info["mission"] = obs["mission"]
        info["grid"] = obs["grid"]
        info['agent_location'] = np.where(obs['grid'] == OBJECT_TO_IDX['agent'])
        assert len(info['agent_location'][0]) == 1
        info['agent_location'] = np.array([info['agent_location'][0][0], info['agent_location'][1][0], self.seed_idx])
        instrs = self.env.get_wrapper_attr("instrs")
        
        if not isinstance(instrs, AfterInstr) and not isinstance(instrs, BeforeInstr):
            first_instr_done = instrs.verify(0)
            if first_instr_done == "success":
                first_instr_done = 1
            else:
                first_instr_done = 0
            second_instr_done = 1 # always true because there is no second instr
            if len(self.goal_str_lst) == 0:
                self.goal_str_lst.append(instrs.surface(self.env))
            info['cur_goal'] = self.goal_str_lst[0]
        else:
            if ", then" in info["mission"]:
                first_instr_done = instrs.a_done # False | continue | success | failure, 
                second_instr_done = instrs.b_done
                first_instr = instrs.instr_a
                second_instr = instrs.instr_b
                
                
            elif "after you" in info["mission"]:
                first_instr_done = instrs.b_done
                second_instr_done = instrs.a_done
                first_instr = instrs.instr_b
                second_instr = instrs.instr_a
            else:
                raise ValueError("The instruction is not supported")
            
            if len(self.goal_str_lst) == 0:
                first_instr_str = first_instr.surface(self.env)
                second_instr_str = second_instr.surface(self.env)
                self.goal_str_lst.append(first_instr_str)
                self.goal_str_lst.append(second_instr_str)
                
            if first_instr_done == "success":
                first_instr_done = 1
            else:
                first_instr_done = 0
            if second_instr_done == "success":
                second_instr_done = 1
            else:
                second_instr_done = 0
            
            if not first_instr_done:
                info['cur_goal'] = self.goal_str_lst[0]
            else:
                info['cur_goal'] = self.goal_str_lst[1]
            
        info["first_instr_done"] = first_instr_done
        info["second_instr_done"] = second_instr_done
        second_instr_done_raw = second_instr.verify(action) == 'success'
        if second_instr_done_raw:
            if not self.get_second_reward_flag:
                self.get_second_reward_flag = True
            else:
                # we do not give reward for the second time
                second_instr_done_raw = False
        info["second_instr_done_raw"] = second_instr_done_raw
        
        return info


class MinigridRGBImgObsWrapper(gym.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation, but reserve the original grid observation.
    This can be used to have the agent to solve the gridworld in pixel space.

    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env = RGBImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
    """

    def __init__(self, env, tile_size=8, auxiliary_info=None):
        super().__init__(env)

        self.tile_size = tile_size
        self.include_grid = False 
        
        # check if auxiliary_info is not None, currently we only support grid
        if auxiliary_info is not None:
            if isinstance(auxiliary_info, list):
                if "grid" in auxiliary_info:
                    self.include_grid = True
                
        

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.height * tile_size, self.env.width * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=True, tile_size=self.tile_size)
        if self.include_grid:
            obs["grid"] = obs["image"]

        return {**obs, "image": rgb_img}

class CrafterCustomWrapper(old_gym.Wrapper):
    def __init__(self, env, lang_rew_machine_type, **kwargs):
        super().__init__(env)
        self.lang_rew_machine_type = lang_rew_machine_type
        self.cur_achievement = [0] * len(CRAFTER_TASKS)
        if lang_rew_machine_type == "false_negative_sim":
            self.false_negative_deactivate_tasks = kwargs['false_negative_deactivate_tasks']
        
    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.cur_achievement = [0] * len(CRAFTER_TASKS)
        return obs
                
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        achievements =  [info["achievements"][task] for task in CRAFTER_TASKS]
        # know which is the new achievement
        new_achievement = [a - b for a, b in zip(achievements, self.cur_achievement)]
        # usually only one task can be completed in one step
        
        # get the index of the new achievement
        if sum(new_achievement) > 0:
            new_achievement_names = []
            for index, value in enumerate(new_achievement):
                if value == 1:
                    new_achievement_names.append(CRAFTER_TASKS[index])
        else:
            new_achievement_names = []
        # update self.cur_achievement
        self.cur_achievement = achievements
        # only oracle case will give frequent reward
        if self.lang_rew_machine_type not in ['standard', 'oracle', 'no_temporal_order_sim', 'false_negative_sim']:
            one_in_flag = False 
            for new_achievement_name in new_achievement_names:
                if new_achievement_name in CRAFTER_STANDARD_CASE_REWARDING_TASKS:
                    one_in_flag = True
                    break
            if not one_in_flag:
                reward = 0.0
                
        if self.lang_rew_machine_type in ['no_temporal_order_sim', 'oracle', 'standard']:
            # it will be the usual case 
            pass
        elif self.lang_rew_machine_type == 'false_negative_sim':
            for new_achievement_name in new_achievement_names:
                if new_achievement_name in self.false_negative_deactivate_tasks:
                    reward = max(0.0, reward - 1.0)
            
        return obs, reward, done, info
  

class VecPyTorch(VecEnvWrapper):
    def __init__(
        self, venv: VecEnv, img_mean, img_std, img_size, env_name, device: str = "cuda"
    ):
        super().__init__(venv)
        self.device = device
        self.img_mean = img_mean
        self.img_std = img_std
        self.img_size = img_size
        self.img_transform = Compose(
            [
                ToTensor(),
                Resize(size=[img_size, img_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
                Normalize(mean=self.img_mean, std=self.img_std),
            ]
        )
        assert env_name in ["crafter", "minigrid", "montezuma"]
        self.env_name = env_name
        self.observation_space = self._transform_observation_space()
        

    def reset(self) -> th.Tensor:
        obs = self.venv.reset()
        obs = self._transform_obs(obs)

        return obs

    def step_async(self, actions: th.Tensor):
        actions = self._transform_actions(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self._transform_obs(obs)
        rewards = self._transform_rewards(rewards)
        dones = self._transform_dones(dones)
        infos = self._transform_infos(infos)

        return obs, rewards, dones, infos

    def _transform_observation_space(self):
        obs_space = self.observation_space
        obs_shape = getattr(obs_space, "shape")
        obs_shape = (obs_shape[2], self.img_size, self.img_size)
        new_obs_space = spaces.Box(low=0, high=1, shape=obs_shape)
        return new_obs_space

    def _transform_obs(self, obs: np.ndarray) -> th.Tensor:
        assert len(obs.shape) == 4  # shape [env_size, H, W, 3]
        obs = th.stack(
            [self.img_transform(o) for o in obs]
        )  # shape [env_size, 3, H, W]
        obs = obs.to(self.device)

        return obs

    def _transform_rewards(self, rewards: np.ndarray) -> th.Tensor:
        assert len(rewards.shape) == 1
        rewards = rewards[:, np.newaxis]  # shape [env_size, 1]
        rewards = th.from_numpy(rewards).float().to(self.device)

        return rewards

    def _transform_dones(self, dones: np.ndarray) -> th.Tensor:
        assert len(dones.shape) == 1
        dones = dones[:, np.newaxis]  # shape [env_size, 1]
        dones = th.from_numpy(dones).float().to(self.device)
        return dones

    def _transform_infos(
        self, infos: Sequence[Dict[str, np.ndarray]]
    ) -> Dict[str, th.Tensor]:
        # save terminal_observation 
        terminal_observation = []
        for info in infos:
            if "terminal_observation" in info:
                transformed_teminal_obs = self.img_transform(info["terminal_observation"]).to(self.device)
                terminal_observation.append(transformed_teminal_obs)
            else:
                terminal_observation.append(th.zeros(3, self.img_size, self.img_size).to(self.device))
        terminal_observation = th.stack(terminal_observation, dim=0)
        
        # Episode lengths and rewards
        episode_lengths = th.zeros(len(infos)).long().to(self.device)
        episode_rewards = th.zeros(len(infos)).float().to(self.device)

        for i, info in enumerate(infos):
            if "episode" in info:
                episode_lengths[i] = int(info["episode"]["l"])
                episode_rewards[i] = float(info["episode"]["r"])

        if self.env_name == "crafter":
            # Achievements
            achievements = [
                [info["achievements"][task] for task in CRAFTER_TASKS] for info in infos
            ] # shape [env_size, len(CRAFTER_TASKS)]
            achievements = np.array(achievements)
            achievements = th.from_numpy(achievements).long().to(self.device)

            # Successes
            successes = (achievements > 0).long()
        elif self.env_name == "minigrid": # minigrid
            # in this environment, achievement will calculate the subgoal completion
            # and success will calculate the final goal completion
            # so if any subgoal is not completed, the final goal is not completed
            first_instr_done = np.array([
                info["first_instr_done"] for info in infos
            ], dtype=np.int64 ) # shape [env_size,]
            second_instr_done = np.array([
                info["second_instr_done"] for info in infos
            ], dtype=np.int64) # shape [env_size,]
            # stack 
            achievements = np.stack([first_instr_done, second_instr_done], axis=-1) # shape [env_size, 2]
            achievements = th.from_numpy(achievements).long().to(self.device)
            
            # success should be the same as achievements in this case
            successes = achievements
            
            # get mission text
            mission_texts = [info["mission"] for info in infos]
            # make it numpy array with shape (env_size, 1), dtype = "<U100"
            mission_texts = np.array(mission_texts, dtype="<U100").reshape(-1, 1)
            
            cur_goals = [info['cur_goal'] for info in infos]
            # make it numpy array with shape (env_size, 1), dtype = "<U100"
            cur_goals = np.array(cur_goals, dtype="<U100").reshape(-1, 1)
            
        elif self.env_name == "montezuma":
            # achievement and success are the same in this case
            # achievement shape [env_size, len(goal_seq_df)]
            achievements = th.full((len(infos), infos[0]['total_goal_num']), 0, dtype=th.int64, device=self.device)
            # get the cur_goal_index
            cur_goal_indices = [info['cur_goal_index'] for info in infos] # shape [env_size,]
            # before this index, the goal is completed
            for i, index in enumerate(cur_goal_indices):
                achievements[i, :index] = 1
            
            mission_texts = [info['cur_goal'] for info in infos]
            # make it numpy array with shape (env_size, 1), dtype = "<U100"
            mission_texts = np.array(mission_texts, dtype="<U100").reshape(-1, 1)
            # copy to cur_goals
            cur_goals = mission_texts # ! cur_goals is the same as mission_texts in Montezuma
            
            successes = achievements
    

        # Infos
        updated_infos = {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "achievements": achievements,
            "successes": successes,
            "terminal_observation": terminal_observation,
        }
        if self.env_name == "minigrid" or self.env_name == "montezuma":
            updated_infos["mission_texts"] = mission_texts
            updated_infos['cur_goals'] = cur_goals
            updated_infos['agent_locations'] = np.stack([info['agent_location'] for info in infos], axis=0) # shape [env_size, 3]
            
        if self.env_name == "crafter":
            # add inventory info 
            inventories = [info["inventory"] for info in infos]
            updated_infos["inventories"] = inventories
            
        if self.env_name == "montezuma":
            # add visited states
            updated_infos['is_state_visited'] = np.stack([i["is_state_visited"] for i in infos], axis=0)
            
        return updated_infos

    def _transform_actions(self, actions: th.Tensor) -> np.ndarray:
        assert (
            len(actions.shape) == 2, f"actions shape should be [env_size, action_dim], got {actions.shape}"
        )  # shape [env_size, 1] from pytorch policy model, and the dtype is Long
        actions = actions.squeeze(dim=-1)  # shape [env_size,]
        actions = actions.cpu().numpy()

        return actions

if __name__ == "__main__":
    print("Initiating the environment wrapper...")