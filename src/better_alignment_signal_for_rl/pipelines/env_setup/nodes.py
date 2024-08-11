"""
This is a boilerplate pipeline 'env_setup'
generated using Kedro 0.19.3
"""

from copy import deepcopy
import time
from omegaconf import OmegaConf, DictConfig
from kedro.io import MemoryDataset
from icecream import ic
import random
import numpy as np
import pandas as pd
import torch as th
from functools import partial
from crafter.env import Env as CrafterEnv
from stable_baselines3.common.vec_env.subproc_vec_env import (
    SubprocVecEnv,
)  # multiprocess env wrapper

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv  # single process env wrapper

from stable_baselines3.common.vec_env.vec_monitor import (
    VecMonitor,
)  # add episode_count episode_returns info

from .env_wrapper import (
    CRAFTER_STANDARD_CASE_REWARDING_TASKS,
    CRAFTER_TASKS,
    CrafterCustomWrapper,
    FourOutputWrapper,
    MinigridRGBImgObsWrapper,
    MinigridCustomWrapper,
    VecPyTorch,
    MontezumaInfoWrapper,
    MaxAndSkipEnv,
    MR_ActionsWrapper,
)
from minigrid.wrappers import ReseedWrapper
from minigrid.wrappers import FullyObsWrapper
import gymnasium as gym
import sys
import os 

from better_alignment_signal_for_rl.agent_components.storage import RolloutStorage
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")
MONTEZUMA_NSTEP_EPOCH_RATIO = 4

def minigrid_env_factory(
    observation,
    max_steps,
    auxiliary_info,
    level_name,
    room_size,
    num_rows,
    num_cols,
    max_difficulty=3,
    is_task_composition=True,
    env_seeds_lst=None,
    lang_rew_machine_type = 'standard',
):
    def _init():
        assert observation in ["full", "partial"]
        if is_task_composition:
            instr_kinds = ["seq"]
        else:
            instr_kinds = ["action"]

        env = gym.make(
            level_name,
            render_mode="rgb_array",
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            instr_kinds=instr_kinds,
            max_steps=max_steps,
        )
        # apply reseed wrapper
        env = ReseedWrapper(env, seeds=env_seeds_lst)
        # apply fully obs wrapper
        if observation == "full":
            env = FullyObsWrapper(env)
        # apply Img Obs wrapper
        env = MinigridRGBImgObsWrapper(env, auxiliary_info=auxiliary_info)
        # apply MinigridCustomWrapper
        env = MinigridCustomWrapper(env, max_difficulty=max_difficulty, lang_rew_machine_type=lang_rew_machine_type)
        # apply FourOutputWrapper
        env = FourOutputWrapper(env)
        return env

    return _init

def crafter_env_factory(partial_func, lang_rew_machine_type, false_negative_deactivate_tasks = None):
        
    def _init():
        env = partial_func()
        # apply custom wrapper
        env = CrafterCustomWrapper(env, lang_rew_machine_type,
                                   false_negative_deactivate_tasks=false_negative_deactivate_tasks)
        return env

    return _init


def montezuma_env_factory(level_name, walkthrough_csv_path,
                          lang_rew_machine_type, false_negative_sim_goal_seq_df = None, oracle_error_rate=None):
    # the montezuma env has no stochasticity, thus we do not need to set seeds
    # check lang_rew_machine_type
    if lang_rew_machine_type == "false_positive_sim":
        print("False Positive Sim is testing")
        
        # create false_positive_reward_locations
        assert oracle_error_rate is not None
        # according to oracle_error_rate, randomly flip the false_positive_reward_locations
        false_positive_reward_locations = np.random.choice([True, False], size=(400, 400), p=[oracle_error_rate, 1 - oracle_error_rate])
    else:
        false_positive_reward_locations = None
    
    def _init():
        env = gym.make(level_name)
        env = MaxAndSkipEnv(env, skip=4)
        env = MR_ActionsWrapper(env)
        goal_seq_df = pd.read_csv(walkthrough_csv_path)
        env = MontezumaInfoWrapper(env, goal_seq_df, lang_rew_machine_type,
                                   false_negative_sim_goal_seq_df=false_negative_sim_goal_seq_df, false_positive_reward_locations=false_positive_reward_locations)
        # apply FourOutputWrapper
        env = FourOutputWrapper(env)
        return env
    
    return _init


def setup_env_helper(general_cfg, env_setup_cfg, env_process_num, seed, env_purpose, lang_rew_machine_type='standard', oracle_error_rate=0.5):
    # ! used for setting up both training and evaluation env
    # when env_process_num = 1, we deem it as eval env
    env_name = general_cfg["env_name"]
    assert env_name in ["crafter", "minigrid", "montezuma"]

    # init seed
    # set seed for random, np and th
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # set env params
    max_steps = general_cfg["max_steps"]
    if env_name == "minigrid":
        max_steps = int(env_setup_cfg['minigrid_env_params']['room_size'] * env_setup_cfg['minigrid_env_params']['room_size'] * env_setup_cfg['minigrid_env_params']['num_rows'] * env_setup_cfg['minigrid_env_params']['num_cols'] * env_setup_cfg['minigrid_env_params']['max_step_ratio'])
    device = general_cfg["device"]

    # constant config
    constant_param = general_cfg["constant"]
    clip_mean = constant_param["clip_mean"]
    clip_std = constant_param["clip_std"]
    env_size = constant_param["env_size"]

    if env_name == "crafter":
        env_params = env_setup_cfg["crafter_env_params"]
        has_ext_reward = env_params["has_ext_reward"]
        # set seeds for env
        seeds = np.random.randint(0, 2**31 - 1, size=env_process_num)  # shape (nproc,)
        # apply to env
        if lang_rew_machine_type == 'false_negative_sim':
            # generate the false_negative_deactivate_tasks
            task_candidates = deepcopy(CRAFTER_TASKS)
            # remove tasks that are in CRAFTER_STANDARD_CASE_REWARDING_TASKS
            for task in CRAFTER_STANDARD_CASE_REWARDING_TASKS:
                if task in task_candidates:
                    task_candidates.remove(task)
            # randomly select error_rate
            random_indexs = np.random.choice(len(task_candidates), int(len(task_candidates) * (1- oracle_error_rate)), replace=False)
            random_indexs = sorted(random_indexs)
            false_negative_deactivate_tasks = [task_candidates[i] for i in random_indexs]
        else:
            false_negative_deactivate_tasks = None
        env_fns = [
            crafter_env_factory(
                partial(
                    CrafterEnv,
                    seed=seed,
                    length=max_steps,
                    size=(env_size, env_size),
                    reward=has_ext_reward,
                ),
                lang_rew_machine_type,
                false_negative_deactivate_tasks=false_negative_deactivate_tasks,
            )
            for seed in seeds
        ]

    elif env_name == "minigrid":  # minigrid
        env_params = env_setup_cfg["minigrid_env_params"]
        # set seeds for env
        if env_process_num > 2: 
            diff_env_num = 50 # ! so far we do not focus on the generalization of minigrid env, thus we limit the env diversity
        else:
            diff_env_num = 80000 # add more seeds for evaluation and training data generation 
        seeds = np.random.randint(0, 2**31 - 1, size=(1, diff_env_num))  # shape (1, diff_env_num)
        # repeat the seeds for nproc times
        seeds = np.repeat(seeds, env_process_num, axis=0)
        # shuffle along the first axis
        for i in range(env_process_num):
            np.random.shuffle(seeds[i])

        print(f"Seeds shape for Minigrid Env is {seeds.shape}")

        # to list
        seeds = seeds.tolist()
        observation = env_params["observation"]
        auxiliary_info = env_params["auxiliary_info"]
        level_name = env_params["level_name"]
        room_size = env_params["room_size"]
        num_rows = env_params["num_rows"]
        num_cols = env_params["num_cols"]
        if env_purpose == "lang":
            is_task_composition = env_params["lang_purpose_params"][
                "is_task_composition"
            ]
        else:
            is_task_composition = env_params["policy_purpose_params"][
                "is_task_composition"
            ]

        max_difficulty = env_params["max_difficulty"]

        env_fns = [
            minigrid_env_factory(
                observation=observation,
                max_steps=max_steps,
                auxiliary_info=auxiliary_info,
                level_name=level_name,
                room_size=room_size,
                num_rows=num_rows,
                num_cols=num_cols,
                max_difficulty=max_difficulty,
                is_task_composition=is_task_composition,
                env_seeds_lst=seed_list,
                lang_rew_machine_type = lang_rew_machine_type,
            )
            for seed_list in seeds
        ]
    elif env_name == "montezuma":
        env_params = env_setup_cfg['montezuma_env_params']
        level_name = env_params['level_name']
        walkthrough_csv_path = os.path.join(os.environ['PWD'], env_params['walkthrough_csv_path'])
        if not os.path.exists(walkthrough_csv_path):
            raise FileNotFoundError(f"File {walkthrough_csv_path} does not exist")

        # get the false_negative_sim_goal_seq_df
        if lang_rew_machine_type == 'false_negative_sim':
            goal_seq_df_original = pd.read_csv(walkthrough_csv_path) 
            current_seed_backup = np.random.get_state()
            np.random.seed(13)
            # the cols are goal,room,x,y
            # we randomly skip (delete) 50% of the goals within the room 
            false_negative_sim_goal_seq_df = pd.DataFrame(columns=['goal', 'room', 'x', 'y'])
            cur_room = 1 # start from room 1
            cur_rows = []
            for index, row in goal_seq_df_original.iterrows():
                if row['room'] == cur_room:
                    cur_rows.append(row)
                else:
                    # randomly select 50% of the goals
                    random_indexs = np.random.choice(len(cur_rows), int(len(cur_rows) * (1- oracle_error_rate)), replace=False)
                    random_indexs = sorted(random_indexs)
                    for i in random_indexs:
                        false_negative_sim_goal_seq_df.loc[len(false_negative_sim_goal_seq_df)] = cur_rows[i]
                    cur_room = row['room']
                    cur_rows = [row]
            # restore the backup seed
            np.random.set_state(current_seed_backup)
        else:
            false_negative_sim_goal_seq_df = None

        env_fns = [montezuma_env_factory(level_name, walkthrough_csv_path, lang_rew_machine_type, false_negative_sim_goal_seq_df, oracle_error_rate) for _ in range(env_process_num)]

    else:
        raise ValueError(f"Invalid env_name: {env_name}")

    if env_process_num == 1:
        venv = DummyVecEnv(env_fns)  # create single process env
    else:
        venv = SubprocVecEnv(env_fns)  # create multiprocess env
    venv = VecMonitor(venv)  # add episode_count episode_returns info
    venv = VecPyTorch(
        venv,
        img_mean=clip_mean,
        img_std=clip_std,
        img_size=env_size,
        env_name=env_name,
        device=device,
        
    )  # convert to pytorch tensor, also resize the image
    return venv

# ! NODE
def setup_env(general_cfg, env_setup_cfg, reward_machine_cfg):
    """
    Set up the environment based on the provided configuration.

    Args:
        cfg (dict): Configuration dictionary containing the following keys:
            - env_name (str): Name of the environment ("crafter" or "minigrid").
            - env_purpose (str): Purpose of the environment ("lang" or "policy").
            - seed (int): Seed value for random number generation.
            - nproc (int): Number of processes for the environment.
            - max_steps (int): Maximum number of steps per episode.
            - device (str): Device to use for computation.
            - constant (dict): Dictionary containing constant configuration parameters.
                - clip_mean (float): Mean value for image clipping.
                - clip_std (float): Standard deviation value for image clipping.
                - env_size (int): Size for image 

    Returns:
        venv (SubprocVecEnv): Multiprocess environment.
        obs (numpy.ndarray): Initial observation from the environment.

    Raises:
        AssertionError: If the provided environment name or purpose is invalid.
    """
    nproc = general_cfg["nproc"]
    seed = general_cfg["seed"]
    env_purpose = general_cfg["env_purpose"]
    assert env_purpose in ["lang", "policy"]
    lang_rew_machine_type = reward_machine_cfg["lang_reward_machine_type"]
    oracle_error_rate = reward_machine_cfg['oracle_error_rate']
    # TODO oracle_error_rate
    assert lang_rew_machine_type in ['standard', 'no_temporal_order', 'oracle', 'false_negative', 'no_temporal_order_sim', 'false_negative_sim', 'false_positive_sim']
    return setup_env_helper(general_cfg, env_setup_cfg, nproc, seed, env_purpose, lang_rew_machine_type, oracle_error_rate)


# ! NODE
def setup_rollout_storage(general_cfg, venv):
    nstep = general_cfg["nstep"]
    nproc = general_cfg["nproc"]
    observation_space = venv.observation_space
    action_space = venv.action_space
    hidsize = general_cfg["hidsize"]
    device = general_cfg["device"]
    clip_embeds_size = general_cfg['constant']['clip_embeds_size']
    
    env_name = general_cfg["env_name"]
    if env_name == "crafter":
        subgoal_num = len(CRAFTER_TASKS)
    elif env_name == "minigrid":
        subgoal_num = 2 # for minigrid, we only have 2 subgoals (first instr, second instr)
    elif env_name == "montezuma":
        subgoal_num = venv.get_attr("total_goal_num")[0]
        # ! decrease nstep for montezuma env
        nstep = int(nstep / MONTEZUMA_NSTEP_EPOCH_RATIO)
    
    # create rollout storage
    rollout_storage = RolloutStorage(
        nstep = nstep,
        nproc = nproc,
        observation_space = observation_space,
        action_space = action_space,
        hidsize = hidsize,
        device = device,
        subgoal_num = subgoal_num,
        clip_embeds_size =clip_embeds_size,
        env_name = env_name,
    )
    
    # init the rollout storage and also init the venv
    obs = venv.reset()
    rollout_storage.obs[0].copy_(obs) # copy without grad
    
    return rollout_storage
