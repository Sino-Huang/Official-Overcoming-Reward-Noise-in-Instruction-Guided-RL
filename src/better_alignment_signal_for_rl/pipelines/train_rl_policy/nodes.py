"""
This is a boilerplate pipeline 'train_rl_policy'
generated using Kedro 0.19.3
"""
import json
import random 
import pandas as pd
import torch as th
import numpy as np
from natsort import natsorted
from glob import glob
import os 
import sys 
import wandb 
from pathlib import Path 
import torch.optim as optim
from better_alignment_signal_for_rl.agent_components.storage import RolloutStorage
from better_alignment_signal_for_rl.pipelines.env_setup.nodes import MONTEZUMA_NSTEP_EPOCH_RATIO
from better_alignment_signal_for_rl.pipelines.eval_lrm.cp_contant import get_cp_threshold
from better_alignment_signal_for_rl.pipelines.eval_lrm.exist_impact_eval_enum import Stage_1_Offline_Evaluation_Type, Stage_1_Online_Evaluation_Type
from better_alignment_signal_for_rl.pipelines.reward_machine_setup.nodes import Reward_Machine
from better_alignment_signal_for_rl.pipelines.rl_policy_setup.int_rew_utils import int_rew_update
from better_alignment_signal_for_rl.pipelines.train_rl_policy.noise_solution_eval_enum import Stage_2_Online_Evaluation_Type
from better_alignment_signal_for_rl.ppo_backbone.model.montezuma_model import RNDModel, RewardForwardFilter, RunningMeanStd
from better_alignment_signal_for_rl.ppo_backbone.algorithm.base import BaseAlgorithm
from better_alignment_signal_for_rl.ppo_backbone.algorithm.ppo import PPOAlgorithm
from better_alignment_signal_for_rl.ppo_backbone.algorithm.motnezuma_ppo import MontezumaPPOAlgorithm
from better_alignment_signal_for_rl.ppo_backbone.algorithm.ppo_ad import PPOADAlgorithm
from better_alignment_signal_for_rl.ppo_backbone.algorithm.ppo_ewc import PPOEWCAlgorithm
from torchvision.transforms.functional import resize, rgb_to_grayscale

from .sample import sample_rollouts
from better_alignment_signal_for_rl.pipelines.env_setup.env_wrapper import CRAFTER_TASKS
from better_alignment_signal_for_rl.agent_components.ir_deir_model import DiscriminatorModel
from tqdm.auto import tqdm 


all_online_eval_list = list(Stage_2_Online_Evaluation_Type) + list(Stage_1_Online_Evaluation_Type)

# ! NODE
def train_rl_main(venv, policy_model, int_rew_model, lang_rew_model, lang_rew_model_load_path, storage : RolloutStorage, general_cfg, env_setup_cfg, rl_policy_setup_cfg, reward_machine_cfg, train_rl_policy_cfg, lang_rew_model_cfg):
    """The main training loop for the RL policy
    policy_model, int_rew_model, lang_rew_model should be initialized and in cpu device
    """

    env_name = general_cfg['env_name']
    nstep = general_cfg['nstep']
    nproc = general_cfg['nproc']
    if env_name == "montezuma":
        nstep = nstep // MONTEZUMA_NSTEP_EPOCH_RATIO
    is_int_rew_activated = rl_policy_setup_cfg['is_int_rew_activated']
    is_lang_rew_activated = rl_policy_setup_cfg['is_lang_rew_activated']
    save_freq = train_rl_policy_cfg['save_freq']
    if env_name == "montezuma":
        save_freq = save_freq * MONTEZUMA_NSTEP_EPOCH_RATIO
    lang_reward_machine_type = reward_machine_cfg['lang_reward_machine_type']
    oracle_error_rate = reward_machine_cfg['oracle_error_rate']
    
    lang_rew_settings = rl_policy_setup_cfg['lang_rew_settings']

    # lang_rew_model cfg
    traj_length = lang_rew_model_cfg['traj_length']
    pretrained_model_cls = lang_rew_model_cfg['model_kwargs']['pretrained_model_cls']
    minigrid_no_pretrain = lang_rew_model_cfg['model_kwargs']['minigrid_no_pretrain']
    if env_name != "minigrid":
        minigrid_no_pretrain = False 

    # lang_rew_settings part
    has_hard_signal = lang_rew_settings['has_hard_signal']
    lang_rew_coef = lang_rew_settings['lang_rew_coef']
    has_boltzmann_rationality_coeff = lang_rew_settings['has_boltzmann_rationality_coeff']
    hard_signal_cp_error_rate = str(lang_rew_settings['hard_signal_cp_error_rate'])
    boltzmann_rationality_offset = lang_rew_settings['boltzmann_rationality_offset']
    lang_reward_function_type = lang_rew_settings['lang_reward_function_type']
    has_reward_capping = lang_rew_settings['has_reward_capping']
    capping_max = lang_rew_settings['capping_max']
    


    # eval type tag, we will do some assertion here
    eval_type_tag = train_rl_policy_cfg['eval_type_tag']
    # find the corresponding eval enum
    if eval_type_tag is not None:
        all_eval_list_names = [e.name for e in all_online_eval_list]
        assert eval_type_tag in all_eval_list_names, f"eval_type_tag {eval_type_tag} not found in all_eval_list"
        # change the eval_type_tag to the corresponding enum
        eval_type_tag_index = all_eval_list_names.index(eval_type_tag)
        eval_type_tag = all_online_eval_list[eval_type_tag_index]
        # check the consistency here
        if eval_type_tag == Stage_1_Online_Evaluation_Type.Extra_1_Oracle:
            assert lang_reward_machine_type == "oracle", "Extra_1_Oracle eval_type_tag should have oracle lang_reward_machine_type"
        elif eval_type_tag == Stage_1_Online_Evaluation_Type.H4_1_False_Negative_Sim:
            assert lang_reward_machine_type == "false_negative_sim", "H4_1_False_Negative_Sim eval_type_tag should have false_negative_sim lang_reward_machine_type"
        # TODO add H4_2_False_Positive_Sim
        elif eval_type_tag == Stage_1_Online_Evaluation_Type.H4_2_False_Positive_Sim:
            assert lang_reward_machine_type == "false_positive_sim", "H4_2_False_Positive_Sim eval_type_tag should have false_positive_sim lang_reward_machine" 
        elif eval_type_tag == Stage_1_Online_Evaluation_Type.H1_1_Compo_No_Temp_Order_Sim:
            assert lang_reward_machine_type == "no_temporal_order_sim", "H1_1_Compo_No_Temp_Order_Sim eval_type_tag should have no_temporal_order_sim lang_reward_machine_type"
        elif eval_type_tag in [Stage_1_Online_Evaluation_Type.H3_1_Compared_With_PPO, Stage_1_Online_Evaluation_Type.H2_1_Partial_Rew_Heatmap, Stage_1_Online_Evaluation_Type.H2_2_Partial_Rew_Offset_To_Goal]:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert not has_hard_signal
            assert not has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "normal"
            assert has_reward_capping
        elif eval_type_tag == Stage_2_Online_Evaluation_Type.H5_1_Hard_Signal_Thres_0_1:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert has_hard_signal
            assert hard_signal_cp_error_rate == "0.1"
            assert not has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "normal"
            assert has_reward_capping
        elif eval_type_tag == Stage_2_Online_Evaluation_Type.H5_2_Hard_Signal_Thres_0_2:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert has_hard_signal
            assert hard_signal_cp_error_rate == "0.2"
            assert not has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "normal"
            assert has_reward_capping
        elif eval_type_tag == Stage_2_Online_Evaluation_Type.H6_1_Threshold_As_Beta_Coeff:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert has_hard_signal
            assert hard_signal_cp_error_rate == "0.1" # ! need to confirm 
            assert has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "normal"
            assert has_reward_capping
        elif eval_type_tag == Stage_2_Online_Evaluation_Type.H7_Condi_Mut_Info_Log_Ver:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert has_hard_signal
            assert hard_signal_cp_error_rate == "0.1" # ! need to confirm 
            assert has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "cmi_log"
            assert has_reward_capping
        elif eval_type_tag == Stage_2_Online_Evaluation_Type.H7_Condi_Mut_Info_Lin_Ver:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert has_hard_signal
            assert hard_signal_cp_error_rate == "0.1" # ! need to confirm 
            assert has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "cmi_linear"
            assert has_reward_capping
            
        # extra for soft reward
        elif eval_type_tag == Stage_2_Online_Evaluation_Type.H6_2_Threshold_As_Beta_Coeff_Soft:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert not has_hard_signal
            assert has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "normal"
            assert has_reward_capping
            
        elif eval_type_tag == Stage_2_Online_Evaluation_Type.H7_Condi_Mut_Info_Log_Ver_Soft:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert not has_hard_signal
            assert hard_signal_cp_error_rate == "0.1"  
            assert has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "cmi_log"
            assert has_reward_capping
            
        elif eval_type_tag == Stage_2_Online_Evaluation_Type.H7_Condi_Mut_Info_Lin_Ver_Soft:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert not has_hard_signal
            assert hard_signal_cp_error_rate == "0.1" 
            assert has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "cmi_linear"
            assert has_reward_capping
            
        # extra for hard reward
        elif eval_type_tag == Stage_2_Online_Evaluation_Type.H8_Condi_Mut_Info_Lin_Ver_No_Beta:
            assert is_lang_rew_activated, f"{eval_type_tag} should have is_lang_rew_activated set to True"
            assert lang_reward_machine_type == "standard"
            assert has_hard_signal
            assert hard_signal_cp_error_rate == "0.1" # ! need to confirm 
            assert not has_boltzmann_rationality_coeff
            assert lang_reward_function_type == "cmi_linear"
            assert has_reward_capping
            

    # ! we still lack language reward machine so for now we should set is_lang_rew_activated to False

    # release memory if not activated
    if not is_int_rew_activated:
        del int_rew_model
        int_rew_model = None
    if not is_lang_rew_activated:
        del lang_rew_model
        lang_rew_model = None 
    elif lang_reward_machine_type in ['oracle', 'no_temporal_order_sim', 'false_negative_sim', 'false_positive_sim']:
        del lang_rew_model
        lang_rew_model = None
    else:
        # load lang_rew_model
        lang_rew_model.load_state_dict(th.load(lang_rew_model_load_path))

    # fix random seed
    seed = general_cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    # CUDA setting
    th.set_num_threads(1)
    device = general_cfg['device']

    continue_training = train_rl_policy_cfg['continue_training']

    # ! wandb setup
    mix_config = dict()
    mix_config.update(general_cfg)
    mix_config.update(rl_policy_setup_cfg)
    mix_config.update(train_rl_policy_cfg)
    mix_config.update(lang_rew_model_cfg)

    tag_lst = [
        f'int_rew_{is_int_rew_activated}',
        f'lang_rew_{is_lang_rew_activated}',
    ] 
    
    if eval_type_tag is not None:
        if eval_type_tag in [Stage_1_Online_Evaluation_Type.H4_1_False_Negative_Sim, Stage_1_Online_Evaluation_Type.H4_2_False_Positive_Sim]:
            # append oracle_error_rate to the tag_lst
            tag_lst.append(f"oracle_error_rate_{oracle_error_rate}")

    if is_lang_rew_activated:
        extra_lst = [
            "soft_signal" if not has_hard_signal else "hard_signal",
            f"lang_rew_coef_{lang_rew_coef}",
            f"rew_mac_{lang_reward_machine_type}",
            f"lang_rew_func_type_{lang_reward_function_type}",
            f"traj_len_{traj_length}"
        ]
        if has_hard_signal or (not has_hard_signal and lang_reward_function_type in ['cmi_log', 'cmi_linear']):
            extra_lst.append(f"cp_r{hard_signal_cp_error_rate}")
        if has_reward_capping:
            extra_lst.append(f"lang_rew_cap_{capping_max}")
        if eval_type_tag is not None:
            extra_lst.append(eval_type_tag.name)
        if has_boltzmann_rationality_coeff:
            cp_threshold = get_cp_threshold(
                env_name, traj_length, pretrained_model_cls, minigrid_no_pretrain
            )['0.2'] # we use 0.2 alpha for now
            extra_lst.append(f"boltzmann_beta_{cp_threshold}+{boltzmann_rationality_offset}")

            beta_ceoff = cp_threshold + boltzmann_rationality_offset

        tag_lst.extend(extra_lst)

    tag_lst = natsorted(tag_lst)
    tag_lst_str = '-'.join(tag_lst)

    save_base_dir = os.path.join(os.environ['PWD'], 'data/06_models',env_name , tag_lst_str)
    save_policy_dir = os.path.join(save_base_dir, 'policy_model')
    save_int_rew_dir = os.path.join(save_base_dir, 'int_rew_model')
    Path(save_policy_dir).mkdir(parents=True, exist_ok=True)
    Path(save_int_rew_dir).mkdir(parents=True, exist_ok=True)

    print(f"Training on Env: {env_name} and Int Rew Model type: {int_rew_model.__class__.__name__} and Lang Rew Model type: {lang_rew_model.__class__.__name__}")

    job_name = f"Train RL Agent on {env_name}"
    if is_int_rew_activated:
        job_name += " with Intrinsic Reward"
    if is_lang_rew_activated:
        job_name += " with Language Reward"
        
    if eval_type_tag is not None:
        if eval_type_tag == Stage_1_Online_Evaluation_Type.H4_1_False_Negative_Sim:
            job_name += " with False Negative Oracle"
        elif eval_type_tag == Stage_1_Online_Evaluation_Type.H4_2_False_Positive_Sim:
            job_name += " with False Positive Oracle"

    # for sim lang rew envs, we internally set is_lang_rew_activated back to False
    if lang_reward_machine_type in ['oracle', 'no_temporal_order_sim', 'false_negative_sim', 'false_positive_sim']:
        is_lang_rew_activated = False


    # ! comment this out to do debugging
    if not general_cfg['debug_mode']:
        wandb.init(
            project="Better Vision Language Alignment Signal for RL",
            name=job_name,
            config=mix_config,
            tags=tag_lst
        )

        wandb.watch(policy_model)

        if is_int_rew_activated and int_rew_model is not None:
            wandb.watch(int_rew_model)

        if is_lang_rew_activated and lang_rew_model is not None:
            wandb.watch(lang_rew_model)
    # ! ---------------

    # load model is continue_training is True
    if continue_training:
        policy_checkpoint_paths = glob(os.path.join(save_policy_dir, 'checkpoint_*.pth'))
        if len(policy_checkpoint_paths) == 0:
            raise FileNotFoundError(f"No policy checkpoint found at {save_policy_dir}")
        policy_checkpoint_paths = natsorted(policy_checkpoint_paths)
        policy_load_path = policy_checkpoint_paths[-1]
        if not os.path.exists(policy_load_path):
            raise FileNotFoundError(f"Policy model not found at {policy_load_path}")
        policy_model.load_state_dict(th.load(policy_load_path))

        if is_int_rew_activated:
            int_rew_checkpoint_paths = glob(os.path.join(save_int_rew_dir, 'checkpoint_*.pth'))
            if len(int_rew_checkpoint_paths) == 0:
                raise FileNotFoundError(f"No int_rew checkpoint found at {save_int_rew_dir}")
            int_rew_checkpoint_paths = natsorted(int_rew_checkpoint_paths)
            int_rew_load_path = int_rew_checkpoint_paths[-1]
            if is_int_rew_activated and not os.path.exists(int_rew_load_path):
                raise FileNotFoundError(f"Int Rew model not found at {int_rew_load_path}")

            int_rew_model.load_state_dict(th.load(int_rew_load_path))

    # policy to device
    policy_model = policy_model.to(device)
    if is_int_rew_activated:
        int_rew_model = int_rew_model.to(device)
    if is_lang_rew_activated:
        lang_rew_model = lang_rew_model.to(device)

    # ! reset venv
    obs = venv.reset() # shape [nproc, *obs_shape]
    # assign the first obs to the rollout storage, on training start
    storage.obs[0].copy_(obs) # copy without grad
    if is_int_rew_activated:
        int_rew_type_str_for_monte = train_rl_policy_cfg['montezuma_env_params']['int_rew_type']
        if env_name == "montezuma" and int_rew_type_str_for_monte == "rnd":
            int_rew_model: RNDModel = int_rew_model
            # init important objects
            reward_rms = RunningMeanStd()
            obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
            discounted_reward = RewardForwardFilter(train_rl_policy_cfg['montezuma_env_params']['gamma'])
            optimizer = optim.Adam(list(policy_model.parameters()) + list(int_rew_model.predictor.parameters()), lr=train_rl_policy_cfg['montezuma_env_params']['algorithm_kwargs']['lr'])
        else:
            int_rew_model: DiscriminatorModel = int_rew_model
            int_rew_model.init_obs_queue(obs.cpu().numpy())
    if not is_int_rew_activated and env_name == "montezuma":
        reward_rms = RunningMeanStd()
        obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        discounted_reward = RewardForwardFilter(train_rl_policy_cfg['montezuma_env_params']['gamma'])
        optimizer = optim.Adam(policy_model.parameters(), lr=train_rl_policy_cfg['montezuma_env_params']['algorithm_kwargs']['lr'])

    if env_name == "crafter":
        algorithm_cls_name = train_rl_policy_cfg['crafter_env_params']['algorithm_cls'] 
        algorithm_kwargs = train_rl_policy_cfg['crafter_env_params']['algorithm_kwargs']
    elif env_name in ["montezuma"]:
        algorithm_cls_name = train_rl_policy_cfg['montezuma_env_params']['algorithm_cls']
        algorithm_kwargs = train_rl_policy_cfg['montezuma_env_params']['algorithm_kwargs']
        algorithm_kwargs.update(
            {
                "optimizer": optimizer,
                "obs_rms": obs_rms,
                "nstep": nstep,
                "nproc": general_cfg['nproc'],
                
            }
        )

        if is_int_rew_activated:
            algorithm_kwargs.update(
                {
                    "rnd_int_rew_model": int_rew_model,
                }
            )
        else:
            algorithm_kwargs.update(
                {
                    "rnd_int_rew_model": None,
                }
            )

    elif env_name in ["minigrid"]:
        algorithm_cls_name = train_rl_policy_cfg['minigrid_env_params']['algorithm_cls']
        algorithm_kwargs = train_rl_policy_cfg['minigrid_env_params']['algorithm_kwargs']
    else:
        raise ValueError(f"env_name {env_name} not supported")

    ppo_nepoch_algo = algorithm_kwargs['ppo_nepoch']
    ppo_nbatch_algo = algorithm_kwargs.get('ppo_nbatch', None)

    algorithm_cls = getattr(sys.modules[__name__], algorithm_cls_name)
    algorithm: BaseAlgorithm = algorithm_cls(
        model=policy_model,
        **algorithm_kwargs,
    )

    nepoch = general_cfg['nepoch']
    if env_name == "montezuma":
        # ! increase the nepoch as we decreases nsteps previously in line 273 in env_setup pipeline
        nepoch = nepoch * MONTEZUMA_NSTEP_EPOCH_RATIO
        # nepoch *= 4 # montezuma is a reward sparse environment, we in total give them 4M steps to train,
        nepoch *= 10 # montezuma is a reward sparse environment, we in total give them 10M steps to train, 

    gamma = train_rl_policy_cfg['gamma']
    gae_lambda = train_rl_policy_cfg['gae_lambda']
    if env_name == "montezuma":
        gamma = train_rl_policy_cfg['montezuma_env_params']['gamma']
        gae_lambda = train_rl_policy_cfg['montezuma_env_params']['gae_lambda']

    # Run algorithm
    if env_name == "crafter":
        TASK_LEN = len(CRAFTER_TASKS)
        TASKS = CRAFTER_TASKS
    elif env_name == "minigrid":
        TASK_LEN = env_setup_cfg['minigrid_env_params']['max_difficulty']
        TASKS = [f"task_{i}" for i in range(TASK_LEN)]
    elif env_name == "montezuma":
        walkthrough_csv_path = env_setup_cfg['montezuma_env_params']['walkthrough_csv_path']

        montezuma_walkthrough_df = pd.read_csv(os.path.join(os.environ['PWD'], walkthrough_csv_path))
        # add oracle_error_rate
        if eval_type_tag == Stage_1_Online_Evaluation_Type.H4_1_False_Negative_Sim:
            current_seed_backup = np.random.get_state()
            np.random.seed(13)
            # the cols are goal,room,x,y
            # we randomly skip (delete) of the goals within the room
            false_negative_sim_goal_seq_df = pd.DataFrame(columns=['goal', 'room', 'x', 'y'])
            cur_room = 1 # start from room 1
            cur_rows = []
            for index, row in montezuma_walkthrough_df.iterrows():
                if row['room'] == cur_room:
                    cur_rows.append(row)
                else:
                    # randomly select (1- oracle_error_rate) of the goals
                    random_indexs = np.random.choice(len(cur_rows), int(len(cur_rows) * (1- oracle_error_rate)), replace=False)
                    random_indexs = sorted(random_indexs)
                    for i in random_indexs:
                        false_negative_sim_goal_seq_df.loc[len(false_negative_sim_goal_seq_df)] = cur_rows[i]
                    cur_room = row['room']
                    cur_rows = [row]
            # restore the backup seed
            np.random.set_state(current_seed_backup) 

            del montezuma_walkthrough_df 
            montezuma_walkthrough_df = false_negative_sim_goal_seq_df

        TASK_LEN = len(montezuma_walkthrough_df)
        TASKS = montezuma_walkthrough_df['goal'].tolist()
        TASKS = [f"{i:02d}_{task}" for i, task in enumerate(TASKS)] # add index to the task name for better sorting
    else:
        raise ValueError(f"env_name {env_name} not supported")
    total_successes = np.zeros((0, TASK_LEN), dtype=np.int32)
    best_mean_episodic_return_ext = -9999
    mean_episodic_return_ext = 0
    mean_episodic_return_int = 0
    mean_episodic_return_lang = 0
    mean_int_rew_per_step = 0

    # ! montezuma specific normalize observation
    if env_name == "montezuma":
        print('Initializes observation normalization...')
        pre_obs_norm_steps = train_rl_policy_cfg['montezuma_env_params']['pre_obs_norm_steps'] 
        next_obs = []
        for step in tqdm(range(nstep * pre_obs_norm_steps)):
            actions = th.randint(0, venv.action_space.n, size=(general_cfg['nproc'], 1), device=device)
            obs, rewards, dones, infos = venv.step(actions)
            obs = resize(rgb_to_grayscale(obs), (84, 84), antialias=False) # shape [nproc, 1, 84, 84]

            next_obs.append(obs)

            if len(next_obs) % (nstep * general_cfg['nproc']) == 0:
                next_obs = th.concatenate(next_obs, axis=0).cpu().numpy() # shape [nstep * nproc, 1, 84, 84]
                obs_rms.update(next_obs)
                next_obs = []

    # ! init reward_machine_lst
    if is_lang_rew_activated:
        
        params = dict(
            reward_machine_type=lang_reward_machine_type,
            env_name = env_name,
            reward_cap = capping_max,
            has_hard_signal = has_hard_signal,
            seed = seed,
        )
        working_cp_threshold = get_cp_threshold(
            env_name, traj_length, pretrained_model_cls, minigrid_no_pretrain
        )[hard_signal_cp_error_rate]
        params['cp_threshold'] = working_cp_threshold
            
        if env_name == "montezuma":
            params['walkthrough_df'] = montezuma_walkthrough_df
            
        reward_machine_lst = []
        for _ in range(general_cfg['nproc']):
            reward_machine = Reward_Machine(**params)
            reward_machine_lst.append(reward_machine)
    else:
        reward_machine_lst = None
        
        
        

    for epoch in (pbar:=tqdm(range(nepoch), desc="RL Training")):
        # ! sample episodes
        input_dict = dict(
            env_name = env_name,
            venv = venv,
            policy_model = policy_model,
            int_rew_model = int_rew_model,
            lang_rew_model = lang_rew_model,
            has_exploration_reward = is_int_rew_activated,
            has_language_reward = is_lang_rew_activated,
            reward_machine_lst = reward_machine_lst, 
            storage = storage,
            nproc=general_cfg['nproc'],
            epoch = epoch,
            tag_lst_str=tag_lst_str,
            lang_rew_coef=lang_rew_coef,
            lang_reward_function_type=lang_reward_function_type,
            traj_length=traj_length,
            resize_size_for_lang_rew_model=general_cfg['constant']['clip_size'],
            has_hard_signal=has_hard_signal,
            seed=seed,
        )
        if env_name == "montezuma":
            input_dict.update(
                {
                    "reward_rms": reward_rms,
                    "obs_rms": obs_rms,
                    "discounted_reward": discounted_reward,
                }
            )
        if has_boltzmann_rationality_coeff: # add beta_ceoff to control the exploration of policy model if boltzmann_rationality_coeff is activated
            input_dict['beta_ceoff'] = beta_ceoff
        rollout_stats = sample_rollouts(
            **input_dict
        ) # it will collect nproc x nsteps steps 

        # ! compute returns
        # ! int rew and lang rew are computed and postprocessed already in sample_rollouts
        storage.compute_returns(
            gamma = gamma,
            gae_lambda = gae_lambda,
            has_exploration_reward = is_int_rew_activated if env_name != "montezuma" else False, # we has separated computation for montezuma
            has_language_reward = is_lang_rew_activated,
        )

        if env_name == "montezuma":
            storage.compute_int_returns_montezuma(
                gamma = gamma,
                gae_lambda = gae_lambda,
            )

        # ! update models (train the model)
        policy_train_stats = algorithm.update(
            storage = storage,
        )
        if is_int_rew_activated:
            if env_name != "montezuma" or (env_name == "montezuma" and int_rew_type_str_for_monte == "deir"):
                int_rew_train_stats = int_rew_update(int_rew_model, storage, ppo_nepoch_algo, ppo_nbatch_algo, env_name)

        # ! reset storage
        storage.reset()
        
        Path(save_int_rew_dir).mkdir(parents=True, exist_ok=True)
        Path(save_policy_dir).mkdir(parents=True, exist_ok=True)

        # do not record epoch 0, because some stats are not accurate
        if epoch == 0:
            continue

        # ! compute score
        successes = rollout_stats["successes"] # shape (episode_size(not_deterministic), TASK_NUM), this is just the stats
        if len(successes) != 0: # if there is no success, we will not update total_successes
            total_successes = np.concatenate([total_successes, successes], axis=0)
        success_rate = 100 * np.mean(total_successes, axis=0)
        score = np.exp(np.mean(np.log(1 + success_rate))) - 1

        # get mean_episodic_return from rollout stats
        if rollout_stats['episode_num'] > 0: # only update if there is at least 1 episode
            mean_episodic_return_ext = rollout_stats['episode_sum_return_ext'] / rollout_stats['episode_num']
            if is_int_rew_activated:
                mean_episodic_return_int = rollout_stats['episode_sum_return_int'] / rollout_stats['episode_num']
                mean_episodic_return_int *= storage.int_rew_coef # scale the intrinsic reward
                mean_int_rew_per_step = rollout_stats['episode_sum_return_int'] / rollout_stats['episode_sum_step'] * storage.int_rew_coef
            else:
                mean_episodic_return_int = 0
            if is_lang_rew_activated:
                mean_episodic_return_lang = rollout_stats['episode_sum_return_lang'] / rollout_stats['episode_num']
            else:
                mean_episodic_return_lang = 0

        if mean_episodic_return_ext > best_mean_episodic_return_ext:
            best_mean_episodic_return_ext = mean_episodic_return_ext
            policy_save_path = os.path.join(save_policy_dir, 'best_mean_ext_rew_checkpoint.pth')
            th.save(policy_model.state_dict(), policy_save_path)
            if is_int_rew_activated:
                int_rew_save_path = os.path.join(save_int_rew_dir, 'best_mean_ext_rew_checkpoint.pth')
                th.save(int_rew_model.state_dict(), int_rew_save_path)

        # Get eval stats
        eval_stats = {
            "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
            "score": score,
            "mean_episodic_return_ext": mean_episodic_return_ext,
            "best_mean_episodic_return_ext": best_mean_episodic_return_ext,
            "mean_episodic_return_int": mean_episodic_return_int,
            "mean_episodic_return_lang": mean_episodic_return_lang,
            "mean_int_rew_per_step": mean_int_rew_per_step,
        }

        # Print stats

        if is_int_rew_activated and "int_rew_train_stats" in locals():
            policy_train_stats['int_rew_dsc_loss'] = int_rew_train_stats['dsc_loss']
        stats_str = f"\nepoch {epoch}, score {round(eval_stats['score'], 3)}, best epi ext return {round(eval_stats['best_mean_episodic_return_ext'], 3)}:\n"

        
        # wandb log
        if env_name == "montezuma" and MONTEZUMA_NSTEP_EPOCH_RATIO > 1:
            # we need to accumulate 4 epochs and log the stats
            if epoch % MONTEZUMA_NSTEP_EPOCH_RATIO == 0:
                for k in accu_policy_train_stats:
                    if isinstance(accu_policy_train_stats[k], dict):
                        accu_policy_train_stats[k] = {kk: vv / MONTEZUMA_NSTEP_EPOCH_RATIO for kk, vv in accu_policy_train_stats[k].items()}
                    else:
                        accu_policy_train_stats[k] /= MONTEZUMA_NSTEP_EPOCH_RATIO

                for k in accu_eval_stats:
                    if isinstance(accu_eval_stats[k], dict):
                        accu_eval_stats[k] = {kk: vv / 4 for kk, vv in accu_eval_stats[k].items()}
                    else:
                        accu_eval_stats[k] /= MONTEZUMA_NSTEP_EPOCH_RATIO

                wandb.log({**accu_policy_train_stats, **accu_eval_stats})
                del accu_policy_train_stats
                del accu_eval_stats
            else:
                # accumulate stats
                if 'accu_policy_train_stats' not in locals():
                    accu_policy_train_stats = policy_train_stats
                    accu_eval_stats = eval_stats
                else:
                    # accumulate policy_train_stats
                    for k in accu_policy_train_stats:
                        if isinstance(accu_policy_train_stats[k], dict):
                            for kk in accu_policy_train_stats[k]:
                                accu_policy_train_stats[k][kk] += policy_train_stats[k][kk]
                        else:
                            accu_policy_train_stats[k] += policy_train_stats[k]
                    # accumulate eval_stats
                    for k in accu_eval_stats:
                        if isinstance(accu_eval_stats[k], dict):
                            for kk in accu_eval_stats[k]:
                                accu_eval_stats[k][kk] += eval_stats[k][kk]
                        else:
                            accu_eval_stats[k] += eval_stats[k]

        else:
            wandb.log({**policy_train_stats, **eval_stats})

        # ! save checkpoint
        # TODO do not save 
        if False:
            if (epoch + 1) % save_freq == 0 or epoch == nepoch - 1:
                policy_save_path = os.path.join(save_policy_dir, f'checkpoint_{epoch + 1}.pth')
                th.save(policy_model.state_dict(), policy_save_path)
                if is_int_rew_activated:
                    int_rew_save_path = os.path.join(save_int_rew_dir, f'checkpoint_{epoch + 1}.pth')
                    th.save(int_rew_model.state_dict(), int_rew_save_path)
    
    
    # postprocessing 
    if reward_machine_lst is not None:
        for i, reward_machine in enumerate(reward_machine_lst):
            reward_machine: Reward_Machine = reward_machine
            reward_machine.save_offset_value(i, tag_lst_str, force_save=True)
    
    # wandb finish
    wandb.finish()

# kedro run --from-nodes "setup_env_node" --to-nodes "train_rl_main_node" --params env=crafter,purpose=policy,train_rl_policy_params.eval_type_tag=H3_1_Compared_With_PPO,reward_machine_params.lang_reward_machine_type=standard,\
# rl_policy_setup_params.is_int_rew_activated=false,rl_policy_setup_params.is_lang_rew_activated=true,\
# lang_rew_model_params.is_markovian=true,lang_rew_model_params.traj_length=2,lang_rew_model_params.has_extra_data_manipulation=false,\
# general.seed=7738