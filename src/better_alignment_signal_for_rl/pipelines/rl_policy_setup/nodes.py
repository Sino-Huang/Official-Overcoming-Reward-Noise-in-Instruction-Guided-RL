"""
This is a boilerplate pipeline 'rl_policy_setup'
generated using Kedro 0.19.3
"""
import os
import sys
import time
from natsort import natsorted
import torch as th
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from better_alignment_signal_for_rl.agent_components.ir_deir_model import DiscriminatorModel
from better_alignment_signal_for_rl.agent_components.deir.algo.common_models.cnns import BatchNormCnnFeaturesExtractor
from better_alignment_signal_for_rl.agent_components.deir.utils.enum_types import NormType
import numpy as np

from better_alignment_signal_for_rl.pipelines.train_lang_rew_model.nodes import setup_lang_rew_model
from better_alignment_signal_for_rl.ppo_backbone.model.base import BaseModel
from better_alignment_signal_for_rl.ppo_backbone.model.montezuma_model import CnnActorCriticNetwork, RNDModel
from better_alignment_signal_for_rl.ppo_backbone.model.ppo import PPOModel
from better_alignment_signal_for_rl.ppo_backbone.model.ppo_ad import PPOADModel
from better_alignment_signal_for_rl.ppo_backbone.model.ppo_rnn import PPORNNModel

from better_alignment_signal_for_rl.pipelines.env_setup.env_wrapper import MINIGRID_TASKS
from torchvision.transforms.functional import resize, rgb_to_grayscale

def INT_REW_MODEL_FEATURES_DIM(env_name):
    if env_name != "montezuma":
        return 128
    else:
        return 512
INT_REW_MODEL_GRU_LAYERS = 1

def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, th.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == th.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm

def lr_lambda(epoch):
    rollout_size = 128
    batch_size = 8 
    total_frame = 512*8*250
    return 1 - min(epoch * rollout_size * batch_size, total_frame) / total_frame

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



def init_int_rew_model(venv, env_name, train_rl_policy_cfg):
    """
    Initializes the intrinsic reward model.

    The intrinsic reward model comes from DEIR work. We will not provide configs in the config file but rather we fix the hyperparameters here.
    DEIR paper: https://arxiv.org/abs/2304.10770
    DEIR code URL: https://github.com/swan-utokyo/deir

    Args:
        venv : The environment.

    Returns:
        DiscriminatorModel: The initialized intrinsic reward model.
    """
    int_rew_type_str_for_monte = train_rl_policy_cfg['montezuma_env_params']['int_rew_type']
    if env_name == "montezuma" and int_rew_type_str_for_monte == "rnd":
        int_rew_model = RNDModel(
            venv.observation_space.shape,
            venv.action_space.n,
        )
    else:
        if env_name == "montezuma":
            model_learning_rate = 1e-3
            model_features_dim = 512
            model_latents_dim = 512
            model_mlp_norm = NormType.NoNorm
            model_cnn_norm = NormType.LayerNorm
        else:
            model_learning_rate = 0.0003
            model_features_dim = 128
            model_latents_dim = 256
            model_mlp_norm = NormType.BatchNorm
            model_cnn_norm = NormType.BatchNorm
            
        int_rew_model_kwargs = {
            "action_space" : venv.action_space,
            "activation_fn": nn.ReLU,
            "gru_layers": INT_REW_MODEL_GRU_LAYERS,
            "max_grad_norm": 0.5,
            "model_cnn_features_extractor_class" : BatchNormCnnFeaturesExtractor,
            "model_cnn_features_extractor_kwargs": {
                "activation_fn": nn.ReLU,
                "features_dim": INT_REW_MODEL_FEATURES_DIM(env_name),
                "model_type": 0,
            },
            "model_cnn_norm": model_cnn_norm,
            'model_features_dim': model_features_dim,
            'model_gru_norm': NormType.NoNorm,
            'model_latents_dim': model_latents_dim,
            'model_learning_rate': model_learning_rate,
            'model_mlp_layers': 1,
            'model_mlp_norm': model_mlp_norm,
            'observation_space': venv.observation_space,
            'optimizer_class': th.optim.Adam,
            'optimizer_kwargs': {
                'betas': (0.9, 0.999),
                'eps': 1e-05,
            },
            'use_model_rnn': 1, # Use GRU
            'use_status_predictor': 0,
            'obs_rng': np.random.default_rng(seed=131),
            'dsc_obs_queue_len': 100000 # The length of the observation queue for the discriminator, looks like a separate buffer from the RolloutStorage buffer
        }
        
        int_rew_model = DiscriminatorModel(**int_rew_model_kwargs)
    return int_rew_model


# the int rew model will go through the following steps
# 1. optimize it using the rollout_data in ppo_rollout_buffer in the training loop # ! should be handled by int_rew_update(int_rew_model, storage), should be similar as the update function in PPOAlgorithm
# 2. get intrinsic rewards in the ppo rollout model, also update the intrinsic reward model obs_queue is obs is novel # ! should be handled inside rollout sampling. check line 218 in ppo_rollout.py in deir project # ! get_intrinsic_rewards is in line 547 in ppo_rollout.py in deir project and 238 in deir.py
# 3. when we init the ppo rollout model, init_obs_queue is needed when we start a new episode (on training start), also we need to setup episodic_obs_emb_history and episodic_trj_emb_history # ! should be handled inside the rollout sampling
# 4. we need to construct clear_on_episode_end function clear_on_episode_end(self, dones, policy_mems, model_mems) # ! should be handled in rollout sampling, should consider the # Update stats section in sample_rollouts
# 5. we need to save a) episodic_obs_emb_history b) episodic_trj_emb_history c) model_mems in our rollout sampling process

# some notes:
# - episodic_obs_emb_history and episodic_trj_emb_history will be updated during get_intrinsic_rewards function
# - model_mems comes from get_intrinsic_rewards function, it will go through the clear_on_episode_end and finally we need to assign it to self._last_model_mems of the ppo rollout model
# - we want the model_mems to be .detach() after each rollout step
# - we need to normalize the int reward when we obtained the reward in each rolloutstep (see ppo_rollout_buffer.compute_intrinsic_rewards() line 714 in ppo_rollout.py in deir project)


def init_rl_policy_model(venv, general_cfg, rl_policy_setup_cfg):
    
    env_name = general_cfg['env_name']
    if env_name == "crafter":
        model_cls_name = rl_policy_setup_cfg['crafter_env_params']['model_cls']
        model_kwargs = rl_policy_setup_cfg['crafter_env_params']['model_kwargs']
    elif env_name in ['minigrid']:
        model_cls_name = rl_policy_setup_cfg['minigrid_env_params']['model_cls']
        model_kwargs = rl_policy_setup_cfg['minigrid_env_params']['model_kwargs']
    elif env_name == "montezuma":
        model = CnnActorCriticNetwork(
            venv.observation_space.shape,
            venv.action_space.n,
        )
        return model 
    else:
        raise ValueError(f"Invalid environment name: {env_name}")
    if env_name == "minigrid":
        model_kwargs['goal_info_dim'] = len(MINIGRID_TASKS)
        
    model_cls = getattr(sys.modules[__name__], model_cls_name)
    model = model_cls(
        observation_space = venv.observation_space,
        action_space = venv.action_space,
        **model_kwargs
    )
    
    return model


def init_lang_rew_model(lang_rew_model_cfg, general_cfg):
    """
    Initializes the language reward model and returns the model along with the model load path.

    Args:
        lang_rew_model_cfg (dict): Configuration parameters for the language reward model.
        general_cfg (dict): General configuration parameters.

    Returns:
        tuple: A tuple containing the initialized language reward model and the model load path.

    Raises:
        FileNotFoundError: If the model checkpoint is not found at the specified load path.
    """
    
    # the function is already implemented in the train_lang_rew_model pipeline
    # in this function we also need to provide the model_load_path for the model 
    lang_rew_model = setup_lang_rew_model(lang_rew_model_cfg, general_cfg)
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
    return lang_rew_model, load_path


# ! NODE
def setup_policy_int_rew_lang_rew_models(venv, general_cfg, lang_rew_model_cfg, rl_policy_setup_cfg, train_rl_policy_cfg):
    
    policy_model = init_rl_policy_model(venv, general_cfg, rl_policy_setup_cfg)
    env_name = general_cfg['env_name']
    int_rew_model = init_int_rew_model(venv, env_name, train_rl_policy_cfg)
    
    lang_rew_model, lang_rew_model_load_path = init_lang_rew_model(lang_rew_model_cfg, general_cfg)
    
    return policy_model, int_rew_model, lang_rew_model, lang_rew_model_load_path
