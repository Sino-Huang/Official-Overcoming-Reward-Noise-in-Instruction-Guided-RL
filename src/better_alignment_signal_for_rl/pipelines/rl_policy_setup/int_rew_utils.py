from typing import Optional
import numpy as np
from better_alignment_signal_for_rl.agent_components.storage import RolloutStorage
from better_alignment_signal_for_rl.agent_components.ir_deir_model import DiscriminatorModel
import torch as th

from better_alignment_signal_for_rl.ppo_backbone.model.montezuma_model import RNDModel
from torchvision.transforms.functional import resize, rgb_to_grayscale

def int_rew_update(int_rew_model:RNDModel| DiscriminatorModel, storage: RolloutStorage, ppo_nepoch_algo: int, ppo_nbatch_algo: int, env_name: str):
    # ! this should follow the PPOAlgorithm update function line 48 in ppo.py in algorithm module 
    
    # Set model to training mode 
    int_rew_model.train()
    int_rew_stats_overall = dict()
    for _ in range(ppo_nepoch_algo):
        data_loader = storage.get_data_loader(ppo_nbatch_algo)
        
        for batch in data_loader:
            int_rew_stats = int_rew_model.optimize(batch, env_name)
            for key, value in int_rew_stats.items():
                if key not in int_rew_stats_overall:
                    int_rew_stats_overall[key] = []
                int_rew_stats_overall[key].append(value)
    
    # return int_rew_stats
    # aggregate the stats
    for key in int_rew_stats_overall:
        if int_rew_stats_overall[key][0] is None:
            int_rew_stats_overall[key] = None
        else:
            if key in ['n_valid_samples', 'n_valid_pos_samples', 'n_valid_neg_samples']:
                int_rew_stats_overall[key] = sum(int_rew_stats_overall[key])
            else:
                int_rew_stats_overall[key] = (sum(int_rew_stats_overall[key]) / len(int_rew_stats_overall[key])).cpu().item()
    
    return int_rew_stats_overall


# ! handle intrinsic reward generation
def create_intrinsic_rewards_deir(last_obs, new_obs, last_model_mems,
                             episodic_obs_emb_history,
                             episodic_trj_emb_history,
                             int_rew_model: DiscriminatorModel,
                             int_rew_stats_mean,
                             is_obs_queue_init,
                             ):
    # check line 514 in ppo_rollout.py in deir project
    # Prepare input tensors for IR generation

    intrinsic_rewards, model_mems = int_rew_model.get_intrinsic_rewards(
        curr_obs = last_obs,
        next_obs = new_obs,
        last_mems = last_model_mems,
        obs_history = episodic_obs_emb_history,
        trj_history = episodic_trj_emb_history,
        plain_dsc = False,
    )

    int_rew_model.update_obs_queue(
        is_obs_queue_init=is_obs_queue_init,
        intrinsic_rewards=intrinsic_rewards,
        ir_mean=int_rew_stats_mean,
        new_obs=new_obs.cpu().numpy(),
    )
    # intrinsic_rewards needs to be reshaped to [env_size, 1] so as to save to storage
    intrinsic_rewards : np.ndarray = intrinsic_rewards 
    intrinsic_rewards = intrinsic_rewards.reshape(-1, 1)

    return intrinsic_rewards, model_mems # model_mems will update last_model_mems


def create_intrinsic_rewards_rnd(
    last_obs,
    new_obs,
    int_rew_model: RNDModel,
    is_state_visited,  # shape [env_size, ] np.ndarray bool
    mode: str,
    obs_rms,
):
    with th.no_grad():
        if mode == "noveld":
            device = new_obs.device
            new_obs = resize(rgb_to_grayscale(new_obs), (84, 84), antialias=False).cpu().numpy()
            new_obs = ((new_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
            new_obs = th.as_tensor(new_obs, device=device, dtype=th.float32)
            
            last_obs = last_obs.cpu().numpy()
            last_obs = ((last_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
            last_obs = th.as_tensor(last_obs, device=device, dtype=th.float32)

            target_next_feature = int_rew_model.target(new_obs)
            predict_next_feature = int_rew_model.predictor(new_obs)

            target_last_feature = int_rew_model.target(last_obs)
            predict_last_feature = int_rew_model.predictor(last_obs)

            int_rew_next = th.norm(predict_next_feature - target_next_feature, dim=-1, p=2)
            int_rew_last = th.norm(predict_last_feature - target_last_feature, dim=-1, p=2)

            # NovelD measures the difference 
            intrinsic_reward = th.clamp(int_rew_next - 0.5 * int_rew_last, min=0).cpu().numpy().reshape(-1, 1) # reshape to [env_size, 1]
            is_state_visited_coef = np.array([1.0 if state_visited else 0 for state_visited in is_state_visited], dtype=np.float32).reshape(-1, 1)
        
            assert intrinsic_reward.shape == is_state_visited_coef.shape
            intrinsic_reward = intrinsic_reward * is_state_visited_coef
        elif mode =="rnd":
            # alternative RND is just the difference between the two features
            device = new_obs.device
            new_obs = resize(rgb_to_grayscale(new_obs), (84, 84), antialias=False).cpu().numpy()
            new_obs = ((new_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)
            new_obs = th.as_tensor(new_obs, device=device, dtype=th.float32)
            target_next_feature = int_rew_model.target(new_obs)
            predict_next_feature = int_rew_model.predictor(new_obs)
            intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
            intrinsic_reward = intrinsic_reward.cpu().numpy().reshape(-1, 1)
        else: 
            raise ValueError(f"Unknown mode {mode}")
        
        
        return intrinsic_reward
