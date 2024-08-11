import math
from typing import Dict, Iterator

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from better_alignment_signal_for_rl.agent_components.running_mean_std import RunningMeanStd
from better_alignment_signal_for_rl.pipelines.env_setup.env_wrapper import MINIGRID_TASKS_ONEHOT_ENCODER
from better_alignment_signal_for_rl.pipelines.rl_policy_setup.nodes import INT_REW_MODEL_FEATURES_DIM, INT_REW_MODEL_GRU_LAYERS
from better_alignment_signal_for_rl.ppo_backbone.model.ppo_rnn import PPORNNModel
from better_alignment_signal_for_rl.ppo_backbone.model.montezuma_model import CnnActorCriticNetwork
from torchvision.transforms.functional import resize, rgb_to_grayscale

from gymnasium import spaces
from icecream import ic
from random import shuffle

def normalize_rewards(norm_type, rewards, mean, std, eps=1e-5):
    """
    Normalize the input rewards using a specified normalization method (norm_type).
    [0] No normalization
    [1] Standardization per mini-batch
    [2] Standardization per rollout buffer
    [3] Standardization without subtracting the average reward
    """
    if norm_type <= 0:
        return rewards

    if norm_type == 1:
        # Standardization
        return (rewards - mean) / (std + eps)

    if norm_type == 2:
        raise NotImplementedError("Standardization per rollout buffer is not implemented yet.")
        # Min-max normalization
        min_int_rew = th.min(rewards)
        max_int_rew = th.max(rewards)
        mean_int_rew = (max_int_rew + min_int_rew) / 2
        return (rewards - mean_int_rew) / (max_int_rew - min_int_rew + eps)

    if norm_type == 3:
        # Standardization without subtracting the mean
        return rewards / (std + eps)


class RolloutStorage:
    def __init__(
        self,
        nstep: int,  # rollout length
        nproc: int,  # number of parallel environments
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,  # hidden size for the semantic encoding of the observations
        device: th.device,
        subgoal_num: int,  # number of subgoals
        clip_embeds_size: int, # used for instruction embedding size, 
        env_name: str, # env name
    ):
        # Params
        self.nstep = nstep
        self.nproc = nproc
        self.device = device
        self.env_name = env_name

        # Get obs shape and action dim
        obs_shape = getattr(observation_space, "shape")
        action_shape = (1,)

        # Buffer Tensors Init
        self.obs = th.zeros(
            nstep + 1, nproc, *obs_shape, device=device
        )  # +1 for the next observation (obs, action, next_obs)
        if env_name == "montezuma":
            self.rnn_obs = th.zeros(
                nstep + 1, nproc, 3, *obs_shape, device=device
            )
        self.actions = th.zeros(nstep, nproc, *action_shape, device=device).long()
        self.rewards = th.zeros(
            nstep, nproc, 1, device=device
        )  # ! Extrinsic / Env reward
        self.exploration_rewards = th.zeros(
            nstep, nproc, 1, device=device
        )  # ! Intrinsic reward
        self.language_rewards = th.zeros(
            nstep, nproc, 1, device=device
        )  # ! Language reward
        self.instr_str = np.full(
            shape=(nstep + 1, nproc, 1), fill_value="", dtype="<U100"
        )  # ! Instruction str, we will obtain this data from venv info['cur_goals'], crafter will receive this info from reward machine  
        
        self.reward_machine_instr_str = np.full(
            shape=(nstep + 1, nproc, 1), fill_value="", dtype="<U100"
        ) # ! instr str from reward machine, therefore it does not guarantee to be the same as the instruction str from the environment.
        self.reward_machine_instr_additional_key = np.full(
            shape=(nstep + 1, nproc, 1), fill_value="", dtype="<U100"   
        ) # instruction_with_room_id (montezuma), or instruction with seed (minigrid) or instruction (crafter)

        self.masks = th.ones(
            nstep + 1, nproc, 1, device=device
        )  # masks = 1 means not done, thus the value of the next state is valid
        self.vpreds = th.zeros(
            nstep + 1, nproc, 1, device=device
        )  # value model prediction

        if env_name == "montezuma":
            self.vpreds_int = th.zeros(
                nstep + 1, nproc, 1, device=device
            ) # for intrinsic reward value prediction

        self.log_probs = th.zeros(
            nstep, nproc, 1, device=device
        )  # log probability of the action
        self.returns = th.zeros(
            nstep, nproc, 1, device=device
        )  # returns = rewards + gamma * returns
        if env_name == "montezuma":
            self.returns_int = th.zeros(
                nstep, nproc, 1, device=device
            )

        self.advs = th.zeros(
            nstep, nproc, 1, device=device
        )  # advantage = returns - vpreds
        if env_name == "montezuma":
            self.advs_int = th.zeros(
                nstep, nproc, 1, device=device
            )
        self.successes = th.zeros(
            nstep + 1, nproc, subgoal_num, device=device
        ).long()  # log the success of the subgoals
        self.timesteps = th.zeros(nstep + 1, nproc, 1, device=device).long()
        self.states = th.zeros(
            nstep + 1, nproc, hidsize, device=device
        )  # store the hidden states (the obs within the same subgoal will be in the same state, see achievement distillation paper for more details)

        self.has_exploration_reward = False # whether the intrinsic reward is used

        # Step
        self.step = 0

    def init_policy_memory(self, rnn_layer_num:int, feature_dim:int, device, policy_model):
        if getattr(self, "policy_mems", None) is None: # if the policy memory is not initialized
            self.policy_mems = th.zeros(self.nstep + 1, self.nproc, rnn_layer_num, feature_dim, device=device) # shape [nstep + 1, nproc, rnn_layer_num, feature_dim]
            self.policy_mems[self.step].copy_(policy_model.init_memory(self.nproc, device)) # initialize the hidden state of the GRU
        return self.policy_mems[self.step]

    def init_int_rew_storage(self, **kwargs):
        if not self.has_exploration_reward:
            # INT REW SECTION
            self.has_exploration_reward = True 
            self.int_rew_last_model_mems= th.zeros(self.nstep, self.nproc, INT_REW_MODEL_GRU_LAYERS, INT_REW_MODEL_FEATURES_DIM(self.env_name), device=self.device)
            self.int_rew_stats = RunningMeanStd(momentum=0.9) # 0.9 comes from deir project's default value
            self.is_obs_queue_init = False
            if self.env_name == "minigrid":
                self.int_rew_coef = 2.0e-2
            elif self.env_name == "crafter":
                self.int_rew_coef = 1.0e-2
            elif self.env_name == "montezuma":
                self.int_rew_coef = 1.0 # adjusted according to mean int rew per step 
            else:
                raise ValueError(f"env_name {self.env_name} is not supported")

            if self.env_name == "montezuma":
                self.int_rew_clip = 5.0
            else:
                self.int_rew_clip = -1 # it comes from deir project's default value, meaning not activated

            if self.env_name == "montezuma":
                self.obs_rms = kwargs["obs_rms"]
        return self.int_rew_last_model_mems[self.step]

    def __getitem__(self, key: str) -> th.Tensor:
        return getattr(self, key)

    def get_inputs(self, step: int):
        """Get inputs and we can let the model to predict the action and then continue the rollout, it is not used for training the model, but for the rollout of the environment"""
        inputs = {"obs": self.obs[step], "states": self.states[step]}
        if self.env_name == "montezuma":
            inputs["rnn_obs"] = self.rnn_obs[step]
        # add instr_str
        inputs['instr_str'] = self.instr_str[step]
        if getattr(self, "policy_mems", None) is not None:
            inputs["policy_mem"] = self.policy_mems[step]
            
        inputs['env_instr_additional_key'] = self.reward_machine_instr_additional_key[step]
        return inputs

    def insert(
        self,
        obs: th.Tensor,  # actually it is the next observation
        latents: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        masks: th.Tensor,
        vpreds: th.Tensor,
        log_probs: th.Tensor,
        successes: th.Tensor,
        exploration_rewards: th.Tensor,
        language_rewards: th.Tensor,
        instr_str: np.ndarray,  # instruction embedding, so far only minigrid environment need this data
        last_model_mems: th.Tensor,  # int model memory gru layers, features dim
        policy_mem: th.Tensor, # policy memory
        goal_info: th.Tensor, # onehot encoding of the instruction
        model,  # policy model
        has_exploration_reward: bool,
        has_language_reward: bool,
        **kwargs,
    ):
        """during sample rollout, the RolloutStorage will store the data from the environment"""
        # Get prev successes, timesteps, and states
        prev_successes = self.successes[self.step]
        prev_states = self.states[
            self.step
        ]  # states is the latent representation of the observation
        prev_timesteps = self.timesteps[self.step]

        # Update timesteps
        timesteps = prev_timesteps + 1

        # Update states if new achievement is unlocked (see the achievement distillation paper for more details)
        success_conds = successes != prev_successes
        success_conds = success_conds.any(
            dim=-1, keepdim=True
        )  # shape: [nproc, 1], just check any new achievement is unlocked
        if (
            success_conds.any()
        ):  # if any new achievement is unlocked in any of the parallel environments
            if not isinstance(model, CnnActorCriticNetwork):
                with th.no_grad():
                    if policy_mem is not None:
                        next_latents, _ = model.encode(obs=obs, policy_mem=policy_mem.detach(), goal_info=goal_info) 
                    else:
                        next_latents = model.encode(obs=obs, goal_info=goal_info) 
                states = (
                    next_latents - latents
                )  # states is the difference between the current and the previous latent representation
                states = F.normalize(states, dim=-1)
                states = th.where(
                    success_conds, states, prev_states
                )  # ! it means we only update the states when a new achievement is unlocked, the states of the other parallel environments remain the same
            else:
                states = prev_states
        else:
            states = prev_states

        # Refresh successes, timesteps, and states if done
        done_conds = masks == 0  # if done, mask = 0, mask = 1 means it is within the same episode

        successes = th.where(
            done_conds, 0, successes
        )  # if done, it means we refresh it
        timesteps = th.where(
            done_conds, 0, timesteps
        )  # if done, it means we refresh it
        states = th.where(done_conds, 0, states)

        # Update tensors (returns and advs are computed later)
        self.obs[self.step + 1].copy_(obs)
        if self.env_name == "montezuma":
            rnn_obs = [self.rnn_obs[self.step, :, -2:], obs.unsqueeze(1)]
            rnn_obs = th.cat(rnn_obs, dim=1) # shape [nproc, 3, obs_shape]
            assert rnn_obs.shape == self.rnn_obs[self.step].shape
            self.rnn_obs[self.step + 1].copy_(rnn_obs)

        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)  # ! Extrinsic reward
        if has_exploration_reward:
            self.exploration_rewards[self.step].copy_(exploration_rewards)
            if last_model_mems is not None:
                self.int_rew_last_model_mems[self.step].copy_(last_model_mems)
        if has_language_reward:
            self.language_rewards[self.step].copy_(language_rewards)

        if policy_mem is not None:
            self.policy_mems[self.step + 1].copy_(policy_mem)

        self.masks[self.step + 1].copy_(masks)
        self.vpreds[self.step].copy_(vpreds)
        if self.env_name == "montezuma":
            self.vpreds_int[self.step].copy_(kwargs["vpreds_int"])
        self.log_probs[self.step].copy_(log_probs)
        self.successes[self.step + 1].copy_(successes)
        self.timesteps[self.step + 1].copy_(timesteps)
        self.states[self.step + 1].copy_(states)

        if instr_str is not None:
            self.instr_str[self.step + 1] = instr_str # we want the shape of instr_str be [nproc, 1]
            
        if "reward_machine_instr_str" in kwargs:
            self.reward_machine_instr_str[self.step + 1] = kwargs["reward_machine_instr_str"]
            
        if 'reward_machine_instr_additional_key' in kwargs:
            self.reward_machine_instr_additional_key[self.step + 1] = kwargs["reward_machine_instr_additional_key"]

        # Update step, if the buffer is full, the oldest data will be replaced
        self.step = (self.step + 1) % self.nstep

    def reset(self):
        """
        Resets the storage by copying the last observation, masks, successes, timesteps, and states.
        """
        # Reset tensors by copying the last observation
        self.obs[0].copy_(self.obs[-1])
        if self.env_name == "montezuma":
            self.rnn_obs[0].copy_(self.rnn_obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.successes[0].copy_(self.successes[-1])
        self.timesteps[0].copy_(self.timesteps[-1])
        self.states[0].copy_(self.states[-1])
        self.instr_str[0] = self.instr_str[-1]
        self.reward_machine_instr_str[0] = self.reward_machine_instr_str[-1]
        self.reward_machine_instr_additional_key[0] = self.reward_machine_instr_additional_key[-1]
        # reset policy memory
        if getattr(self, "policy_mems", None) is not None:
            self.policy_mems[0].copy_(self.policy_mems[-1])

        # Reset step
        self.step = 0

    def compute_returns(
        self,
        gamma: float,
        gae_lambda: float,
        has_exploration_reward: bool,
        has_language_reward: bool,
    ):
        # ref: https://nn.labml.ai/rl/ppo/gae.html
        # Compute returns
        gae = th.zeros_like(self.rewards[0], device=self.device, dtype=th.float32)
        for step in reversed(range(self.rewards.shape[0])):
            cur_rew = self.rewards[step]
            # set the max external reward to 3
            cur_rew = th.clamp(cur_rew, -3.0, 3.0)  # clip the external reward # this is just for montezuma, for other envs, the reward is clipped in the env
            if has_exploration_reward:
                cur_rew += self.exploration_rewards[step] # they have been normalized in the rollout sampling function
            if has_language_reward:
                cur_rew += self.language_rewards[step]

            delta = (  # temporal difference error (TD error) is calculated
                cur_rew
                + gamma
                * self.vpreds[step + 1]
                * self.masks[
                    step + 1
                ]  # masks = 1 means not done, thus the value of the next state is valid
                - self.vpreds[step]
            )  # delta = (cur_reward) + future_value_estimate - cur_value_estimate (bootstrapping)
            gae = (
                delta + gamma * gae_lambda * self.masks[step + 1] * gae
            )  # gae = delta + gamma * lambda * next_gae
            self.returns[step] = (
                gae + self.vpreds[step]
            )  # this is the target value for the value model, V_target = A_t + V_{old}(s_t)
            self.advs[step] = gae

        # Compute advantages
        if self.env_name != "montezuma": # for montezuma, we do not normalize the advantages, ref: https://github.com/swan-utokyo/deir/issues/3
            # ! we may avoid normalizing the advantages for montezuma
            self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)

    def compute_int_returns_montezuma(
        self,
        gamma: float,
        gae_lambda: float,
    ):
        assert math.isclose(gamma, 0.99, abs_tol=1e-5)
        assert math.isclose(gae_lambda, 0.95, abs_tol=1e-5)

        gae = th.zeros_like(self.rewards[0], device=self.device, dtype=th.float32)
        for t in reversed(range(self.rewards.shape[0])):

            delta = self.exploration_rewards[t] + gamma * self.vpreds_int[t + 1] * self.masks[t + 1] - self.vpreds_int[t]
            gae = delta + gamma * gae_lambda * self.masks[t + 1] * gae

            self.returns_int[t] = gae + self.vpreds_int[t]

        # For Actor
        self.advs_int = self.returns_int - self.vpreds_int[:-1]

    def get_data_loader(self, nbatch: int) -> Iterator[Dict[str, th.Tensor]]:
        """Different from get_inputs, this function is used for training the model.
        During updating the model, we need to create a data loader from the RolloutStorage, yield the data in mini-batches"""

        # if PPO RNN, we need to provide last mem
        # consider minigrid env extra instr onehot encoding
        # Get sampler
        ndata = self.nstep * self.nproc
        assert ndata >= nbatch # nbatch is the PPO minibatch size default: 8, so the data loader will yield 8 mini-batches, and do the gradient update 8 times
        batch_size = ndata // nbatch
        sampler = SubsetRandomSampler(range(ndata)) # going to extract data with size self.nstep * self.nproc,
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        # Sample batch
        # obs and states have nstep+1 length, thus we need to remove the last observation and state
        obs = self.obs[:-1].view(-1, *self.obs.shape[2:]) # *self.obs.shape[2:] is the image shape [3, 64, 64], this is flattening the batch and mini-batch dimensions
        if self.env_name == "montezuma":
            rnn_obs = self.rnn_obs[:-1].view(-1, *self.rnn_obs.shape[2:])
        if getattr(self, "policy_mems", None) is not None:
            policy_mem = self.policy_mems[:-1].view(-1, *self.policy_mems.shape[2:]) # shape [nstep * nproc, gru_layers, hidsize]

        states = self.states[:-1].view(-1, *self.states.shape[2:])
        actions = self.actions.view(-1, *self.actions.shape[2:])
        vtargs = self.returns.view(-1, *self.returns.shape[2:])
        if self.has_exploration_reward and self.env_name == "montezuma":
            v_int_targs = self.returns_int.view(-1, *self.returns_int.shape[2:])

        log_probs = self.log_probs.view(-1, *self.log_probs.shape[2:])
        advs = self.advs.view(-1, *self.advs.shape[2:])
        if self.has_exploration_reward and self.env_name == "montezuma":
            advs_int = self.advs_int.view(-1, *self.advs_int.shape[2:])

        if self.env_name == "minigrid":
            instr_str = self.instr_str[:-1].reshape(-1, 1) # shape [nstep * nproc, 1]
        if self.has_exploration_reward:
            new_obs = self.obs[1:].view(-1, *self.obs.shape[2:]) # shape torch.Size([512, 3, 64, 64])
            if self.env_name == "montezuma":
                # meanwhile, update the obs_rms
                self.obs_rms.update(resize(rgb_to_grayscale(obs), (84, 84), antialias=False).cpu().numpy())
            int_rew_last_model_mems = self.int_rew_last_model_mems.view(-1, *self.int_rew_last_model_mems.shape[2:])
            dones = self.masks[1:].view(-1, *self.masks.shape[2:]).long() ^ 1 # 1 means not done, thus we need to flip the value

        for indices in sampler: # sampler will shuffle the order of the observations
            # len(indices) == batch_size (default 512), the sampler fuse data from different processes
            if self.env_name == "minigrid":
                goal_info = MINIGRID_TASKS_ONEHOT_ENCODER.transform(instr_str[indices]) # shape [batch_size, dim], dim = 55
                goal_info = th.from_numpy(goal_info).float().to(self.device) # shape [batch_size, dim]

            else:
                goal_info = None
            batch = {
                "obs": obs[indices], # shape [batch_size, *obs_shape]
                "states": states[indices],
                "actions": actions[indices],
                "vtargs": vtargs[indices], # same as the returns
                "log_probs": log_probs[indices],
                "advs": advs[indices], # shape [batch_size, 1]
                "goal_info": goal_info,
                "rnn_obs": rnn_obs[indices] if self.env_name == "montezuma" else None,
                
            }

            if self.has_exploration_reward and self.env_name == "montezuma":
                batch.update({
                    "v_int_targs": v_int_targs[indices],
                    "advs_int": advs_int[indices],
                    "total_adv": advs[indices] + advs_int[indices],
                })

            if getattr(self, "policy_mems", None) is not None:
                batch["policy_mem"] = policy_mem[indices]

            # the int rew model require the following data
            if self.has_exploration_reward:
                extra_data = {
                    "new_obs": new_obs[indices], # shape torch.Size([512, 3, 64, 64])
                    "last_model_mems": int_rew_last_model_mems[indices], # shape torch.Size([512, 1, 128])
                    "episode_dones": dones[indices], # shape torch.Size([512, 1])
                }
                batch.update(extra_data)

            # if minigrid, give instr_str
            if self.env_name == "minigrid":
                batch["instr_str"] = instr_str[indices]

            yield batch

    def get_ewc_data_loader(self, nbatch:int, task_id_dict: dict) -> Iterator[Dict[str, th.Tensor]]:
        """for each batch, yield all the data from the same task"""
        # obs and states have nstep+1 length, thus we need to remove the last observation and state
        obs = self.obs[:-1].view(-1, *self.obs.shape[2:]) # *self.obs.shape[2:] is the image shape [3, 64, 64], this is flattening the batch and mini-batch dimensions
        states = self.states[:-1].view(-1, *self.states.shape[2:])
        actions = self.actions.view(-1, *self.actions.shape[2:])
        vtargs = self.returns.view(-1, *self.returns.shape[2:])
        log_probs = self.log_probs.view(-1, *self.log_probs.shape[2:])
        advs = self.advs.view(-1, *self.advs.shape[2:])
        if self.env_name == "minigrid":
            instr_str = self.instr_str[:-1].reshape(-1, 1) # shape [nstep * nproc, 1]
        if self.has_exploration_reward:
            new_obs = self.obs[1:].view(-1, *self.obs.shape[2:])
            int_rew_last_model_mems = self.int_rew_last_model_mems.view(-1, *self.int_rew_last_model_mems.shape[2:])
            dones = self.masks[1:].view(-1, *self.masks.shape[2:]).long() ^ 1 # 1 means not done, thus we need to flip the value
        task_ids = [task_id_dict[task] for task in instr_str.reshape(-1)]    
        # assign the index to each task group
        task_indices = {} # key: task_id, value: list of indices
        for i, task_id in enumerate(task_ids):
            if task_id not in task_indices:
                task_indices[task_id] = []
            task_indices[task_id].append(i)

        # shuffle task_indices
        task_indices_keys = list(task_indices.keys())
        shuffle(task_indices_keys)
        task_indices = {key: task_indices[key] for key in task_indices_keys}

        # Get sampler
        ndata = self.nstep * self.nproc
        assert ndata >= nbatch # nbatch is the PPO minibatch size default: 8, so the data loader will yield 8 mini-batches, and do the gradient update 8 times
        batch_size = ndata // nbatch # we need to divide the batch size by the number of tasks

        for task_id, indices in task_indices.items():
            # shuffle indices
            shuffle(indices)

            if self.env_name == "minigrid":
                goal_info = MINIGRID_TASKS_ONEHOT_ENCODER.transform(instr_str[indices]) # shape [batch_size, dim], dim = 55
                goal_info = th.from_numpy(goal_info).float().to(self.device) # shape [batch_size, dim]

            else:
                goal_info = None
            batch = {
                "obs": obs[indices], # shape [batch_size, *obs_shape]
                "states": states[indices],
                "actions": actions[indices],
                "vtargs": vtargs[indices], # same as the returns
                "log_probs": log_probs[indices],
                "advs": advs[indices], # shape [batch_size, 1]
                "goal_info": goal_info,
                
            }
            # the int rew model require the following data
            if self.has_exploration_reward:
                extra_data = {
                    "new_obs": new_obs[indices], # shape torch.Size([512, 3, 64, 64])
                    "last_model_mems": int_rew_last_model_mems[indices], # shape torch.Size([512, 1, 128])
                    "episode_dones": dones[indices], # shape torch.Size([512, 1])
                }
                batch.update(extra_data)

            # if minigrid, give instr_str
            if self.env_name == "minigrid":
                batch["instr_str"] = instr_str[indices]

            yield task_id, batch_size, batch 

    # ! handling postprocessing of intrinsic reward
    def compute_intrinsic_rewards(self):
        # check line 93 in ppo_buffer.py in deir project
        self.int_rew_stats.update(self.exploration_rewards.cpu().numpy().reshape(-1))
        int_rew_mean = th.tensor(self.int_rew_stats.mean, device=self.device, dtype=th.float32)
        int_rew_std = th.tensor(self.int_rew_stats.std, device=self.device, dtype=th.float32)

        self.exploration_rewards = normalize_rewards(
            norm_type=1, # 1 means standardization 
            rewards=self.exploration_rewards,
            mean=int_rew_mean,
            std=int_rew_std,
            eps=1.0e-5, # 1.0e-5 comes from deir project's default value
        )

        # Rescale by IR coef
        self.exploration_rewards *= self.int_rew_coef

        # Clip after normalization
        if self.int_rew_clip > 0:
            self.exploration_rewards = th.clamp(self.exploration_rewards, -self.int_rew_clip, self.int_rew_clip)

    def compute_intrinsic_rewards_montezuma(self, reward_rms, discounted_reward):
        total_int_reward = self.exploration_rewards.cpu().numpy().squeeze(-1) # shape [nstep, nproc]
        total_int_reward = total_int_reward.transpose() # shape [nproc, nstep]
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in total_int_reward.T]) 
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count)

        total_int_reward /= np.sqrt(reward_rms.var) # shape [nproc, nstep]
        # put it back to the original shape
        total_int_reward = total_int_reward.transpose() # shape [nstep, nproc]
        self.exploration_rewards = th.tensor(total_int_reward, device=self.device, dtype=th.float32).unsqueeze(-1) # shape [nstep, nproc, 1]

        self.exploration_rewards *= self.int_rew_coef
        assert math.isclose(self.int_rew_coef, 1.0, abs_tol=1e-5)

    def compute_language_rewards(self, lang_rew_coef):
        self.language_rewards = th.clamp(self.language_rewards, min=0.0)
        self.language_rewards *= lang_rew_coef # multiply the language reward coefficient