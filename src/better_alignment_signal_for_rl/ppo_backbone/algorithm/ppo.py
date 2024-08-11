import torch.nn as nn
import torch as th
import torch.optim as optim
from torchvision.transforms.functional import resize

from better_alignment_signal_for_rl.pipelines.rl_policy_setup.nodes import global_grad_norm_
from better_alignment_signal_for_rl.ppo_backbone.model.ppo import PPOModel
from better_alignment_signal_for_rl.ppo_backbone.model.ppo_rnn import PPORNNModel

from better_alignment_signal_for_rl.ppo_backbone.algorithm.base import BaseAlgorithm
from better_alignment_signal_for_rl.agent_components.storage import RolloutStorage


class PPOAlgorithm(BaseAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm implementation, responsible for updating the weight of the model.

    Args:
        model (PPOModel | PPORNNModel): The PPO model used for training.
        ppo_nepoch (int): Number of times to repeat the update for each epoch within the current RolloutStorage.
        ppo_nbatch (int): Number of minibatches for each epoch.
        clip_param (float): Clipping parameter for PPO loss calculation.
        vf_loss_coef (float): Coefficient for the value function loss.
        ent_coef (float): Coefficient for the entropy loss.
        lr (float): Learning rate for the optimizer.
        max_grad_norm (float): Maximum gradient norm for gradient clipping.

    """    
    def __init__(
        self,
        model: PPOModel | PPORNNModel,
        ppo_nepoch: int,
        ppo_nbatch: int,
        clip_param: float,
        vf_loss_coef: float,
        ent_coef: float,
        lr: float,
        max_grad_norm: float,
        **kwargs,
    ):
        super().__init__(model)
        self.model: PPOModel | PPORNNModel

        # PPO params
        self.clip_param = clip_param
        self.ppo_nepoch = ppo_nepoch # repeat the update for nepoch
        self.ppo_nbatch = ppo_nbatch # minibatch number for each epoch
        self.vf_loss_coef = vf_loss_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def update(self, storage: RolloutStorage, rnd_int_rew_model = None,should_train_int_rew_model = False):
        # Set model to training mode
        self.model.train()

        # Run PPO
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        nupdate = 0
        if should_train_int_rew_model and rnd_int_rew_model is not None:
            rnd_int_rew_model.train()
            int_rew_stats_overall = dict()
            int_rew_stats_overall['dsc_loss'] = [] 
            is_int_rew_model_trained = False
        else:
            int_rew_stats_overall = None
            

        for _ in range(self.ppo_nepoch):
            # Get data loader
            data_loader = storage.get_data_loader(self.ppo_nbatch)
            if should_train_int_rew_model and rnd_int_rew_model is not None:
                int_rew_loss = None  
            for batch in data_loader:
                # Compute loss
                losses = self.model.compute_losses(**batch, clip_param=self.clip_param)
                pi_loss = losses["pi_loss"]
                vf_loss = losses["vf_loss"]
                entropy = losses["entropy"]
                loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy # want entropy to be high so as to smooth the policy

                # Update parameter
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # clip the gradient to prevent exploding gradient
                self.optimizer.step()
                
                if should_train_int_rew_model and rnd_int_rew_model is not None and not is_int_rew_model_trained:
                    # ! update RNDModel too
                    rnd_int_rew_model.optimizer.zero_grad()
                    next_obs_batch = batch['new_obs']
                    # only want some portion of the data to update the predictor
                    rand_indices = th.randperm(len(next_obs_batch))[:int(len(next_obs_batch) * rnd_int_rew_model.update_proportion)]
                    next_obs_batch = next_obs_batch[rand_indices]
                    next_obs_batch = resize(next_obs_batch, (84, 84), antialias=True)
                    predict_next_state_feature, target_next_state_feature = rnd_int_rew_model(next_obs_batch)

                    # Case 1: MASK LOSS for original RND paper
                    # forward_loss = rnd_int_rew_model.forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1) 

                    # # Proportion to exp used for updating the predictor
                    # mask = th.rand(len(forward_loss)).to(next_obs_batch.device)

                    # mask = (mask < rnd_int_rew_model.update_proportion).float()
                    # forward_loss = (forward_loss * mask).sum() / th.max(mask.sum(), th.tensor(1.0).to(next_obs_batch.device))
                    
                    # Case 2: NovelD loss
                    forward_loss = th.norm(predict_next_state_feature - target_next_state_feature.detach(), dim=-1, p=2).mean()
                    if int_rew_loss is None:
                        int_rew_loss = forward_loss
                    else:
                        int_rew_loss += forward_loss
                        

                    

                # Update stats
                pi_loss_epoch += pi_loss.item()
                vf_loss_epoch += vf_loss.item()
                entropy_epoch += entropy.item()
                nupdate += 1
                
            if should_train_int_rew_model and rnd_int_rew_model is not None and not is_int_rew_model_trained:
                # no minibatch update, only update once
                int_rew_loss /= nupdate
                int_rew_loss *= 10.0 # according to NovelD codebase 
                int_rew_loss.backward()
                
                int_rew_stats_overall['dsc_loss'].append(int_rew_loss.item())
                # ! clip the gradient to prevent exploding gradient
                # nn.utils.clip_grad_norm_(rnd_int_rew_model.predictor.parameters(), 40.) # follow NovelD codebase
                global_grad_norm_(rnd_int_rew_model.predictor.parameters())
                rnd_int_rew_model.optimizer.step()
                is_int_rew_model_trained = True # only update once
                
                
        if should_train_int_rew_model and rnd_int_rew_model is not None:
            rnd_int_rew_model.lr_scheduler.step()
                
        # aggregate the stats
        if int_rew_stats_overall is not None:
            int_rew_stats_overall['dsc_loss'] = sum(int_rew_stats_overall['dsc_loss']) / len(int_rew_stats_overall['dsc_loss'])

        # Compute average stats
        pi_loss_epoch /= nupdate
        vf_loss_epoch /= nupdate
        entropy_epoch /= nupdate

        # Define train stats
        train_stats = {
            "pi_loss": pi_loss_epoch,
            "vf_loss": vf_loss_epoch,
            "entropy": entropy_epoch,
        }
        if should_train_int_rew_model and int_rew_stats_overall is not None:
            return train_stats, int_rew_stats_overall
        else:
            return train_stats
