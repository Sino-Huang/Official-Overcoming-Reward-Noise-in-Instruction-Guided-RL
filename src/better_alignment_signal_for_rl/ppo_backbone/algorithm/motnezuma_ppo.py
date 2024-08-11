import numpy as np
import torch.nn as nn
import torch as th
import torch.optim as optim

from better_alignment_signal_for_rl.pipelines.rl_policy_setup.nodes import global_grad_norm_
from better_alignment_signal_for_rl.pipelines.train_rl_policy.sample import postprocess_inputs_montezuma
from better_alignment_signal_for_rl.ppo_backbone.model.ppo import PPOModel
from better_alignment_signal_for_rl.ppo_backbone.model.ppo_rnn import PPORNNModel

from better_alignment_signal_for_rl.ppo_backbone.algorithm.base import BaseAlgorithm
from better_alignment_signal_for_rl.agent_components.storage import RolloutStorage
from torchvision.transforms.functional import resize, rgb_to_grayscale


class MontezumaPPOAlgorithm(BaseAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm implementation, responsible for updating the weight of the model.

    Args:
        model: The PPO model used for training.
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
        model,
        rnd_int_rew_model,
        optimizer, 
        obs_rms,
        update_proportion,
        nstep, 
        nproc,
        ppo_nepoch: int,
        ppo_batch_size: int,
        clip_param: float,
        vf_loss_coef: float,
        ent_coef: float,
        **kwargs,
    ):
        super().__init__(model)

        # PPO params
        self.clip_param = clip_param
        self.ppo_nepoch = ppo_nepoch # repeat the update for nepoch
        self.ppo_batch_size = ppo_batch_size
        self.ppo_nbatch = nstep * nproc // ppo_batch_size # minibatch number for each epoch
        self.vf_loss_coef = vf_loss_coef
        self.ent_coef = ent_coef
        self.update_proportion = update_proportion
        assert ppo_nepoch == 3
        assert update_proportion == 0.25
        assert clip_param == 0.1
        assert vf_loss_coef == 0.5
        assert ent_coef == 0.001
        
        # Optimizer
        self.optimizer = optimizer
        self.obs_rms = obs_rms
        self.forward_mse = nn.MSELoss(reduction='none')

        # int rew model 
        self.rnd_int_rew_model = rnd_int_rew_model
        
    def update(self, storage: RolloutStorage,):
        # Set model to training mode
        self.model.train()

        # Run PPO
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        dsc_loss_epoch = 0 # for intrinsic reward model
        nupdate = 0
            
        if self.rnd_int_rew_model is not None:
            has_int_rew = True
        else:
            has_int_rew = False

        for _ in range(self.ppo_nepoch):
            # Get data loader
            data_loader = storage.get_data_loader(self.ppo_nbatch)
            for batch in data_loader:
                postprocess_inputs_montezuma(batch)
                
                # same for new_obs
                if "new_obs" in batch:
                    new_obs = resize(rgb_to_grayscale(batch['new_obs']), (84, 84), antialias=False).cpu().numpy()
                    new_obs = ((new_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5)
                    new_obs = th.as_tensor(new_obs, device=batch['new_obs'].device, dtype=th.float32)
                    batch['new_obs'] = new_obs
                
         
                # Compute loss
                batch.update({
                    "has_int_rew": has_int_rew,
                })
                losses = self.model.compute_losses(**batch, clip_param=self.clip_param)
                pi_loss = losses["pi_loss"]
                vf_loss = losses["vf_loss"]
                entropy = losses["entropy"]
                
                
                # ! --- RND Intrinsic Reward Model --- ! #
                if self.rnd_int_rew_model is not None: 
                    predict_next_state_feature, target_next_state_feature = self.rnd_int_rew_model(batch['new_obs'])
                    rnd_loss = self.forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                    # Proportion to exp used for updating the predictor
                    mask = th.rand(len(rnd_loss)).to(batch['new_obs'].device)
                    mask = (mask < self.update_proportion).float()
                    rnd_loss = (rnd_loss * mask).sum() / th.max(mask.sum(), th.tensor([1.0]).to(batch['new_obs'].device))
                    
                
                    loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy + rnd_loss # want entropy to be high so as to smooth the policy
                else:
                    loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy

                # Update parameter
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # clip the gradient to prevent exploding gradient
                self.optimizer.step()
                

                # Update stats
                pi_loss_epoch += pi_loss.item()
                vf_loss_epoch += vf_loss.item()
                entropy_epoch += entropy.item()
                if "rnd_loss" in locals():
                    dsc_loss_epoch += rnd_loss.item()
                nupdate += 1
                

        # Compute average stats
        pi_loss_epoch /= nupdate
        vf_loss_epoch /= nupdate
        entropy_epoch /= nupdate
        dsc_loss_epoch /= nupdate

        # Define train stats
        train_stats = {
            "pi_loss": pi_loss_epoch,
            "vf_loss": vf_loss_epoch,
            "entropy": entropy_epoch,
            "int_rew_dsc_loss": dsc_loss_epoch,
        }
        return train_stats