import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch as th
import torch.nn.functional as F


from better_alignment_signal_for_rl.ppo_backbone.model.ppo import PPOModel
from better_alignment_signal_for_rl.ppo_backbone.algorithm.base import BaseAlgorithm
from better_alignment_signal_for_rl.agent_components.storage import RolloutStorage
from better_alignment_signal_for_rl.pipelines.env_setup.env_wrapper import MINIGRID_TASKS

MINIGRID_TASKS_ID_DICT = {task: i for i, task in enumerate(MINIGRID_TASKS)}

class PPOEWCAlgorithm(BaseAlgorithm):
    """Proximal Policy Optimization (PPO) plus Elastic Weights Consolidation (EWC) Strategy algorithm implementation, responsible for updating the weight of the model.
    However, as tested in preliminary experiments, the EWC strategy does not improve the performance of the model.

    Args:
        model (PPOModel): The PPO model used for training.
        ppo_nepoch (int): Number of times to repeat the update for each epoch within the current RolloutStorage.
        ppo_nbatch (int): Number of minibatches for each epoch.
        clip_param (float): Clipping parameter for PPO loss calculation.
        vf_loss_coef (float): Coefficient for the value function loss.
        ent_coef (float): Coefficient for the entropy loss.
        lr (float): Learning rate for the optimizer.
        max_grad_norm (float): Maximum gradient norm for gradient clipping.
        ewc_lambda (float): EWC lambda value.

    """    
    def __init__(
        self,
        model: PPOModel,
        ppo_nepoch: int,
        ppo_nbatch: int,
        clip_param: float,
        vf_loss_coef: float,
        ent_coef: float,
        lr: float,
        max_grad_norm: float,
        ewc_lambda: float,
    ):
        super().__init__(model)
        self.model: PPOModel

        # PPO params
        self.clip_param = clip_param
        self.ppo_nepoch = ppo_nepoch # repeat the update for nepoch
        self.ppo_nbatch = ppo_nbatch # minibatch number for each epoch
        self.vf_loss_coef = vf_loss_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.ewc_lambda = ewc_lambda
        
        self.fisher_dict = {}
        self.optpar_dict = {}

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def on_task_update(self, task_id, batch_size, large_batch):

        self.model.train()
        self.optimizer.zero_grad()
        
        # accumulating gradients
        large_batch_data_size = len(large_batch['obs'])
        for start in range(0, large_batch_data_size, batch_size):
            end = start + batch_size
            batch = {k: v[start:end] for k, v in large_batch.items()}
            # Compute loss
            losses = self.model.compute_losses(**batch, clip_param=self.clip_param)
            pi_loss = losses["pi_loss"]
            vf_loss = losses["vf_loss"]
            entropy = losses["entropy"]
            loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy # want entropy to be high so as to smooth the policy
            loss.backward()

        self.fisher_dict[task_id] = {}
        self.optpar_dict[task_id] = {}

        # gradients accumulated can be used to calculate fisher
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.optpar_dict[task_id][name] = param.data.clone()
                self.fisher_dict[task_id][name] = param.grad.data.clone().pow(2)


    def update(self, storage: RolloutStorage):
        # Set model to training mode
        self.model.train()

        # Run PPO
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        nupdate = 0
        learned_tasks = set()
        
        # Get data loader
        data_loader = storage.get_ewc_data_loader(self.ppo_nbatch, MINIGRID_TASKS_ID_DICT)
        for task_id, batch_size, large_batch in data_loader:
            large_batch_data_size = len(large_batch['obs'])
            for _ in range(self.ppo_nepoch): # ! train each task for ppo_nepoch times is more stable than train all tasks for ppo_nepoch times. This is tested in preliminary experiments. 
                for start in range(0, large_batch_data_size, batch_size):
                    end = start + batch_size
                    batch = {k: v[start:end] for k, v in large_batch.items()}
                    # Update parameter
                    self.optimizer.zero_grad()
                
                    # Compute loss
                    losses = self.model.compute_losses(**batch, clip_param=self.clip_param)
                    pi_loss = losses["pi_loss"]
                    vf_loss = losses["vf_loss"]
                    entropy = losses["entropy"]
                    loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy # want entropy to be high so as to smooth the policy
                    
                    # EWC magic here 
                    for task in learned_tasks:
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                fisher = self.fisher_dict[task][name]
                                optpar = self.optpar_dict[task][name]
                                loss += (fisher * (optpar - param).pow(2)).sum() * self.ewc_lambda
                                
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # clip the gradient to prevent exploding gradient
                    self.optimizer.step()
                    
                    
                    # Update stats
                    pi_loss_epoch += pi_loss.item()
                    vf_loss_epoch += vf_loss.item()
                    entropy_epoch += entropy.item()
                    nupdate += 1

            learned_tasks.add(task_id)
                
            # on_task_update 
            self.on_task_update(task_id, batch_size, large_batch)
                

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

        return train_stats
