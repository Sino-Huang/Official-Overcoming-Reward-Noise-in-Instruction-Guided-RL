from typing import Dict
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
import torch as th
from torch.distributions.categorical import Categorical


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(CnnActorCriticNetwork, self).__init__()

        if use_noisy_net:
            print('Use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            linear(7 * 7 * 64, 256),
            nn.ReLU(),
            linear(256, 448),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            linear(448, 448),
            nn.ReLU(),
            linear(448, output_size)
        )

        self.extra_layer = nn.Sequential(
            linear(448, 448),
            nn.ReLU()
        )

        self.critic_ext = linear(448, 1)
        self.critic_int = linear(448, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()

        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer)):
            if type(self.extra_layer[i]) == nn.Linear:
                init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()

    def forward(
        self,
        obs: th.Tensor,
        beta: th.Tensor = None,
        **kwargs
    ):
        x = self.feature(obs)
        action_scores = self.actor(x)
        if beta is not None:
            action_scores = action_scores * beta
        action_probs = F.softmax(action_scores, dim=1)
        value_ext = self.critic_ext(self.extra_layer(x) + x)
        value_int = self.critic_int(self.extra_layer(x) + x)
        outputs = {
            "latents": x,
            "pi_latents": x,
            "vf_latents": x,
            "pi_logits": action_probs,
            "vpreds": value_ext,
            "vpreds_int": value_int,
        }
        return outputs

    @th.no_grad()
    def act(
        self,
        obs: th.Tensor,
        beta: th.Tensor = None,
        is_deterministic: bool = False,
        **kwargs
    ) -> Dict[str, th.Tensor]:
        outputs = self.forward(obs, beta, **kwargs)
        action_dist = Categorical(outputs['pi_logits'])
        
        if is_deterministic:
            action = th.argmax(outputs['pi_logits'], dim=-1, keepdim=True) # shape = (batch_size, 1)
        else:
            action = action_dist.sample().unsqueeze(-1)
            
        outputs.update(
            {
                "actions": action,
                "log_probs": action_dist.log_prob(action.squeeze(-1)).unsqueeze(-1), # shape (nproc, 1)
            }
        )
        assert outputs['log_probs'].shape == (obs.shape[0], 1)
        return outputs
    
    
    def compute_losses( # calculate the loss for the policy and value function
        self,
        has_int_rew: bool,
        obs: th.Tensor,
        actions: th.Tensor,
        log_probs: th.Tensor,
        vtargs: th.Tensor, # shape (batch, 1)
        advs: th.Tensor,
        clip_param: float = 0.1,
        beta = None, # Human Rationality Level parameter
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        if has_int_rew:
            total_adv = kwargs['total_adv']
            v_int_targs = kwargs['v_int_targs']
            advs_int = kwargs['advs_int']
        # Pass through model
        outputs = self.forward(obs, beta=beta, **kwargs)
        action_dist = Categorical(outputs['pi_logits'])
        new_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) # shape (batch, 1)
        
        ratio = th.exp(new_log_probs - log_probs) # shape (batch, 1)
        if has_int_rew:
            adv_c = total_adv
        else:
            adv_c = advs # shape (batch, 1)
        surr1 = ratio * adv_c
        surr2 = th.clamp(
            ratio,
            1.0 - clip_param,
            1.0 + clip_param
        ) * adv_c
        
        pi_loss = -th.min(surr1, surr2).mean()
        if has_int_rew:
            critic_loss = F.mse_loss(outputs["vpreds"].sum(1), vtargs.squeeze()) + F.mse_loss(outputs["vpreds_int"].sum(1), v_int_targs.squeeze())
        else:
            critic_loss = F.mse_loss(outputs["vpreds"].sum(1), vtargs.squeeze())
            
        entropy = action_dist.entropy().mean()
        
        # Define losses
        losses = {"pi_loss": pi_loss, "vf_loss": critic_loss, "entropy": entropy}

        return losses


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        # Initialize weights    
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        # Set target parameters as untrainable
        for param in self.target.parameters():
            param.requires_grad = False
    
        

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature

