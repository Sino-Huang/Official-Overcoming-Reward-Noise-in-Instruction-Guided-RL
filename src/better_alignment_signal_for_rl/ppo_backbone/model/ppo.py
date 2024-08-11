from typing import Dict

import torch as th

from gym import spaces

from better_alignment_signal_for_rl.ppo_backbone.model.base import BaseModel
from better_alignment_signal_for_rl.agent_components.impala_cnn import ImpalaCNN
from better_alignment_signal_for_rl.agent_components.action_head import CategoricalActionHead
from better_alignment_signal_for_rl.agent_components.mse_head import ScaledMSEHead
from better_alignment_signal_for_rl.agent_components.torch_util import FanInInitReLULayer


class PPOModel(BaseModel):
    """PPO model implementation, responsible for defining the neural network architecture and forward pass.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        impala_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
        goal_info_dim: int = None, # support minigrid env extra instr onehot encoding
    ):
        super().__init__(observation_space, action_space)

        # Encoder
        obs_shape = getattr(self.observation_space, "shape")
        self.enc = ImpalaCNN(
            obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs, # this will decide whether to add norm layer
            **impala_kwargs,
        )
        outsize = impala_kwargs["outsize"]
        self.goal_info_dim = goal_info_dim
        if self.goal_info_dim is not None:
            encoding_input_size = outsize + self.goal_info_dim
        else:
            encoding_input_size = outsize
        
        self.linear = FanInInitReLULayer( # ! used for encoding latent only 
            encoding_input_size,
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.hidsize = hidsize

        # Heads
        num_actions = getattr(self.action_space, "n")
        self.pi_head = CategoricalActionHead(
            insize=hidsize,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=hidsize,
            outsize=1,
            **mse_head_kwargs,
        )

    @th.no_grad()
    def act(self, obs: th.Tensor, beta: th.Tensor= None,
                goal_info: th.Tensor = None, is_deterministic:bool = False, **kwargs) -> Dict[str, th.Tensor]:
        # Check training mode
        assert not self.training

        # Pass through model
        # ! goal_info is the onehot encoding of the instruction in minigrid env
        outputs = self.forward(obs=obs, beta=beta, goal_info=goal_info, **kwargs)

        # Sample actions
        pi_logits = outputs["pi_logits"] # shape [env_size, action_dim]
        actions = self.pi_head.sample(pi_logits, deterministic=is_deterministic) # shape [env_size, 1]

        # Compute log probs
        log_probs = self.pi_head.log_prob(pi_logits, actions)

        # Denormalize vpreds
        vpreds = outputs["vpreds"]
        vpreds = self.vf_head.denormalize(vpreds)

        # Update outputs
        outputs.update({"actions": actions, "log_probs": log_probs, "vpreds": vpreds})

        return outputs

    def forward(self, obs: th.Tensor, beta: th.Tensor= None,
                goal_info: th.Tensor = None, **kwargs) -> Dict[str, th.Tensor]:
        # Pass through encoder
        # goal info is the onehot encoding of the instruction in minigrid env
        latents = self.encode(obs, goal_info)
        # Pass through heads
        pi_latents = vf_latents = latents # shape [env_size, hidsize]
        pi_logits = self.pi_head(pi_latents, beta) # ! beta is the Human Rationality Level parameter
        vpreds = self.vf_head(vf_latents)

        # Define outputs
        outputs = {
            "latents": latents,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
        }

        return outputs

    def encode(self, obs: th.Tensor, goal_info: th.Tensor = None) -> th.Tensor:
        
        # Pass through encoder
        x = self.enc(obs)
        if self.goal_info_dim is not None:
            assert goal_info is not None
            # concat x and goal_info
            x = th.cat([x, goal_info], dim=-1)
        x = self.linear(x)

        return x

    def compute_losses( # calculate the loss for the policy and value function
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        log_probs: th.Tensor,
        vtargs: th.Tensor,
        advs: th.Tensor,
        clip_param: float = 0.2,
        beta = None, # Human Rationality Level parameter
        goal_info: th.Tensor = None,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        # Pass through model
        outputs = self.forward(obs, beta=beta, goal_info=goal_info, **kwargs)

        # Compute policy loss
        pi_logits = outputs["pi_logits"]
        new_log_probs = self.pi_head.log_prob(pi_logits, actions)
        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -th.min(advs * ratio, advs * ratio_clipped).mean()

        # Compute entropy
        entropy = self.pi_head.entropy(pi_logits).mean()

        # Compute value loss
        vpreds = outputs["vpreds"]
        vf_loss = self.vf_head.mse_loss(vpreds, vtargs).mean() # vtargs is the target value which is gae + V(s)

        # Define losses
        losses = {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}

        return losses
