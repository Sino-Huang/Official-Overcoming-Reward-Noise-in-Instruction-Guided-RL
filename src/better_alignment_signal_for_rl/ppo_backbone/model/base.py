from __future__ import annotations

import abc
from typing import Dict

import torch as th
import torch.nn as nn

from gymnasium import spaces


class BaseModel(nn.Module, abc.ABC):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
    ):
        super().__init__()

        # Observation and action spaces
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def act(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        pass

    @abc.abstractmethod
    def forward(self, obs: th.Tensor) -> Dict[str, th.Tensor]:
        pass

    @abc.abstractmethod
    def encode(self, obs: th.Tensor) -> th.Tensor:
        pass

    @abc.abstractmethod
    def compute_losses(self, **kwargs) -> Dict[str, th.Tensor]:
        pass
