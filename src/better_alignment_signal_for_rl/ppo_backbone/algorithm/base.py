import abc
from typing import Dict

import torch as th

from better_alignment_signal_for_rl.ppo_backbone.model.base import BaseModel
from better_alignment_signal_for_rl.agent_components.storage import RolloutStorage


class BaseAlgorithm(abc.ABC):
    def __init__(self, model: BaseModel):
        self.model = model

    @abc.abstractmethod
    def update(self, storage: RolloutStorage, *args, **kwargs) -> Dict[str, th.Tensor]:
        pass
