import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CategoricalActionHead(nn.Module):
    def __init__(
        self,
        insize: int,
        num_actions: int,
        init_scale: float = 0.01,
    ):
        super().__init__()

        # Layer
        self.linear = nn.Linear(insize, num_actions)

        # Initialization
        init.orthogonal_(self.linear.weight, gain=init_scale)
        init.constant_(self.linear.bias, val=0.0)

    def forward(self, x: th.Tensor, beta: th.Tensor=None) -> th.Tensor:
        x = self.linear(x) # shape = (batch_size, num_actions)
        if beta is not None:
            x = x * beta # ! for Human Rationality Level Beta parameter, 0 meaning the reward signal is random, 1 meaning the reward signal is perfect
        logits = F.log_softmax(x, dim=-1)
        return logits

    def log_prob(self, logits: th.Tensor, actions: th.Tensor) -> th.Tensor:
        # logits shape = (batch_size, num_actions)
        # actions shape = (batch_size, 1)
        log_prob = th.gather(logits, dim=-1, index=actions) # shape = (batch_size, 1)
        return log_prob

    def entropy(self, logits: th.Tensor) -> th.Tensor:
        probs = th.exp(logits)
        entropy = -th.sum(probs * logits, dim=-1, keepdim=True) # entropy = sum p log p, where logits = log p
        return entropy # shape = (batch_size, 1)

    def sample(self, logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # logits shape (env_num, action_dim)
        if deterministic:
            actions = th.argmax(logits, dim=-1, keepdim=True) # shape = (batch_size, 1)
        else:
            u = th.rand_like(logits)
            u[u == 1.0] = 0.999 # to avoid log(0)
            gumbels = logits - th.log(-th.log(u)) # The Gumbel distribution is particularly suitable for modeling the distribution of extreme values, so "Gumbel noise" might be used to introduce randomness in a way that mimics the behavior of extreme events or to model the maximum/minimum of a set of variables in a system. that is, it gives all actions a chance to be selected. 
            actions = th.argmax(gumbels, dim=-1, keepdim=True) # shape = (batch_size, 1)
        return actions

    def kl_divergence(self, logits_q: th.Tensor, logits_p: th.Tensor) -> th.Tensor: 
        # this looks like a static method as it does not use self
        kl = th.sum(th.exp(logits_q) * (logits_q - logits_p), dim=-1, keepdim=True)
        return kl
