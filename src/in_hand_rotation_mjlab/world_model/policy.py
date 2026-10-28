"""Actor and critic for policy learning through the world model (SuperMimic-style: a deterministic policy trained by
backpropagating imagined returns, a privileged value function bootstrapping past the imagination horizon)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
  """Observation normalizer (running mean / variance over everything seen, frozen inside gradient steps)."""

  def __init__(self, dim: int, eps: float = 1e-2):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(1e-4))
    self.eps = eps

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.reshape(-1, x.shape[-1]).float()
    b_mean, b_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    tot = self.count + n
    delta = b_mean - self.mean
    self.mean += delta * n / tot
    self.var = (self.var * self.count + b_var * n + delta.pow(2) * self.count * n / tot) / tot
    self.count = tot

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / torch.sqrt(self.var + self.eps)


def mlp(din: int, hidden: tuple[int, ...], dout: int, act: type[nn.Module], layer_norm: bool = False) -> nn.Sequential:
  layers, d = [], din
  for h in hidden:
    layers += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if layer_norm else []) + [act()]
    d = h
  layers.append(nn.Linear(d, dout))
  return nn.Sequential(*layers)


class Actor(nn.Module):
  """a = tanh(MLP(norm(obs))): bounded in the normalized action range [-1, 1] with a nonzero gradient everywhere."""

  def __init__(self, obs_dim: int, act_dim: int, hidden=(512, 512, 256)):
    super().__init__()
    self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, tuple(hidden)
    self.norm = RunningNorm(obs_dim)
    self.net = mlp(obs_dim, self.hidden, act_dim, nn.ELU)
    nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
    nn.init.zeros_(self.net[-1].bias)

  def forward(self, obs: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
    pre = self.net(self.norm(obs))
    if noise > 0:  # reparameterized Gaussian before the squash: a smoothed objective for the imagined gradient
      pre = pre + noise * torch.randn_like(pre)
    return torch.tanh(pre)

  def checkpoint(self, **extra) -> dict:
    return dict(kind="wm_actor", obs_dim=self.obs_dim, act_dim=self.act_dim, hidden=self.hidden,
                state_dict={k: v.detach().cpu() for k, v in self.state_dict().items()}, **extra)

  @classmethod
  def from_checkpoint(cls, ck: dict) -> "Actor":
    a = cls(ck["obs_dim"], ck["act_dim"], ck["hidden"])
    a.load_state_dict(ck["state_dict"])
    return a


class Critic(nn.Module):
  def __init__(self, in_dim: int, hidden=(256, 128)):
    super().__init__()
    self.norm = RunningNorm(in_dim)
    self.net = mlp(in_dim, tuple(hidden), 1, nn.SiLU, layer_norm=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(self.norm(x)).squeeze(-1)
