"""GPU ring buffer of collected env steps for online world-model fitting and imagination starts.

Row (global step g, env e) holds the transition of that step: state s, the true next state (the terminal state when
the env was reset by the step), executed action, done (1 terminated, 2 time-out), episode id, the episode's context,
the newest frame of the policy observation the action was taken on, the episode info (task-specific: reset joints,
anchor pose; ``info_ok`` once known), the env's per-term reward rates, the episode step of s.
"""

from __future__ import annotations

import torch


class Replay:
  def __init__(self, capacity: int, n_envs: int, d_state: int, d_act: int, d_ctx: int, d_frame: int, d_info: int,
               n_rew: int, device: str, d_plant: int = 0):
    L, N = capacity, n_envs
    f = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
    self.L, self.N, self.device = L, N, device
    self.s, self.s_next = f(L, N, d_state), f(L, N, d_state)
    self.a = f(L, N, d_act)
    self.ctx = f(L, N, d_ctx)
    self.frame = f(L, N, d_frame)
    self.info = f(L, N, d_info)
    self.rew = f(L, N, n_rew)
    self.plant = f(L, N, max(d_plant, 1))
    self.done = torch.zeros(L, N, dtype=torch.uint8, device=device)
    self.info_ok = torch.zeros(L, N, dtype=torch.bool, device=device)
    self.epi = torch.full((L, N), -1, dtype=torch.long, device=device)
    self.ep_step = torch.zeros(L, N, dtype=torch.long, device=device)
    self.G = 0  # global steps written

  def add(self, **row) -> None:
    i = self.G % self.L
    for k, v in row.items():
      getattr(self, k)[i] = v
    self.G += 1

  def _rows(self, g: torch.Tensor) -> torch.Tensor:
    return g % self.L

  def sample_windows(self, B: int, k: int):
    """(g, e) of B windows of k steps inside one episode (the last step may end it)."""
    lo, hi = max(0, self.G - self.L), self.G - k
    assert hi > lo, "not enough data"
    out_g, out_e, n = [], [], 0
    while n < B:
      g = torch.randint(lo, hi + 1, (2 * B,), device=self.device)
      e = torch.randint(0, self.N, (2 * B,), device=self.device)
      j = torch.arange(k, device=self.device)[:, None]
      ep = self.epi[self._rows(g[None] + j), e[None]]
      ok = (ep == ep[:1]).all(0) & (ep[0] >= 0)
      out_g.append(g[ok]); out_e.append(e[ok]); n += int(ok.sum())
    return torch.cat(out_g)[:B], torch.cat(out_e)[:B]

  def window(self, g, e, k):
    j = torch.arange(k, device=self.device)[:, None]
    r = self._rows(g[None] + j)
    return self.s[r[0], e], self.a[r, e], self.s_next[r, e], self.ctx[r[0], e], self.done[r, e]

  def sample_starts(self, B: int, history: int):
    """(g, e) of B imagination starts whose episode has ``history`` earlier steps in the buffer and known info."""
    lo, hi = max(0, self.G - self.L) + history, self.G - 1
    out_g, out_e, n = [], [], 0
    tries = 0
    while n < B and tries < 50:
      g = torch.randint(lo, hi + 1, (2 * B,), device=self.device)
      e = torch.randint(0, self.N, (2 * B,), device=self.device)
      r = self._rows(g)
      ok = (self.ep_step[r, e] >= history) & self.info_ok[r, e]
      out_g.append(g[ok]); out_e.append(e[ok]); n += int(ok.sum()); tries += 1
    g, e = torch.cat(out_g)[:B], torch.cat(out_e)[:B]
    return g, e

  def history(self, field: str, g, e, n: int):
    """(B, n, ...) rows g-n+1 .. g of ``field`` (oldest -> newest)."""
    j = torch.arange(-n + 1, 1, device=self.device)[None]
    return getattr(self, field)[self._rows(g[:, None] + j), e[:, None]]
