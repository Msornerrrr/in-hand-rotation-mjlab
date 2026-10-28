"""Deterministic hand + object world model (SuperMimic-style: an MLP fit to multi-step open-loop rollouts).

One control step: (s_t, cmd_{t+1}, ctx) -> s_{t+1}. ``cmd_{t+1}`` is the joint target in effect during the step; it is
not learned but computed from the action by ``CommandMap`` (the delta-position action term: integrate the delta on the
previous target, clamp the lead around the measured joints, clip to the soft limits), so imagined rollouts produce it
the way the env does. The network predicts normalized increments that are integrated outside it (as the reproductions
do, so positions stay consistent with velocities):

  qd' = qd + s_qd * o_qd        q' = q + dt qd' + s_qc * o_qc
  v'  = v  + s_v  * o_v         p' = p + dt v'  + s_pc * o_pc
  w'  = w  + s_w  * o_w         quat' = exp(dt w' + s_rc * o_rc) * quat     (w in the hand frame)

The last layer is zero-initialized: before training the model extrapolates at constant velocity. A separate small
head predicts the probability that the step ends the episode by termination (a drop), from the predicted state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .record import state_slices


# --- rotations (wxyz) ------------------------------------------------------------------------------------------------

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], -1)


def quat_exp(r: torch.Tensor) -> torch.Tensor:
  """Rotation vector -> unit quaternion; smooth at 0 (Taylor branch)."""
  th2 = (r * r).sum(-1, keepdim=True)
  th = torch.sqrt(th2.clamp_min(1e-12))
  small = th2 < 1e-8
  half_sinc = torch.where(small, 0.5 - th2 / 48.0, torch.sin(0.5 * th) / th)
  w = torch.where(small, 1.0 - th2 / 8.0, torch.cos(0.5 * th))
  return torch.cat([w, r * half_sinc], -1)


def quat_log(q: torch.Tensor) -> torch.Tensor:
  """Unit quaternion -> rotation vector (shortest), with a finite gradient at the identity."""
  q = torch.where(q[..., :1] < 0, -q, q)
  v = q[..., 1:]
  s = torch.sqrt((v * v).sum(-1, keepdim=True).clamp_min(1e-16))
  ang = 2.0 * torch.atan2(s, q[..., :1])
  scale = torch.where(s < 1e-6, 2.0 / q[..., :1].clamp_min(1e-6), ang / s)
  return v * scale


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  c0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
  c1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
  return torch.cat([c0, c1], -1)


def geodesic(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Angle (rad) between two unit quaternions."""
  d = (q1 * q2).sum(-1).abs().clamp(max=1.0)
  return 2.0 * torch.acos(d)


def twist_z(q_rel: torch.Tensor) -> torch.Tensor:
  """Signed rotation angle about the hand z axis of a relative rotation (swing-twist decomposition)."""
  return 2.0 * torch.atan2(q_rel[..., 3], q_rel[..., 0])


# --- the command the action term holds -------------------------------------------------------------------------------

class CommandMap(nn.Module):
  """cmd_{t+1} = clip_limits(q + clamp(cmd_t + delta(a_t) - q, +-lead)), as ``JointPositionDeltaAction``."""

  def __init__(self, meta: dict):
    super().__init__()
    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    self.register_buffer("dmin", t(meta["delta_min"]))
    self.register_buffer("dmax", t(meta["delta_max"]))
    lead = meta.get("max_lead")
    self.register_buffer("lead", t(lead) if lead is not None else torch.full_like(t(meta["delta_min"]), math.inf))
    lim = t(meta["soft_limits"])
    self.register_buffer("lo", lim[:, 0] if meta.get("clip_to_joint_limits", True) else torch.full_like(lim[:, 0], -math.inf))
    self.register_buffer("hi", lim[:, 1] if meta.get("clip_to_joint_limits", True) else torch.full_like(lim[:, 1], math.inf))

  def forward(self, cmd: torch.Tensor, a: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    a = a.clamp(-1.0, 1.0)
    target = cmd + self.dmin + 0.5 * (a + 1.0) * (self.dmax - self.dmin)
    target = q + (target - q).clamp(-self.lead, self.lead)
    return torch.maximum(torch.minimum(target, self.hi), self.lo)


# --- the model -------------------------------------------------------------------------------------------------------

@dataclass
class WMConfig:
  nj: int = 16
  dc: int = 0  # context width (0 = no context input)
  hidden: tuple[int, ...] = (1024, 1024, 1024)
  term_hidden: tuple[int, ...] = (256, 256)
  dt: float = 0.05
  use_prev_cmd: bool = True
  layer_norm: bool = True
  bf16: bool = False  # run the dynamics MLP under bfloat16 autocast (CUDA); integration stays float32
  extra: dict = field(default_factory=dict)


def mlp(din: int, hidden: tuple[int, ...], dout: int, layer_norm: bool) -> nn.Sequential:
  layers, d = [], din
  for h in hidden:
    layers += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if layer_norm else []) + [nn.SiLU()]
    d = h
  layers.append(nn.Linear(d, dout))
  return nn.Sequential(*layers)


class WorldModel(nn.Module):
  def __init__(self, cfg: WMConfig):
    super().__init__()
    self.cfg = cfg
    nj = cfg.nj
    self.S = state_slices(nj)
    self.din = nj * (3 + int(cfg.use_prev_cmd)) + 3 + 6 + 3 + 3 + cfg.dc
    self.dout = 2 * nj + 12
    self.net = mlp(self.din, cfg.hidden, self.dout, cfg.layer_norm)
    nn.init.zeros_(self.net[-1].weight)
    nn.init.zeros_(self.net[-1].bias)
    self.term = mlp(2 * nj + 3 + 6 + 3 + 3, cfg.term_hidden, 1, cfg.layer_norm)
    # normalization statistics (set from data by ``fit_stats``)
    self.register_buffer("x_mu", torch.zeros(self.din))
    self.register_buffer("x_sd", torch.ones(self.din))
    self.register_buffer("t_mu", torch.zeros(2 * nj + 15))
    self.register_buffer("t_sd", torch.ones(2 * nj + 15))
    self.register_buffer("o_sd", torch.ones(self.dout))

  # features
  def _feat(self, s: torch.Tensor, cmd_next: torch.Tensor, ctx: torch.Tensor | None) -> torch.Tensor:
    S = self.S
    q, qd, cmd = s[..., S["q"]], s[..., S["qd"]], s[..., S["cmd"]]
    parts = [q, qd, cmd_next - q]
    if self.cfg.use_prev_cmd:
      parts.append(cmd - q)
    parts += [s[..., S["p"]], quat_to_rot6d(s[..., S["quat"]]), s[..., S["v"]], s[..., S["w"]]]
    if self.cfg.dc:
      parts.append(ctx)
    return torch.cat(parts, -1)

  def _term_feat(self, s: torch.Tensor) -> torch.Tensor:
    S = self.S
    return torch.cat([s[..., S["q"]], s[..., S["qd"]], s[..., S["p"]], quat_to_rot6d(s[..., S["quat"]]),
                      s[..., S["v"]], s[..., S["w"]]], -1)

  def forward(self, s: torch.Tensor, cmd_next: torch.Tensor, ctx: torch.Tensor | None = None):
    """(s_{t+1}, termination logit of the step)."""
    S, dt, nj = self.S, self.cfg.dt, self.cfg.nj
    x = (self._feat(s, cmd_next, ctx) - self.x_mu) / self.x_sd
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.bf16 and x.is_cuda):
      o = self.net(x)
    o = o.float() * self.o_sd
    o_qd, o_qc = o[..., :nj], o[..., nj:2 * nj]
    o_v, o_pc, o_w, o_rc = o[..., 2 * nj:2 * nj + 3], o[..., 2 * nj + 3:2 * nj + 6], o[..., 2 * nj + 6:2 * nj + 9], \
        o[..., 2 * nj + 9:2 * nj + 12]
    qd = s[..., S["qd"]] + o_qd
    q = s[..., S["q"]] + dt * qd + o_qc
    v = s[..., S["v"]] + o_v
    p = s[..., S["p"]] + dt * v + o_pc
    w = s[..., S["w"]] + o_w
    quat = quat_mul(quat_exp(dt * w + o_rc), s[..., S["quat"]])
    quat = quat / quat.norm(dim=-1, keepdim=True)
    s2 = torch.cat([q, qd, cmd_next, p, quat, v, w], -1)
    logit = self.term((self._term_feat(s2) - self.t_mu) / self.t_sd).squeeze(-1)
    return s2, logit

  @torch.no_grad()
  def fit_stats(self, s: torch.Tensor, cmd_next: torch.Tensor, ctx: torch.Tensor | None, s_next: torch.Tensor):
    """Input / output normalization from a batch of one-step transitions (s, cmd_{t+1}, ctx, s_{t+1})."""
    S, dt, nj = self.S, self.cfg.dt, self.cfg.nj
    x = self._feat(s, cmd_next, ctx)
    self.x_mu.copy_(x.mean(0))
    self.x_sd.copy_(x.std(0).clamp_min(1e-4))
    tf = self._term_feat(s_next)
    self.t_mu.copy_(tf.mean(0))
    self.t_sd.copy_(tf.std(0).clamp_min(1e-4))
    d_qd = s_next[:, S["qd"]] - s[:, S["qd"]]
    d_qc = s_next[:, S["q"]] - s[:, S["q"]] - dt * s_next[:, S["qd"]]
    d_v = s_next[:, S["v"]] - s[:, S["v"]]
    d_pc = s_next[:, S["p"]] - s[:, S["p"]] - dt * s_next[:, S["v"]]
    d_w = s_next[:, S["w"]] - s[:, S["w"]]
    d_rc = quat_log(quat_mul(s_next[:, S["quat"]], quat_conj(s[:, S["quat"]]))) - dt * s_next[:, S["w"]]
    rms = lambda d: d.pow(2).mean(0).sqrt().clamp_min(1e-6)
    # one shared scale per object quantity (frame-free), per joint for the hand
    sd = torch.cat([rms(d_qd), rms(d_qc), rms(d_v).mean().expand(3), rms(d_pc).mean().expand(3),
                    rms(d_w).mean().expand(3), rms(d_rc).mean().expand(3)])
    self.o_sd.copy_(sd)


# --- multi-step error ------------------------------------------------------------------------------------------------

class StateError(nn.Module):
  """Squared error of predicted vs true states, each quantity in units of its typical one-step change."""

  def __init__(self, nj: int):
    super().__init__()
    self.S = state_slices(nj)
    self.register_buffer("scale", torch.ones(5))  # q, qd, p, rot, v / w are folded in below
    self.register_buffer("sq", torch.ones(nj))
    self.register_buffer("sqd", torch.ones(nj))
    self.register_buffer("sv", torch.ones(()))
    self.register_buffer("sw", torch.ones(()))
    self.register_buffer("sp", torch.ones(()))
    self.register_buffer("sr", torch.ones(()))

  @torch.no_grad()
  def fit(self, s: torch.Tensor, s_next: torch.Tensor):
    S = self.S
    r = lambda d: d.pow(2).mean().sqrt().clamp_min(1e-6)
    self.sq.copy_((s_next[:, S["q"]] - s[:, S["q"]]).pow(2).mean(0).sqrt().clamp_min(1e-5))
    self.sqd.copy_((s_next[:, S["qd"]] - s[:, S["qd"]]).pow(2).mean(0).sqrt().clamp_min(1e-4))
    self.sp.copy_(r(s_next[:, S["p"]] - s[:, S["p"]]))
    self.sv.copy_(r(s_next[:, S["v"]] - s[:, S["v"]]))
    self.sw.copy_(r(s_next[:, S["w"]] - s[:, S["w"]]))
    self.sr.copy_(r(quat_log(quat_mul(s_next[:, S["quat"]], quat_conj(s[:, S["quat"]])))))

  def forward(self, sh: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    S = self.S
    e = ((sh[..., S["q"]] - s[..., S["q"]]) / self.sq).pow(2).sum(-1)
    e = e + ((sh[..., S["qd"]] - s[..., S["qd"]]) / self.sqd).pow(2).sum(-1)
    e = e + ((sh[..., S["p"]] - s[..., S["p"]]) / self.sp).pow(2).sum(-1)
    e = e + ((sh[..., S["v"]] - s[..., S["v"]]) / self.sv).pow(2).sum(-1)
    e = e + ((sh[..., S["w"]] - s[..., S["w"]]) / self.sw).pow(2).sum(-1)
    rot = quat_log(quat_mul(sh[..., S["quat"]], quat_conj(s[..., S["quat"]])))
    return e + (rot / self.sr).pow(2).sum(-1)
