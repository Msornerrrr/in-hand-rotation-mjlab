"""The palm-down 22-object fingertip rotation task (``-PalmDown-MixTeacherF`` rewards) seen from world-model states.

Reward terms, as rates (per second; the reward manager multiplies by dt) with the task's weights:
  rotate           +10  (1 - a) clip(w . n, -0.3, 0.3) + a clip(new / dt, max 0.3), new = rotation beyond the
                        episode's high-water mark (the ratchet); x 0.1 when the object drifted > 2 cm from its reset
  obj_linvel       -1.8 |v_world|_1
  pose_diff        -1.8 sum (q - q_reset)^2
  torque, work     -0.6 sum tau^2, -3 (sum tau qdot)^2       tau: the gear servo law (``gear_yaw.servo``)
  pad_force_excess -0.1 sum clamp(F_pad - 8, 0, 5)
  dropped          -400 [drift > 3 cm] (the episode ends)
  support          +1   pads_low: pad-force-weighted mean of clamp(-h, 0, 1), h the pad contact's height above the
                        object's centre in half extents
  link_contact     -0.5 [a finger link other than the fingertip presses > 0.2 N]
  finger_idle      -1.5 number of fingers whose pad has not pressed (> 0.1 N) for over 1 s
  palm_contact     -2   [the palm touches the object > 0.05 N]
The contact quantities (pad forces and heights, link and palm force) are not in the state: imagination predicts them
with a learned head (``ContactHead``); the indicators become sigmoids there so the gradient sees them.

Actor observation (the deployable ``-PalmDown-Mix`` actor, history 16, per term oldest -> newest): joint_pos -
default (+ encoder bias, U(+-0.01), lag 0/1), the commanded target, commanded - measured (+ bias, U(+-0.01), same lag).
The privileged actor adds the current object state and descriptors (``priv``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import quat_conj, quat_log, quat_mul, quat_to_rot6d
from .record import state_slices

H_OBS = 16
CONTACT_DIM = 10  # pad force (4), pad height (4), link force (1), palm force (1)
TERMS = ("rotate", "obj_linvel", "pose_diff", "torque", "work", "pad_force_excess", "dropped", "support",
         "link_contact", "finger_idle", "palm_contact")
CLIP_MIN, CLIP_MAX, DRIFT_GATE, GATE_OUT, DROP_DRIFT = -0.3, 0.3, 0.02, 0.1, 0.03
PAD_FREE, PAD_CAP, PRESS_N, LINK_N, PALM_N, IDLE_S = 8.0, 5.0, 0.1, 0.2, 0.05, 1.0
OBS_NOISE = 0.01
LEAD_CLAMP = 0.15


def contact_signals(env, palm: str = "down") -> torch.Tensor:
  """(N, 10) the true contact quantities after the last physics step (read from the task's sensors)."""
  from in_hand_rotation_mjlab.tasks.fingertip_rotate import rl_mdp as M
  from in_hand_rotation_mjlab.tasks.fingertip_rotate.env_cfg import NONPAD_SENSOR, PAD_SENSOR, PALM_SENSOR, pad_contact_pos, pad_net_force

  st = M.ft(env, palm)
  pad = env.scene.sensors[PAD_SENSOR]
  F = pad_net_force(pad).norm(dim=-1)
  P = pad_contact_pos(pad)
  c = env.scene["cube"].data.root_link_pos_w
  h = (P[..., 2] - c[:, None, 2]) / st.extent[:, None].clamp(min=1e-3)
  h = torch.where(F > PRESS_N, h, torch.zeros_like(h))
  link = env.scene.sensors[NONPAD_SENSOR].data.force.norm(dim=-1)[:, M._link_columns(env)].amax(-1)
  palm_f = env.scene.sensors[PALM_SENSOR].data.force.norm(dim=-1).amax(-1)
  return torch.cat([F, h.clamp(-3, 3), link[:, None], palm_f[:, None]], -1)


class ContactHead(nn.Module):
  """Contact quantities at the end of a step from the predicted state: pad forces (softplus), pad heights, and the
  logits of link contact > 0.2 N and palm contact > 0.05 N."""

  def __init__(self, nj: int, dc: int, hidden=(512, 512)):
    super().__init__()
    din = 4 * nj + 15 + dc
    layers, d = [], din
    for h in hidden:
      layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
      d = h
    layers.append(nn.Linear(d, CONTACT_DIM))
    self.net = nn.Sequential(*layers)
    self.register_buffer("mu", torch.zeros(din))
    self.register_buffer("sd", torch.ones(din))
    self.S = state_slices(nj)

  def features(self, s2, cmd, ctx):
    S = self.S
    parts = [s2[..., S["q"]], s2[..., S["qd"]], cmd - s2[..., S["q"]], s2[..., S["cmd"]] - s2[..., S["q"]],
             s2[..., S["p"]], quat_to_rot6d(s2[..., S["quat"]]), s2[..., S["v"]], s2[..., S["w"]]]
    if ctx is not None:
      parts.append(ctx)
    return torch.cat(parts, -1)

  def forward(self, s2, cmd, ctx):
    o = self.net((self.features(s2, cmd, ctx) - self.mu) / self.sd)
    return dict(F=nn.functional.softplus(o[..., :4]) * 2.0, h=o[..., 4:8], link_logit=o[..., 8], palm_logit=o[..., 9])

  def loss(self, pred, target):
    F, h = target[..., :4], target[..., 4:8]
    lf = nn.functional.smooth_l1_loss(torch.log1p(pred["F"]), torch.log1p(F))
    pressed = (F > PRESS_N).float()
    lh = ((pred["h"] - h).pow(2) * pressed).sum() / pressed.sum().clamp(min=1)
    ll = nn.functional.binary_cross_entropy_with_logits(pred["link_logit"], (target[..., 8] > LINK_N).float())
    lp = nn.functional.binary_cross_entropy_with_logits(pred["palm_logit"], (target[..., 9] > PALM_N).float())
    return lf + lh + ll + lp


class FailHead(nn.Module):
  """Logit that the step entering state ``s2`` terminates the episode (a drop). Unlike the world model's own head it
  sees what the drop rule is made of: the object's drift from its reset position (``info``), its size and the
  episode context. Trained (Dex-Supermimic style) on true AND world-model-predicted next states with importance
  weights that undo the oversampling of drop windows, so its probabilities need no correction."""

  def __init__(self, nj: int, dc: int, hidden=(256, 128)):
    super().__init__()
    din = 2 * nj + 15 + 5 + dc
    layers, d = [], din
    for h in hidden:
      layers += [nn.Linear(d, h), nn.ELU()]
      d = h
    layers.append(nn.Linear(d, 1))
    self.net = nn.Sequential(*layers)
    self.register_buffer("mu", torch.zeros(din))
    self.register_buffer("sd", torch.ones(din))
    self.nj, self.S = nj, state_slices(nj)

  def features(self, s2, info, ctx):
    S, nj = self.S, self.nj
    drift = s2[..., S["p"]] - info[..., nj:nj + 3]
    parts = [s2[..., S["q"]], s2[..., S["qd"]], s2[..., S["p"]], quat_to_rot6d(s2[..., S["quat"]]), s2[..., S["v"]],
             s2[..., S["w"]], drift, drift.norm(dim=-1, keepdim=True), info[..., nj + 3:nj + 4]]
    if ctx is not None:
      parts.append(ctx)
    return torch.cat(parts, -1)

  @torch.no_grad()
  def fit_stats(self, s2, info, ctx):
    f = self.features(s2, info, ctx)
    self.mu.copy_(f.mean(0)); self.sd.copy_(f.std(0).clamp_min(1e-4))

  def forward(self, s2, info, ctx):
    return self.net((self.features(s2, info, ctx) - self.mu) / self.sd).squeeze(-1)


class StateEstimator(nn.Module):
  """Proprio history -> the privileged actor's features (``FtrotMix.priv``: object drift and velocities, orientation,
  shape, size, mass, pad friction), standardized. Trained supervised on the replay; the deployable actor reads its
  output next to the history (both run on joint readings only)."""

  def __init__(self, obs_dim: int, out_dim: int, hidden=(512, 256)):
    super().__init__()
    layers, d = [], obs_dim
    for h in hidden:
      layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
      d = h
    layers.append(nn.Linear(d, out_dim))
    self.net = nn.Sequential(*layers)
    self.register_buffer("x_mu", torch.zeros(obs_dim))
    self.register_buffer("x_sd", torch.ones(obs_dim))
    self.register_buffer("y_mu", torch.zeros(out_dim))
    self.register_buffer("y_sd", torch.ones(out_dim))
    self.obs_dim, self.out_dim, self.hidden = obs_dim, out_dim, tuple(hidden)

  @torch.no_grad()
  def fit_stats(self, x, y):
    self.x_mu.copy_(x.mean(0)); self.x_sd.copy_(x.std(0).clamp_min(1e-4))
    self.y_mu.copy_(y.mean(0)); self.y_sd.copy_(y.std(0).clamp_min(1e-4))

  def forward(self, x):
    """Standardized feature estimate."""
    return self.net((x - self.x_mu) / self.x_sd)

  def loss(self, x, y):
    return (self.forward(x) - (y - self.y_mu) / self.y_sd).pow(2).mean()


class FtrotMix:
  def __init__(self, nj: int, q_default, base_q, n_hand, dt: float, weights: dict, ctx_names: list[str]):
    self.nj, self.S, self.dt = nj, state_slices(nj), dt
    self.q_default, self.base_q, self.n_hand = q_default, base_q, n_hand
    self.w = {k: float(weights[k]) for k in TERMS}
    idx = lambda name: ctx_names.index(name) if name in ctx_names else None  # noqa: E731
    self.enc_idx = [idx(f"encoder_bias[{j}]") for j in range(nj)]
    self.shape_idx = idx("shape_id")
    self.dims_idx = [idx(f"shape_dim{k}") for k in range(3)]
    self.mass_idx = next((i for i, n in enumerate(ctx_names) if n.startswith("body_mass[cube")), None)
    self.padfr_idx = next((i for i, n in enumerate(ctx_names) if n.startswith("geom_friction[robot/if_tip")), None)
    self.frame_dim = 3 * nj
    self.obs_dim = H_OBS * self.frame_dim
    self.info_dim = nj + 3 + 1  # q_reset, p0 (hand frame), extent (half extent along the palm normal)
    self.aux_dim = 1 + 4  # ratchet gap (qyaw - qyaw_max <= 0), finger idle clocks

  # --- observation ---
  def enc(self, ctx_raw):
    cols = [i for i in self.enc_idx if i is not None]
    if len(cols) != self.nj:
      return torch.zeros(ctx_raw.shape[0], self.nj, device=ctx_raw.device)
    return ctx_raw[:, cols]

  def obs_frame(self, q_meas_lagged, cmd, ctx_raw, noise: bool = True):
    """One 48-D frame: (q + bias - default + n1, cmd, cmd - (q + bias) + n2), q from the (possibly lagged) state."""
    qb = q_meas_lagged + self.enc(ctx_raw)
    n1 = (torch.rand_like(qb) * 2 - 1) * OBS_NOISE if noise else 0.0
    n2 = (torch.rand_like(qb) * 2 - 1) * OBS_NOISE if noise else 0.0
    return torch.cat([qb - self.q_default + n1, cmd, cmd - qb + n2], -1)

  def flatten(self, hist):
    """(B, H, 3 nj) frames -> the env's per-term layout (joint_pos x H, command x H, error x H)."""
    B, nj = hist.shape[0], self.nj
    return torch.cat([hist[..., :nj].reshape(B, -1), hist[..., nj:2 * nj].reshape(B, -1),
                      hist[..., 2 * nj:].reshape(B, -1)], -1)

  def priv(self, s, info, ctx_raw):
    """Privileged actor inputs: object position from its reset, the reset position, velocities, shape, mass,
    pad friction (17 + 4 + ...)."""
    S, nj = self.S, self.nj
    p0 = info[:, nj:nj + 3]
    parts = [(s[..., S["p"]] - p0) * 20.0, p0 * 10.0, s[..., S["v"]], s[..., S["w"]] * 0.3,
             quat_to_rot6d(s[..., S["quat"]])]
    if self.shape_idx is not None:
      parts.append(nn.functional.one_hot(ctx_raw[:, self.shape_idx].round().long().clamp(0, 3), 4).float())
      parts.append(ctx_raw[:, self.dims_idx] * 20.0)
    if self.mass_idx is not None:
      parts.append(ctx_raw[:, self.mass_idx:self.mass_idx + 1] * 10.0)
    if self.padfr_idx is not None:
      parts.append(ctx_raw[:, self.padfr_idx:self.padfr_idx + 1])
    return torch.cat(parts, -1)

  # --- reward ---
  def twist_increment(self, s, s2):
    rel = quat_mul(s2[..., self.S["quat"]], quat_conj(s[..., self.S["quat"]]))
    return quat_log(rel) @ self.n_hand

  def torque(self, s2, cmd, plant):
    nj, S = self.nj, self.S
    G, Gd, Imax, Kt = plant[:, :nj], plant[:, nj:2 * nj], plant[:, 2 * nj:3 * nj], plant[:, 3 * nj:]
    q, qd = s2[..., S["q"]], s2[..., S["qd"]]
    cur = torch.maximum(torch.minimum(G * (cmd - q).clamp(-LEAD_CLAMP, LEAD_CLAMP) - Gd * qd, Imax), -Imax)
    return Kt * cur

  def reward_rates(self, s, s2, cmd, info, aux, contact, plant, a_mix: float, p_drop=None, soft: bool = False):
    """Per-term rates of the step s -> s2. ``contact``: the true signals (N, 10) or the head's prediction (dict).
    ``aux`` (N, 5) = ratchet gap, idle clocks at s. Returns (terms, aux at s2)."""
    S, nj, dt = self.S, self.nj, self.dt
    q2, qd2 = s2[..., S["q"]], s2[..., S["qd"]]
    p0, extent = info[:, nj:nj + 3], info[:, nj + 3]
    drift = (s2[..., S["p"]] - p0).norm(dim=-1)
    gate = torch.where(drift > DRIFT_GATE, torch.full_like(drift, GATE_OUT), torch.ones_like(drift))
    rate = s2[..., S["w"]] @ self.n_hand
    gap = aux[:, 0] + self.twist_increment(s, s2)
    new = gap.clamp(min=0.0)
    gap2 = gap.clamp(max=0.0)
    rot = ((1 - a_mix) * rate.clamp(CLIP_MIN, CLIP_MAX) + a_mix * (new / dt).clamp(max=CLIP_MAX)) * gate
    from mjlab.utils.lab_api.math import quat_apply

    v_w = quat_apply(self.base_q.expand(s2.shape[0], 4), s2[..., S["v"]])
    tau = self.torque(s2, cmd, plant)
    if isinstance(contact, dict):
      F, h = contact["F"], contact["h"]
      link_p, palm_p = torch.sigmoid(contact["link_logit"]), torch.sigmoid(contact["palm_logit"])
    else:
      F, h = contact[:, :4], contact[:, 4:8]
      link_p, palm_p = (contact[:, 8] > LINK_N).float(), (contact[:, 9] > PALM_N).float()
    if soft:
      pressed = torch.sigmoid((F - PRESS_N) / 0.03)
    else:
      pressed = (F > PRESS_N).float()
    wgt = F * pressed
    support = (wgt * (-h).clamp(0.0, 1.0)).sum(-1) / wgt.sum(-1).clamp(min=1e-6)
    idle = (1 - pressed) * (aux[:, 1:] + dt)
    idle_count = torch.sigmoid((idle - IDLE_S) / 0.05).sum(-1) if soft else (idle > IDLE_S).float().sum(-1)
    if p_drop is None:
      p_drop = (drift > DROP_DRIFT).float()
    terms = dict(
      rotate=self.w["rotate"] * rot,
      obj_linvel=self.w["obj_linvel"] * v_w.abs().sum(-1),
      pose_diff=self.w["pose_diff"] * (q2 - info[:, :nj]).pow(2).sum(-1),
      torque=self.w["torque"] * tau.pow(2).sum(-1),
      work=self.w["work"] * (tau * qd2).sum(-1).pow(2),
      pad_force_excess=self.w["pad_force_excess"] * (F - PAD_FREE).clamp(0, PAD_CAP).sum(-1),
      dropped=self.w["dropped"] * p_drop,
      support=self.w["support"] * support,
      link_contact=self.w["link_contact"] * link_p,
      finger_idle=self.w["finger_idle"] * idle_count,
      palm_contact=self.w["palm_contact"] * palm_p,
    )
    return terms, torch.cat([gap2[:, None], idle], -1)

  def critic_features(self, s, info, aux, ctx):
    S, nj = self.S, self.nj
    parts = [s[..., S["q"]], s[..., S["qd"]], s[..., S["cmd"]] - s[..., S["q"]], s[..., S["p"]] - info[:, nj:nj + 3],
             quat_to_rot6d(s[..., S["quat"]]), s[..., S["v"]], s[..., S["w"]], s[..., S["q"]] - info[:, :nj],
             info[:, nj + 3:nj + 4] * 20.0, aux[:, :1], aux[:, 1:].clamp(max=2.0)]
    if ctx is not None:
      parts.append(ctx)
    return torch.cat(parts, -1)
