"""Transition recording on mjlab in-hand tasks, for fitting a world model (SuperMimic-style).

A transition is (state s_t, executed action a_t, next state s_{t+1}, done_t) at the control rate. The state comes from
the simulator's qpos / qvel (fresh, never a physics substep stale), in the hand's base frame (the robot's root link):

  q (nj)      hand joint positions (true, without encoder bias)
  qd (nj)     hand joint velocities
  cmd (nj)    the joint target the delta-position action term holds, i.e. the command in effect during the step that
              produced this state (the delta is integrated on it; the gear servo tracks cmd - encoder_bias)
  p (3)       object position
  quat (4)    object orientation, wxyz, sign-fixed to w >= 0
  v (3)       object linear velocity
  w (3)       object angular velocity

``STATE_SLICES`` names the columns. When an env terminates (drop) or times out, the step's next state is captured by
``TerminalRecorder`` just before mjlab resets that env; the row after it is the new episode's first state.

The context is every per-env model parameter the task's randomization changed (columns that differ across envs after
the first reset: geom friction and size, body mass / inertia / COM, armature, damping), the gear servo's per-env
tensors, the command lag, the encoder bias and, for the multi-shape fingertip_rotate object, its shape id.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.recorder_manager import RecorderTerm
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul, quat_conjugate


def state_slices(nj: int) -> dict[str, slice]:
  names = (("q", nj), ("qd", nj), ("cmd", nj), ("p", 3), ("quat", 4), ("v", 3), ("w", 3))
  out, i = {}, 0
  for n, d in names:
    out[n] = slice(i, i + d)
    i += d
  return out


@dataclass
class Layout:
  """Indices resolved once per env: hand joints, the object's free joint, the hand base pose, the action term."""

  jq: torch.Tensor  # (nj,) qpos addresses of the hand joints, in the action term's target order
  jv: torch.Tensor
  oq: torch.Tensor  # (7,) object free-joint qpos addresses
  ov: torch.Tensor  # (6,)
  base_p: torch.Tensor  # (N, 3) hand base position, world
  base_q: torch.Tensor  # (N, 4) hand base orientation, wxyz
  term: object  # the delta-position action term
  nj: int

  @property
  def dim(self) -> int:
    return 3 * self.nj + 13


def layout(env, robot: str = "robot", obj: str = "cube", action: str = "joint_pos") -> Layout:
  r, o = env.scene[robot], env.scene[obj]
  term = env.action_manager.get_term(action)
  tid = term.target_ids if isinstance(term.target_ids, torch.Tensor) else torch.as_tensor(term.target_ids)
  tid = tid.to(env.device).long()
  jq = r.indexing.joint_q_adr.to(env.device).long()[tid]
  jv = r.indexing.joint_v_adr.to(env.device).long()[tid]
  oq = o.indexing.free_joint_q_adr.to(env.device).long()
  ov = o.indexing.free_joint_v_adr.to(env.device).long()
  assert oq.numel() == 7 and ov.numel() == 6, "the object must have one free joint"
  return Layout(jq=jq, jv=jv, oq=oq, ov=ov, base_p=r.data.root_link_pos_w.clone(),
                base_q=r.data.root_link_quat_w.clone(), term=term, nj=int(tid.numel()))


def read_state(env, L: Layout, env_ids: torch.Tensor | None = None) -> torch.Tensor:
  """(n, 3 nj + 13) the state rows of ``env_ids`` (all envs if None), hand base frame."""
  qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
  ids = slice(None) if env_ids is None else env_ids
  bp, bq = L.base_p[ids], L.base_q[ids]
  q = qpos[ids][:, L.jq]
  qd = qvel[ids][:, L.jv]
  cmd = L.term._target[ids]
  op = qpos[ids][:, L.oq[:3]]
  oquat = qpos[ids][:, L.oq[3:]]
  ov = qvel[ids][:, L.ov[:3]]  # free joint: linear velocity in world
  ow_body = qvel[ids][:, L.ov[3:]]  # angular velocity in the object's own frame
  p = quat_apply_inverse(bq, op - bp)
  quat = quat_mul(quat_conjugate(bq), oquat)
  quat = torch.where(quat[:, :1] < 0, -quat, quat)
  v = quat_apply_inverse(bq, ov)
  w = quat_apply_inverse(bq, _quat_apply(oquat, ow_body))
  return torch.cat([q, qd, cmd, p, quat, v, w], -1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  from mjlab.utils.lab_api.math import quat_apply

  return quat_apply(q, v)


# --- context -------------------------------------------------------------------------------------------------------

_MODEL_FIELDS = ("geom_friction", "geom_size", "body_mass", "body_ipos", "body_inertia", "dof_armature",
                 "dof_damping")


def context_full(env, robot: str = "robot") -> tuple[torch.Tensor, list[str]]:
  """(N, D) every candidate context value, with a name per column."""
  import mujoco

  m, mjm, N = env.sim.model, env.sim.mj_model, env.num_envs
  cols, names = [], []
  for f in _MODEL_FIELDS:
    x = getattr(m, f, None)
    if x is None:
      continue
    if not isinstance(x, torch.Tensor):  # mjlab's TorchArray view of a warp array
      x = x[:]
    if not isinstance(x, torch.Tensor) or x.shape[0] != N:
      continue
    x = x.reshape(N, x.shape[1], -1).float()
    if f == "geom_friction":
      x = x[:, :, :1]  # sliding friction only (torsion / roll are not randomized)
    obj = {"geom": mujoco.mjtObj.mjOBJ_GEOM, "body": mujoco.mjtObj.mjOBJ_BODY}.get(f.split("_")[0])
    for i in range(x.shape[1]):
      if obj is not None:
        nm = mujoco.mj_id2name(mjm, obj, i) or f"{i}"
      else:
        nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, int(mjm.dof_jntid[i])) or f"{i}"
      for k in range(x.shape[2]):
        names.append(f"{f}[{nm}][{k}]")
      cols.append(x[:, i])
  r = env.scene[robot]
  for a in r.actuators:
    for f in ("current_gain", "current_damping", "current_limit", "torque_constant", "gear_friction", "dry_friction",
              "stiction_friction"):
      x = getattr(a, f, None)
      if isinstance(x, torch.Tensor) and x.shape[0] == N:
        cols.append(x.float())
        names += [f"gear.{f}[{j}]" for j in range(x.shape[1])]
    buf = getattr(a, "_delay_buffer", None)
    if buf is not None:
      cols.append(buf.current_lags.float()[:, None])
      names.append("lag_substeps")
  eb = getattr(r.data, "encoder_bias", None)
  if isinstance(eb, torch.Tensor) and eb.shape[0] == N:
    cols.append(eb.float())
    names += [f"encoder_bias[{j}]" for j in range(eb.shape[1])]
  st = getattr(env, "_ft", None)
  if st is not None:
    cols.append(st.shape.float()[:, None])
    names.append("shape_id")
    cols.append(st.dims.float())
    names += ["shape_dim0", "shape_dim1", "shape_dim2"]
  return torch.cat(cols, -1), names


def varying_columns(x: torch.Tensor, names: list[str], tol: float = 1e-9) -> tuple[torch.Tensor, list[str]]:
  """The columns that differ across envs (the randomized ones); shape descriptors are always kept."""
  span = x.max(0).values - x.min(0).values
  keep = span > tol
  for i, n in enumerate(names):
    if n.startswith("shape_") or n == "lag_substeps":
      keep[i] = True
  idx = keep.nonzero().squeeze(-1).tolist()
  uniq = []  # drop exact duplicates (a per-env scalar copied to every joint, e.g. K_t)
  for i in idx:
    if not any(torch.equal(x[:, i], x[:, j]) for j in uniq):
      uniq.append(i)
  idx = torch.tensor(uniq, dtype=torch.long, device=x.device)
  return idx, [names[i] for i in idx.tolist()]


class TerminalRecorder(RecorderTerm):
  """Captures the true next state of envs that are about to be reset (mjlab resets inside ``step``)."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.L: Layout | None = None
    self.ids: torch.Tensor | None = None
    self.rows: torch.Tensor | None = None

  def clear(self) -> None:
    self.ids, self.rows = None, None

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    if self.L is None:
      return
    ids = env_ids.to(self._env.device).long()
    self.ids, self.rows = ids, read_state(self._env, self.L, ids)
