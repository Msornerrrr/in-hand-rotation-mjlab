"""MDP terms for the fingertip-only yaw rotation policy (RL).

Frames: the hand frame is the robot root (``env_cfg.MOUNTS``); the palm normal n (hand +z, world
direction fixed by the mount) is the rotation axis. Positive yaw = counter-clockwise about n, the
direction of the planner trajectories the episodes reset from.

Per-env task state lives on ``env._ft`` (``FtState``): the object's reset position, its radius,
the episode's accumulated yaw, and the tactile sensor randomization.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul

from in_hand_rotation_mjlab.mdp.events import _resolve_geom_ids

from .env_cfg import MOUNTS, NONPAD_SENSOR, PAD_SENSOR, PALM_SENSOR, pad_contact_pos, pad_net_force
from .objects import OBJECTS

FINGER_TIPS = ("robot/if_tip", "robot/mf_tip", "robot/rf_tip", "robot/th_tip")  # pad sensor order


class FtState:
  def __init__(self, env, palm: str):
    m, dev, B = MOUNTS[palm], env.device, env.num_envs
    self.palm = palm
    self.R = torch.tensor(m.R, dtype=torch.float32, device=dev)  # hand -> world rotation
    self.p_mount = torch.tensor(m.pos, dtype=torch.float32, device=dev)
    self.q_mount = torch.tensor(m.rot, dtype=torch.float32, device=dev)
    self.n_w = self.R[:, 2].clone()  # palm normal, world
    self.p0 = torch.zeros(B, 3, device=dev)  # object position at reset, hand frame
    self.radius = torch.full((B,), 0.03, device=dev)
    # the object of the episode (multi-shape object; a primitive pool fills extent only): shape id
    # (objects.SHAPE_ID), geom size (sphere r; box half extents; cylinder radius, half height), half extent along n
    self.shape = torch.zeros(B, dtype=torch.long, device=dev)
    self.dims = torch.full((B, 3), 0.03, device=dev)
    self.extent = torch.full((B,), 0.03, device=dev)
    self.stage = 1
    self.idle = torch.zeros(B, 4, device=dev)  # s since each finger's pad last pressed (``finger_idle``)
    self.yaw = torch.zeros(B, device=dev)  # accumulated twist about n this episode
    self.rows = torch.zeros(B, dtype=torch.long, device=dev)  # the pool row each env reset from
    self.tac_gain = torch.ones(B, 4, device=dev)  # per-pad tactile gain (sensor calibration)
    self.tac_bias = torch.zeros(B, 4, 3, device=dev)  # per-pad tactile offset [N]
    self.push = torch.zeros(B, 3, device=dev)  # random push force on the object (world frame) [N]
    # net twist about n from the object's orientation (``rot_progress_window``): a sampled angular velocity
    # integrated at 20 Hz overstates a fast back-and-forth by ~1/3
    self.qyaw = torch.zeros(B, device=dev)
    self.qyaw_max = torch.zeros(B, device=dev)  # the episode's high-water mark of qyaw (``rot_progress_ratchet``)
    self.q_prev = torch.zeros(B, 4, device=dev)
    self.fresh = torch.ones(B, dtype=torch.bool, device=dev)  # q_prev not yet set this episode
    self.yhist: torch.Tensor | None = None  # (B, window steps) ring buffer of qyaw
    self.hptr = 0
    mjm = env.sim.mj_model
    self.pad_geoms = torch.tensor([mjm.geom(n).id for n in FINGER_TIPS], device=dev)


def ft(env, palm: str) -> FtState:
  st = getattr(env, "_ft", None)
  if st is None:
    st = env._ft = FtState(env, palm)
  return st


def obj_pos_hand(env, st: FtState) -> torch.Tensor:
  p = env.scene["cube"].data.root_link_pos_w - env.scene.env_origins - st.p_mount
  return p @ st.R


def obj_angvel_w(env) -> torch.Tensor:
  return env.scene["cube"].data.root_link_ang_vel_w


def yaw_rate(env, st: FtState) -> torch.Tensor:
  return obj_angvel_w(env) @ st.n_w


def pad_forces_w(env) -> torch.Tensor:
  """(B, 4, 3) net contact force of each pad on the object, world frame."""
  return pad_net_force(env.scene.sensors[PAD_SENSOR])


def pad_forces_local(env, st: FtState) -> torch.Tensor:
  """(B, 4, 3) pad forces in each pad geom's frame: x = along the pad normal (face is -x), y/z = shear."""
  R = env.sim.data.geom_xmat[:, st.pad_geoms]  # (B, 4, 3, 3), geom -> world
  return torch.einsum("bpji,bpj->bpi", R, pad_forces_w(env))


def nonpad_force(env) -> torch.Tensor:
  return env.scene.sensors[NONPAD_SENSOR].data.force.norm(dim=-1).amax(-1)


# --------------------------------------------------------------------------------------------
# reset: a state from the planner pools (settled grasps + trajectory states), with its object size
_POOLS: dict = {}


def _pool(pattern: str, palm: str, device):
  key = (pattern, palm, str(device))
  if key not in _POOLS:
    files = sorted(f for g in pattern.split(";") for f in glob.glob(g))
    parts = []
    for f in files:
      d = np.load(f)
      if str(d["palm"]) != palm:
        continue
      n = len(d["source"])
      ws = d["qacc_warmstart"] if "qacc_warmstart" in d.files else np.zeros((n, 22), dtype=np.float32)
      parts.append({k: d[k] for k in ("joint_pos", "joint_vel", "joint_target", "object_pos_hand", "object_quat_hand",
                                      "object_linvel_hand", "object_angvel_body", "yaw")}
                   | {"qacc_warmstart": ws}
                   | {"radius": np.full(n, OBJECTS[str(d["obj"])].size[0], dtype=np.float32),
                      "source": d["source"].astype(np.float32)})
    if not parts:
      raise FileNotFoundError(f"no reset pools for palm={palm} match {pattern}")
    pool = {k: torch.as_tensor(np.concatenate([p[k] for p in parts]), dtype=torch.float32, device=device)
            for k in parts[0]}
    # size-balanced sampling: rows sorted by radius, a size drawn uniformly, then a row of that size
    order = torch.argsort(pool["radius"], stable=True)
    pool = {k: v[order] for k, v in pool.items()}
    radii, counts = torch.unique_consecutive(pool["radius"], return_counts=True)
    pool["_size_count"] = counts.to(torch.float32)
    pool["_size_offset"] = torch.cumsum(counts, 0) - counts
    _POOLS[key] = pool
    print(f"[fingertip_rotate] reset pool palm={palm}: {len(pool['radius'])} states from {len(parts)} files, diameters "
          f"{[round(float(r) * 200, 2) for r in radii]} cm x {counts.tolist()} states (sampled size-balanced), "
          f"{int((pool['source'] == 0).sum())} grasps", flush=True)
  return _POOLS[key]


@requires_model_fields("geom_size")
def reset_from_pool(env, env_ids, palm: str, pool_glob: str, cube_cfg: SceneEntityCfg,
                    tactile_gain: tuple[float, float] = (1.0, 1.0), tactile_bias: float = 0.0):
  """Restore a full hold: joints, the commanded target (the squeeze lead), the object pose and
  velocity, and the object radius it was recorded with."""
  st = ft(env, palm)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return
  pool = _pool(pool_glob, palm, env.device)
  k = torch.randint(len(pool["_size_count"]), (len(env_ids),), device=env.device)
  rows = pool["_size_offset"][k] + (torch.rand(len(env_ids), device=env.device) * pool["_size_count"][k]).long()
  gid = _resolve_geom_ids(env, cube_cfg)
  r = pool["radius"][rows]
  env.sim.model.geom_size[env_ids[:, None], gid[None, :], :] = r[:, None, None].expand(-1, len(gid), 3)
  _restore_hold(env, env_ids, st, pool, rows, tactile_gain, tactile_bias)
  st.radius[env_ids] = r
  st.extent[env_ids] = r


def _restore_hold(env, env_ids, st, pool, rows, tactile_gain, tactile_bias):
  """Write a pool row's hold into the sim (after the object's geometry is set: the forward pass computes its
  contacts) and reset the per-episode task state."""
  robot, cube = env.scene["robot"], env.scene["cube"]
  robot.write_joint_state_to_sim(pool["joint_pos"][rows], pool["joint_vel"][rows], env_ids=env_ids)
  q_w = quat_mul(st.q_mount.expand(len(rows), 4), pool["object_quat_hand"][rows])
  pose = torch.cat([pool["object_pos_hand"][rows] @ st.R.T + st.p_mount + env.scene.env_origins[env_ids], q_w], -1)
  vel = torch.cat([pool["object_linvel_hand"][rows] @ st.R.T, quat_apply(q_w, pool["object_angvel_body"][rows])], -1)
  cube.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  cube.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
  # the solver warm start (hand dofs, then the object's free joint), recorded on this mount's own world frame
  # (same dof layout as the planner env: 16 hand joints, then the object's free joint)
  ws = pool["qacc_warmstart"][rows]
  if env.sim.data.qacc_warmstart.shape[1] == ws.shape[1]:
    env.sim.data.qacc_warmstart[env_ids] = ws
  # the recorded command, in measured (encoder) coordinates: the physical setpoint target - bias is unchanged
  term = env.action_manager.get_term("joint_pos")
  tid = term._target_ids
  term._target[env_ids] = pool["joint_target"][rows] + robot.data.encoder_bias[env_ids][:, tid]
  term.stage_restored_reset_target(env_ids)
  # The robot-side setpoint too: env.reset() writes actuator controls right after the reset events,
  # which pushes this target into the gear servo's (just cleared, so back-filled) command-delay ring.
  # Left at the scene reset's zeros, the servos would track 0 rad for the first 25-75 ms and drop
  # the object.
  q = robot.data.joint_pos[env_ids][:, tid]
  phys = pool["joint_target"][rows]
  lead = getattr(term, "_max_lead", None)
  if lead is not None:
    phys = q + torch.clamp(phys - q, -lead, lead)
  robot.set_joint_position_target(phys, joint_ids=tid[None, :] if torch.is_tensor(tid) else tid, env_ids=env_ids[:, None])
  # refresh kinematics, contact forces and gravity compensation for the restored configuration: the gear
  # servo's friction and link-weight terms read the previous step's forces on the first substep
  env.sim.forward()
  st.rows[env_ids] = rows
  st.p0[env_ids] = pool["object_pos_hand"][rows]
  st.yaw[env_ids] = 0.0
  st.push[env_ids] = 0.0
  st.qyaw[env_ids] = 0.0
  st.qyaw_max[env_ids] = 0.0
  st.fresh[env_ids] = True
  st.idle[env_ids] = 0.0
  if st.yhist is not None:
    st.yhist[env_ids] = 0.0
  n = len(env_ids)
  st.tac_gain[env_ids] = tactile_gain[0] + (tactile_gain[1] - tactile_gain[0]) * torch.rand(n, 4, device=env.device)
  st.tac_bias[env_ids] = tactile_bias * (2 * torch.rand(n, 4, 3, device=env.device) - 1)


def _mixed_pool(path: str, palm: str, device):
  """A mixed-shape pool (``export_mixed_resets.py``): per row the hold, its object (shape id, the multi-shape
  object's three geom sizes, the half extent along the palm normal) and the curriculum stages' sampling weights."""
  from .objects import OBJECTS, SHAPE_ID, mixed_geom_sizes

  key = ("mixed", path, palm, str(device))
  if key not in _POOLS:
    if not Path(path).is_file():
      raise FileNotFoundError(f"mixed reset pool {path} not found: it ships with the package "
                              "(tasks/fingertip_rotate/data); FTROT_MIXED_RESETS points elsewhere")
    d = np.load(path)
    assert str(d["palm"]) == palm, (path, str(d["palm"]), palm)
    names = [str(x) for x in d["obj_names"]]
    objs = [OBJECTS[n] for n in names]
    oi = d["obj_idx"].astype(np.int64)
    n = len(oi)
    pool = {k: torch.as_tensor(d[k], dtype=torch.float32, device=device)
            for k in ("joint_pos", "joint_vel", "joint_target", "object_pos_hand", "object_quat_hand",
                      "object_linvel_hand", "object_angvel_body", "qacc_warmstart", "weight_stage1", "weight_stage2")}
    pool["geom_size"] = torch.as_tensor(np.stack([mixed_geom_sizes(o) for o in objs])[oi], device=device)  # (n, 3, 3)
    pool["dims"] = torch.as_tensor(np.stack([np.asarray(o.geom_size, np.float32) for o in objs])[oi], device=device)
    pool["shape"] = torch.as_tensor(np.array([SHAPE_ID[o.shape] for o in objs])[oi], device=device)
    pool["extent"] = torch.as_tensor(np.array([o.extent for o in objs], np.float32)[oi], device=device)
    pool["obj_idx"] = torch.as_tensor(oi, device=device)
    pool["names"] = names
    # rows are grouped by object (export order): each object's first row and count, for per-env object assignment
    _, cnt = torch.unique_consecutive(pool["obj_idx"], return_counts=True)
    assert len(cnt) == len(objs), "pool rows are not grouped by object"
    pool["_obj_count"] = cnt
    pool["_obj_start"] = torch.cumsum(cnt, 0) - cnt
    _POOLS[key] = pool
    by = {s: int(sum((oi == i).sum() for i, o in enumerate(objs) if o.shape == s)) for s in SHAPE_ID}
    print(f"[fingertip_rotate] mixed reset pool palm={palm}: {n} holds of {len(objs)} objects {by}", flush=True)
  return _POOLS[key]


def mixed_stage(env, stage2_step: int) -> int:
  """Curriculum stage of the mixed pool: 1 (spheres + cylinders) until ``stage2_step`` control steps, then 2."""
  return 1 if env.common_step_counter < stage2_step else 2


@requires_model_fields("geom_size", "body_inertia", "geom_rbound", "geom_aabb")
def reset_from_mixed_pool(env, env_ids, palm: str, pool_path: str, stage2_step: int,
                          tactile_gain: tuple[float, float] = (1.0, 1.0), tactile_bias: float = 0.0,
                          tight_bounds: bool = True, assign_objects: bool = False):
  """``reset_from_pool`` for the multi-shape object (``objects.MixedObject``): a hold drawn with the curriculum stage's
  weights (shape, then size, then hold), its primitive made active (geom sizes) with the body inertia of that
  primitive at the env's current (randomized) mass. Runs after the mass randomization. ``assign_objects``
  (evaluation): env e holds object e mod (number of objects) instead."""
  from .objects import MIXED_GEOM_ORDER, SHAPES, principal_inertia

  st = ft(env, palm)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return
  pool = _mixed_pool(pool_path, palm, env.device)
  stage = mixed_stage(env, stage2_step)
  if assign_objects:  # evaluation: env e always holds object e mod (number of objects), a random start of it
    k = env_ids % len(pool["_obj_count"])
    rows = pool["_obj_start"][k] + (torch.rand(len(env_ids), device=env.device) * pool["_obj_count"][k]).long()
  else:
    rows = torch.multinomial(pool[f"weight_stage{stage}"], len(env_ids), replacement=True)
  cube = env.scene["cube"]
  mjm = env.sim.mj_model
  gid = torch.tensor([mjm.geom(f"cube/{g}").id for g in MIXED_GEOM_ORDER], device=env.device)
  gsz = pool["geom_size"][rows]  # (n, 3 geoms: sphere, box, cylinder, 3)
  env.sim.model.geom_size[env_ids[:, None], gid[None, :], :] = gsz
  if tight_bounds:
    # the collision bounds (per env in mujoco_warp) follow the sizes: the shrunken geoms drop out of the broadphase
    # instead of sending every pad piece near the object into box / cylinder narrowphase against a 1 mm geom
    half = gsz.clone()
    half[:, 0] = gsz[:, 0, :1].expand(-1, 3)  # sphere: r, r, r
    half[:, 2] = torch.stack([gsz[:, 2, 0], gsz[:, 2, 0], gsz[:, 2, 1]], -1)  # cylinder: r, r, half height
    rb = torch.stack([gsz[:, 0, 0], gsz[:, 1].norm(dim=-1), gsz[:, 2, :2].norm(dim=-1)], -1)
    env.sim.model.geom_rbound[env_ids[:, None], gid[None, :]] = rb
    aabb = torch.zeros(len(env_ids), 3, 2, 3, device=env.device)
    aabb[:, :, 1] = half
    env.sim.model.geom_aabb[env_ids[:, None], gid[None, :]] = aabb
  body = int(cube.indexing.body_ids[0])
  mass = env.sim.model.body_mass[env_ids, body]
  shp, dims = pool["shape"][rows], pool["dims"][rows]
  inertia = torch.zeros(len(env_ids), 3, device=env.device)
  for k, name in enumerate(SHAPES):
    m = shp == k
    if m.any():
      inertia[m] = principal_inertia(name, dims[m], mass[m])
  env.sim.model.body_inertia[env_ids, body] = inertia
  _restore_hold(env, env_ids, st, pool, rows, tactile_gain, tactile_bias)
  st.shape[env_ids] = shp
  st.dims[env_ids] = dims
  st.extent[env_ids] = pool["extent"][rows]
  st.radius[env_ids] = pool["extent"][rows]
  st.stage = stage


def reset_gravity_curriculum(env, env_ids, palm: str, g0: float, ramp_steps: int):
  """Object weight curriculum: an upward world-frame force (1 - s) m g on the object, s ramping linearly from
  ``g0`` to 1 over ``ramp_steps`` control steps (env.common_step_counter). Exploring a regrasp (lifting a
  finger, letting the thumb slide) mostly drops a full-weight object; a light one forgives it. Applied per
  episode at reset (xfrc_applied persists until the next sim reset)."""
  st = ft(env, palm)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return
  s = min(1.0, g0 + (1.0 - g0) * env.common_step_counter / max(1, ramp_steps))
  st.g_scale = s
  cube = env.scene["cube"]
  body = int(cube.indexing.body_ids[0])
  m = env.sim.model.body_mass[env_ids, body]
  f = torch.zeros(len(env_ids), 1, 3, device=env.device)
  f[:, 0, 2] = (1.0 - s) * m * 9.81
  cube.write_external_wrench_to_sim(f, torch.zeros_like(f), env_ids=env_ids, body_ids=[0])


def random_object_push(env, env_ids, palm: str, scale: float, prob: float, decay: float = 0.9,
                       decay_interval: float = 0.08):
  """HORA's object perturbation (step event): each step the push decays by ``decay`` per ``decay_interval``
  seconds, and with probability ``prob`` an env gets a fresh push N(0, 1) * mass * ``scale`` per axis. Forces
  the policy to keep a grasp that survives disturbances instead of one balanced only for the nominal case."""
  st = ft(env, palm)
  cube = env.scene["cube"]
  body = int(cube.indexing.body_ids[0])
  st.push *= decay ** (env.step_dt / decay_interval)
  new = torch.rand(env.num_envs, device=env.device) < prob
  if new.any():
    m = env.sim.model.body_mass[:, body]
    st.push[new] = torch.randn(int(new.sum()), 3, device=env.device) * (m[new] * scale)[:, None]
  ids = torch.arange(env.num_envs, device=env.device)
  cube.write_external_wrench_to_sim(st.push[:, None, :], torch.zeros_like(st.push)[:, None, :], env_ids=ids, body_ids=[0])


def m_push(env, palm: str):
  return ft(env, palm).push.norm(dim=-1)


def obj_linvel_l1(env):
  return env.scene["cube"].data.root_link_lin_vel_w.abs().sum(-1)


def m_gravity_scale(env, palm: str):
  return torch.full((env.num_envs,), getattr(ft(env, palm), "g_scale", 1.0), device=env.device)


# --------------------------------------------------------------------------------------------
# rewards
def _drift(env, st):
  return (obj_pos_hand(env, st) - st.p0).norm(dim=-1)


def rot_vel_clipped(env, palm: str, clip_min: float, clip_max: float, drift_threshold: float = 0.02,
                    outside_factor: float = 0.1):
  """Yaw rate about the palm normal, clipped to [clip_min, clip_max] (slow, steady rotation), scaled
  down to ``outside_factor`` once the object drifted more than ``drift_threshold`` from its reset."""
  st = ft(env, palm)
  w = yaw_rate(env, st)
  st.yaw += w * env.step_dt  # the episode's accumulated yaw (this term runs once per step)
  r = w.clamp(clip_min, clip_max)
  gate = torch.where(_drift(env, st) > drift_threshold, outside_factor, 1.0)
  return r * gate


def _update_qyaw(env, st: FtState):
  """Advance the orientation-based net twist ``st.qyaw`` by this step's rotation about n (once per step)."""
  st.yaw += yaw_rate(env, st) * env.step_dt  # the logged / observed episode yaw, as with rot_vel_clipped
  q = env.scene["cube"].data.root_link_quat_w
  st.q_prev = torch.where(st.fresh[:, None], q, st.q_prev)
  st.qyaw += axis_angle_from_quat(quat_mul(q, quat_conjugate(st.q_prev))) @ st.n_w
  st.q_prev = q.clone()
  st.fresh[:] = False


def rot_progress_ratchet(env, palm: str, clip_max: float, drift_threshold: float = 0.02, outside_factor: float = 0.1,
                         **_):
  """New yaw beyond the episode's high-water mark per second, clipped at ``clip_max`` (from the object's
  orientation). The episode's sum is the high-water mark: rotation given back (a snap-back while regrasping) earns
  nothing until it is re-covered, so a back-and-forth pays only its net progress, and a still object nothing.
  Speed beyond the clip is wasted (the mark still advances). ``rot_vel_clipped`` charged a 500 deg/s snap-back
  only its clip. Same drift gate."""
  st = ft(env, palm)
  _update_qyaw(env, st)
  new = (st.qyaw - st.qyaw_max).clamp(min=0.0)
  st.qyaw_max = torch.maximum(st.qyaw_max, st.qyaw)
  r = (new / env.step_dt).clamp(max=clip_max)
  gate = torch.where(_drift(env, st) > drift_threshold, outside_factor, 1.0)
  return r * gate


def rot_progress_mix(env, palm: str, clip_min: float, clip_max: float, ramp_steps: int, drift_threshold: float = 0.02,
                     outside_factor: float = 0.1, ramp_delay: int = 0, **_):
  """(1 - a) x the clipped yaw rate + a x the ratchet, a ramping 0 -> 1 over ``ramp_steps`` control steps
  (env.common_step_counter, from this process's start) after ``ramp_delay`` steps of the rate alone (from scratch:
  the rate first finds a gait). Switching a wind-forward / snap-back gait straight to the ratchet (or to a full
  backward charge) made it park within ~100 iterations; ramping lets it adapt instead."""
  st = ft(env, palm)
  w = yaw_rate(env, st)
  _update_qyaw(env, st)
  new = (st.qyaw - st.qyaw_max).clamp(min=0.0)
  st.qyaw_max = torch.maximum(st.qyaw_max, st.qyaw)
  a = min(1.0, max(0.0, (env.common_step_counter - ramp_delay) / max(1, ramp_steps)))
  r = (1.0 - a) * w.clamp(clip_min, clip_max) + a * (new / env.step_dt).clamp(max=clip_max)
  gate = torch.where(_drift(env, st) > drift_threshold, outside_factor, 1.0)
  return r * gate


def rot_progress_window(env, palm: str, clip_min: float, clip_max: float, window_s: float = 2.0,
                        drift_threshold: float = 0.02, outside_factor: float = 0.1):
  """Net yaw about the palm normal over the last ``window_s`` per second, clipped, from the object's orientation.
  Summed over an episode it telescopes to the net yaw (up to the clip), so winding forward and letting the object
  snap back earns only the net progress, and a still object earns nothing, as long as the window's average stays
  inside the clip (a 2 s window under a 30 deg/s forward wind does not: prefer ``rot_progress_ratchet``).
  Same drift gate."""
  st = ft(env, palm)
  _update_qyaw(env, st)
  n = max(1, round(window_s / env.step_dt))
  if st.yhist is None or st.yhist.shape[1] != n:
    st.yhist = torch.zeros(env.num_envs, n, device=env.device)
    st.hptr = 0
  old = st.yhist[:, st.hptr].clone()
  st.yhist[:, st.hptr] = st.qyaw
  st.hptr = (st.hptr + 1) % n
  r = ((st.qyaw - old) / (n * env.step_dt)).clamp(clip_min, clip_max)
  gate = torch.where(_drift(env, st) > drift_threshold, outside_factor, 1.0)
  return r * gate


def angvel_perp_l2(env, palm: str):
  """Rotation about any axis but the palm normal (the object rolling / wobbling off-axis)."""
  st = ft(env, palm)
  w = obj_angvel_w(env)
  perp = w - (w @ st.n_w)[:, None] * st.n_w
  return perp.pow(2).sum(-1)


def yaw_rate_excess_sq(env, palm: str, free: float):
  st = ft(env, palm)
  return (yaw_rate(env, st).abs() - free).clamp(min=0).pow(2)


def obj_linvel_l2(env):
  return env.scene["cube"].data.root_link_lin_vel_w.pow(2).sum(-1)


def obj_drift_l2(env, palm: str):
  return _drift(env, ft(env, palm)).pow(2)


def pad_contact_frac(env, f_min: float):
  """Fraction of the four pads pressing the object: keep every PaXini pad in contact."""
  return (pad_forces_w(env).norm(dim=-1) > f_min).float().mean(-1)


def pads_beyond_two(env, f_min: float):
  """0 with two or fewer pads pressing, 0.5 with three, 1 with all four: rewards a third and fourth
  PaXini pad on the object (richer tactile signal) without paying for the basic thumb-finger pinch."""
  n = (pad_forces_w(env).norm(dim=-1) > f_min).float().sum(-1)
  return ((n - 2.0) / 2.0).clamp(0.0, 1.0)


def joint_limit_margin_sq(env, asset_cfg: SceneEntityCfg, margin: float):
  """sum over the selected joints of ((margin - distance to the nearer soft limit) / margin)^2 inside the margin.
  On the abduction joints: a stroke that sweeps a finger sideways until its joint runs out has to reset
  that finger (re-grasp) before the stop, instead of pinning it there and stalling the rotation."""
  robot = env.scene[asset_cfg.name]
  q = robot.data.joint_pos[:, asset_cfg.joint_ids]
  lim = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  d = torch.minimum(q - lim[..., 0], lim[..., 1] - q)
  return ((margin - d).clamp(min=0) / margin).pow(2).sum(-1)


def few_pads(env, f_min: float, n_min: int = 3):
  """1 while fewer than ``n_min`` pads press the object. A thumb-finger pinch (2 pads, opposed normals) cannot
  make the thumb slide (the finger's friction at best matches it), so the sphere only turns as far as the thumb
  travels; fingers spread around the sphere carry more normal force in total and can."""
  return ((pad_forces_w(env).norm(dim=-1) > f_min).float().sum(-1) < n_min).float()


def pad_force_excess(env, free: float, cap: float = 5.0):
  """Squeezing harder than ``free`` N on any pad (gentle contact, sensor-safe): linear and capped per
  pad, so contact spikes cannot dominate the return."""
  return (pad_forces_w(env).norm(dim=-1) - free).clamp(min=0, max=cap).sum(-1)


def nonpad_contact(env, f_min: float):
  """1 while anything but a pad (links, palm) touches the object."""
  return (nonpad_force(env) > f_min).float()


def normal_support(env, palm: str):
  """Share of the object's weight its contacts' NORMAL forces carry (world up; every hand geom touching it, pads and
  links): sum_k (F_k . n_k)+ n_k,z / (m g), n_k the sphere's inward normal at contact k (centre - contact point).
  ~0 for a pinch at the equator (friction holds it up), ~1 for a cradle (the object rests on contacts below its
  equator), < 0 for a pinch above it (normals push it down). A cradle stays up when friction is lower than in sim
  or a finger lifts to regrasp; a pinch does not. Clamped to [-1, 1.5]."""
  c = env.scene["cube"].data.root_link_pos_w
  up = torch.zeros(3, device=env.device)
  up[2] = 1.0
  num = torch.zeros(env.num_envs, device=env.device)
  for name in (PAD_SENSOR, NONPAD_SENSOR):
    d = env.scene.sensors[name].data
    F, P = d.force, d.pos
    live = F.norm(dim=-1) > 1e-4
    n = c[:, None] - P
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    fn = (F * n).sum(-1).clamp(min=0.0)
    num += torch.where(live, fn * (n @ up), 0.0).sum(-1)
  body = int(env.scene["cube"].indexing.body_ids[0])
  w = env.sim.model.body_mass[:, body] * 9.81
  return (num / w).clamp(-1.0, 1.5)


def normal_support_capped(env, palm: str):
  """``normal_support`` capped at 1 (the reward): carrying the full weight on normals is the goal; a harder squeeze
  from below (friction then pulls down, > 1) earns nothing more."""
  return normal_support(env, palm).clamp(max=1.0)


def support_obs(env, palm: str):
  return normal_support(env, palm)[:, None]


def palm_contact(env, f_min: float = 0.05):
  """1 while the palm touches the object (it must be carried by the fingers)."""
  return (env.scene.sensors[PALM_SENSOR].data.force.norm(dim=-1).amax(-1) > f_min).float()


def m_support(env, palm: str):
  return normal_support(env, palm)


def m_palm(env):
  return palm_contact(env)


def obj_dropped_penalty(env, palm: str, max_drift: float):
  st = ft(env, palm)
  d = _drift(env, st)
  return ((d > max_drift) | ~torch.isfinite(d)).float()


# --------------------------------------------------------------------------------------------
# terminations
def object_dropped(env, palm: str, max_drift: float):
  st = ft(env, palm)
  d = _drift(env, st)
  return (d > max_drift) | ~torch.isfinite(d)


# --------------------------------------------------------------------------------------------
# observations
def pad_tactile(env, palm: str, scale: float = 0.5, clip: float = 3.0, deadband: float = 0.05,
                randomize: bool = True):
  """(B, 12) each pad's net force in its own frame (normal + 2 shear), what a PaXini pad reports
  aggregated over its taxels. Per-episode gain / offset randomization (sensor calibration), a dead
  band (sensor threshold), scaled by ``scale`` per N and clipped."""
  st = ft(env, palm)
  f = pad_forces_local(env, st)
  if randomize:
    f = f * st.tac_gain[..., None] + st.tac_bias
  f = torch.where(f.abs() < deadband, torch.zeros_like(f), f)
  return (f * scale).clamp(-clip, clip).reshape(f.shape[0], -1)


def pad_contact_flags(env, f_min: float = 0.1):
  return (pad_forces_w(env).norm(dim=-1) > f_min).float()


def obj_pos_rel_hand(env, palm: str):
  st = ft(env, palm)
  return torch.cat([obj_pos_hand(env, st) - st.p0, st.p0], -1)


def obj_vel_hand(env, palm: str):
  st = ft(env, palm)
  c = env.scene["cube"].data
  return torch.cat([c.root_link_lin_vel_w @ st.R, c.root_link_ang_vel_w @ st.R], -1)


def obj_radius(env, palm: str):
  return ft(env, palm).radius[:, None] * 20.0  # ~0.5-0.6


def obj_shape_dims(env, palm: str):
  """(B, 7) the object of the episode: shape one-hot (sphere, cube, cylinder, cuboid) and geom size x 20 (~0.5-0.75)."""
  st = ft(env, palm)
  return torch.cat([torch.nn.functional.one_hot(st.shape, 4).float(), st.dims * 20.0], -1)


def pad_friction(env):
  """(B, 1) the pads' sliding friction this episode (randomized per env, ``dr_split_contact_friction``). The pads have
  contact priority 2, so this alone sets the pad-object friction (the object geom's own friction does not count)."""
  gid = getattr(env, "_ft_pad_gid", None)
  if gid is None:
    import mujoco

    m = env.sim.mj_model
    gid = env._ft_pad_gid = next(g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(
      ("if_tip_cvx0", "if_tip")))
  return env.sim.model.geom_friction[:, gid, :1]


def pads_low(env, palm: str, f_min: float = 0.1):
  """How low on the object the pads press, shape-agnostic: the pad-force-weighted mean of clip(-h, 0, 1), h the pad
  contact's height along world up above the object's centre in half extents (0 at the middle, -1 at the bottom).
  ~0.75 for the template's grip on the bottom quarter of the sides, ~0.5 for a sphere cradled below its equator,
  0 at or above the middle. 0 with no pad pressing. Gravity is world -z on both mounts."""
  st = ft(env, palm)
  sensor = env.scene.sensors[PAD_SENSOR]
  F = pad_net_force(sensor).norm(dim=-1)  # (B, 4)
  P = pad_contact_pos(sensor)  # (B, 4, 3) world
  c = env.scene["cube"].data.root_link_pos_w
  h = (P[..., 2] - c[:, None, 2]) / st.extent[:, None].clamp(min=1e-3)
  w = torch.where(F > f_min, F, torch.zeros_like(F))
  return (w * (-h).clamp(0.0, 1.0)).sum(-1) / w.sum(-1).clamp(min=1e-6)


def finger_idle(env, palm: str, idle_s: float = 1.0, f_min: float = 0.1):
  """Number of fingers whose pad has not pressed the object (> ``f_min`` N) for longer than ``idle_s``: all four
  fingers must take part. A finger may lift to step (the clock restarts at its next press); it may not sit out.
  Updates the per-finger clocks (``FtState.idle``), so it runs once per step as a reward term."""
  st = ft(env, palm)
  pressed = pad_net_force(env.scene.sensors[PAD_SENSOR]).norm(dim=-1) > f_min
  st.idle = torch.where(pressed, torch.zeros_like(st.idle), st.idle + env.step_dt)
  return (st.idle > idle_s).float().sum(-1)


def _link_columns(env):
  """Columns of the non-pad sensor that are finger links other than the fingertip (not ``*_ds_*``, not the palm)."""
  cols = getattr(env, "_ft_link_cols", None)
  if cols is None:
    names = list(env.scene.sensors[NONPAD_SENSOR].primary_names)
    cols = env._ft_link_cols = torch.tensor([i for i, n in enumerate(names) if "_ds_" not in n and not n.startswith("palm")],
                                            device=env.device)
  return cols


def link_contact(env, f_min: float = 0.2):
  """1 while a finger link other than the fingertip (base / proximal / middle phalanx) presses the object harder than
  ``f_min``: the rotation should be done with the fingertips (their pads, or their sides at times)."""
  F = env.scene.sensors[NONPAD_SENSOR].data.force.norm(dim=-1)[:, _link_columns(env)]
  return (F > f_min).any(-1).float()


def m_link_touching(env, f_min: float = 0.2):
  return link_contact(env, f_min)


def m_fingers_active(env, palm: str, idle_s: float = 1.0):
  return (ft(env, palm).idle <= idle_s).float().sum(-1)


def m_index_active(env, palm: str, idle_s: float = 1.0):
  return (ft(env, palm).idle[:, 0] <= idle_s).float()


def pads_low_obs(env, palm: str):
  return pads_low(env, palm)[:, None]


def m_pads_low(env, palm: str):
  return pads_low(env, palm)


def m_stage(env, palm: str):
  return torch.full((env.num_envs,), float(ft(env, palm).stage), device=env.device)


def pad_forces_true(env, palm: str):
  st = ft(env, palm)
  return (pad_forces_local(env, st) * 0.5).clamp(-5, 5).reshape(env.num_envs, -1)


def nonpad_force_obs(env):
  return nonpad_force(env).clamp(max=5.0)[:, None]


def episode_yaw(env, palm: str):
  return ft(env, palm).yaw[:, None]


def yaw_deficit(env, palm: str):
  """How far the object's net twist sits below the episode's high-water mark [rad] (critic only). The ratchet pays
  nothing until that gap is re-covered, so without it the critic cannot tell a state just after a snap-back from a
  normal one, and the snap's cost never reaches the advantage."""
  st = ft(env, palm)
  return (st.qyaw_max - st.qyaw).clamp(0.0, 3.0)[:, None]


# --------------------------------------------------------------------------------------------
# metrics (per-step values, averaged per episode by the metrics manager)
def m_yaw_rate(env, palm: str):
  return yaw_rate(env, ft(env, palm))


def m_episode_yaw(env, palm: str):
  return ft(env, palm).yaw


def m_drift(env, palm: str):
  return _drift(env, ft(env, palm))


def m_pads(env, f_min: float = 0.1):
  return (pad_forces_w(env).norm(dim=-1) > f_min).float().sum(-1)


def m_nonpad(env):
  return (nonpad_force(env) > 0.05).float()
