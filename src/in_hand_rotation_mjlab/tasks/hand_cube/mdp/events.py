from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import re
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

from in_hand_rotation_mjlab.mdp.events import _resolve_env_ids, _resolve_geom_ids

from .commands import GOAL

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_CUBE_CFG = SceneEntityCfg("cube", geom_names=("cube_geom",))


def _load_grasp_cache(
  env: ManagerBasedRlEnv,
  cache_file: str,
) -> dict[str, torch.Tensor] | None:
  cache_path = str(Path(cache_file).expanduser())
  cached_path = getattr(env, "_hand_cube_grasp_cache_path", None)
  cached_obj = getattr(env, "_hand_cube_grasp_cache", None)
  if cached_obj is not None and cached_path == cache_path:
    return cached_obj

  if not Path(cache_path).exists():
    if not getattr(env, "_hand_cube_grasp_cache_missing_warned", False):
      print(
        f"[hand_cube] grasp cache missing at '{cache_path}', using default reset events."
      )
      setattr(env, "_hand_cube_grasp_cache_missing_warned", True)
    setattr(env, "_hand_cube_grasp_cache", None)
    setattr(env, "_hand_cube_grasp_cache_path", cache_path)
    return None

  with np.load(cache_path, allow_pickle=False) as data:
    size_key = next(
      (
        k
        for k in (
          "cube_size",
          "cube_half_size",
          "cube_size_half",
          "size",
          "sizes",
        )
        if k in data
      ),
      None,
    )
    if size_key is None:
      raise ValueError(
        f"Grasp cache '{cache_path}' missing cube size key. "
        "Expected one of: cube_size/cube_half_size/size/sizes."
      )

    if "cube_pose_rel" in data:
      cube_pose_rel = data["cube_pose_rel"]
    elif "cube_pose" in data:
      cube_pose_rel = data["cube_pose"]
    elif "cube_pos_rel" in data and "cube_quat" in data:
      cube_pose_rel = np.concatenate([data["cube_pos_rel"], data["cube_quat"]], axis=-1)
    else:
      raise ValueError(
        f"Grasp cache '{cache_path}' missing cube pose key. "
        "Expected cube_pose_rel or cube_pose (Nx7)."
      )

    sizes = np.asarray(data[size_key], dtype=np.float32).reshape(-1)
    cube_pose_rel = np.asarray(cube_pose_rel, dtype=np.float32)

  if cube_pose_rel.ndim != 2 or cube_pose_rel.shape[1] != 7:
    raise ValueError(f"cube pose must be shape (N, 7), got {cube_pose_rel.shape}")
  n = sizes.shape[0]
  if cube_pose_rel.shape[0] != n:
    raise ValueError(
      "Grasp cache arrays must have same first dimension: "
      f"sizes={sizes.shape}, cube_pose={cube_pose_rel.shape}"
    )

  cache = {
    "cube_size": torch.from_numpy(sizes).to(device=env.device),
    "cube_pose_rel": torch.from_numpy(cube_pose_rel).to(device=env.device),
  }
  setattr(env, "_hand_cube_grasp_cache", cache)
  setattr(env, "_hand_cube_grasp_cache_path", cache_path)
  print(f"[hand_cube] loaded grasp cache '{cache_path}' with {n} entries")
  return cache


@requires_model_fields("geom_size")
def randomize_cube_size(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  size_scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> None:
  """Randomize cube geom size per environment using isotropic scale.

  This is intended for startup/reset events. When used with ``mode="startup"``,
  each parallel env gets one persistent cube size for the full run.
  """
  env_ids = _resolve_env_ids(env, env_ids)
  geom_ids = _resolve_geom_ids(env, asset_cfg)

  default_geom_size = env.sim.get_default_field("geom_size")[geom_ids]
  default_geom_size = default_geom_size.unsqueeze(0).expand(len(env_ids), -1, -1)

  scale = sample_uniform(
    size_scale_range[0],
    size_scale_range[1],
    (len(env_ids), 1, 1),
    device=env.device,
  )
  env.sim.model.geom_size[env_ids[:, None], geom_ids[None, :], :] = (
    default_geom_size * scale
  )


@requires_model_fields("geom_size")
def reset_from_grasp_cache(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  cache_file: str,
  pose_range: dict[str, tuple[float, float]] | None = None,
  scale_list: tuple[float, ...] | list[float] | None = None,
  scale_jitter: float = 0.025,
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> None:
  """Reset cube size+pose from cache.

  The cache must contain: ``cube_size`` and ``cube_pose_rel`` (or ``cube_pose``),
  where pose is relative to env origin. Robot joints are not touched.
  """
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return

  cache = _load_grasp_cache(env, cache_file)
  if cache is None:
    # Missing cache: leave default reset events in control.
    return

  cube: Entity = env.scene[cube_cfg.name]

  cache_sizes = cache["cube_size"]
  cache_cube_pose_rel = cache["cube_pose_rel"]
  if cache_sizes.numel() == 0:
    raise ValueError("Grasp cache is empty.")

  geom_ids = _resolve_geom_ids(env, cube_cfg)
  if scale_list is not None and len(scale_list) > 0:
    scales = torch.tensor(scale_list, device=env.device, dtype=cache_sizes.dtype)
    env_bucket = env_ids.to(torch.long) % len(scales)
    base_scale = scales[env_bucket]
    sampled_scale = sample_uniform(
      base_scale - scale_jitter,
      base_scale + scale_jitter,
      (len(env_ids),),
      device=env.device,
    )

    default_geom_size = env.sim.get_default_field("geom_size")[geom_ids, 0].mean()
    selected_sizes = sampled_scale * default_geom_size

    # Pick nearest cache pose for sampled size using sorted search.
    sorted_sizes, sorted_order = torch.sort(cache_sizes)
    insert_idx = torch.searchsorted(sorted_sizes, selected_sizes)
    left_idx = torch.clamp(insert_idx - 1, min=0)
    right_idx = torch.clamp(insert_idx, max=sorted_sizes.numel() - 1)
    left_diff = torch.abs(sorted_sizes[left_idx] - selected_sizes)
    right_diff = torch.abs(sorted_sizes[right_idx] - selected_sizes)
    nearest_sorted_idx = torch.where(left_diff <= right_diff, left_idx, right_idx)
    chosen_cache_ids = sorted_order[nearest_sorted_idx]
  else:
    chosen_cache_ids = torch.randint(
      cache_sizes.numel(), (len(env_ids),), device=env.device
    )
    selected_sizes = cache_sizes[chosen_cache_ids]

  # Set per-env cube half-size. Applies isotropically to all target geoms.
  selected_geom_sizes = selected_sizes[:, None, None].expand(-1, len(geom_ids), 3)
  env.sim.model.geom_size[env_ids[:, None], geom_ids[None, :], :] = selected_geom_sizes

  cube_pose_rel = cache_cube_pose_rel[chosen_cache_ids].clone()
  cube_pose_w = cube_pose_rel.clone()
  cube_pose_w[:, 0:3] += env.scene.env_origins[env_ids]

  if pose_range is None:
    pose_range = {}
  pos_delta = torch.stack(
    [
      sample_uniform(*pose_range.get("x", (0.0, 0.0)), len(env_ids), device=env.device),
      sample_uniform(*pose_range.get("y", (0.0, 0.0)), len(env_ids), device=env.device),
      sample_uniform(*pose_range.get("z", (0.0, 0.0)), len(env_ids), device=env.device),
    ],
    dim=-1,
  )
  cube_pose_w[:, 0:3] += pos_delta

  roll_delta = sample_uniform(
    *pose_range.get("roll", (0.0, 0.0)),
    len(env_ids),
    device=env.device,
  )
  pitch_delta = sample_uniform(
    *pose_range.get("pitch", (0.0, 0.0)),
    len(env_ids),
    device=env.device,
  )
  yaw_delta = sample_uniform(
    *pose_range.get("yaw", (0.0, 0.0)),
    len(env_ids),
    device=env.device,
  )
  delta_quat = quat_from_euler_xyz(roll_delta, pitch_delta, yaw_delta)
  cube_pose_w[:, 3:7] = quat_mul(cube_pose_w[:, 3:7], delta_quat)

  cube.write_root_link_pose_to_sim(cube_pose_w, env_ids=env_ids)
  cube.write_root_link_velocity_to_sim(
    torch.zeros((len(env_ids), 6), device=env.device),
    env_ids=env_ids,
  )


@requires_model_fields("geom_friction")
def randomize_shared_contact_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  friction_range: tuple[float, float],
  hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", geom_names=(".*",)),
  cube_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
  axes: tuple[int, ...] = (0,),
) -> None:
  """Randomize contact friction with one shared sample per env.

  Applies the same sampled friction value to all selected hand and cube geoms
  for each environment, similar to LEAPSim's hand/object shared-friction
  randomization.
  """
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return

  hand_geom_ids = _resolve_geom_ids(env, hand_cfg)
  cube_geom_ids = _resolve_geom_ids(env, cube_cfg)
  if len(hand_geom_ids) == 0 or len(cube_geom_ids) == 0:
    return

  friction = sample_uniform(
    friction_range[0],
    friction_range[1],
    (len(env_ids), 1),
    device=env.device,
  )

  for axis in axes:
    env.sim.model.geom_friction[env_ids[:, None], hand_geom_ids[None, :], axis] = (
      friction
    )
    env.sim.model.geom_friction[env_ids[:, None], cube_geom_ids[None, :], axis] = (
      friction
    )


def inject_random_cube_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
  position_noise_m: float = 0.02,
  rotation_noise_rad: float = 0.1,
  probability: float = 0.1,
) -> None:
  """Occasionally inject random pose offsets to mimic estimator glitches."""
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return

  prob = float(torch.clamp(torch.tensor(probability), min=0.0, max=1.0).item())
  if prob <= 0.0:
    return
  if prob < 1.0:
    valid = torch.rand(len(env_ids), device=env.device) < prob
    env_ids = env_ids[valid]
    if len(env_ids) == 0:
      return

  asset: Entity = env.scene[asset_cfg.name]
  pose_w = asset.data.root_link_pose_w[env_ids].clone()
  pos_delta = sample_uniform(
    -position_noise_m, position_noise_m, (len(env_ids), 3), device=env.device
  )
  euler_delta = sample_uniform(
    -rotation_noise_rad, rotation_noise_rad, (len(env_ids), 3), device=env.device
  )
  quat_delta = quat_from_euler_xyz(
    euler_delta[:, 0], euler_delta[:, 1], euler_delta[:, 2]
  )
  pose_w[:, 0:3] += pos_delta
  pose_w[:, 3:7] = quat_mul(pose_w[:, 3:7], quat_delta)
  asset.write_root_link_pose_to_sim(pose_w, env_ids=env_ids)


def apply_random_cube_wrench(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("cube", body_names=("cube",)),
  force_abs_max: float = 0.0,
  torque_abs_max: float = 0.0,
  probability: float = 1.0,
) -> None:
  """Apply random external force/torque perturbations to cube bodies."""
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  if force_abs_max <= 0.0 and torque_abs_max <= 0.0:
    return

  prob = float(torch.clamp(torch.tensor(probability), min=0.0, max=1.0).item())
  if prob <= 0.0:
    return
  if prob < 1.0:
    valid = torch.rand(len(env_ids), device=env.device) < prob
    env_ids = env_ids[valid]
    if len(env_ids) == 0:
      return

  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  if isinstance(body_ids, slice):
    local_ids = torch.arange(asset.num_bodies, device=env.device)[body_ids]
  elif isinstance(body_ids, torch.Tensor):
    local_ids = body_ids.to(env.device, dtype=torch.long)
  elif isinstance(body_ids, list):
    local_ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)
  else:
    local_ids = torch.tensor([body_ids], device=env.device, dtype=torch.long)

  size = (len(env_ids), len(local_ids), 3)
  force = sample_uniform(-force_abs_max, force_abs_max, size, device=env.device)
  torque = sample_uniform(-torque_abs_max, torque_abs_max, size, device=env.device)
  asset.write_external_wrench_to_sim(
    forces=force,
    torques=torque,
    body_ids=local_ids.tolist(),
    env_ids=env_ids,
  )

# ---------------------------------------------------------------------------
# SO(3) cube reorientation (reorient_env_cfg.py)
@requires_model_fields("geom_friction")
def randomize_split_contact_friction(env, env_ids, pad_range, other_range):
  """Sliding friction per surface class: the fingertip pads (``robot/*_tip``) get one sample,
  every other hand geom and the cube share another (MuJoCo mixes same-priority geoms by max;
  the pads, priority 2, win theirs). Geoms are resolved by their compiled names: the entity's
  ``geom_names`` selector maps the tip meshes one geom off."""
  import mujoco

  from in_hand_rotation_mjlab.robots.leap_hand.pad_model import is_pad_geom

  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  ids = getattr(env, "_split_friction_ids", None)
  if ids is None:
    m = env.sim.mj_model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    hand = [i for i, n in enumerate(names) if n.startswith("robot/")]
    # the tip meshes, or their convex pieces under a decomposed pad model (robots/leap_hand/pad_model.py)
    pads = [i for i in hand if is_pad_geom(names[i])]
    # the object's geoms: "cube/cube_geom", plus the box / cylinder geoms of the multi-shape object
    other = [i for i in hand if i not in pads] + [i for i, n in enumerate(names) if re.fullmatch(r"cube/cube_geom(_\w+)?", n)]
    assert len(pads) % 4 == 0 and pads, pads
    ids = env._split_friction_ids = (torch.tensor(pads, device=env.device), torch.tensor(other, device=env.device))
  pads, other = ids
  n = len(env_ids)
  f_other = other_range[0] + (other_range[1] - other_range[0]) * torch.rand(n, 1, device=env.device)
  f_pad = pad_range[0] + (pad_range[1] - pad_range[0]) * torch.rand(n, 1, device=env.device)
  env.sim.model.geom_friction[env_ids[:, None], other[None, :], 0] = f_other
  env.sim.model.geom_friction[env_ids[:, None], pads[None, :], 0] = f_pad


def reset_cube_freeze(env, env_ids, p_fh: tuple[float, float], p_hh: tuple[float, float]):
  B, dev = env.num_envs, env.device
  if not hasattr(env, "_cube_freeze"):
    env._cube_freeze = dict(step=-1, held=torch.zeros(B, dtype=torch.bool, device=dev),
                            p_fh=torch.zeros(B, device=dev), p_hh=torch.zeros(B, device=dev),
                            pos=torch.zeros(B, 3, device=dev), quat=torch.zeros(B, 4, device=dev))
    env._cube_freeze["quat"][:, 0] = 1.0
  st = env._cube_freeze
  if env_ids is None:
    env_ids = torch.arange(B, device=dev)
  n = len(env_ids)
  st["held"][env_ids] = False  # a fresh estimate first; the chain may freeze it from the next step
  st["p_fh"][env_ids] = p_fh[0] + (p_fh[1] - p_fh[0]) * torch.rand(n, device=dev)
  st["p_hh"][env_ids] = p_hh[0] + (p_hh[1] - p_hh[0]) * torch.rand(n, device=dev)


# ---------------------------------------------------------------------------
# Reverse curriculum from the sampling-MPC expert: start an episode from an expert state,
# with the goal the expert was pursuing there and reached ``sts`` steps later.
_RC_POOL: dict = {}


def _rc_pool(pattern: str, device):
  if pattern not in _RC_POOL:
    import glob
    import numpy as np

    keys = ("joint_pos", "cmd", "cube_pos", "cube_quat", "goal_quat")
    keep = {k: [] for k in keys + ("sts",)}
    files = sorted(f for g in pattern.split(";") for f in glob.glob(g))
    for f in files:
      d = np.load(f)
      alive, succ, gid = d["alive"], d["success"], d["goal_id"]
      T, M = alive.shape
      by_student = d["by_student"] if "by_student" in d.files else np.zeros_like(alive)
      sts = np.full((T, M), np.inf, dtype=np.float32)
      last = np.full(M, -1)
      for t in range(T - 1, -1, -1):
        # success[t]: the step taken from row t reached the goal of row t; a goal change
        # after row t without success is a timeout (the expert never reached that goal)
        changed = np.zeros(M, dtype=bool) if t == T - 1 else gid[t + 1] != gid[t]
        last = np.where(succ[t], t, np.where(changed, -1, last))
        sts[t] = np.where(last >= 0, last - t + 1, np.inf)
      ok = alive & np.isfinite(sts) & ~by_student.astype(bool)
      for k in keys:
        keep[k].append(d[k][ok])
      keep["sts"].append(sts[ok])
    if not files or not sum(len(v) for v in keep["sts"]):
      raise FileNotFoundError(f"no usable demo shards match {pattern}")
    pool = {k: torch.as_tensor(np.concatenate(v), dtype=torch.float32, device=device) for k, v in keep.items()}
    order = pool["sts"].argsort()
    _RC_POOL[pattern] = {k: v[order] for k, v in pool.items()}
    print(f"[reverse curriculum] {len(order)} expert states from {len(files)} shards; "
          f"steps-to-goal median {float(pool['sts'].median()):.0f}, max {float(pool['sts'].max()):.0f}", flush=True)
  return _RC_POOL[pattern]


def reset_reverse_curriculum(env, env_ids, shard_glob: str, prob: float, cube_cfg: SceneEntityCfg):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if not hasattr(env, "_rc_pending"):
    env._rc_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._rc_goal = torch.zeros(env.num_envs, 4, device=env.device)
  if len(env_ids) == 0:
    return
  pick = torch.rand(len(env_ids), device=env.device) < prob
  env._rc_pending[env_ids] = pick
  ids = env_ids[pick]
  if len(ids) == 0:
    return
  pool = _rc_pool(shard_glob, env.device)
  try:
    rc_steps = env.command_manager.get_term(GOAL).rc_steps
  except Exception:  # managers not built yet (first reset during construction)
    rc_steps = 10.0
  n_ok = max(1, int(torch.searchsorted(pool["sts"], torch.tensor(rc_steps, device=env.device), right=True)))
  rows = torch.randint(n_ok, (len(ids),), device=env.device)
  robot, cube = env.scene["robot"], env.scene["cube"]
  robot.write_joint_state_to_sim(pool["joint_pos"][rows], torch.zeros_like(pool["joint_pos"][rows]), env_ids=ids)
  state = torch.zeros(len(ids), 13, device=env.device)
  state[:, :3] = pool["cube_pos"][rows] + env.scene.env_origins[ids]  # demos were recorded with zero env spacing
  state[:, 3:7] = pool["cube_quat"][rows]
  cube.write_root_state_to_sim(state, env_ids=ids)
  # the demos used the nominal cube; undo the grasp-cache event's size randomization
  geom_ids = _resolve_geom_ids(env, cube_cfg)
  nominal = env.sim.get_default_field("geom_size")[geom_ids]
  env.sim.model.geom_size[ids[:, None], geom_ids[None, :], :] = nominal[None]
  term = env.action_manager.get_term("joint_pos")
  term._target[ids] = pool["cmd"][rows]
  term.stage_restored_reset_target(ids)
  env._rc_goal[ids] = pool["goal_quat"][rows]
