from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_gaussian,
  sample_log_uniform,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_CUBE_CFG = SceneEntityCfg("cube", geom_names=("cube_geom",))


def _resolve_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if isinstance(env_ids, slice):
    start, stop, step = env_ids.indices(env.num_envs)
    return torch.arange(start, stop, step, device=env.device, dtype=torch.int)
  return env_ids.to(env.device, dtype=torch.int)


def _resolve_geom_ids(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  if asset_cfg.geom_ids is None:
    raise ValueError("asset_cfg.geom_ids must be resolved.")
  if isinstance(asset_cfg.geom_ids, list):
    return torch.tensor(asset_cfg.geom_ids, device=env.device, dtype=torch.int)
  if isinstance(asset_cfg.geom_ids, torch.Tensor):
    return asset_cfg.geom_ids.to(env.device, dtype=torch.int)
  if isinstance(asset_cfg.geom_ids, slice):
    entity: Entity = env.scene[asset_cfg.name]
    return entity.indexing.geom_ids[asset_cfg.geom_ids]
  return torch.tensor([asset_cfg.geom_ids], device=env.device, dtype=torch.int)


def _resolve_body_ids(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  if asset_cfg.body_ids is None:
    raise ValueError("asset_cfg.body_ids must be resolved.")
  if isinstance(asset_cfg.body_ids, list):
    return torch.tensor(asset_cfg.body_ids, device=env.device, dtype=torch.int)
  if isinstance(asset_cfg.body_ids, torch.Tensor):
    return asset_cfg.body_ids.to(env.device, dtype=torch.int)
  if isinstance(asset_cfg.body_ids, slice):
    entity: Entity = env.scene[asset_cfg.name]
    return entity.indexing.body_ids[asset_cfg.body_ids]
  return torch.tensor([asset_cfg.body_ids], device=env.device, dtype=torch.int)


def _resolve_ctrl_ids(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  raw_ctrl_ids = asset.indexing.ctrl_ids
  if raw_ctrl_ids is None:
    raise ValueError(f"Entity '{asset_cfg.name}' has no actuators.")
  ctrl_ids = raw_ctrl_ids.to(env.device, dtype=torch.long)

  def _select_from_ctrl_ids(ids: torch.Tensor) -> torch.Tensor | None:
    if ids.numel() == 0:
      return ctrl_ids.new_empty((0,), dtype=torch.long)
    if bool(torch.all((ids >= -ctrl_ids.numel()) & (ids < ctrl_ids.numel()))):
      return ctrl_ids[ids]
    return None

  actuator_ids = asset_cfg.actuator_ids
  if isinstance(actuator_ids, slice):
    return ctrl_ids[actuator_ids]
  if isinstance(actuator_ids, list):
    actuator_ids_t = torch.tensor(actuator_ids, device=env.device, dtype=torch.long)
  elif isinstance(actuator_ids, torch.Tensor):
    actuator_ids_t = actuator_ids.to(env.device, dtype=torch.long)
  else:
    actuator_ids_t = torch.tensor([actuator_ids], device=env.device, dtype=torch.long)

  selected = _select_from_ctrl_ids(actuator_ids_t)
  if selected is not None:
    return selected

  # Fallback for grouped/wrapped actuator configurations: resolve through
  # entity actuator objects and collect their MuJoCo control IDs.
  num_actuators = len(asset.actuators)
  if num_actuators > 0:
    valid = (actuator_ids_t >= -num_actuators) & (actuator_ids_t < num_actuators)
    if bool(torch.any(valid)):
      obj_ids = actuator_ids_t[valid]
      obj_ids = torch.where(obj_ids < 0, obj_ids + num_actuators, obj_ids)
      gathered = []
      for idx in obj_ids.tolist():
        gathered.append(asset.actuators[int(idx)].global_ctrl_ids.to(env.device))
      if len(gathered) > 0:
        return torch.unique(torch.cat(gathered).to(torch.long), sorted=True)

  return ctrl_ids


def _sample_distribution(
  distribution: str,
  lower: float,
  upper: float,
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  low = torch.tensor(lower, device=device, dtype=torch.float32)
  high = torch.tensor(upper, device=device, dtype=torch.float32)
  if distribution == "uniform":
    return sample_uniform(low, high, shape, device=device)
  if distribution == "log_uniform":
    return sample_log_uniform(low, high, shape, device=device)
  if distribution == "gaussian":
    return sample_gaussian(low, high, shape, device=device)
  raise ValueError(f"Unsupported distribution: {distribution}")


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


@requires_model_fields("body_mass")
def randomize_body_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  mass_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  distribution: str = "uniform",
  operation: str = "scale",
  min_mass: float = 1e-5,
) -> None:
  """Randomize body masses for selected asset bodies.

  Supports ``operation`` in ``{"scale", "abs", "add"}``.
  """
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  body_ids = _resolve_body_ids(env, asset_cfg)
  if len(body_ids) == 0:
    return

  current = env.sim.model.body_mass[env_ids[:, None], body_ids[None, :]]
  if operation == "scale":
    base = env.sim.get_default_field("body_mass")[body_ids].unsqueeze(0).expand_as(current)
    scale = _sample_distribution(
      distribution=distribution,
      lower=mass_range[0],
      upper=mass_range[1],
      shape=current.shape,
      device=env.device,
    )
    out = base * scale
  elif operation == "abs":
    out = _sample_distribution(
      distribution=distribution,
      lower=mass_range[0],
      upper=mass_range[1],
      shape=current.shape,
      device=env.device,
    )
  elif operation == "add":
    base = env.sim.get_default_field("body_mass")[body_ids].unsqueeze(0).expand_as(current)
    delta = _sample_distribution(
      distribution=distribution,
      lower=mass_range[0],
      upper=mass_range[1],
      shape=current.shape,
      device=env.device,
    )
    out = base + delta
  else:
    raise ValueError(f"Unsupported operation '{operation}'.")

  env.sim.model.body_mass[env_ids[:, None], body_ids[None, :]] = torch.clamp(
    out, min=min_mass
  )


@requires_model_fields("actuator_forcerange", "actuator_forcelimited")
def set_actuator_effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  effort_limit: float | tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", actuator_names=(".*",)),
) -> None:
  """Set actuator effort limits and enable force limiting in MuJoCo."""
  from mjlab.actuator import IdealPdActuator
  from mjlab.actuator.delayed_actuator import DelayedActuator

  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  asset: Entity = env.scene[asset_cfg.name]
  ctrl_ids = _resolve_ctrl_ids(env, asset_cfg)
  if len(ctrl_ids) == 0:
    return

  forcerange = env.sim.model.actuator_forcerange
  forcelimited = env.sim.model.actuator_forcelimited

  if forcerange.ndim == 3:
    max_ctrl_id = int(forcerange.shape[1])
  elif forcerange.ndim == 2:
    max_ctrl_id = int(forcerange.shape[0])
  else:
    raise RuntimeError(
      f"Unsupported actuator_forcerange shape {tuple(forcerange.shape)}; expected 2D or 3D."
    )

  valid_ctrl = (ctrl_ids >= 0) & (ctrl_ids < max_ctrl_id)
  if not bool(torch.all(valid_ctrl)):
    ctrl_ids = ctrl_ids[valid_ctrl]
  if len(ctrl_ids) == 0:
    return

  if forcerange.ndim == 3:
    if isinstance(effort_limit, tuple):
      limits = sample_uniform(
        effort_limit[0],
        effort_limit[1],
        (len(env_ids), len(ctrl_ids)),
        device=env.device,
      )
    else:
      limits = torch.full(
        (len(env_ids), len(ctrl_ids)),
        float(effort_limit),
        dtype=torch.float32,
        device=env.device,
      )
    limits = torch.clamp(limits, min=1e-6)
    forcerange[env_ids[:, None], ctrl_ids[None, :], 0] = -limits
    forcerange[env_ids[:, None], ctrl_ids[None, :], 1] = limits
  elif forcerange.ndim == 2:
    if isinstance(effort_limit, tuple):
      limits = sample_uniform(
        effort_limit[0],
        effort_limit[1],
        (len(ctrl_ids),),
        device=env.device,
      )
    else:
      limits = torch.full(
        (len(ctrl_ids),),
        float(effort_limit),
        dtype=torch.float32,
        device=env.device,
      )
    limits = torch.clamp(limits, min=1e-6)
    forcerange[ctrl_ids, 0] = -limits
    forcerange[ctrl_ids, 1] = limits
  else:
    raise RuntimeError(
      f"Unsupported actuator_forcerange shape {tuple(forcerange.shape)}; expected 2D or 3D."
    )

  # actuator_forcelimited is often shared across envs (shape: [nu]) even when
  # actuator_forcerange is expanded per-env.
  if forcelimited.ndim == 1:
    max_force_limited = int(forcelimited.shape[0])
    valid_force_limited = (ctrl_ids >= 0) & (ctrl_ids < max_force_limited)
    if bool(torch.any(valid_force_limited)):
      forcelimited[ctrl_ids[valid_force_limited]] = 1
  elif forcelimited.ndim == 2:
    max_force_limited = int(forcelimited.shape[1])
    valid_force_limited = (ctrl_ids >= 0) & (ctrl_ids < max_force_limited)
    if bool(torch.any(valid_force_limited)):
      valid_ctrl_ids = ctrl_ids[valid_force_limited]
      forcelimited[env_ids[:, None], valid_ctrl_ids[None, :]] = 1
  else:
    raise RuntimeError(
      f"Unsupported actuator_forcelimited shape {tuple(forcelimited.shape)}; expected 1D or 2D."
    )

  # Keep custom IdealPd actuator internal limits in sync with model forcerange.
  # Entity-level actuator objects may be grouped (e.g., one object controlling
  # multiple MuJoCo actuators), so we match by ctrl-id intersection instead of
  # indexing asset.actuators with manager-resolved actuator IDs.
  actuators = list(asset.actuators)

  ctrl_index = {int(cid.item()): i for i, cid in enumerate(ctrl_ids)}
  for actuator in actuators:
    base_actuator = (
      actuator.base_actuator if isinstance(actuator, DelayedActuator) else actuator
    )
    if not isinstance(base_actuator, IdealPdActuator):
      continue

    act_ctrl_ids = [int(cid.item()) for cid in base_actuator.global_ctrl_ids]
    cols = [ctrl_index[cid] for cid in act_ctrl_ids if cid in ctrl_index]
    if len(cols) == 0:
      continue

    if forcerange.ndim == 3:
      act_limits = limits[:, cols]
    else:
      act_limits = limits[cols].unsqueeze(0).expand(len(env_ids), -1)
    base_actuator.set_effort_limit(env_ids, effort_limit=act_limits)


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
