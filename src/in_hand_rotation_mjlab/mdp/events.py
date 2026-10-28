"""Robot-plant randomization events shared by every task: actuator gains, effort limits and
command delay, joint armature / friction / damping, body masses and centers of mass.

Selections use SceneEntityCfg (joint_names / actuator_names / body_names), so a
task can randomize one part of a robot, e.g. only the LEAP hand of the xArm7 composition.
The measured LEAP ranges are in robots/leap_hand/leap_plant.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_gaussian, sample_log_uniform, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "randomize_body_ipos",
  "randomize_body_mass",
  "randomize_dof_armature",
  "randomize_dof_damping",
  "randomize_dof_frictionloss",
  "randomize_field",
  "randomize_leap_gear_servo",
  "randomize_pd_gains",
  "set_actuator_effort_limits",
  "sync_actuator_delays",
]


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


def _resolve_joint_ids(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  if asset_cfg.joint_ids is None:
    raise ValueError("asset_cfg.joint_ids must be resolved.")
  if isinstance(asset_cfg.joint_ids, list):
    return torch.tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.int)
  if isinstance(asset_cfg.joint_ids, torch.Tensor):
    return asset_cfg.joint_ids.to(env.device, dtype=torch.int)
  if isinstance(asset_cfg.joint_ids, slice):
    entity: Entity = env.scene[asset_cfg.name]
    return entity.indexing.joint_ids[asset_cfg.joint_ids]
  return torch.tensor([asset_cfg.joint_ids], device=env.device, dtype=torch.int)


def _resolve_actuator_object_ids(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  num_actuators = len(asset.actuators)
  actuator_ids = asset_cfg.actuator_ids
  if isinstance(actuator_ids, list):
    return torch.tensor(actuator_ids, device=env.device, dtype=torch.long)
  if isinstance(actuator_ids, torch.Tensor):
    return actuator_ids.to(env.device, dtype=torch.long)
  if isinstance(actuator_ids, slice):
    return torch.arange(num_actuators, device=env.device, dtype=torch.long)[
      actuator_ids
    ]
  return torch.tensor([actuator_ids], device=env.device, dtype=torch.long)


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


def _resolve_model_field_ids(
  env: ManagerBasedRlEnv, field: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  if field.startswith("body_"):
    return _resolve_body_ids(env, asset_cfg).to(torch.long)
  if field.startswith("geom_"):
    return _resolve_geom_ids(env, asset_cfg).to(torch.long)
  if field.startswith("dof_") or field.startswith("jnt_"):
    return _resolve_joint_ids(env, asset_cfg).to(torch.long)
  if field.startswith("actuator_"):
    return _resolve_ctrl_ids(env, asset_cfg).to(torch.long)
  raise ValueError(f"Cannot infer indexed component type for model field '{field}'.")


def _default_for_selection(
  env: ManagerBasedRlEnv,
  field: str,
  env_ids: torch.Tensor,
  ids: torch.Tensor,
  ndim: int,
) -> torch.Tensor:
  default = env.sim.get_default_field(field)
  if field in env.sim.per_world_default_fields:
    selected = default[env_ids[:, None], ids[None, :]]
  else:
    selected = default[ids].unsqueeze(0).expand(len(env_ids), *default[ids].shape)
  if ndim == 2:
    return selected
  return selected


def _sample_like(
  ranges: tuple[float, float] | dict[int, tuple[float, float]],
  distribution: str,
  shape: torch.Size,
  device: str,
) -> torch.Tensor:
  if isinstance(ranges, dict):
    if len(shape) < 1:
      raise ValueError("dict ranges require an indexed trailing dimension.")
    out = torch.zeros(shape, device=device)
    for axis, axis_range in ranges.items():
      out[..., axis] = _sample_distribution(
        distribution,
        axis_range[0],
        axis_range[1],
        shape[:-1],
        device,
      )
    return out
  return _sample_distribution(
    distribution,
    ranges[0],
    ranges[1],
    tuple(shape),
    device,
  )


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def randomize_body_ipos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  ranges: dict[int, tuple[float, float]],
  distribution: str = "uniform",
  operation: str = "add",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=(".*",)),
) -> None:
  randomize_field(
    env,
    env_ids,
    field="body_ipos",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
  )


@requires_model_fields("dof_frictionloss")
def randomize_dof_frictionloss(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  ranges: tuple[float, float],
  distribution: str = "uniform",
  operation: str = "scale",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> None:
  randomize_field(
    env,
    env_ids,
    field="dof_frictionloss",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
  )


@requires_model_fields("dof_damping")
def randomize_dof_damping(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  ranges: tuple[float, float],
  distribution: str = "uniform",
  operation: str = "scale",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> None:
  randomize_field(
    env,
    env_ids,
    field="dof_damping",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
  )


@requires_model_fields("dof_armature", recompute=RecomputeLevel.set_const_0)
def randomize_dof_armature(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  ranges: tuple[float, float],
  distribution: str = "uniform",
  operation: str = "abs",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> None:
  randomize_field(
    env,
    env_ids,
    field="dof_armature",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
  )


def randomize_field(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  field: str,
  ranges: tuple[float, float] | dict[int, tuple[float, float]],
  distribution: str = "uniform",
  operation: str = "scale",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Randomize an expanded MuJoCo model field for selected entity components."""
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  ids = _resolve_model_field_ids(env, field, asset_cfg)
  if len(ids) == 0:
    return

  model_field = getattr(env.sim.model, field)
  if model_field.ndim == 2:
    current = model_field[env_ids[:, None], ids[None, :]]
  elif model_field.ndim == 3:
    current = model_field[env_ids[:, None], ids[None, :], :]
  else:
    raise RuntimeError(
      f"Unsupported model field shape for '{field}': {tuple(model_field.shape)}."
    )

  sample = _sample_like(ranges, distribution, current.shape, env.device)
  if operation == "abs":
    out = sample
  else:
    base = _default_for_selection(env, field, env_ids, ids, model_field.ndim)
    if operation == "scale":
      out = base * sample
    elif operation == "add":
      out = base + sample
    else:
      raise ValueError(f"Unsupported operation '{operation}'.")

  if model_field.ndim == 2:
    model_field[env_ids[:, None], ids[None, :]] = out
  else:
    model_field[env_ids[:, None], ids[None, :], :] = out


def randomize_pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  distribution: str = "uniform",
  operation: str = "scale",
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Randomize in-memory ideal PD actuator gains."""
  from mjlab.actuator import IdealPdActuator

  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  asset: Entity = env.scene[asset_cfg.name]
  selected = set(_resolve_actuator_object_ids(env, asset_cfg).tolist())

  for actuator_id, actuator in enumerate(asset.actuators):
    if actuator_id not in selected or not isinstance(actuator, IdealPdActuator):
      continue
    assert actuator.default_stiffness is not None
    assert actuator.default_damping is not None
    base_kp = actuator.default_stiffness[env_ids]
    base_kd = actuator.default_damping[env_ids]
    kp_sample = _sample_distribution(
      distribution, kp_range[0], kp_range[1], base_kp.shape, env.device
    )
    kd_sample = _sample_distribution(
      distribution, kd_range[0], kd_range[1], base_kd.shape, env.device
    )
    if operation == "scale":
      kp = base_kp * kp_sample
      kd = base_kd * kd_sample
    elif operation == "add":
      kp = base_kp + kp_sample
      kd = base_kd + kd_sample
    elif operation == "abs":
      kp = kp_sample
      kd = kd_sample
    else:
      raise ValueError(f"Unsupported operation '{operation}'.")
    actuator.set_gains(env_ids, kp=kp, kd=kd)


@requires_model_fields("dof_frictionloss")
def randomize_leap_gear_servo(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  torque_constant: tuple[float, float],
  gear_friction: tuple[float, float],
  current_gain_scale: tuple[float, float] = (1.0, 1.0),
  current_damping_scale: tuple[float, float] = (1.0, 1.0),
  dry_friction_scale: tuple[float, float] = (1.0, 1.0),
  lag: tuple[int, int] | None = None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  stiction_friction: tuple[float, float] | None = None,
) -> None:
  """Per-env parameters of the measured LEAP servo (robots/leap_hand/gear_servo.py), one sample
  per env shared by all its joints (one motor type): K_t and mu absolute, the rest as scales
  of the version's values; ``lag`` in physics steps. ``stiction_friction`` (N m, absolute; a
  version with stiction joints, gear_2026_10b on): one sample per env and stiction joint (each
  joint's own gear train), zero on the other joints; None leaves the version's value.

  Every task on that actuator needs this event, randomized or not: it is what expands
  dof_frictionloss per env, which the actuator rewrites every step.
  """
  from in_hand_rotation_mjlab.robots.leap_hand.gear_servo import LeapGearServoActuator

  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  asset: Entity = env.scene[asset_cfg.name]
  n = (len(env_ids), 1)
  draw = lambda r: sample_uniform(torch.tensor(r[0], device=env.device), torch.tensor(r[1], device=env.device), n,
                                  device=env.device)
  for actuator in asset.actuators:
    if not isinstance(actuator, LeapGearServoActuator):
      continue
    d = actuator.defaults
    actuator.torque_constant[env_ids] = draw(torque_constant).expand(-1, d["torque_constant"].shape[1])
    actuator.gear_friction[env_ids] = draw(gear_friction).expand(-1, d["gear_friction"].shape[1])
    actuator.current_gain[env_ids] = d["current_gain"][env_ids] * draw(current_gain_scale)
    actuator.current_damping[env_ids] = d["current_damping"][env_ids] * draw(current_damping_scale)
    actuator.dry_friction[env_ids] = d["dry_friction"][env_ids] * draw(dry_friction_scale)
    if stiction_friction is not None:
      nj = actuator.stiction_friction.shape[1]
      per_joint = sample_uniform(torch.tensor(stiction_friction[0], device=env.device),
                                 torch.tensor(stiction_friction[1], device=env.device), (len(env_ids), nj),
                                 device=env.device)
      actuator.stiction_friction[env_ids] = per_joint * actuator.stiction_joints
    if lag is not None and actuator.has_delay:
      actuator.set_lags(torch.randint(int(lag[0]), int(lag[1]) + 1, (len(env_ids),), device=env.device), env_ids)


def sync_actuator_delays(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  lag_range: tuple[int, int],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", actuator_names=(".*",)),
) -> None:
  """Sample and set command delay lags on configured actuators."""
  env_ids = _resolve_env_ids(env, env_ids)
  if len(env_ids) == 0:
    return
  asset: Entity = env.scene[asset_cfg.name]
  selected = set(_resolve_actuator_object_ids(env, asset_cfg).tolist())
  low, high = int(lag_range[0]), int(lag_range[1])
  for actuator_id, actuator in enumerate(asset.actuators):
    if actuator_id not in selected or not actuator.has_delay:
      continue
    lags = torch.randint(low, high + 1, (len(env_ids),), device=env.device)
    actuator.set_lags(lags, env_ids)


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
    if not isinstance(actuator, IdealPdActuator):
      continue

    act_ctrl_ids = [int(cid.item()) for cid in actuator.global_ctrl_ids]
    cols = [ctrl_index[cid] for cid in act_ctrl_ids if cid in ctrl_index]
    if len(cols) == 0:
      continue

    if forcerange.ndim == 3:
      act_limits = limits[:, cols]
    else:
      act_limits = limits[cols].unsqueeze(0).expand(len(env_ids), -1)
    actuator.set_effort_limit(env_ids, effort_limit=act_limits)
