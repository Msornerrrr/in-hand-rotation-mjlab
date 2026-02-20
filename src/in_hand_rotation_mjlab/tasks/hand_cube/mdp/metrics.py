from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi
from .numerics import sanitize_to_range

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_CUBE_CFG = SceneEntityCfg("cube")


def contact_found_mean(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Mean raw `found` value across sensor primaries for each environment."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.float().mean(dim=-1)


def contact_found_max(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Max raw `found` value across sensor primaries for each environment."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.float().amax(dim=-1)


def contact_fraction(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Fraction of primaries in contact in [0, 1] for each environment."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  return in_contact.mean(dim=-1)


def contact_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Number of primaries in contact for each environment."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  return in_contact.sum(dim=-1)


def object_linear_speed(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Object linear speed magnitude in world frame (m/s)."""
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
  return sanitize_to_range(speed, 0.0, 1e6, nan_default=0.0)


class object_rotation_progress:
  """Per-step task progress in [0, 1] from yaw speed and pose stability.

  Progress = yaw_score * position_score * tilt_score, where:
    yaw_score      encourages target yaw rotation direction/speed
    position_score penalizes xyz drift from reset
    tilt_score     penalizes roll/pitch drift from reset
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_CUBE_CFG)
    self._asset: Entity = env.scene[self._asset_cfg.name]
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)

    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
    target_yaw_rate: float = 0.20,
    position_threshold: float = 0.02,
    tilt_threshold: float = 0.35,
  ) -> torch.Tensor:
    del env, asset_cfg  # Bound once in __init__.

    # Positive score for target clockwise rotation direction in this task.
    yaw_rate = -self._asset.data.root_link_ang_vel_w[:, 2]
    yaw_score = torch.clamp(yaw_rate / max(target_yaw_rate, 1e-6), min=0.0, max=1.0)

    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )

    pos_score = torch.clamp(
      1.0 - pos_error / max(position_threshold, 1e-6), min=0.0, max=1.0
    )
    tilt_score = torch.clamp(
      1.0 - tilt_error / max(tilt_threshold, 1e-6), min=0.0, max=1.0
    )

    progress = sanitize_to_range(
      yaw_score * pos_score * tilt_score,
      0.0,
      1.0,
      nan_default=0.0,
    )
    return torch.where(self._has_init, progress, torch.zeros_like(progress))


class object_pose_rp_error_from_reset:
  """Object error from episode reset state for position or roll/pitch tilt."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_CUBE_CFG)
    self._component = cast(
      Literal["position", "tilt"], cfg.params.get("component", "position")
    )
    if self._component not in ("position", "tilt"):
      raise ValueError(
        f"Unknown component '{self._component}'. Expected 'position' or 'tilt'."
      )

    self._asset: Entity = env.scene[self._asset_cfg.name]
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)

    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    component: Literal["position", "tilt"] = "position",
    asset_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
  ) -> torch.Tensor:
    del env, component, asset_cfg  # Bound once in __init__.

    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)

    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )

    metric = pos_error if self._component == "position" else tilt_error
    metric = sanitize_to_range(metric, 0.0, 1e6, nan_default=0.0)
    return torch.where(self._has_init, metric, torch.zeros_like(metric))
