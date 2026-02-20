from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_CUBE_CFG = SceneEntityCfg("cube")


def object_linear_speed_above(
  env: ManagerBasedRlEnv,
  max_linear_speed: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Terminate when object linear speed exceeds a threshold."""
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
  return speed > max_linear_speed


class object_pose_rp_position_deviation_from_reset:
  """Terminate when object drifts from episode-initial pose (pos + roll/pitch)."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self._env = env
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
    max_position_error: float,
    max_tilt_error: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
  ) -> torch.Tensor:
    del env, asset_cfg  # Bound once in __init__.

    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )
    return self._has_init & (
      (pos_error > max_position_error) | (tilt_error > max_tilt_error)
    )
