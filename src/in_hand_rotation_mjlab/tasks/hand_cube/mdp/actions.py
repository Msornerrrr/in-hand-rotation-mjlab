"""
Our custom action space for LEAP Hand
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseActionCfg, BaseAction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class JointPositionDeltaActionCfg(BaseActionCfg):
  """Configuration for normalized command-space delta joint position control.

  The policy action is interpreted as normalized input in [-1, 1], mapped to a
  per-step delta range, and integrated against the previous command:
    delta = map_to_delta_range(action_norm)
    target_t = target_{t-1} + delta
  """

  use_default_offset: bool = False
  """If True, initialize target from default joint positions on reset."""

  clip_to_joint_limits: bool = True
  """If True, clamp targets to joint limits after each update."""

  use_soft_joint_pos_limits: bool = True
  """If True, clamp with soft limits. Otherwise clamp with hard limits."""

  delta_min: float = -0.03
  """Minimum command delta (radians) when action is -1."""

  delta_max: float = 0.03
  """Maximum command delta (radians) when action is +1."""

  interpolate_decimation: bool = False
  """If True, linearly interpolate the position target across decimation
  substeps instead of applying a constant target. This ramps the setpoint
  from the previous env-step target to the current one, producing smoother
  torques and reducing sim-to-real jerkiness."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointPositionDeltaAction:
    return JointPositionDeltaAction(self, env)


class JointPositionDeltaAction(BaseAction):
  """Control joints via normalized command deltas integrated on previous command."""

  cfg: JointPositionDeltaActionCfg

  def __init__(self, cfg: JointPositionDeltaActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    if cfg.delta_max <= cfg.delta_min:
      raise ValueError(
        f"delta_max must be > delta_min, got ({cfg.delta_min}, {cfg.delta_max})"
      )
    if cfg.scale != 1.0 or cfg.offset != 0.0:
      raise ValueError(
        "JointPositionDeltaAction expects raw normalized action in [-1, 1]. "
        "Set scale=1.0 and offset=0.0."
      )

    self._target = torch.zeros_like(self._raw_actions)
    # Interpolation state.
    self._prev_target = torch.zeros_like(self._raw_actions)
    self._substep_counter = 0
    self._decimation = env.cfg.decimation
    self._initialize_target(slice(None))

  def _joint_limits(
    self, env_ids: torch.Tensor | slice
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if self.cfg.use_soft_joint_pos_limits:
      limits = self._entity.data.soft_joint_pos_limits
    else:
      limits = self._entity.data.joint_pos_limits
    lower = limits[env_ids][:, self._target_ids, 0]
    upper = limits[env_ids][:, self._target_ids, 1]
    return lower, upper

  def _apply_target_limits(self, env_ids: torch.Tensor | slice) -> None:
    if not self.cfg.clip_to_joint_limits:
      return
    lower, upper = self._joint_limits(env_ids)
    self._target[env_ids] = torch.clamp(self._target[env_ids], min=lower, max=upper)

  def _initialize_target(self, env_ids: torch.Tensor | slice) -> None:
    if self.cfg.use_default_offset:
      source = self._entity.data.default_joint_pos
    else:
      source = self._entity.data.joint_pos
    self._target[env_ids] = source[env_ids][:, self._target_ids]
    self._apply_target_limits(env_ids)
    self._processed_actions[env_ids] = self._target[env_ids]
    self._prev_target[env_ids] = self._target[env_ids]

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    # Save current target as the interpolation start point.
    self._prev_target[:] = self._target
    self._substep_counter = 0
    normalized = torch.clamp(self._raw_actions, min=-1.0, max=1.0)
    delta = self.cfg.delta_min + 0.5 * (normalized + 1.0) * (
      self.cfg.delta_max - self.cfg.delta_min
    )
    self._target = self._target + delta
    self._apply_target_limits(slice(None))
    self._processed_actions = self._target

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._initialize_target(env_ids)

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    if self.cfg.interpolate_decimation:
      alpha = (self._substep_counter + 1) / self._decimation
      interp = (1.0 - alpha) * self._prev_target + alpha * self._processed_actions
      target = interp - encoder_bias
      self._substep_counter += 1
    else:
      target = self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
