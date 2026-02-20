from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  JointPositionDeltaActionCfg,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

from in_hand_rotation_mjlab.policy_server.contracts import ActionTermMetadata, ServerActionPacket


@dataclass
class _ActionTermSlice:
  term_name: str
  action_slice: slice
  mapper: "_BaseJointPositionMapper"


class _BaseJointPositionMapper:
  def __init__(self, cfg: JointPositionActionCfg | JointPositionDeltaActionCfg, metadata: ActionTermMetadata, device: str):
    self.cfg = cfg
    self.metadata = metadata
    self.device = device
    self.action_dim = len(metadata.target_joint_names)

    self.scale = self._resolve_scale(cfg.scale)
    self.offset = self._resolve_offset(cfg.offset)

  def _resolve_scale(self, scale: float | dict[str, float]) -> torch.Tensor:
    if isinstance(scale, (float, int)):
      return torch.full((1, self.action_dim), float(scale), device=self.device)
    out = torch.ones((1, self.action_dim), device=self.device)
    idx, _, values = resolve_matching_names_values(scale, list(self.metadata.target_joint_names))
    out[:, idx] = torch.tensor(values, device=self.device, dtype=torch.float32)
    return out

  def _resolve_offset(self, offset: float | dict[str, float]) -> torch.Tensor:
    if isinstance(offset, (float, int)):
      return torch.full((1, self.action_dim), float(offset), device=self.device)
    out = torch.zeros((1, self.action_dim), device=self.device)
    idx, _, values = resolve_matching_names_values(offset, list(self.metadata.target_joint_names))
    out[:, idx] = torch.tensor(values, device=self.device, dtype=torch.float32)
    return out

  def reset(self, current_joint_pos: torch.Tensor | None) -> None:
    del current_joint_pos

  def map_action(self, raw_action: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


class _JointPositionMapper(_BaseJointPositionMapper):
  cfg: JointPositionActionCfg

  def __init__(self, cfg: JointPositionActionCfg, metadata: ActionTermMetadata, device: str):
    super().__init__(cfg, metadata, device)
    if cfg.use_default_offset:
      self.offset = metadata.default_joint_pos.to(device=self.device)

  def map_action(self, raw_action: torch.Tensor) -> torch.Tensor:
    return raw_action * self.scale + self.offset


class _JointPositionDeltaMapper(_BaseJointPositionMapper):
  cfg: JointPositionDeltaActionCfg

  def __init__(
    self,
    cfg: JointPositionDeltaActionCfg,
    metadata: ActionTermMetadata,
    device: str,
  ):
    super().__init__(cfg, metadata, device)
    if cfg.delta_max <= cfg.delta_min:
      raise ValueError(
        f"delta_max must be > delta_min, got ({cfg.delta_min}, {cfg.delta_max})."
      )
    if not torch.allclose(self.scale, torch.ones_like(self.scale)):
      raise ValueError(
        "JointPositionDeltaAction expects scale=1.0 for server-side mapping."
      )
    if not torch.allclose(self.offset, torch.zeros_like(self.offset)):
      raise ValueError(
        "JointPositionDeltaAction expects offset=0.0 for server-side mapping."
      )
    self._target = torch.zeros((1, self.action_dim), device=self.device)

  def _joint_limits(self) -> torch.Tensor:
    if self.cfg.use_soft_joint_pos_limits:
      return self.metadata.soft_joint_pos_limits.to(device=self.device)
    return self.metadata.hard_joint_pos_limits.to(device=self.device)

  def _apply_target_limits(self) -> None:
    if not self.cfg.clip_to_joint_limits:
      return
    limits = self._joint_limits()
    self._target = torch.clamp(self._target, min=limits[:, 0], max=limits[:, 1])

  def reset(self, current_joint_pos: torch.Tensor | None) -> None:
    if self.cfg.use_default_offset:
      self._target = self.metadata.default_joint_pos.to(device=self.device).clone()
    else:
      if current_joint_pos is None:
        raise ValueError(
          f"Missing current joint position for action term '{self.metadata.term_name}'."
        )
      self._target = current_joint_pos.to(device=self.device).clone()
    self._apply_target_limits()

  def map_action(self, raw_action: torch.Tensor) -> torch.Tensor:
    normalized = torch.clamp(raw_action, min=-1.0, max=1.0)
    delta = self.cfg.delta_min + 0.5 * (normalized + 1.0) * (
      self.cfg.delta_max - self.cfg.delta_min
    )
    self._target = self._target + delta
    self._apply_target_limits()
    return self._target.clone()


class JointPositionActionMapper:
  """Server-side policy-action to joint-position target mapper."""

  def __init__(
    self,
    actions_cfg: dict[str, Any],
    term_metadata: dict[str, ActionTermMetadata],
    device: str,
  ):
    self.device = device
    self._term_slices: dict[str, slice] = {}
    self._term_mappers: list[_ActionTermSlice] = []
    self._total_action_dim = 0

    for term_name, term_cfg in actions_cfg.items():
      if term_name not in term_metadata:
        raise KeyError(
          f"Missing action metadata for action term '{term_name}'. "
          f"Available metadata: {sorted(term_metadata.keys())}"
        )
      metadata = term_metadata[term_name]
      mapper = self._build_mapper(term_cfg, metadata)
      sl = slice(self._total_action_dim, self._total_action_dim + mapper.action_dim)
      self._term_slices[term_name] = sl
      self._term_mappers.append(
        _ActionTermSlice(term_name=term_name, action_slice=sl, mapper=mapper)
      )
      self._total_action_dim += mapper.action_dim

  @property
  def total_action_dim(self) -> int:
    return self._total_action_dim

  def _build_mapper(
    self,
    term_cfg: Any,
    metadata: ActionTermMetadata,
  ) -> _BaseJointPositionMapper:
    if isinstance(term_cfg, JointPositionDeltaActionCfg):
      return _JointPositionDeltaMapper(term_cfg, metadata, self.device)
    if isinstance(term_cfg, JointPositionActionCfg):
      return _JointPositionMapper(term_cfg, metadata, self.device)
    raise NotImplementedError(
      "Server-side joint-position mapping supports only joint-position action spaces. "
      f"Got unsupported action cfg: {type(term_cfg).__name__}"
    )

  def reset(self, current_joint_pos_by_term: dict[str, torch.Tensor]) -> None:
    for term in self._term_mappers:
      current = current_joint_pos_by_term.get(term.term_name)
      term.mapper.reset(current)

  def map_policy_action(self, raw_action: torch.Tensor) -> ServerActionPacket:
    action = raw_action.to(device=self.device).detach()
    if action.ndim == 1:
      action = action.unsqueeze(0)
    if action.shape != (1, self.total_action_dim):
      raise ValueError(
        "Invalid policy action shape for server mapping. "
        f"Expected (1, {self.total_action_dim}), got {tuple(action.shape)}."
      )

    commands: dict[str, torch.Tensor] = {}
    for term in self._term_mappers:
      raw_term = action[:, term.action_slice]
      commands[term.term_name] = term.mapper.map_action(raw_term)

    return ServerActionPacket(
      raw_action=action,
      joint_position_commands=commands,
    )
