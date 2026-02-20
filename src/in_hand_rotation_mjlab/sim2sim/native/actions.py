from __future__ import annotations

from typing import Any, Protocol

import mujoco
import torch

from mjlab.actuator import DelayedActuatorCfg, IdealPdActuatorCfg
from mjlab.envs.mdp.actions import (
  JointEffortActionCfg,
  JointPositionActionCfg,
  JointPositionDeltaActionCfg,
  JointVelocityActionCfg,
)
from mjlab.utils.lab_api.string import resolve_matching_names, resolve_matching_names_values

from in_hand_rotation_mjlab.policy_server.contracts import ActionTermMetadata
from in_hand_rotation_mjlab.sim2sim.native.state import _NativeState


class _ActionTerm(Protocol):
  action_dim: int

  def reset(self, state: _NativeState) -> None: ...

  def process(self, raw_action: torch.Tensor) -> None: ...

  def set_raw_action(self, raw_action: torch.Tensor) -> None: ...

  def set_commanded_joint_pos(self, command: torch.Tensor) -> None: ...

  def apply_substep(self, data: mujoco.MjData, substep_idx: int, decimation: int) -> None: ...

  @property
  def raw_action(self) -> torch.Tensor: ...

  @property
  def target_local_joint_ids(self) -> torch.Tensor: ...

  def commanded_joint_pos(self) -> torch.Tensor: ...


class _BaseJointActionTerm:
  def __init__(
    self,
    cfg: JointPositionActionCfg | JointPositionDeltaActionCfg | JointVelocityActionCfg | JointEffortActionCfg,
    state: _NativeState,
  ):
    self.cfg = cfg
    self.state = state
    self.device = state.device
    self.entity_name = cfg.entity_name
    self.entity = state.scene.entities[cfg.entity_name]
    self.entity_rt = state.entity(cfg.entity_name)

    _, target_names = resolve_matching_names(
      cfg.actuator_names,
      self.entity_rt.actuated_joint_names_in_order,
      preserve_order=cfg.preserve_order,
    )
    if len(target_names) == 0:
      raise ValueError(
        f"Action term for entity '{cfg.entity_name}' resolved no joints "
        f"from pattern(s): {cfg.actuator_names}"
      )
    self.target_names = target_names
    joint_name_to_local_id = {n: i for i, n in enumerate(self.entity_rt.joint_names)}
    self.target_local_joint_ids = torch.tensor(
      [joint_name_to_local_id[n] for n in target_names],
      device=self.device,
      dtype=torch.long,
    )
    self.ctrl_ids = torch.tensor(
      [self.entity_rt.joint_name_to_ctrl_id[n] for n in target_names],
      device=self.device,
      dtype=torch.long,
    )

    biastype = torch.as_tensor(
      state.model.actuator_biastype,
      device=self.device,
      dtype=torch.int32,
    )[self.ctrl_ids]
    all_none = torch.all(biastype == int(mujoco.mjtBias.mjBIAS_NONE))
    all_affine = torch.all(biastype == int(mujoco.mjtBias.mjBIAS_AFFINE))
    if not (all_none or all_affine):
      raise NotImplementedError(
        "Native sim2sim requires homogeneous actuator bias types per action term "
        "(all motor or all position)."
      )
    self._action_dim = len(target_names)
    self._uses_motor_pd = bool(all_none.item())
    self._pd_kp: torch.Tensor | None = None
    self._pd_kd: torch.Tensor | None = None
    self._pd_effort_limit: torch.Tensor | None = None
    self._pd_kp_default: torch.Tensor | None = None
    self._pd_kd_default: torch.Tensor | None = None
    self._pd_effort_limit_default: torch.Tensor | None = None
    if self._uses_motor_pd:
      self._init_motor_pd_from_entity_cfg()

    self._raw_action = torch.zeros((1, self._action_dim), device=self.device)
    self._processed_action = torch.zeros_like(self._raw_action)

    self.scale = self._resolve_scale(cfg.scale)
    self.offset = self._resolve_offset(cfg.offset)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  @property
  def target_local_joint_ids(self) -> torch.Tensor:
    return self._target_local_joint_ids

  @target_local_joint_ids.setter
  def target_local_joint_ids(self, value: torch.Tensor) -> None:
    self._target_local_joint_ids = value

  def commanded_joint_pos(self) -> torch.Tensor:
    # Match training observation semantics: commanded joint positions are kept
    # in the policy command frame (before encoder-bias compensation).
    return self._processed_action

  def _resolve_scale(self, scale: float | dict[str, float]) -> torch.Tensor:
    if isinstance(scale, (float, int)):
      return torch.full((1, self._action_dim), float(scale), device=self.device)
    out = torch.ones((1, self._action_dim), device=self.device)
    idx, _, values = resolve_matching_names_values(scale, self.target_names)
    out[:, idx] = torch.tensor(values, device=self.device, dtype=torch.float32)
    return out

  def _resolve_offset(self, offset: float | dict[str, float]) -> torch.Tensor:
    if isinstance(offset, (float, int)):
      return torch.full((1, self._action_dim), float(offset), device=self.device)
    out = torch.zeros((1, self._action_dim), device=self.device)
    idx, _, values = resolve_matching_names_values(offset, self.target_names)
    out[:, idx] = torch.tensor(values, device=self.device, dtype=torch.float32)
    return out

  def reset(self, state: _NativeState) -> None:
    del state
    self._raw_action.zero_()
    self._processed_action.zero_()

  def process(self, raw_action: torch.Tensor) -> None:
    self._raw_action = raw_action.to(device=self.device)
    self._processed_action = self._raw_action * self.scale + self.offset

  def set_raw_action(self, raw_action: torch.Tensor) -> None:
    action = raw_action.to(device=self.device)
    if action.shape != (1, self._action_dim):
      raise ValueError(
        f"Invalid raw action shape for '{self.entity_name}'. "
        f"Expected (1, {self._action_dim}), got {tuple(action.shape)}."
      )
    self._raw_action = action.clone()

  def set_commanded_joint_pos(self, command: torch.Tensor) -> None:
    target = command.to(device=self.device)
    if target.shape != (1, self._action_dim):
      raise ValueError(
        f"Invalid commanded joint position shape for '{self.entity_name}'. "
        f"Expected (1, {self._action_dim}), got {tuple(target.shape)}."
      )
    self._processed_action = target.clone()

  def _write_ctrl(self, data: mujoco.MjData, command: torch.Tensor) -> None:
    data.ctrl[self.ctrl_ids.cpu().numpy()] = command.squeeze(0).cpu().numpy()

  def _init_motor_pd_from_entity_cfg(self) -> None:
    articulation = self.entity.cfg.articulation
    if articulation is None:
      raise ValueError(
        f"Entity '{self.entity_name}' has motor actuators but no articulation config."
      )

    kp = torch.full((1, self._action_dim), float("nan"), device=self.device)
    kd = torch.full((1, self._action_dim), float("nan"), device=self.device)
    effort_limit = torch.full((1, self._action_dim), float("nan"), device=self.device)

    def _unwrap_cfg(cfg: Any) -> Any:
      # DelayedActuatorCfg wraps the underlying actuator in base_cfg.
      while isinstance(cfg, DelayedActuatorCfg):
        cfg = cfg.base_cfg
      return cfg

    for actuator_cfg in articulation.actuators:
      base_cfg = _unwrap_cfg(actuator_cfg)
      if not isinstance(base_cfg, IdealPdActuatorCfg):
        continue
      local_ids, _ = resolve_matching_names(
        base_cfg.target_names_expr,
        self.target_names,
        preserve_order=False,
      )
      if len(local_ids) == 0:
        continue
      ids_t = torch.tensor(local_ids, device=self.device, dtype=torch.long)
      kp[:, ids_t] = float(base_cfg.stiffness)
      kd[:, ids_t] = float(base_cfg.damping)
      effort_limit[:, ids_t] = float(base_cfg.effort_limit)

    if torch.any(torch.isnan(kp)) or torch.any(torch.isnan(kd)) or torch.any(
      torch.isnan(effort_limit)
    ):
      raise ValueError(
        f"Failed to resolve IdealPd gains/limits for action term on '{self.entity_name}'."
      )

    self._pd_kp = kp
    self._pd_kd = kd
    self._pd_effort_limit = effort_limit
    # Preserve per-joint defaults for reset/startup "scale" randomization semantics.
    self._pd_kp_default = kp.clone()
    self._pd_kd_default = kd.clone()
    self._pd_effort_limit_default = effort_limit.clone()

  def _write_motor_pd_ctrl(
    self,
    data: mujoco.MjData,
    position_target: torch.Tensor,
  ) -> None:
    if self._pd_kp is None or self._pd_kd is None or self._pd_effort_limit is None:
      raise RuntimeError("Motor PD parameters are not initialized.")

    encoder_bias = self.entity_rt.encoder_bias[:, self.target_local_joint_ids]
    target = position_target - encoder_bias
    joint_pos = self.state.joint_pos(self.entity_name)[:, self.target_local_joint_ids]
    joint_vel = self.state.joint_vel(self.entity_name)[:, self.target_local_joint_ids]
    pos_error = target - joint_pos
    vel_error = -joint_vel
    effort = self._pd_kp * pos_error + self._pd_kd * vel_error
    effort = torch.clamp(
      effort,
      min=-self._pd_effort_limit,
      max=self._pd_effort_limit,
    )
    self._write_ctrl(data, effort)


class _JointPositionActionTerm(_BaseJointActionTerm):
  cfg: JointPositionActionCfg

  def __init__(self, cfg: JointPositionActionCfg, state: _NativeState):
    super().__init__(cfg, state)
    if cfg.use_default_offset:
      self.offset = self.entity_rt.default_joint_pos[:, self.target_local_joint_ids]

  def apply_substep(
    self,
    data: mujoco.MjData,
    substep_idx: int,
    decimation: int,
  ) -> None:
    del substep_idx, decimation
    if self._uses_motor_pd:
      self._write_motor_pd_ctrl(data, self._processed_action)
    else:
      encoder_bias = self.entity_rt.encoder_bias[:, self.target_local_joint_ids]
      self._write_ctrl(data, self._processed_action - encoder_bias)


class _JointVelocityActionTerm(_BaseJointActionTerm):
  cfg: JointVelocityActionCfg

  def __init__(self, cfg: JointVelocityActionCfg, state: _NativeState):
    super().__init__(cfg, state)
    if cfg.use_default_offset:
      self.offset = self.entity_rt.default_joint_vel[:, self.target_local_joint_ids]

  def apply_substep(
    self,
    data: mujoco.MjData,
    substep_idx: int,
    decimation: int,
  ) -> None:
    del substep_idx, decimation
    self._write_ctrl(data, self._processed_action)

  def set_commanded_joint_pos(self, command: torch.Tensor) -> None:
    del command
    raise TypeError(
      "Cannot apply server joint-position commands to a joint-velocity action term."
    )


class _JointEffortActionTerm(_BaseJointActionTerm):
  cfg: JointEffortActionCfg

  def apply_substep(
    self,
    data: mujoco.MjData,
    substep_idx: int,
    decimation: int,
  ) -> None:
    del substep_idx, decimation
    self._write_ctrl(data, self._processed_action)

  def set_commanded_joint_pos(self, command: torch.Tensor) -> None:
    del command
    raise TypeError(
      "Cannot apply server joint-position commands to a joint-effort action term."
    )


class _JointPositionDeltaActionTerm(_BaseJointActionTerm):
  cfg: JointPositionDeltaActionCfg

  def __init__(self, cfg: JointPositionDeltaActionCfg, state: _NativeState):
    super().__init__(cfg, state)
    if cfg.delta_max <= cfg.delta_min:
      raise ValueError(
        f"delta_max must be > delta_min, got ({cfg.delta_min}, {cfg.delta_max})."
      )
    if not torch.allclose(self.scale, torch.ones_like(self.scale)):
      raise ValueError(
        "JointPositionDeltaAction expects scale=1.0 for native sim2sim."
      )
    if not torch.allclose(self.offset, torch.zeros_like(self.offset)):
      raise ValueError(
        "JointPositionDeltaAction expects offset=0.0 for native sim2sim."
      )

    self._target = torch.zeros((1, self.action_dim), device=self.device)
    self._prev_target = torch.zeros_like(self._target)
    self._substep_counter = 0
    self._initialize_target(state)

  def _joint_limits(self) -> torch.Tensor:
    if self.cfg.use_soft_joint_pos_limits:
      return self.entity_rt.soft_joint_pos_limits[self.target_local_joint_ids]
    return self.entity_rt.hard_joint_pos_limits[self.target_local_joint_ids]

  def _apply_target_limits(self) -> None:
    if not self.cfg.clip_to_joint_limits:
      return
    limits = self._joint_limits()
    self._target = torch.clamp(self._target, min=limits[:, 0], max=limits[:, 1])

  def _initialize_target(self, state: _NativeState) -> None:
    if self.cfg.use_default_offset:
      source = self.entity_rt.default_joint_pos
    else:
      source = state.joint_pos(self.entity_name)
    self._target = source[:, self.target_local_joint_ids].clone()
    self._apply_target_limits()
    self._prev_target = self._target.clone()
    self._processed_action = self._target.clone()
    self._substep_counter = 0

  def reset(self, state: _NativeState) -> None:
    super().reset(state)
    self._initialize_target(state)

  def process(self, raw_action: torch.Tensor) -> None:
    self._raw_action = raw_action.to(device=self.device)
    self._prev_target = self._target.clone()
    self._substep_counter = 0
    normalized = torch.clamp(self._raw_action, min=-1.0, max=1.0)
    delta = self.cfg.delta_min + 0.5 * (normalized + 1.0) * (
      self.cfg.delta_max - self.cfg.delta_min
    )
    self._target = self._target + delta
    self._apply_target_limits()
    self._processed_action = self._target.clone()

  def set_commanded_joint_pos(self, command: torch.Tensor) -> None:
    target = command.to(device=self.device)
    if target.shape != (1, self.action_dim):
      raise ValueError(
        "Invalid commanded joint position shape for delta term. "
        f"Expected (1, {self.action_dim}), got {tuple(target.shape)}."
      )
    self._prev_target = self._target.clone()
    self._target = target.clone()
    self._apply_target_limits()
    self._processed_action = self._target.clone()
    self._substep_counter = 0

  def apply_substep(
    self,
    data: mujoco.MjData,
    substep_idx: int,
    decimation: int,
  ) -> None:
    del substep_idx
    if self.cfg.interpolate_decimation:
      alpha = (self._substep_counter + 1) / decimation
      command = (1.0 - alpha) * self._prev_target + alpha * self._processed_action
      self._substep_counter += 1
    else:
      command = self._processed_action
    if self._uses_motor_pd:
      self._write_motor_pd_ctrl(data, command)
    else:
      encoder_bias = self.entity_rt.encoder_bias[:, self.target_local_joint_ids]
      self._write_ctrl(data, command - encoder_bias)


class _ActionAdapter:
  def __init__(self, actions_cfg: dict[str, Any], state: _NativeState):
    self.device = state.device
    self._term_names: list[str] = []
    self._terms: list[_ActionTerm] = []
    self._term_slices: dict[str, slice] = {}

    offset = 0
    for term_name, term_cfg in actions_cfg.items():
      term = self._build_term(term_cfg, state)
      self._term_names.append(term_name)
      self._terms.append(term)
      self._term_slices[term_name] = slice(offset, offset + term.action_dim)
      offset += term.action_dim
    self.total_action_dim = offset
    self.action = torch.zeros((1, self.total_action_dim), device=self.device)

  def _build_term(self, term_cfg: Any, state: _NativeState) -> _ActionTerm:
    if isinstance(term_cfg, JointPositionDeltaActionCfg):
      return _JointPositionDeltaActionTerm(term_cfg, state)
    if isinstance(term_cfg, JointPositionActionCfg):
      return _JointPositionActionTerm(term_cfg, state)
    if isinstance(term_cfg, JointVelocityActionCfg):
      return _JointVelocityActionTerm(term_cfg, state)
    if isinstance(term_cfg, JointEffortActionCfg):
      return _JointEffortActionTerm(term_cfg, state)
    raise NotImplementedError(
      f"Unsupported action term in native sim2sim: {type(term_cfg).__name__}"
    )

  def reset(self, state: _NativeState) -> None:
    self.action.zero_()
    for term in self._terms:
      term.reset(state)

  def process_action(self, action: torch.Tensor) -> None:
    action = action.to(device=self.device)
    if action.ndim == 1:
      action = action.unsqueeze(0)
    if action.shape != (1, self.total_action_dim):
      raise ValueError(
        "Invalid policy action shape. "
        f"Expected (1, {self.total_action_dim}), got {tuple(action.shape)}."
      )
    self.action = action
    for term_name, term in zip(self._term_names, self._terms, strict=True):
      sl = self._term_slices[term_name]
      term.process(action[:, sl])

  def apply_server_action(
    self,
    raw_action: torch.Tensor,
    commanded_joint_pos: dict[str, torch.Tensor],
  ) -> None:
    raw_action = raw_action.to(device=self.device)
    if raw_action.ndim == 1:
      raw_action = raw_action.unsqueeze(0)
    if raw_action.shape != (1, self.total_action_dim):
      raise ValueError(
        "Invalid server raw action shape. "
        f"Expected (1, {self.total_action_dim}), got {tuple(raw_action.shape)}."
      )
    self.action = raw_action
    for term_name, term in zip(self._term_names, self._terms, strict=True):
      if term_name not in commanded_joint_pos:
        raise KeyError(
          f"Missing commanded joint position for action term '{term_name}'. "
          f"Available terms: {sorted(commanded_joint_pos.keys())}"
        )
      term.set_raw_action(raw_action[:, self._term_slices[term_name]])
      term.set_commanded_joint_pos(commanded_joint_pos[term_name])

  def apply_substep(self, data: mujoco.MjData, substep_idx: int, decimation: int) -> None:
    for term in self._terms:
      term.apply_substep(data, substep_idx, decimation)

  def term_raw_action(self, action_name: str) -> torch.Tensor:
    if action_name not in self._term_slices:
      raise KeyError(f"Unknown action term: {action_name}")
    sl = self._term_slices[action_name]
    return self.action[:, sl]

  def term_commanded_joint_pos(
    self,
    action_name: str,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if action_name not in self._term_slices:
      raise KeyError(f"Unknown action term: {action_name}")
    term = self._terms[self._term_names.index(action_name)]
    if not hasattr(term, "target_local_joint_ids") or not hasattr(
      term, "commanded_joint_pos"
    ):
      raise TypeError(
        f"Action term '{action_name}' does not expose joint-position commands."
      )
    return term.target_local_joint_ids, term.commanded_joint_pos()

  def term_current_joint_pos(self, action_name: str) -> torch.Tensor:
    if action_name not in self._term_slices:
      raise KeyError(f"Unknown action term: {action_name}")
    term = self._terms[self._term_names.index(action_name)]
    if not hasattr(term, "target_local_joint_ids"):
      raise TypeError(
        f"Action term '{action_name}' does not expose target joint ids."
      )
    return term.state.joint_pos(term.entity_name)[:, term.target_local_joint_ids].clone()

  def current_joint_pos_by_term(self) -> dict[str, torch.Tensor]:
    return {
      term_name: self.term_current_joint_pos(term_name) for term_name in self._term_names
    }

  def build_server_action_metadata(self) -> dict[str, ActionTermMetadata]:
    metadata: dict[str, ActionTermMetadata] = {}
    for term_name, term in zip(self._term_names, self._terms, strict=True):
      target_ids = term.target_local_joint_ids.clone()
      metadata[term_name] = ActionTermMetadata(
        term_name=term_name,
        target_joint_names=tuple(term.target_names),
        target_local_joint_ids=target_ids,
        default_joint_pos=term.entity_rt.default_joint_pos[:, target_ids].clone(),
        hard_joint_pos_limits=term.entity_rt.hard_joint_pos_limits[target_ids].clone(),
        soft_joint_pos_limits=term.entity_rt.soft_joint_pos_limits[target_ids].clone(),
      )
    return metadata
