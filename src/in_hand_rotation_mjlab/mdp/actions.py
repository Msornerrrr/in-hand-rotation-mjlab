"""The LEAP hand's joint action, shared by every task.

Normalized deltas integrated on the previous command (a persistent setpoint decoupled from
the measured joints), with per-joint delta ranges, an optional in-loop action EMA,
sub-step setpoint interpolation and the command-lead clamp that the real LEAP driver applies
(``max_command_lead``; robots/leap_hand/leap_plant.py LEAP_DRIVER_LEAD_CLAMP).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseActionCfg, BaseAction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "JointPositionDeltaAction",
  "JointPositionDeltaActionCfg",
  "resolve_delta_ranges",
  "resolve_ema_alphas",
  "resolve_max_command_lead",
  "resolve_target_limit_overrides",
]


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

  delta_overrides: dict[str, tuple[float, float]] | None = None
  """Optional per-joint (delta_min, delta_max) overrides keyed by joint-name
  regex, e.g. ``{"RHand_WR[ZY]_joint": (-0.05, 0.05)}``. Joints not matched by
  any pattern keep (delta_min, delta_max)."""

  interpolate_decimation: bool = False
  """If True, linearly interpolate the position target across decimation
  substeps instead of applying a constant target. This ramps the setpoint
  from the previous env-step target to the current one, producing smoother
  torques and reducing sim-to-real jerkiness."""

  ema_alpha: float = 1.0
  """In-loop EMA smoothing on the normalized [-1,1] action BEFORE the delta
  decode: ``a_ema = ema_alpha*a + (1-ema_alpha)*a_ema_prev``. 1.0 (default) = OFF
  (passthrough). Smaller = smoother (0.2 ~= 5-step memory). Structural jitter
  suppression that a gradient-dead action-rate penalty can't provide. Per-env
  state, reset to 0 (neutral) on episode reset. NOTE: this is part of the action
  DECODE -- deployment-side decoders must replicate the
  same stateful EMA or a trained policy will slew joints differently on hardware."""

  ema_overrides: dict[str, float] | None = None
  """Optional per-joint EMA alphas keyed by joint-name regex.

  This lets a heavy wrist carrier use more smoothing than the fingers without
  splitting or reordering the checkpoint-compatible hand action vector. Joints
  not matched by an override keep ``ema_alpha``.
  """

  max_command_lead: float | None = None
  """Hard cap [rad] on |position target - measured joint angle| ("command lead").

  None (default) = OFF: the setpoint integrates freely past a blocked joint, so
  grip force builds without bound up to the actuator effort cap. That is the
  historical behaviour, and on a soft sim plant it is how policies learn to
  grip at all -- but the lead they learn (measured 0.1-0.8 rad) is an
  interference fit on the real, far stiffer hand: instant rated-torque
  clamping, excessive contact force, and a violent snap when contact slips.

  When set, the clamp is applied ANTI-WINDUP style to the stored integrator
  state, not merely to the emitted command: a target that would sit beyond the
  window is pulled back to the window edge and *kept* there, so releasing takes
  one step instead of unwinding many. Static joint torque is then bounded by
  ``max_command_lead * Kp * Kv``; choose it as a force budget.

  Credit-assignment note: clamping is degenerate only where the dynamics are
  already flat. Pick a window at or above the torque-saturation lead
  (cap/(Kp*Kv)) so every distinguishable force level stays reachable inside it;
  beyond saturation, extra lead changes no physical quantity anyway.

  DEPLOYMENT: like the delta ranges and the EMA, this is part of the action
  decode. Every deployment decoder must apply the identical clamp against
  measured joint angles, or the hardware policy will command leads the trained
  policy never could.
  """

  max_command_lead_overrides: dict[str, float] | None = None
  """Optional per-joint command-lead caps [rad] keyed by joint-name regex.

  Joints not matched by any pattern keep ``max_command_lead``. Only meaningful
  when ``max_command_lead`` is set."""

  target_limit_overrides: dict[str, tuple[float | None, float | None]] | None = None
  """Optional per-joint (lower, upper) target bounds [rad] keyed by joint-name
  regex, intersected with the (soft) joint limits the target is clamped to; None
  leaves that side at the joint limit. For a joint the hardware reaches less far
  than its model range (e.g. the real LEAP ring splay, ``joint_8``, stalls at
  ~-0.73 rad while the MJCF allows -1.047). Only applied with
  ``clip_to_joint_limits``; every deployment decoder must apply the same bounds."""

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> JointPositionDeltaAction:
    return JointPositionDeltaAction(self, env)


def resolve_delta_ranges(
  cfg: JointPositionDeltaActionCfg,
  target_names: tuple[str, ...] | list[str],
  device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-joint (delta_min, delta_max) tensors honoring ``cfg.delta_overrides``.

  Single source of truth for the action decode: the training env action term
  AND every deployment-side decoder must map normalized actions through the SAME per-joint ranges,
  or trained policies slew joints at the wrong rate outside the trainer.
  """
  num_targets = len(target_names)
  delta_min = torch.full((num_targets,), cfg.delta_min, device=device)
  delta_max = torch.full((num_targets,), cfg.delta_max, device=device)
  overrides = getattr(cfg, "delta_overrides", None)
  if overrides:
    for pattern, (lo, hi) in overrides.items():
      if hi <= lo:
        raise ValueError(
          f"delta_overrides[{pattern!r}] must have max > min, got ({lo}, {hi})"
        )
      matched = [
        i for i, name in enumerate(target_names) if re.fullmatch(pattern, name)
      ]
      if not matched:
        raise ValueError(
          f"delta_overrides pattern {pattern!r} matches no target joint "
          f"(targets: {list(target_names)})"
        )
      for i in matched:
        delta_min[i] = lo
        delta_max[i] = hi
  return delta_min, delta_max


def resolve_max_command_lead(
  cfg: JointPositionDeltaActionCfg,
  target_names: tuple[str, ...] | list[str],
  device: str | torch.device = "cpu",
) -> torch.Tensor | None:
  """Per-joint command-lead cap tensor, or None when the clamp is off.

  Shared decode contract, exactly like ``resolve_delta_ranges``: a deployment
  decoder that skips this clamp lets hardware command leads the trained policy
  was structurally unable to produce.
  """
  base = getattr(cfg, "max_command_lead", None)
  overrides = getattr(cfg, "max_command_lead_overrides", None)
  if base is None and not overrides:
    return None
  if base is not None and base <= 0.0:
    raise ValueError(f"max_command_lead must be > 0, got {base}")
  # An override-only config leaves unmatched joints unclamped (inf).
  fill = float("inf") if base is None else float(base)
  lead = torch.full((len(target_names),), fill, device=device)
  if overrides:
    for pattern, value in overrides.items():
      if value <= 0.0:
        raise ValueError(
          f"max_command_lead_overrides[{pattern!r}] must be > 0, got {value}"
        )
      matched = [
        i for i, name in enumerate(target_names) if re.fullmatch(pattern, name)
      ]
      if not matched:
        raise ValueError(
          f"max_command_lead_overrides pattern {pattern!r} matches no target "
          f"joint (targets: {list(target_names)})"
        )
      for i in matched:
        lead[i] = value
  return lead


def resolve_ema_alphas(
  cfg: JointPositionDeltaActionCfg,
  target_names: tuple[str, ...] | list[str],
  device: str | torch.device = "cpu",
) -> torch.Tensor:
  """Per-joint EMA-alpha tensor honoring ``cfg.ema_overrides``."""
  if not 0.0 < cfg.ema_alpha <= 1.0:
    raise ValueError(f"ema_alpha must be in (0, 1], got {cfg.ema_alpha}")
  alphas = torch.full((len(target_names),), cfg.ema_alpha, device=device)
  overrides = getattr(cfg, "ema_overrides", None)
  if overrides:
    for pattern, alpha in overrides.items():
      if not 0.0 < alpha <= 1.0:
        raise ValueError(
          f"ema_overrides[{pattern!r}] must be in (0, 1], got {alpha}"
        )
      matched = [
        i for i, name in enumerate(target_names) if re.fullmatch(pattern, name)
      ]
      if not matched:
        raise ValueError(
          f"ema_overrides pattern {pattern!r} matches no target joint "
          f"(targets: {list(target_names)})"
        )
      for i in matched:
        alphas[i] = alpha
  return alphas


def resolve_target_limit_overrides(
  cfg: JointPositionDeltaActionCfg,
  target_names: tuple[str, ...] | list[str],
  device: str | torch.device = "cpu",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  """Per-target (lower, upper) bounds from ``cfg.target_limit_overrides`` (-inf / +inf where unset), or
  (None, None) without overrides."""
  overrides = getattr(cfg, "target_limit_overrides", None)
  if not overrides:
    return None, None
  lo = torch.full((len(target_names),), -torch.inf, device=device)
  hi = torch.full((len(target_names),), torch.inf, device=device)
  for pattern, (a, b) in overrides.items():
    if a is not None and b is not None and b <= a:
      raise ValueError(f"target_limit_overrides[{pattern!r}] must have upper > lower, got ({a}, {b})")
    matched = [i for i, name in enumerate(target_names) if re.fullmatch(pattern, name)]
    if not matched:
      raise ValueError(
        f"target_limit_overrides pattern {pattern!r} matches no target joint (targets: {list(target_names)})"
      )
    for i in matched:
      if a is not None:
        lo[i] = a
      if b is not None:
        hi[i] = b
  return lo, hi


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
    # Per-joint delta ranges: scalar defaults, then regex overrides.
    self._delta_min, self._delta_max = resolve_delta_ranges(
      cfg, self._target_names, device=self._raw_actions.device
    )
    # Interpolation state.
    self._prev_target = torch.zeros_like(self._raw_actions)
    # Demo resets run before ActionManager.reset(). The reset event can stage a
    # captured command target here so contact-force command lead survives that
    # manager reset; ordinary resets leave the mask false.
    self._reset_target_override = torch.zeros_like(self._raw_actions)
    self._reset_target_override_mask = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self._raw_actions.device
    )
    self._substep_counter = 0
    self._decimation = env.cfg.decimation
    # In-loop action EMA state (per-env, per-joint). Reset to 0 each episode.
    self._ema_alpha = resolve_ema_alphas(
      cfg, self._target_names, device=self._raw_actions.device
    )
    self._ema_enabled = bool(torch.any(self._ema_alpha < 1.0))
    self._ema_action = torch.zeros_like(self._raw_actions)
    # Optional anti-windup command-lead clamp (None = off).
    self._max_lead = resolve_max_command_lead(
      cfg, self._target_names, device=self._raw_actions.device
    )
    self._limit_lo, self._limit_hi = resolve_target_limit_overrides(
      cfg, self._target_names, device=self._raw_actions.device
    )
    # Set by a training perturbation event (e.g. leap_mdp.leap_holder_miss): (num_envs, action_dim) raw actions that
    # replace the policy's at the next process_actions, NaN = keep the policy's; None = off (deployment never sets it).
    self.forced_raw: torch.Tensor | None = None
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
    if self._limit_lo is not None:
      lower = torch.maximum(lower, self._limit_lo)
      upper = torch.minimum(upper, self._limit_hi)
    return lower, upper

  def _apply_target_limits(self, env_ids: torch.Tensor | slice) -> None:
    if not self.cfg.clip_to_joint_limits:
      return
    lower, upper = self._joint_limits(env_ids)
    self._target[env_ids] = torch.clamp(self._target[env_ids], min=lower, max=upper)

  def _apply_command_lead_limit(self, env_ids: torch.Tensor | slice) -> None:
    """Pull the stored target back into the per-joint lead window around q.

    Anti-windup: this rewrites ``_target`` itself, so a blocked joint cannot
    accumulate lead it would have to unwind before it can release.
    """
    if self._max_lead is None:
      return
    q = self._entity.data.joint_pos[env_ids][:, self._target_ids]
    lead = torch.clamp(self._target[env_ids] - q, -self._max_lead, self._max_lead)
    self._target[env_ids] = q + lead

  def _initialize_target(self, env_ids: torch.Tensor | slice) -> None:
    if self.cfg.use_default_offset:
      source = self._entity.data.default_joint_pos
    else:
      source = self._entity.data.joint_pos
    self._target[env_ids] = source[env_ids][:, self._target_ids]
    self._apply_command_lead_limit(env_ids)
    self._apply_target_limits(env_ids)
    self._processed_actions[env_ids] = self._target[env_ids]
    self._prev_target[env_ids] = self._target[env_ids]
    # Clear the action-EMA state so the filter never carries a stale command
    # across an episode boundary.
    if hasattr(self, "_ema_action"):
      self._ema_action[env_ids] = 0.0

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    forced = getattr(self, "forced_raw", None)  # (terms built around __init__, e.g. contract doubles, lack it)
    if forced is not None:
      self._raw_actions[:] = torch.where(torch.isnan(forced), self._raw_actions, forced)
    # Save current target as the interpolation start point.
    self._prev_target[:] = self._target
    self._substep_counter = 0
    normalized = torch.clamp(self._raw_actions, min=-1.0, max=1.0)
    # In-loop EMA on the normalized action (structural smoothness); alpha==1.0
    # -> passthrough. Applied before the delta decode so it slews the joint
    # setpoints smoothly instead of chattering.
    if self._ema_enabled:
      self._ema_action.mul_(1.0 - self._ema_alpha).add_(
        normalized * self._ema_alpha
      )
      normalized = self._ema_action
    delta = self._delta_min + 0.5 * (normalized + 1.0) * (
      self._delta_max - self._delta_min
    )
    self._target = self._target + delta
    # Lead clamp first, then joint limits: q is inside the joint limits, so the
    # limit clamp can only pull the target back toward q, i.e. shrink the lead.
    # Both invariants therefore hold after this pair, in this order only.
    self._apply_command_lead_limit(slice(None))
    self._apply_target_limits(slice(None))
    self._processed_actions = self._target

  def stage_restored_reset_target(self, env_ids: torch.Tensor) -> None:
    """Preserve the currently restored command through ActionManager.reset()."""
    env_ids = env_ids.to(device=self._raw_actions.device, dtype=torch.long)
    self._reset_target_override[env_ids] = self._target[env_ids]
    self._reset_target_override_mask[env_ids] = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    reset_ids = env_ids
    if reset_ids is None or isinstance(reset_ids, slice):
      all_ids = torch.arange(self.num_envs, device=self._raw_actions.device)
      selected_ids = all_ids if reset_ids is None else all_ids[reset_ids]
    else:
      selected_ids = reset_ids.to(device=self._raw_actions.device, dtype=torch.long)
    override_mask = self._reset_target_override_mask[selected_ids]
    override_ids = selected_ids[override_mask]
    override_targets = self._reset_target_override[override_ids].clone()

    super().reset(env_ids)
    self._initialize_target(slice(None) if env_ids is None else env_ids)
    if len(override_ids) > 0:
      # Demo-override envs keep the captured joint target (command-lead) instead
      # of the physics-seeded one. INVARIANT: correct only if the demo state
      # restored _target -- restore_state_batch writes it `if key in states`.
      # pool_v1 carries hand_target; a pool omitting it would jerk the hand to
      # a stale target here, so any new demo source must include it.
      self._target[override_ids] = override_targets
      # Demo pools were materialized WITHOUT the lead clamp (and on a softer
      # plant), so their captured targets can carry leads far outside the
      # window. Re-clamp here or every clamped episode would start in
      # violation, at cap force, from step zero.
      self._apply_command_lead_limit(override_ids)
      self._apply_target_limits(override_ids)
      self._processed_actions[override_ids] = self._target[override_ids]
      self._prev_target[override_ids] = self._target[override_ids]
    self._reset_target_override_mask[selected_ids] = False

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    if self.cfg.interpolate_decimation:
      alpha = (self._substep_counter + 1) / self._decimation
      interp = (1.0 - alpha) * self._prev_target + alpha * self._processed_actions
      target = interp - encoder_bias
      self._substep_counter += 1
    else:
      target = self._processed_actions - encoder_bias
    if self._max_lead is not None:
      # The integrator clamp in process_actions runs once per CONTROL step; the
      # physics then takes `decimation` substeps during which contact can push
      # the joint away from its target, re-opening lead the policy never
      # commanded. Re-clamp the emitted setpoint against live q so the bound
      # holds at every substep, not just at command time.
      # Free by construction on the fingers: static torque is lead*Kp*Kv, so a
      # 0.1 rad window allows 1.5-9.1 N.m while the effort cap is 1.24 N.m --
      # the cap still binds first and peak grip strength is untouched.
      q = self._entity.data.joint_pos[:, self._target_ids]
      target = q + torch.clamp(target - q, -self._max_lead, self._max_lead)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
