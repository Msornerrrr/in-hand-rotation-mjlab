from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_inv, wrap_to_pi

from .commands import GOAL

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def actuator_work_l2_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
  use_joint_acc: bool = False,
  max_abs_power: float | None = None,
  max_abs_torque: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """Penalize squared actuation work-like signal.

  By default this uses tau*qdot. For LEAP-style behavior, set
  ``use_joint_acc=True`` to use tau*qacc (their finite-difference proxy).
  Optional clipping prevents rare physics spikes from dominating rewards.
  """
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.joint_ids]
  if max_abs_torque is not None:
    tau = torch.clamp(tau, min=-max_abs_torque, max=max_abs_torque)
  motion_term = (
    asset.data.joint_acc[:, asset_cfg.joint_ids]
    if use_joint_acc
    else asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  mech_power = torch.sum(tau * motion_term, dim=-1)
  if max_abs_power is not None:
    mech_power = torch.clamp(mech_power, min=-max_abs_power, max=max_abs_power)
  work = torch.square(mech_power)
  if normalize and max_abs_power is not None and max_abs_power > 0.0:
    work = work / (max_abs_power**2)
  return work


class object_yaw_finite_diff_clipped:
  """Yaw finite-difference reward term with optional temporal averaging.

  This computes signed yaw-rate from wrapped finite differences and can smooth
  it over the latest ``history_steps`` transitions:

    yaw_rate_t = ±mean_t-H+1..t(delta_yaw) / dt

  The sign is controlled by ``negate_yaw_rate``:
  - For left hand (clockwise rotation positive): negate_yaw_rate=True
  - For right hand (counter-clockwise rotation positive): negate_yaw_rate=False

  Optionally, the yaw reward is multiplied by a drift factor based on object
  translation and roll/pitch deviation from reset pose. This suppresses reward
  when the cube drifts via non-target motion.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self.history_steps = int(cfg.params.get("history_steps", 1))
    if self.history_steps < 1:
      raise ValueError(f"history_steps must be >= 1, got {self.history_steps}")
    self.prev_yaw = torch.zeros(env.num_envs, device=env.device)
    self._env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    self._delta_hist = torch.zeros(
      (env.num_envs, self.history_steps), device=env.device
    )
    self._hist_ptr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._hist_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._env = env
    self.reset(None)

  def _current_yaw(self) -> torch.Tensor:
    cube: Entity = self._env.scene[self.object_name]
    _, _, yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)
    return yaw

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    cube: Entity = self._env.scene[self.object_name]
    yaw = self._current_yaw()
    self.prev_yaw[env_ids] = yaw[env_ids]
    self._delta_hist[env_ids] = 0.0
    self._hist_ptr[env_ids] = 0
    self._hist_count[env_ids] = 0
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    self._init_pos_w[env_ids] = cube.data.root_link_pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    history_steps: int | None = None,
    negate_yaw_rate: bool = True,
    absolute_yaw_rate: bool = False,
    drift_position_threshold: float = 0.02,
    drift_tilt_threshold: float = 0.35,
    drift_inside_factor: float = 1.0,
    drift_outside_factor: float = 0.1,
    drift_mode: str = "step",
    drift_decay_k_pos: float = 30.0,
    drift_decay_k_tilt: float = 3.0,
    drift_min_factor: float = 0.1,
  ) -> torch.Tensor:
    del object_name  # Bound in __init__ for consistency across calls.
    if history_steps is not None and history_steps != self.history_steps:
      raise ValueError(
        f"history_steps={history_steps} does not match initialized value "
        f"{self.history_steps}. Set it via reward params."
      )

    yaw = self._current_yaw()
    # On first post-reset step, anchor the finite-diff reference.
    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
      self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))

    delta_yaw = wrap_to_pi(yaw - self.prev_yaw)
    self.prev_yaw[:] = yaw

    # Rolling mean over latest history_steps finite differences.
    hist_ptr = self._hist_ptr
    self._delta_hist[self._env_ids, hist_ptr] = delta_yaw
    self._hist_ptr = (hist_ptr + 1) % self.history_steps
    self._hist_count = torch.clamp(self._hist_count + 1, max=self.history_steps)

    hist_len = torch.clamp(self._hist_count, min=1).to(delta_yaw.dtype)
    avg_delta_yaw = self._delta_hist.sum(dim=-1) / hist_len

    # Compute yaw rate with optional negation based on hand embodiment.
    # Left hand: negate_yaw_rate=True (clockwise rotation -> positive reward)
    # Right hand: negate_yaw_rate=False (counter-clockwise rotation -> positive reward)
    yaw_rate = avg_delta_yaw / max(env.step_dt, 1e-6)
    if negate_yaw_rate:
      yaw_rate = -yaw_rate
    if absolute_yaw_rate:
      yaw_rate = yaw_rate.abs()
    yaw_reward = torch.clamp(yaw_rate, min=clip_min, max=clip_max)

    cube: Entity = self._env.scene[self.object_name]
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(
      cube.data.root_link_pos_w - self._init_pos_w, dim=-1
    )
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )

    if drift_mode == "step":
      in_bounds = (pos_error <= drift_position_threshold) & (
        tilt_error <= drift_tilt_threshold
      )
      drift_factor = torch.where(
        in_bounds,
        torch.full_like(yaw_reward, drift_inside_factor),
        torch.full_like(yaw_reward, drift_outside_factor),
      )
    elif drift_mode == "exp":
      pos_excess = torch.clamp(pos_error - drift_position_threshold, min=0.0)
      tilt_excess = torch.clamp(tilt_error - drift_tilt_threshold, min=0.0)
      drift_factor = torch.exp(
        -drift_decay_k_pos * pos_excess - drift_decay_k_tilt * tilt_excess
      )
      drift_factor = torch.clamp(drift_factor, min=drift_min_factor, max=1.0)
    else:
      raise ValueError(
        f"Unknown drift_mode '{drift_mode}'. Expected one of: 'step', 'exp'."
      )

    drift_factor = torch.where(
      self._has_init, drift_factor, torch.ones_like(drift_factor)
    )
    return yaw_reward * drift_factor


class object_abs_cumulative_yaw_progress:
  """Reward new net yaw rotation progress in either direction.

  Unlike absolute instantaneous yaw speed, this only rewards increases in the
  episode's best absolute accumulated yaw. Back-and-forth swaying earns reward
  on the first swing, then stops paying unless the object reaches a new net
  rotation amount.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self.prev_yaw = torch.zeros(env.num_envs, device=env.device)
    self.cumulative_yaw = torch.zeros(env.num_envs, device=env.device)
    self.best_abs_yaw = torch.zeros(env.num_envs, device=env.device)
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._env = env
    self.reset(None)

  def _current_yaw(self) -> torch.Tensor:
    cube: Entity = self._env.scene[self.object_name]
    _, _, yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)
    return yaw

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    yaw = self._current_yaw()
    cube: Entity = self._env.scene[self.object_name]
    self.prev_yaw[env_ids] = yaw[env_ids]
    self.cumulative_yaw[env_ids] = 0.0
    self.best_abs_yaw[env_ids] = 0.0
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    self._init_pos_w[env_ids] = cube.data.root_link_pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    clip_max: float = 0.5,
    drift_position_threshold: float | None = None,
    drift_tilt_threshold: float | None = None,
    drift_inside_factor: float = 1.0,
    drift_outside_factor: float = 0.0,
    drift_mode: str = "step",
    drift_decay_k_pos: float = 30.0,
    drift_decay_k_tilt: float = 3.0,
    drift_min_factor: float = 0.0,
  ) -> torch.Tensor:
    del object_name  # Bound in __init__ for consistency across calls.

    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
      self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))

    yaw = self._current_yaw()
    delta_yaw = wrap_to_pi(yaw - self.prev_yaw)
    self.prev_yaw[:] = yaw
    self.cumulative_yaw += delta_yaw

    abs_yaw = self.cumulative_yaw.abs()
    progress = torch.clamp(abs_yaw - self.best_abs_yaw, min=0.0)
    self.best_abs_yaw = torch.maximum(self.best_abs_yaw, abs_yaw)

    progress_rate = progress / max(env.step_dt, 1e-6)
    reward = torch.clamp(progress_rate, max=clip_max)
    if drift_position_threshold is None and drift_tilt_threshold is None:
      return reward

    cube: Entity = self._env.scene[self.object_name]
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(
      cube.data.root_link_pos_w - self._init_pos_w, dim=-1
    )
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )

    pos_threshold = (
      float("inf") if drift_position_threshold is None else drift_position_threshold
    )
    tilt_threshold = (
      float("inf") if drift_tilt_threshold is None else drift_tilt_threshold
    )
    if drift_mode == "step":
      in_bounds = (pos_error <= pos_threshold) & (tilt_error <= tilt_threshold)
      drift_factor = torch.where(
        in_bounds,
        torch.full_like(reward, drift_inside_factor),
        torch.full_like(reward, drift_outside_factor),
      )
    elif drift_mode == "exp":
      pos_excess = torch.clamp(pos_error - pos_threshold, min=0.0)
      tilt_excess = torch.clamp(tilt_error - tilt_threshold, min=0.0)
      drift_factor = torch.exp(
        -drift_decay_k_pos * pos_excess - drift_decay_k_tilt * tilt_excess
      )
      drift_factor = torch.clamp(drift_factor, min=drift_min_factor, max=1.0)
    else:
      raise ValueError(
        f"Unknown drift_mode '{drift_mode}'. Expected one of: 'step', 'exp'."
      )

    drift_factor = torch.where(
      self._has_init, drift_factor, torch.ones_like(drift_factor)
    )
    return reward * drift_factor


class object_abs_cumulative_palm_axis_rotation_progress:
  """Reward new net object rotation about one palm-frame axis in either direction."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self.hand_cfg: SceneEntityCfg = cfg.params["hand_cfg"]
    self.axis_index = int(cfg.params.get("axis_index", 1))
    if self.axis_index not in (0, 1, 2):
      raise ValueError(f"axis_index must be 0, 1, or 2, got {self.axis_index}")
    self.cumulative_angle = torch.zeros(env.num_envs, device=env.device)
    self.best_abs_angle = torch.zeros(env.num_envs, device=env.device)
    self._env = env
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.cumulative_angle[env_ids] = 0.0
    self.best_abs_angle[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    hand_cfg: SceneEntityCfg,
    axis_index: int = 1,
    clip_max: float = 0.5,
  ) -> torch.Tensor:
    del object_name, hand_cfg, axis_index  # Bound in __init__ for stable IDs.

    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
      self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))

    hand: Entity = self._env.scene[self.hand_cfg.name]
    cube: Entity = self._env.scene[self.object_name]
    palm_quat_w = hand.data.body_link_quat_w[:, self.hand_cfg.body_ids].squeeze(1)
    palm_ang_vel_w = hand.data.body_link_vel_w[:, self.hand_cfg.body_ids].squeeze(1)[:, 3:]
    cube_ang_vel_w = cube.data.root_link_ang_vel_w
    rel_ang_vel_palm = quat_apply(quat_inv(palm_quat_w), cube_ang_vel_w - palm_ang_vel_w)

    delta_angle = rel_ang_vel_palm[:, self.axis_index] * env.step_dt
    self.cumulative_angle += delta_angle

    abs_angle = self.cumulative_angle.abs()
    progress = torch.clamp(abs_angle - self.best_abs_angle, min=0.0)
    self.best_abs_angle = torch.maximum(self.best_abs_angle, abs_angle)

    progress_rate = progress / max(env.step_dt, 1e-6)
    return torch.clamp(progress_rate, max=clip_max)


def object_linvel_l1(
  env: ManagerBasedRlEnv,
  object_name: str,
  clip_max: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """LEAPSim linear velocity penalty term: ||v_obj||_1."""
  cube: Entity = env.scene[object_name]
  val = torch.norm(cube.data.root_link_lin_vel_w, p=1, dim=-1)
  if clip_max is not None:
    val = torch.clamp(val, max=clip_max)
  if normalize and clip_max is not None and clip_max > 0.0:
    val = val / clip_max
  return val


def object_angvel_l1(
  env: ManagerBasedRlEnv,
  object_name: str,
  clip_max: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """Object angular velocity penalty term: ||w_obj||_1."""
  cube: Entity = env.scene[object_name]
  val = torch.norm(cube.data.root_link_ang_vel_w, p=1, dim=-1)
  if clip_max is not None:
    val = torch.clamp(val, max=clip_max)
  if normalize and clip_max is not None and clip_max > 0.0:
    val = val / clip_max
  return val


class object_position_l2_from_reset:
  """Object position deviation from its episode reset pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self.cube: Entity = env.scene[self.object_name]
    self.init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.init_pos_w[env_ids] = self.cube.data.root_link_pos_w[env_ids]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    tolerance: float = 0.0,
    clip_max: float | None = None,
    normalize: bool = False,
  ) -> torch.Tensor:
    del env, object_name
    pos_diff = torch.linalg.vector_norm(
      self.cube.data.root_link_pos_w - self.init_pos_w, dim=-1
    )
    if tolerance > 0.0:
      pos_diff = torch.clamp(pos_diff - tolerance, min=0.0)
    val = torch.square(pos_diff)
    if clip_max is not None:
      val = torch.clamp(val, max=clip_max)
    if normalize and clip_max is not None and clip_max > 0.0:
      val = val / clip_max
    return val


def fingertip_cube_contact_fraction(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contacts: int = 3,
) -> torch.Tensor:
  """Contact reward in [0, 1], saturating at max_contacts fingertips.

  Full reward (1.0) is given when max_contacts (default 3) or more fingertips are
  in contact, discouraging the policy from needing all 4 fingertips simultaneously.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  count = torch.sum(in_contact, dim=-1)
  return torch.clamp(count, max=max_contacts) / max_contacts


def object_fallen(
  env: ManagerBasedRlEnv,
  object_name: str,
  minimum_height: float,
) -> torch.Tensor:
  """Indicator that object center fell below minimum height."""
  cube: Entity = env.scene[object_name]
  return (cube.data.root_link_pos_w[:, 2] < minimum_height).float()


def joint_torque_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
  max_abs_torque: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """LEAPSim torque penalty term: sum(tau^2)."""
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.joint_ids]
  if max_abs_torque is not None:
    tau = torch.clamp(tau, min=-max_abs_torque, max=max_abs_torque)
  val = torch.sum(torch.square(tau), dim=-1)
  if normalize:
    if max_abs_torque is not None and max_abs_torque > 0.0:
      denom = tau.shape[1] * (max_abs_torque**2)
    else:
      denom = max(tau.shape[1], 1)
    val = val / denom
  return val


class pose_diff_l2_from_reset:
  """Pose-diff penalty from reset with optional per-joint dead-zone."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self.asset: Entity = env.scene[self.asset_cfg.name]
    num_joints = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids].shape[1]
    self.init_joint_pos = torch.zeros(
      (env.num_envs, num_joints),
      device=env.device,
      dtype=self.asset.data.joint_pos.dtype,
    )
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.init_joint_pos[env_ids] = self.asset.data.joint_pos[env_ids][
      :, self.asset_cfg.joint_ids
    ]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    average_per_joint: bool = True,
    joint_tolerance: float = 0.0,
    clip_max: float | None = None,
    normalize: bool = False,
  ) -> torch.Tensor:
    del env, asset_cfg
    q = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids]
    joint_diff = torch.abs(q - self.init_joint_pos)
    if joint_tolerance > 0.0:
      # Ignore small deviations around reset; only penalize excess beyond tolerance.
      joint_diff = torch.clamp(joint_diff - joint_tolerance, min=0.0)
    val = torch.sum(torch.square(joint_diff), dim=-1)
    if average_per_joint:
      val = val / max(q.shape[1], 1)
    if clip_max is not None:
      val = torch.clamp(val, max=clip_max)
    if normalize and clip_max is not None and clip_max > 0.0:
      val = val / clip_max
    return val

# ---------------------------------------------------------------------------
# SO(3) cube reorientation (reorient_env_cfg.py)
class goal_rot_progress:
  """Per-step reduction of the rotation distance to the goal (0 on the step a new goal appears)."""

  def __init__(self, cfg: RewardTermCfg, env):
    self.prev = torch.zeros(env.num_envs, device=env.device)
    self.prev_goal = torch.zeros(env.num_envs, 4, device=env.device)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.prev_goal[ids] = 0.0

  def __call__(self, env, command_name: str = GOAL, clip: float = 0.2):
    term = env.command_manager.get_term(command_name)
    d = term.dist()
    same = (term.goal == self.prev_goal).all(-1)
    r = torch.where(same, (self.prev - d).clamp(-clip, clip), torch.zeros_like(d))
    self.prev, self.prev_goal = d.clone(), term.goal.clone()
    return r


def goal_success(env, command_name: str = GOAL):
  term = env.command_manager.get_term(command_name)
  return (term.dist() < term.cfg.success_threshold).float()


def goal_rot_dist(env, command_name: str = GOAL):
  return env.command_manager.get_term(command_name).dist()


class cube_pos_from_start:
  """Squared cube displacement from where it is on the first step of the episode.

  ``object_position_l2_from_reset`` captures its reference inside the reset block, where
  kinematics are still stale (the previous episode's final pose); this re-captures it on
  the first step, like the ``cube_far`` termination does.
  """

  def __init__(self, cfg: RewardTermCfg, env):
    self.cube = env.scene["cube"]
    self.ref = torch.zeros(env.num_envs, 3, device=env.device)
    # rsl-rl randomizes the initial episode lengths, so the first episode may never see
    # step 1: capture the reference on the first call as well
    self.has_ref = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None):
    pass

  def __call__(self, env):
    pos = self.cube.data.root_link_pos_w
    first = (env.episode_length_buf <= 1) | ~self.has_ref
    self.ref[first] = pos[first]
    self.has_ref |= first
    return ((pos - self.ref) ** 2).sum(-1)


class cube_pos_from_home:
  """Squared cube distance from its nominal resting position on the palm (per cube size).

  Anchoring to the episode start lets the cube walk off over many goals (each episode
  re-anchors, and expert starts may already be off-center); this keeps pulling it back.
  """

  def __init__(self, cfg: RewardTermCfg, env):
    self.cube = env.scene["cube"]
    self.home = torch.tensor(cfg.params["home"], device=env.device)
    self.origins = env.scene.env_origins

  def reset(self, env_ids=None):
    pass

  def __call__(self, env, home, margin: float = 0.0):
    d2 = ((self.cube.data.root_link_pos_w - self.origins - self.home) ** 2).sum(-1)
    if margin > 0.0:  # only the excess beyond ``margin`` (a steep wall against excursions)
      return (d2.sqrt() - margin).clamp(min=0) ** 2
    return d2


def peak_force_excess_sq(env, key: str, free: float):
  peak = getattr(env, key, None)
  if peak is None:
    return torch.zeros(env.num_envs, device=env.device)
  setattr(env, key, None)
  return (peak - free).clamp(min=0) ** 2


def contact_force_excess_sq(env, sensor_name: str, free: float):
  f = env.scene.sensors[sensor_name].data.force.norm(dim=-1).amax(-1)
  return (f - free).clamp(min=0) ** 2


class torque_near_limit:
  """Sum over actuators of the squared excess of |torque| / the torque it can produce beyond ``frac``.

  That torque is the IdealPd force limit, or for the LEAP gear servo the lead-clamp-limited motor
  torque (gear_servo.actuator_effort_limits): either way, how hard the motor is driven against
  what it can sustain.
  """

  def __init__(self, cfg: RewardTermCfg, env):
    self.robot = env.scene["robot"]

  def reset(self, env_ids=None):
    pass

  def __call__(self, env, frac: float):
    from in_hand_rotation_mjlab.robots.leap_hand.gear_servo import actuator_effort_limits

    eff = actuator_effort_limits(self.robot)
    return ((self.robot.data.actuator_force.abs() / eff.clamp(min=1e-6)) - frac).clamp(min=0).pow(2).sum(-1)


def cube_angvel_excess_sq(env, free: float):
  return (env.scene["cube"].data.root_link_ang_vel_w.norm(dim=-1) - free).clamp(min=0) ** 2
