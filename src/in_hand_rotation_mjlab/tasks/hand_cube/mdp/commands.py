from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  matrix_from_quat,
  sample_uniform,
  wrap_to_pi,
)

from in_hand_rotation_mjlab.tasks.hand_cube.quat_utils import axis_angle_quat, qmul, random_quat, rot_distance

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _viz_array(value: torch.Tensor):
  return value.detach().cpu().numpy()


class InHandYawCommand(CommandTerm):
  cfg: InHandYawCommandCfg

  def __init__(self, cfg: InHandYawCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.target_yaw = torch.zeros(self.num_envs, device=self.device)

    self.metrics["yaw_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_yaw.unsqueeze(-1)

  def _cube_yaw(self) -> torch.Tensor:
    _, _, yaw = euler_xyz_from_quat(self.object.data.root_link_quat_w)
    return yaw

  def _update_metrics(self) -> None:
    yaw_error = wrap_to_pi(self.target_yaw - self._cube_yaw()).abs()
    self.metrics["yaw_error"] = yaw_error
    self.metrics["at_goal"] = (yaw_error < self.cfg.success_threshold).float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    current_yaw = self._cube_yaw()[env_ids]
    delta_yaw = sample_uniform(
      self.cfg.delta_yaw_range[0],
      self.cfg.delta_yaw_range[1],
      (len(env_ids),),
      device=self.device,
    )
    self.target_yaw[env_ids] = wrap_to_pi(current_yaw + delta_yaw)

  def _update_command(self) -> None:
    pass


class InHandRotationDirectionCommand(CommandTerm):
  """Command that specifies a rotation direction (+1 CCW, -1 CW) for in-hand cube rotation."""

  cfg: InHandRotationDirectionCommandCfg

  def __init__(self, cfg: InHandRotationDirectionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]

    self.direction = torch.zeros(self.num_envs, device=self.device)
    self.prev_yaw = torch.zeros(self.num_envs, device=self.device)
    self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)
    self.step_delta_yaw = torch.zeros(self.num_envs, device=self.device)

    self.metrics["total_rotation"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["directed_rotation"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.direction.unsqueeze(-1)

  def _cube_yaw(self) -> torch.Tensor:
    _, _, yaw = euler_xyz_from_quat(self.object.data.root_link_quat_w)
    return yaw

  def _update_metrics(self) -> None:
    self.metrics["total_rotation"] = self.cumulative_rotation.abs()
    self.metrics["directed_rotation"] = self.cumulative_rotation * self.direction

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.direction[env_ids] = (
      2.0 * torch.randint(0, 2, (len(env_ids),), device=self.device).float() - 1.0
    )
    self.prev_yaw[env_ids] = self._cube_yaw()[env_ids]
    self.cumulative_rotation[env_ids] = 0.0

  def _update_command(self) -> None:
    current_yaw = self._cube_yaw()
    self.step_delta_yaw = wrap_to_pi(current_yaw - self.prev_yaw)
    self.cumulative_rotation += self.step_delta_yaw
    self.prev_yaw = current_yaw.clone()


class HandCubeFrameVizCommand(CommandTerm):
  """Debug-only command term that visualizes key task coordinate frames."""

  cfg: HandCubeFrameVizCommandCfg

  def __init__(self, cfg: HandCubeFrameVizCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.hand: Entity = env.scene[cfg.hand_name]
    self.object: Entity = env.scene[cfg.object_name]

    palm_body_ids, _ = self.hand.find_bodies(cfg.palm_body_name, preserve_order=True)
    if len(palm_body_ids) != 1:
      raise ValueError(
        f"Expected exactly one palm body for pattern '{cfg.palm_body_name}', "
        f"got {len(palm_body_ids)}."
      )
    self._palm_body_id = int(palm_body_ids[0])

    palm_geom_ids, _ = self.hand.find_geoms(
      cfg.palm_center_geom_expr, preserve_order=True
    )
    self._palm_center_geom_ids = torch.tensor(
      palm_geom_ids, dtype=torch.long, device=self.device
    )

    self._command = torch.zeros((self.num_envs, 1), device=self.device)
    self._identity_rotm = torch.eye(3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _palm_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
    palm_pos_w = self.hand.data.body_link_pos_w[:, self._palm_body_id]
    palm_quat_w = self.hand.data.body_link_quat_w[:, self._palm_body_id]
    return palm_pos_w, palm_quat_w

  def _palm_center_pos_w(self, palm_pos_w: torch.Tensor) -> torch.Tensor:
    if self._palm_center_geom_ids.numel() == 0:
      return palm_pos_w
    return self.hand.data.geom_pos_w[:, self._palm_center_geom_ids].mean(dim=1)

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    palm_pos_w, palm_quat_w = self._palm_pose_w()
    palm_center_pos_w = self._palm_center_pos_w(palm_pos_w)
    cube_pos_w = self.object.data.root_link_pos_w
    cube_quat_w = self.object.data.root_link_quat_w

    palm_rotm_w = matrix_from_quat(palm_quat_w)
    cube_rotm_w = matrix_from_quat(cube_quat_w)

    viz = self.cfg.viz
    for env_id in env_indices:
      if viz.show_world_frame:
        visualizer.add_frame(
          position=_viz_array(self._env.scene.env_origins[env_id]),
          rotation_matrix=_viz_array(self._identity_rotm),
          scale=viz.world_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.world_axis_colors,
          label=f"world_{env_id}",
        )

      if viz.show_palm_body_frame:
        visualizer.add_frame(
          position=_viz_array(palm_pos_w[env_id]),
          rotation_matrix=_viz_array(palm_rotm_w[env_id]),
          scale=viz.palm_body_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.palm_body_axis_colors,
          label=f"palm_body_{env_id}",
        )

      if viz.show_palm_center_frame:
        visualizer.add_frame(
          position=_viz_array(palm_center_pos_w[env_id]),
          rotation_matrix=_viz_array(palm_rotm_w[env_id]),
          scale=viz.palm_center_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.palm_center_axis_colors,
          label=f"palm_center_{env_id}",
        )

      if viz.show_cube_frame:
        visualizer.add_frame(
          position=_viz_array(cube_pos_w[env_id]),
          rotation_matrix=_viz_array(cube_rotm_w[env_id]),
          scale=viz.cube_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.cube_axis_colors,
          label=f"cube_{env_id}",
        )

      if viz.show_palm_to_cube_arrow:
        visualizer.add_arrow(
          start=_viz_array(palm_center_pos_w[env_id]),
          end=_viz_array(cube_pos_w[env_id]),
          color=viz.palm_to_cube_arrow_color,
          width=viz.axis_radius,
          label=f"palm_to_cube_{env_id}",
        )

      if viz.show_frame_origins:
        visualizer.add_sphere(
          center=_viz_array(palm_center_pos_w[env_id]),
          radius=viz.origin_sphere_radius,
          color=viz.palm_center_origin_color,
          label=f"palm_center_origin_{env_id}",
        )
        visualizer.add_sphere(
          center=_viz_array(cube_pos_w[env_id]),
          radius=viz.origin_sphere_radius,
          color=viz.cube_origin_color,
          label=f"cube_origin_{env_id}",
        )


@dataclass(kw_only=True)
class InHandYawCommandCfg(CommandTermCfg):
  entity_name: str
  delta_yaw_range: tuple[float, float] = (-1.57, 1.57)
  success_threshold: float = 0.15

  def build(self, env: ManagerBasedRlEnv) -> InHandYawCommand:
    return InHandYawCommand(self, env)


@dataclass(kw_only=True)
class InHandRotationDirectionCommandCfg(CommandTermCfg):
  entity_name: str

  def build(self, env: ManagerBasedRlEnv) -> InHandRotationDirectionCommand:
    return InHandRotationDirectionCommand(self, env)


@dataclass(kw_only=True)
class HandCubeFrameVizCommandCfg(CommandTermCfg):
  hand_name: str = "robot"
  object_name: str = "cube"
  palm_body_name: str = "palm"
  palm_center_geom_expr: str = "palm_collision_.*"

  @dataclass
  class VizCfg:
    axis_radius: float = 0.006
    world_frame_scale: float = 0.05
    palm_body_frame_scale: float = 0.05
    palm_center_frame_scale: float = 0.075
    cube_frame_scale: float = 0.06
    origin_sphere_radius: float = 0.006

    show_world_frame: bool = True
    show_palm_body_frame: bool = True
    show_palm_center_frame: bool = True
    show_cube_frame: bool = True
    show_palm_to_cube_arrow: bool = True
    show_frame_origins: bool = True

    world_axis_colors: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
    ] = ((0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8))
    palm_body_axis_colors: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
    ] = ((1.0, 0.4, 0.4), (0.4, 1.0, 0.4), (0.4, 0.4, 1.0))
    palm_center_axis_colors: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
    ] = ((1.0, 0.65, 0.2), (0.2, 1.0, 0.8), (0.8, 0.2, 1.0))
    cube_axis_colors: tuple[
      tuple[float, float, float],
      tuple[float, float, float],
      tuple[float, float, float],
    ] = ((1.0, 0.25, 0.25), (0.25, 1.0, 0.25), (0.25, 0.6, 1.0))
    palm_to_cube_arrow_color: tuple[float, float, float, float] = (
      1.0,
      0.85,
      0.1,
      0.9,
    )
    palm_center_origin_color: tuple[float, float, float, float] = (
      1.0,
      0.65,
      0.15,
      0.85,
    )
    cube_origin_color: tuple[float, float, float, float] = (0.95, 0.3, 0.2, 0.85)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> HandCubeFrameVizCommand:
    return HandCubeFrameVizCommand(self, env)

# ---------------------------------------------------------------------------
# SO(3) cube reorientation (reorient_env_cfg.py)
GOAL = "goal"  # the reorientation goal command's name


class CubeGoalCommand(CommandTerm):
  cfg: CubeGoalCommandCfg

  def __init__(self, cfg: CubeGoalCommandCfg, env):
    super().__init__(cfg, env)
    self.cube = env.scene[cfg.entity_name]
    self.goal = torch.zeros(self.num_envs, 4, device=self.device)
    self.goal[:, 0] = 1.0
    self.metrics["rot_dist"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["successes"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["max_goal_angle"] = torch.zeros(self.num_envs, device=self.device)
    # goal-distance curriculum: goals are the current orientation rotated by an angle in
    # [min_goal_angle, max_angle]; max_angle grows while the success rate is high (pi = uniform SO(3))
    self.max_angle = cfg.init_max_angle
    self._attempts = 0.0
    self._hits = 0.0
    # Reverse curriculum (``reverse_curriculum``): ``reset_reverse_curriculum`` starts episodes
    # from expert states whose goal the expert reached within ``rc_steps`` steps; rc_steps grows
    # with the success rate on those first goals. Attempt kinds: 0 follow-up goal (angle
    # curriculum), 1 expert-state first goal, 2 scratch first goal (uniform SO(3), the real task),
    # 3 closed (counted; the next goal opens a new attempt).
    self.rc_steps = float(cfg.rc_init_steps)
    self.kind = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._kind_att = [0.0, 0.0, 0.0]
    self._kind_hit = [0.0, 0.0, 0.0]
    self._kind_rate = [0.0, 0.0, 0.0]
    self._in_reset = False
    if cfg.reverse_curriculum:
      for k in ("rc_steps", "rc_rate", "scratch_rate", "followup_rate"):
        self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.goal

  def dist(self) -> torch.Tensor:
    return rot_distance(self.cube.data.root_link_quat_w, self.goal)

  def _update_metrics(self) -> None:
    self.metrics["rot_dist"] = self.dist()
    self.metrics["max_goal_angle"][:] = self.max_angle
    if self.cfg.reverse_curriculum:
      self.metrics["rc_steps"][:] = self.rc_steps
      self.metrics["rc_rate"][:] = self._kind_rate[1]
      self.metrics["scratch_rate"][:] = self._kind_rate[2]
      self.metrics["followup_rate"][:] = self._kind_rate[0]

  def _close_attempts(self, env_ids: torch.Tensor, hit: bool) -> None:
    """Count the open goal attempts of ``env_ids`` as reached / failed, per kind."""
    if len(env_ids) == 0:
      return
    kinds = self.kind[env_ids]  # kind 3 = already closed: not counted again
    for k in range(3):
      n = float((kinds == k).sum())
      self._kind_att[k] += n
      self._kind_hit[k] += n if hit else 0.0
    self.kind[env_ids] = 3
    # rates over windows of attempts; the expert-state rate drives rc_steps
    for k in range(3):
      if self._kind_att[k] >= self.cfg.rc_window:
        self._kind_rate[k] = self._kind_hit[k] / self._kind_att[k]
        if k == 1:
          if self._kind_rate[1] > self.cfg.curriculum_up:
            self.rc_steps = min(self.cfg.rc_max_steps, self.rc_steps * self.cfg.rc_factor)
          elif self._kind_rate[1] < self.cfg.curriculum_down:
            self.rc_steps = max(self.cfg.rc_init_steps, self.rc_steps / self.cfg.rc_factor)
        self._kind_att[k] = self._kind_hit[k] = 0.0

  def reset(self, env_ids):
    if self.cfg.reverse_curriculum and env_ids is not None:
      self._close_attempts(torch.as_tensor(env_ids, device=self.device), hit=False)  # episode ended first
    self._in_reset = True
    try:
      return super().reset(env_ids)
    finally:
      self._in_reset = False

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if self.cfg.reverse_curriculum:
      if self._in_reset:
        # Episode start. Never read the cube pose here: kinematics are refreshed only after
        # the reset block. Expert-state envs get the expert's goal, the rest a uniform one.
        demo = getattr(self._env, "_rc_pending", None)
        demo = demo[env_ids] if demo is not None else torch.zeros(n, dtype=torch.bool, device=self.device)
        goal = random_quat(n, self.device)
        if demo.any():
          goal[demo] = self._env._rc_goal[env_ids[demo]]
        self.goal[env_ids] = goal
        self.kind[env_ids] = torch.where(demo, 1, 2)
        return
      self._close_attempts(env_ids, hit=False)  # timed out (reached goals were closed already)
      self.kind[env_ids] = 0
    if self.max_angle >= torch.pi - 1e-3:
      self.goal[env_ids] = random_quat(n, self.device)
      return
    axis = torch.randn(n, 3, device=self.device)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    lo = self.cfg.min_goal_angle
    hi = max(self.max_angle, lo)
    ang = lo + (hi - lo) * torch.rand(n, device=self.device)
    self.goal[env_ids] = qmul(axis_angle_quat(axis, ang), self.cube.data.root_link_quat_w[env_ids])

  def _update_command(self) -> None:
    hit = (self.dist() < self.cfg.success_threshold).nonzero().flatten()
    timed_out_mask = self.time_left <= self._env.step_dt
    if self.cfg.reverse_curriculum:  # the angle curriculum sees follow-up goals only
      self._attempts += float((self.kind[hit] == 0).sum()) + float((timed_out_mask & (self.kind == 0)).sum())
      self._hits += float((self.kind[hit] == 0).sum())
      self._close_attempts(hit, hit=True)
    else:
      self._attempts += len(hit) + timed_out_mask.sum().item()
      self._hits += len(hit)
    if self._attempts >= self.cfg.curriculum_window:
      rate = self._hits / self._attempts
      if rate > self.cfg.curriculum_up:
        self.max_angle = min(torch.pi, self.max_angle + self.cfg.curriculum_step)
      elif rate < self.cfg.curriculum_down:
        self.max_angle = max(self.cfg.init_max_angle, self.max_angle - self.cfg.curriculum_step)
      self._attempts = self._hits = 0.0
    if len(hit):
      self.metrics["successes"][hit] += 1
      self._resample(hit)


@dataclass(kw_only=True)
class CubeGoalCommandCfg(CommandTermCfg):
  entity_name: str = "cube"
  success_threshold: float = 0.4
  init_max_angle: float = 0.8  # set to pi for uniform SO(3) goals from the start (evaluation)
  min_goal_angle: float = 0.5
  curriculum_window: int = 2000  # goal attempts per curriculum decision
  curriculum_up: float = 0.5
  curriculum_down: float = 0.15
  curriculum_step: float = 0.1
  reverse_curriculum: bool = False
  rc_init_steps: float = 10.0  # expert steps-to-goal of the easiest starts (0.5 s)
  rc_max_steps: float = 400.0
  rc_factor: float = 1.25  # rc_steps *= factor above curriculum_up, /= below curriculum_down
  rc_window: int = 1000  # attempts of a kind per rate estimate / curriculum decision

  def build(self, env) -> CubeGoalCommand:
    return CubeGoalCommand(self, env)
