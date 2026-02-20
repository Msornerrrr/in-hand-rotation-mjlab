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

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


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
          position=self._env.scene.env_origins[env_id],
          rotation_matrix=self._identity_rotm,
          scale=viz.world_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.world_axis_colors,
          label=f"world_{env_id}",
        )

      if viz.show_palm_body_frame:
        visualizer.add_frame(
          position=palm_pos_w[env_id],
          rotation_matrix=palm_rotm_w[env_id],
          scale=viz.palm_body_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.palm_body_axis_colors,
          label=f"palm_body_{env_id}",
        )

      if viz.show_palm_center_frame:
        visualizer.add_frame(
          position=palm_center_pos_w[env_id],
          rotation_matrix=palm_rotm_w[env_id],
          scale=viz.palm_center_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.palm_center_axis_colors,
          label=f"palm_center_{env_id}",
        )

      if viz.show_cube_frame:
        visualizer.add_frame(
          position=cube_pos_w[env_id],
          rotation_matrix=cube_rotm_w[env_id],
          scale=viz.cube_frame_scale,
          axis_radius=viz.axis_radius,
          axis_colors=viz.cube_axis_colors,
          label=f"cube_{env_id}",
        )

      if viz.show_palm_to_cube_arrow:
        visualizer.add_arrow(
          start=palm_center_pos_w[env_id],
          end=cube_pos_w[env_id],
          color=viz.palm_to_cube_arrow_color,
          width=viz.axis_radius,
          label=f"palm_to_cube_{env_id}",
        )

      if viz.show_frame_origins:
        visualizer.add_sphere(
          center=palm_center_pos_w[env_id],
          radius=viz.origin_sphere_radius,
          color=viz.palm_center_origin_color,
          label=f"palm_center_origin_{env_id}",
        )
        visualizer.add_sphere(
          center=cube_pos_w[env_id],
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
