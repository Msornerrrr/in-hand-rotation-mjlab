from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_inv,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_HAND_CFG = SceneEntityCfg("robot", body_names=("palm",))
_DEFAULT_PALM_CENTER_GEOM_EXPR = "palm_collision_.*"
_DEFAULT_JOINT_ASSET_CFG = SceneEntityCfg("robot", joint_names=(".*",))


def _joint_position_command(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  action_name: str = "joint_pos",
) -> torch.Tensor:
  """Resolve commanded joint position for selected joints.

  Prefer the action term's internal command target (for delta-position action),
  and fall back to ``joint_pos_target`` when unavailable.
  """
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  term = env.action_manager.get_term(action_name)

  target_ids = getattr(term, "target_ids", None)
  command_value = getattr(term, "_target", None)
  if command_value is None:
    command_value = getattr(term, "_processed_actions", None)
  if target_ids is None or command_value is None:
    return asset.data.joint_pos_target[:, jnt_ids]

  target_ids = target_ids.to(device=env.device, dtype=torch.long)
  commanded_full = asset.data.joint_pos.clone()
  commanded_full[:, target_ids] = command_value
  return commanded_full[:, jnt_ids]


def joint_pos_commanded(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  asset_cfg: SceneEntityCfg = _DEFAULT_JOINT_ASSET_CFG,
) -> torch.Tensor:
  """Commanded joint positions from the action controller."""
  return _joint_position_command(env=env, asset_cfg=asset_cfg, action_name=action_name)


def joint_pos_command_error(
  env: ManagerBasedRlEnv,
  action_name: str = "joint_pos",
  biased: bool = True,
  asset_cfg: SceneEntityCfg = _DEFAULT_JOINT_ASSET_CFG,
) -> torch.Tensor:
  """Joint tracking error: commanded position minus measured position."""
  asset: Entity = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  commanded = _joint_position_command(
    env=env,
    asset_cfg=asset_cfg,
    action_name=action_name,
  )
  measured = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return commanded - measured[:, jnt_ids]


def palm_center_pose_w(
  env: ManagerBasedRlEnv,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = _DEFAULT_PALM_CENTER_GEOM_EXPR,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return palm-center position and palm-body orientation in world frame.

  Palm-center is computed as the centroid of matched palm collision geoms.
  Falls back to palm body origin if no geoms match.
  """
  hand: Entity = env.scene[hand_cfg.name]
  palm_pos_w = hand.data.body_link_pos_w[:, hand_cfg.body_ids].squeeze(1)
  palm_quat_w = hand.data.body_link_quat_w[:, hand_cfg.body_ids].squeeze(1)

  cache_key = (
    f"_hand_cube_palm_center_geom_ids::{hand_cfg.name}::{palm_center_geom_expr}"
  )
  palm_geom_ids = getattr(env, cache_key, None)
  if palm_geom_ids is None:
    palm_geom_ids, _ = hand.find_geoms(palm_center_geom_expr, preserve_order=True)
    palm_geom_ids = torch.tensor(palm_geom_ids, dtype=torch.long, device=env.device)
    setattr(env, cache_key, palm_geom_ids)

  if palm_geom_ids.numel() == 0:
    return palm_pos_w, palm_quat_w

  palm_center_w = hand.data.geom_pos_w[:, palm_geom_ids].mean(dim=1)
  return palm_center_w, palm_quat_w


def cube_pose_in_palm_frame(
  env: ManagerBasedRlEnv,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = _DEFAULT_PALM_CENTER_GEOM_EXPR,
) -> torch.Tensor:
  """Cube pose in palm-center frame: [x, y, z, qw, qx, qy, qz]."""
  cube: Entity = env.scene[object_name]
  palm_center_w, palm_quat_w = palm_center_pose_w(
    env=env,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  q_inv = quat_inv(palm_quat_w)
  cube_pos_w = cube.data.root_link_pos_w
  cube_quat_w = cube.data.root_link_quat_w
  pos_palm = quat_apply(q_inv, cube_pos_w - palm_center_w)
  quat_palm = quat_mul(q_inv, cube_quat_w)
  return torch.cat([pos_palm, quat_palm], dim=-1)


def cube_lin_vel_in_palm_frame(
  env: ManagerBasedRlEnv,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = _DEFAULT_PALM_CENTER_GEOM_EXPR,
) -> torch.Tensor:
  hand: Entity = env.scene[hand_cfg.name]
  cube: Entity = env.scene[object_name]
  palm_pos_w = hand.data.body_link_pos_w[:, hand_cfg.body_ids].squeeze(1)
  palm_quat_w = hand.data.body_link_quat_w[:, hand_cfg.body_ids].squeeze(1)
  palm_vel_w = hand.data.body_link_vel_w[:, hand_cfg.body_ids].squeeze(1)
  palm_lin_vel_w = palm_vel_w[:, :3]
  palm_ang_vel_w = palm_vel_w[:, 3:]

  palm_center_w, _ = palm_center_pose_w(
    env=env,
    hand_cfg=hand_cfg,
    palm_center_geom_expr=palm_center_geom_expr,
  )
  center_offset_w = palm_center_w - palm_pos_w
  palm_center_lin_vel_w = palm_lin_vel_w + torch.cross(
    palm_ang_vel_w, center_offset_w, dim=-1
  )

  cube_lin_vel_w = cube.data.root_link_lin_vel_w
  rel_lin_vel_w = cube_lin_vel_w - palm_center_lin_vel_w
  return quat_apply(quat_inv(palm_quat_w), rel_lin_vel_w)


def cube_ang_vel_in_palm_frame(
  env: ManagerBasedRlEnv,
  object_name: str,
  hand_cfg: SceneEntityCfg = _DEFAULT_HAND_CFG,
  palm_center_geom_expr: str = _DEFAULT_PALM_CENTER_GEOM_EXPR,
) -> torch.Tensor:
  del palm_center_geom_expr  # Orientation is defined by palm body frame.
  hand: Entity = env.scene[hand_cfg.name]
  cube: Entity = env.scene[object_name]
  palm_quat_w = hand.data.body_link_quat_w[:, hand_cfg.body_ids].squeeze(1)
  palm_ang_vel_w = hand.data.body_link_vel_w[:, hand_cfg.body_ids].squeeze(1)[:, 3:]
  cube_ang_vel_w = cube.data.root_link_ang_vel_w
  rel_ang_vel_w = cube_ang_vel_w - palm_ang_vel_w
  return quat_apply(quat_inv(palm_quat_w), rel_ang_vel_w)


def cube_size(
  env: ManagerBasedRlEnv,
  object_name: str,
  geom_name: str = "cube_geom",
) -> torch.Tensor:
  """Cube half-size from the MuJoCo geom_size field (shape: [num_envs, 1])."""
  cube: Entity = env.scene[object_name]
  cache_key = f"_hand_cube_size_geom_id::{object_name}::{geom_name}"
  geom_world_id = getattr(env, cache_key, None)
  if geom_world_id is None:
    geom_local_ids, _ = cube.find_geoms(geom_name, preserve_order=True)
    if len(geom_local_ids) != 1:
      raise ValueError(
        f"Expected exactly one cube geom matching '{geom_name}', got {len(geom_local_ids)}."
      )
    geom_world_id = int(cube.indexing.geom_ids[geom_local_ids[0]].item())
    setattr(env, cache_key, geom_world_id)

  size = env.sim.model.geom_size[:, geom_world_id, 0]
  return size.unsqueeze(-1)


def cube_mass(
  env: ManagerBasedRlEnv,
  object_name: str,
  body_name: str = "cube",
) -> torch.Tensor:
  """Cube mass from MuJoCo body_mass field (shape: [num_envs, 1])."""
  cube: Entity = env.scene[object_name]
  cache_key = f"_hand_cube_mass_body_id::{object_name}::{body_name}"
  body_world_id = getattr(env, cache_key, None)
  if body_world_id is None:
    body_local_ids, _ = cube.find_bodies(body_name, preserve_order=True)
    if len(body_local_ids) != 1:
      raise ValueError(
        f"Expected exactly one cube body matching '{body_name}', got {len(body_local_ids)}."
      )
    body_world_id = int(cube.indexing.body_ids[body_local_ids[0]].item())
    setattr(env, cache_key, body_world_id)

  mass = env.sim.model.body_mass[:, body_world_id]
  return mass.unsqueeze(-1)


def cube_com_offset_b(
  env: ManagerBasedRlEnv,
  object_name: str,
  body_name: str = "cube",
) -> torch.Tensor:
  """Cube body COM offset in body frame from MuJoCo body_ipos (shape: [N, 3])."""
  cube: Entity = env.scene[object_name]
  cache_key = f"_hand_cube_com_body_id::{object_name}::{body_name}"
  body_world_id = getattr(env, cache_key, None)
  if body_world_id is None:
    body_local_ids, _ = cube.find_bodies(body_name, preserve_order=True)
    if len(body_local_ids) != 1:
      raise ValueError(
        f"Expected exactly one cube body matching '{body_name}', got {len(body_local_ids)}."
      )
    body_world_id = int(cube.indexing.body_ids[body_local_ids[0]].item())
    setattr(env, cache_key, body_world_id)

  return env.sim.model.body_ipos[:, body_world_id, :]


def cube_friction_coeff(
  env: ManagerBasedRlEnv,
  object_name: str,
  geom_name: str = "cube_geom",
  axis: int = 0,
) -> torch.Tensor:
  """Cube geom friction coefficient for selected axis (shape: [num_envs, 1])."""
  cube: Entity = env.scene[object_name]
  cache_key = f"_hand_cube_friction_geom_id::{object_name}::{geom_name}"
  geom_world_id = getattr(env, cache_key, None)
  if geom_world_id is None:
    geom_local_ids, _ = cube.find_geoms(geom_name, preserve_order=True)
    if len(geom_local_ids) != 1:
      raise ValueError(
        f"Expected exactly one cube geom matching '{geom_name}', got {len(geom_local_ids)}."
      )
    geom_world_id = int(cube.indexing.geom_ids[geom_local_ids[0]].item())
    setattr(env, cache_key, geom_world_id)

  friction = env.sim.model.geom_friction[:, geom_world_id, axis]
  return friction.unsqueeze(-1)
